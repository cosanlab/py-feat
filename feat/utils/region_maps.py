"""Facial AU / ARKit-blendshape region overlays on the MediaPipe-478 mesh.

This is the dense-mesh, *non-overlapping* successor to the dlib-68 muscle
polygons in ``feat.plotting.draw_muscles`` and the overlapping muscle map in
``feat.utils.muscle_to_landmark``. It ships two maps:

* ``au_region_map.json``        — the 20 FACS AUs, left+right merged per AU.
* ``blendshape_region_map.json`` — the spatially-distinct ARKit blendshapes,
  Left/Right kept independent (ARKit pre-splits ``...Left``/``...Right``).

Both are built by a **winner-take-all geodesic-Voronoi partition** of the mesh:
each vertex is assigned to the single nearest region seed by walking the mesh
surface (Dijkstra over the triangle-edge graph, capped at ``TIGHTNESS`` of the
face span), with the eye/mouth aperture rings walling growth off. The partition
is non-overlapping *by construction*, which fixes the heavy region overlap of
the geodesic-grow muscle map (where 323/394 covered verts sat in >=2 muscles).

The AU <-> muscle <-> blendshape correspondence is grounded in:
* FACS (Ekman & Friesen) — AU -> muscle.
* Melinda Ozel's ARKit-to-FACS cheat sheet (melindaozel.com/arkit-to-facs-cheat-sheet).
* pooyadeperson's "Ultimate Guide to ARKit's 52 Facial Blendshapes" (anatomy refs).

Rendering smooths the coarse 468-vertex mesh by midpoint-subdividing it
``SUBDIV_LEVELS`` times before filling triangles. The subdivision topology and
the dense per-vertex region labels are fixed in index space, so the same assets
overlay a live detected 478-mesh — only the dense vertex *positions* are
recomputed per frame (``dense_positions``). See ``feat.plotting.plot_face_regions``.
"""

from __future__ import annotations

import copy
import heapq
import json
import os
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from feat.utils.io import get_resource_path
from feat.utils.mp_plotting import FaceLandmarksConnections as _F

# --- partition / render tuning (the user-approved look) ---
TIGHTNESS = 0.10      # max geodesic reach of a seed, as a fraction of face span
SUBDIV_LEVELS = 2     # midpoint subdivisions applied before rendering (smoothing)

AU_MAP_FILENAME = "au_region_map.json"
BLENDSHAPE_MAP_FILENAME = "blendshape_region_map.json"


# ---------------------------------------------------------------------------
# Aperture rings — the eye/mouth *openings*. Growth never enters these and no
# region may cover them (orbicularis oculi/oris are annuli around the holes).
# Eye rings come from MediaPipe's own canonical groups; the inner-lip ring has
# no dedicated group (FACE_LANDMARKS_LIPS bundles inner+outer) so it's explicit.
# ---------------------------------------------------------------------------
def _group_verts(name: str) -> set[int]:
    return {v for c in getattr(_F, name) for v in (c.start, c.end)}


_INNER_LIP = {78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
              95, 88, 178, 87, 14, 317, 402, 318, 324}
APERTURE = (_group_verts("FACE_LANDMARKS_RIGHT_EYE")
            | _group_verts("FACE_LANDMARKS_LEFT_EYE")
            | _INNER_LIP)

_OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
              146, 91, 181, 84, 17, 314, 405, 321, 375]


# ---------------------------------------------------------------------------
# Seed tables — anatomical SKIN-insertion vertices per region (MP-478 indices).
# A region = its seeds grown over the surface to the Voronoi boundary.
# ---------------------------------------------------------------------------
# muscle -> (AU, [seed verts]). Sides: _L = subject-left (263/291/336 family),
# _R = subject-right (33/61/107 family). Mirrors map_muscles_canonical_geodesic.
MUSCLE_SEEDS: dict[str, tuple[str, list[int]]] = {
    "frontalis_inner_R": ("AU01", [107, 66, 105, 69]),
    "frontalis_inner_L": ("AU01", [336, 296, 334, 299]),
    "frontalis_outer_R": ("AU02", [105, 63, 70, 71, 54, 68]),
    "frontalis_outer_L": ("AU02", [334, 293, 300, 301, 284, 298]),
    "corrugator_R": ("AU04", [107, 55, 65, 9]),
    "corrugator_L": ("AU04", [336, 285, 295, 8]),
    "procerus": ("AU04", [8, 9, 168, 6]),
    "levator_palpebrae_R": ("AU05", [223, 222, 221, 189, 56]),
    "levator_palpebrae_L": ("AU05", [443, 442, 441, 413, 286]),
    "orb_oculi_orbital_R": ("AU06", [117, 118, 119, 100, 50, 36, 111, 31]),
    "orb_oculi_orbital_L": ("AU06", [346, 347, 348, 329, 280, 266, 340, 261]),
    "orb_oculi_palpebral_R": ("AU07", [226, 110, 24, 23, 22, 26, 112, 190]),
    "orb_oculi_palpebral_L": ("AU07", [446, 339, 254, 253, 252, 256, 341, 414]),
    "llsan_R": ("AU09", [115, 48, 49, 64, 102, 219]),
    "llsan_L": ("AU09", [344, 278, 279, 294, 331, 439]),
    "levator_labii_R": ("AU10", [205, 36, 142, 203, 206, 207]),
    "levator_labii_L": ("AU10", [425, 266, 371, 423, 426, 427]),
    "zygomaticus_minor_R": ("AU11", [116, 123, 117, 50]),
    "zygomaticus_minor_L": ("AU11", [345, 352, 346, 280]),
    "zygomaticus_major_R": ("AU12", [50, 101, 205, 187, 207, 61]),
    "zygomaticus_major_L": ("AU12", [280, 330, 425, 411, 427, 291]),
    "levator_anguli_oris_R": ("AU13", [205, 203, 206, 61]),
    "levator_anguli_oris_L": ("AU13", [425, 423, 426, 291]),
    "buccinator_R": ("AU14", [212, 202, 57, 43, 204, 207]),
    "buccinator_L": ("AU14", [432, 422, 287, 273, 424, 427]),
    "depressor_anguli_oris_R": ("AU15", [43, 204, 210, 135, 169, 61]),
    "depressor_anguli_oris_L": ("AU15", [273, 424, 430, 364, 394, 291]),
    "depressor_labii_R": ("AU16", [84, 181, 91, 146, 61, 77]),
    "depressor_labii_L": ("AU16", [314, 405, 321, 375, 291, 307]),
    "mentalis": ("AU17", [175, 199, 200, 18, 83, 313, 152, 148, 377]),
    "risorius_R": ("AU20", [212, 202, 210, 169, 150]),
    "risorius_L": ("AU20", [432, 422, 430, 394, 379]),
    "orbicularis_oris": ("AU24", list(_OUTER_LIP)),
    "masseter_R": ("AU26", [58, 172, 136, 132, 215, 138]),
    "masseter_L": ("AU26", [288, 397, 365, 361, 435, 367]),
    "nasalis_R": ("AU38", [48, 49, 64, 98, 240]),
    "nasalis_L": ("AU38", [278, 279, 294, 327, 460]),
}

# ARKit blendshape -> (AU, muscle, side, [seed verts]). Side in {"L","R","C"}
# (ARKit Left/Right = subject's left/right). The 8 eyeLook* (gaze) and tongueOut
# have no facial-skin footprint and are intentionally absent. The pure
# lip-closure modes (mouthPress/Close/Roll*) deform the same orbicularis-oris
# skin as Pucker/Funnel/ShrugUpper, so they can't be a distinct cell in a
# non-overlapping partition and are folded into those lip regions.
BLENDSHAPE_SEEDS: dict[str, tuple[str, str, str, list[int]]] = {
    # seeded on the frontalis (medial forehead, ABOVE the corrugator/browDown
    # band) so the partition can separate inner-brow RAISE from browDown's LOWER.
    "browInnerUp": ("AU01", "frontalis_medial", "C", [151, 108, 337, 67, 297, 109, 338, 10]),
    "browOuterUpRight": ("AU02", "frontalis_lateral", "R", [105, 63, 70, 71, 54]),
    "browOuterUpLeft": ("AU02", "frontalis_lateral", "L", [334, 293, 300, 301, 284]),
    "browDownRight": ("AU04", "corrugator+procerus", "R", [107, 55, 65, 9, 66]),
    "browDownLeft": ("AU04", "corrugator+procerus", "L", [336, 285, 295, 8, 296]),
    "eyeWideRight": ("AU05", "levator_palpebrae", "R", [223, 222, 221, 189]),
    "eyeWideLeft": ("AU05", "levator_palpebrae", "L", [443, 442, 441, 413]),
    "cheekSquintRight": ("AU06", "orb_oculi_orbital", "R", [117, 118, 50, 101, 119]),
    "cheekSquintLeft": ("AU06", "orb_oculi_orbital", "L", [346, 347, 280, 330, 348]),
    "eyeSquintRight": ("AU07", "orb_oculi_palpebral", "R", [226, 110, 24, 23, 22, 112]),
    "eyeSquintLeft": ("AU07", "orb_oculi_palpebral", "L", [446, 339, 254, 253, 252, 341]),
    # lid SKIN just outside the eye ring (ring verts are aperture-blocked)
    "eyeBlinkRight": ("AU45", "orb_oculi_palpebral", "R", [247, 30, 29, 27, 28, 56, 112, 26, 22]),
    "eyeBlinkLeft": ("AU45", "orb_oculi_palpebral", "L", [467, 260, 259, 257, 258, 286, 341, 256, 252]),
    "noseSneerRight": ("AU09", "llsan", "R", [48, 115, 49, 64, 220]),
    "noseSneerLeft": ("AU09", "llsan", "L", [278, 344, 279, 294, 440]),
    "mouthUpperUpRight": ("AU10", "levator_labii", "R", [205, 36, 40, 39, 142]),
    "mouthUpperUpLeft": ("AU10", "levator_labii", "L", [425, 266, 270, 269, 371]),
    "mouthSmileRight": ("AU12", "zygomaticus_major", "R", [50, 101, 205, 61, 187]),
    "mouthSmileLeft": ("AU12", "zygomaticus_major", "L", [280, 330, 425, 291, 411]),
    "mouthDimpleRight": ("AU14", "buccinator", "R", [212, 57, 43, 202]),
    "mouthDimpleLeft": ("AU14", "buccinator", "L", [432, 287, 273, 422]),
    "mouthStretchRight": ("AU20", "risorius", "R", [212, 202, 210, 169]),
    "mouthStretchLeft": ("AU20", "risorius", "L", [432, 422, 430, 394]),
    "mouthFrownRight": ("AU15", "depressor_anguli_oris", "R", [43, 204, 210, 135]),
    "mouthFrownLeft": ("AU15", "depressor_anguli_oris", "L", [273, 424, 430, 364]),
    "mouthLowerDownRight": ("AU16", "depressor_labii", "R", [84, 181, 91, 146]),
    "mouthLowerDownLeft": ("AU16", "depressor_labii", "L", [314, 405, 321, 375]),
    "mouthShrugLower": ("AU17", "mentalis", "C", [175, 199, 200, 18, 83, 313]),
    "mouthShrugUpper": ("AU17", "mentalis_upper", "C", [164, 165, 391, 167, 393]),
    # NB: mouthPressLeft/Right (AU24) and other pure lip-closure modes
    # (mouthClose/Roll*) deform the SAME orbicularis-oris/lip skin as Pucker/
    # Funnel/ShrugUpper and so can't be a distinct cell in a non-overlapping
    # partition — they're represented by those lip regions.
    "mouthPucker": ("AU18", "orbicularis_oris", "C", [0, 17, 37, 267, 84, 314]),
    "mouthFunnel": ("AU22", "orbicularis_oris", "C", [61, 291, 40, 270, 91, 321]),
    "cheekPuff": ("AD34", "buccinator", "C", [212, 432, 202, 422, 57, 287]),
    "jawOpen": ("AU26", "masseter+pterygoid", "C", [152, 175, 199, 58, 288, 200]),
}


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _canonical_geometry():
    """(V[N,3] float64, tris[M,3] int) — bundled MediaPipe canonical mesh +
    its full triangulation. Natively upright (no frontalization)."""
    from feat.utils.face_pose import load_canonical_face_model

    V = load_canonical_face_model().detach().cpu().numpy().astype(np.float64)
    tess_path = os.path.join(get_resource_path(), "canonical_face_tessellation.json")
    with open(tess_path) as f:
        tris = np.array(json.load(f)["triangles"], dtype=int)
    return V, tris


def project_xy(V: np.ndarray) -> np.ndarray:
    """Frontal x/y projection, flipped so the forehead (v10) is above the chin
    (v152). ``V`` may be the canonical mesh or a detected mesh (N>=153, x/y in
    cols 0/1). Used both for geodesic distances and for plotting."""
    xy = np.asarray(V, dtype=np.float64)[:, :2].copy()
    if xy[10, 1] < xy[152, 1]:
        xy[:, 1] = -xy[:, 1]
    return xy


def build_adjacency(tris: np.ndarray, n: int) -> dict[int, list[int]]:
    adj: dict[int, set] = defaultdict(set)
    for a, b, c in tris:
        for u, w in ((a, b), (b, c), (a, c)):
            adj[int(u)].add(int(w))
            adj[int(w)].add(int(u))
    return {k: sorted(v) for k, v in adj.items()}


def geodesic_voronoi(seeds_by_region, adj, xy, aperture, max_geo):
    """Winner-take-all surface partition: assign each vertex to the nearest
    seed's region by Dijkstra over the edge graph (euclidean edge weights on
    ``xy``), capped at ``max_geo``, never crossing ``aperture`` verts.

    Returns ``{vertex_index: region_name}``. Non-overlapping by construction —
    a vertex is claimed once, by whichever region reaches it first/closest."""
    label: dict[int, str] = {}
    dist: dict[int, float] = {}
    pq: list = []
    for region, seeds in seeds_by_region.items():
        for s in seeds:
            if s in aperture or s >= len(xy):
                continue
            heapq.heappush(pq, (0.0, int(s), region))
    while pq:
        d, v, region = heapq.heappop(pq)
        if v in label and dist.get(v, np.inf) <= d:
            continue
        label[v] = region
        dist[v] = d
        for w in adj.get(v, []):
            if w in aperture:
                continue
            nd = d + float(np.linalg.norm(xy[v] - xy[w]))
            if nd <= max_geo and (w not in dist or nd < dist[w]):
                dist[w] = nd
                heapq.heappush(pq, (nd, w, region))
    return label


# vertices on the facial midline (define the mirror axis; stay put when mirrored)
_MIDLINE_SEEDS = [1, 168, 9, 8, 0, 17, 152, 10, 151, 175, 199, 200, 164, 18, 6, 4]


def _mirror_name(name):
    for a, b in (("Right", "Left"), ("Left", "Right"), ("_R", "_L"), ("_L", "_R")):
        if name.endswith(a):
            return name[: -len(a)] + b
    return name  # center / unsided


def _mirror_map(V):
    """``({v: mirror_v}, midline_set, axis_x)`` — each vertex paired with its
    nearest neighbour under reflection across the facial midline."""
    idx = [i for i in _MIDLINE_SEEDS if i < len(V)]
    axis = float(V[idx, 0].mean())
    refl = V.copy()
    refl[:, 0] = 2 * axis - refl[:, 0]
    M = cKDTree(V).query(refl)[1]
    span = float(V[:, 0].max() - V[:, 0].min())
    midline = {int(v) for v in range(len(V))
               if int(M[v]) == v or abs(V[v, 0] - axis) < 0.02 * span}
    return {int(v): int(M[v]) for v in range(len(V))}, midline, axis


def _symmetrize_partition(label, V):
    """Force exact L/R mirror symmetry on a partition: keep the subject-right
    half's assignment and mirror it onto the left (``Left``↔``Right`` swapped),
    so winner-take-all boundary noise can't make paired regions asymmetric.
    Center (unsided) regions are mirrored about the midline too.

    Returns ``{vertex: {region, ...}}``. Almost every vertex maps to a single
    region; a midline-seam vertex on a SIDED region is shared by both mirror
    partners (e.g. the nose-bridge column joins noseSneerLeft+Right) so the two
    halves stay contiguous instead of leaving an unassigned gap. ``label``/``V``
    use final region names + the same vertex set."""
    V = np.asarray(V, dtype=np.float64)
    M, midline, axis = _mirror_map(V)
    # primary half = the side a known subject-right seed (v107) sits on
    primary_neg = bool((V[107, 0] - axis) < 0) if len(V) > 107 else True
    out: dict[int, set] = defaultdict(set)
    for v, name in label.items():
        if v in midline:
            # center region keeps the seam vertex; a sided region shares it with
            # its mirror partner so L/R meet at the midline (no gap).
            out[v].add(name)
            out[v].add(_mirror_name(name))
        elif bool((V[v, 0] - axis) < 0) == primary_neg:
            out[v].add(name)
            out[M[v]].add(_mirror_name(name))
    return out


def subdivide(V, tris, aperture, levels):
    """``levels`` rounds of 1-to-4 midpoint subdivision. Original vertices keep
    their indices (so seeds and stored vertex ids stay valid). Returns
    ``(V_dense, tris_dense, aperture_dense, parents)`` where ``parents`` lists,
    in creation order, the ``(a, b)`` parent pair of each appended midpoint — so
    a deformed mesh's dense positions can be rebuilt with ``dense_positions``.
    A midpoint inherits aperture status only if BOTH parents are aperture."""
    V = [np.asarray(p, dtype=np.float64) for p in V]
    ap = set(aperture)
    parents: list[tuple[int, int]] = []
    for _ in range(levels):
        cache: dict[tuple[int, int], int] = {}
        new_tris = []

        def mid(a, b):
            k = (a, b) if a < b else (b, a)
            idx = cache.get(k)
            if idx is None:
                idx = len(V)
                cache[k] = idx
                V.append((V[a] + V[b]) / 2.0)
                parents.append((a, b))
                if a in ap and b in ap:
                    ap.add(idx)
            return idx

        for a, b, c in tris:
            a, b, c = int(a), int(b), int(c)
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_tris += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        tris = np.array(new_tris, dtype=int)
    return np.array(V), tris, ap, parents


def dense_positions(V, parents):
    """Rebuild subdivided vertex positions for an arbitrary base mesh ``V``
    (e.g. a detected 478-mesh) given the ``parents`` list from ``subdivide``.
    ``V`` supplies the original vertices; midpoints are filled by averaging."""
    pos = [np.asarray(p, dtype=np.float64) for p in V]
    for a, b in parents:
        pos.append((pos[a] + pos[b]) / 2.0)
    return np.array(pos)


# ---------------------------------------------------------------------------
# map construction (used by scripts/build_region_maps.py + the loaders' fallback)
# ---------------------------------------------------------------------------
def _build_partition_labels(seeds_by_region, V=None, tris=None):
    """Run the geodesic-Voronoi partition on the (original) canonical mesh.
    Returns ``(label, V, tris)``."""
    if V is None or tris is None:
        V, tris = _canonical_geometry()
    xy = project_xy(V)
    adj = build_adjacency(tris, len(V))
    face = float(np.linalg.norm(xy.max(0) - xy.min(0)))
    aperture = {a for a in APERTURE if a < len(V)}
    label = geodesic_voronoi(seeds_by_region, adj, xy, aperture, face * TIGHTNESS)
    label = _symmetrize_partition(label, V)
    return label, V, tris


def build_au_region_map():
    """``{AU: {muscles, mp478_vertices, n_vertices}}`` — muscle partition merged
    to AU (left+right share an AU)."""
    seeds = {m: s for m, (au, s) in MUSCLE_SEEDS.items()}
    label, _, _ = _build_partition_labels(seeds)
    muscle_au = {m: au for m, (au, s) in MUSCLE_SEEDS.items()}
    by_au: dict[str, set] = defaultdict(set)
    muscles_by_au: dict[str, set] = defaultdict(set)
    for v, muscles in label.items():
        for muscle in muscles:
            by_au[muscle_au[muscle]].add(v)
            muscles_by_au[muscle_au[muscle]].add(muscle)
    out = {}
    for au in sorted(by_au):
        verts = sorted(by_au[au])
        out[au] = dict(muscles=sorted(muscles_by_au[au]),
                       mp478_vertices=verts, n_vertices=len(verts))
    return out


def _feature_of(blendshape: str) -> str:
    """``noseSneerLeft`` -> ``noseSneer``; center shapes unchanged. The L/R
    pair share one symmetric mesh FEATURE, split by the midline at the end."""
    for suf in ("Left", "Right"):
        if blendshape.endswith(suf):
            return blendshape[: -len(suf)]
    return blendshape


def _feature_seeds() -> dict:
    feat: dict[str, list] = defaultdict(list)
    for bs, (au, muscle, side, s) in BLENDSHAPE_SEEDS.items():
        feat[_feature_of(bs)].extend(s)
    return dict(feat)


def build_blendshape_region_map():
    """``{blendshape: {au, muscle, side, mp478_vertices, n_vertices}}``.

    L/R pairs share one symmetric mesh FEATURE (their seeds merged) grown by the
    geodesic-Voronoi partition; each sided blendshape is then the half of that
    feature on its side of the facial midline (center shapes keep the whole,
    bilateral feature). So L/R divide cleanly down the middle — no winner-take-
    all asymmetry, no shared midline seam. Non-overlapping by construction."""
    label, V, _ = _build_partition_labels(_feature_seeds())
    feat_verts: dict[str, set] = defaultdict(set)
    for v, names in label.items():
        for f in names:
            feat_verts[f].add(v)
    _, _, axis = _mirror_map(np.asarray(V, dtype=np.float64))
    eps = 0.01 * float(V[:, 0].max() - V[:, 0].min())
    out = {}
    for bs, (au, muscle, side, _seeds) in BLENDSHAPE_SEEDS.items():
        fv = feat_verts[_feature_of(bs)]
        if side == "L":          # subject-left half (x > midline)
            verts = sorted(v for v in fv if V[v, 0] > axis + eps)
        elif side == "R":        # subject-right half (x < midline)
            verts = sorted(v for v in fv if V[v, 0] < axis - eps)
        else:                    # center: whole bilateral feature
            verts = sorted(fv)
        out[bs] = dict(au=au, muscle=muscle, side=side,
                       mp478_vertices=verts, n_vertices=len(verts))
    return out


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
_CACHE: dict[str, dict] = {}


def _load(filename: str) -> dict:
    if filename not in _CACHE:
        path = os.path.join(get_resource_path(), filename)
        with open(path) as f:
            _CACHE[filename] = json.load(f)["regions"]
    return copy.deepcopy(_CACHE[filename])


def load_au_region_map() -> dict:
    """Bundled non-overlapping AU region map (20 AUs, L+R merged). Each value:
    ``{"muscles": [...], "mp478_vertices": [...], "n_vertices": int}``."""
    return _load(AU_MAP_FILENAME)


def load_blendshape_region_map() -> dict:
    """Bundled non-overlapping blendshape region map (L/R independent). Each
    value: ``{"au", "muscle", "side", "mp478_vertices", "n_vertices"}``."""
    return _load(BLENDSHAPE_MAP_FILENAME)


# ---------------------------------------------------------------------------
# render assets — subdivided topology + dense per-vertex labels, computed once
# on the canonical mesh and cached. Reused for any 478-mesh (canonical or live).
# ---------------------------------------------------------------------------
_RENDER_ASSETS: dict[str, dict] = {}


def render_assets(kind: str = "au") -> dict:
    """Cached rendering assets for ``kind`` in {"au", "blendshape"}:

    ``{"parents": [(a,b)...], "tris": dense_tris[K,3],
       "region_verts": {region: set(dense vert ids)}, "axis": float, "n_base": 468}``

    Regions are resolved on the SUBDIVIDED canonical mesh (smooth boundaries),
    keyed by AU for ``"au"`` and by FEATURE (L/R merged) for ``"blendshape"`` —
    the renderer splits a feature into Left/Right by the midline ``axis`` using
    triangle centroids. ``parents``/``tris``/``region_verts`` are fixed in index
    space, so a deformed 478-mesh just needs ``dense_positions(mesh, parents)``.
    """
    if kind not in _RENDER_ASSETS:
        if kind not in ("au", "blendshape"):
            raise ValueError(f"kind must be 'au' or 'blendshape', got {kind!r}")

        V, tris = _canonical_geometry()
        n_base = len(V)
        aperture = {a for a in APERTURE if a < n_base}
        Vd, tris_d, ap_d, parents = subdivide(V, tris, aperture, SUBDIV_LEVELS)
        xy = project_xy(Vd)
        adj = build_adjacency(tris_d, len(Vd))
        face = float(np.linalg.norm(xy.max(0) - xy.min(0)))
        max_geo = face * TIGHTNESS

        if kind == "au":
            seeds = {m: s for m, (au, s) in MUSCLE_SEEDS.items()}
            muscle_au = {m: au for m, (au, s) in MUSCLE_SEEDS.items()}
            raw = geodesic_voronoi(seeds, adj, xy, ap_d, max_geo)
            label = {v: muscle_au[m] for v, m in raw.items()}
        else:
            label = geodesic_voronoi(_feature_seeds(), adj, xy, ap_d, max_geo)
        label = _symmetrize_partition(label, Vd)
        region_verts: dict[str, set] = defaultdict(set)
        for v, names in label.items():
            for name in names:
                region_verts[name].add(v)

        _RENDER_ASSETS[kind] = dict(parents=parents, tris=tris_d,
                                    region_verts=dict(region_verts),
                                    axis=float(_mirror_map(Vd)[2]), n_base=n_base)
    a = _RENDER_ASSETS[kind]
    return dict(parents=list(a["parents"]), tris=a["tris"],
                region_verts={k: set(v) for k, v in a["region_verts"].items()},
                axis=a["axis"], n_base=a["n_base"])
