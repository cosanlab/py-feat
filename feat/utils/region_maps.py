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
is non-overlapping *by construction*.

The per-vertex partition is then resolved to a **per-triangle** assignment and
cleaned so the region silhouettes are whole rather than ragged, while staying
strictly on the 478-mesh triangle edges (no subdivision, no curve fitting — the
boundaries never become smoother than the mesh itself):

1. each triangle takes the region of its majority vertex (the eye/mouth aperture
   triangles are permanent holes);
2. **boundary-length (perimeter) descent** flips boundary triangles to shorten
   the zigzag region↔region edges, bounded so no region's area drifts past
   ``CAP`` and tiny regions (<= ``SMALL_REGION`` triangles) are frozen;
3. the partition is **L/R symmetrized** with a geometric (triangle-centroid)
   mirror map;
4. small disconnected specks are removed symmetrically (each speck and its mirror
   reassigned together), killing stray "flipped" triangles.

The per-triangle assignment is fixed in index space (triangle indices into the
bundled canonical tessellation), so the same map overlays a live detected
478-mesh — only the vertex *positions* change per frame. See
``feat.plotting.plot_face_regions``.

The AU <-> muscle <-> blendshape correspondence is grounded in:
* FACS (Ekman & Friesen) — AU -> muscle.
* Melinda Ozel's ARKit-to-FACS cheat sheet (melindaozel.com/arkit-to-facs-cheat-sheet).
* pooyadeperson's "Ultimate Guide to ARKit's 52 Facial Blendshapes" (anatomy refs).
"""

from __future__ import annotations

import copy
import heapq
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

from feat.utils.io import get_resource_path
from feat.utils.mp_plotting import FaceLandmarksConnections as _F

# --- partition / cleanup tuning (the user-approved look) ---
TIGHTNESS = 0.10       # max geodesic reach of a seed, as a fraction of face span
CAP = 0.40             # max fractional triangle-count change per region in cleanup
SMALL_REGION = 6       # regions with <= this many triangles are frozen (no erosion)
ISLAND_MAX = 4         # disconnected specks <= this many triangles are reassigned

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
    """Vertex adjacency over the triangle-edge graph."""
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


# vertices on the facial midline (define the mirror axis)
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


def _feature_of(blendshape: str) -> str:
    """``noseSneerLeft`` -> ``noseSneer``; center shapes unchanged. The L/R
    pair share one symmetric mesh FEATURE before the partition splits them."""
    for suf in ("Left", "Right"):
        if blendshape.endswith(suf):
            return blendshape[: -len(suf)]
    return blendshape


def _feature_seeds() -> dict:
    feat: dict[str, list] = defaultdict(list)
    for bs, (au, muscle, side, s) in BLENDSHAPE_SEEDS.items():
        feat[_feature_of(bs)].extend(s)
    return dict(feat)


# ---------------------------------------------------------------------------
# triangle graph + per-triangle cleanup
# ---------------------------------------------------------------------------
def triangle_adjacency(tris):
    """``{triangle_index: [neighbour triangle indices]}`` — triangles sharing an
    edge (two vertices)."""
    edge_tris: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ti, t in enumerate(tris):
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        for u, w in ((a, b), (b, c), (a, c)):
            edge_tris[(u, w) if u < w else (w, u)].append(ti)
    nb: dict[int, set] = defaultdict(set)
    for ts in edge_tris.values():
        for i in ts:
            for j in ts:
                if i != j:
                    nb[i].add(j)
    return {k: sorted(v) for k, v in nb.items()}


def aperture_triangles(tris, aperture):
    """Triangles wholly inside an eye/mouth aperture (all 3 verts on a ring) —
    the OPENINGS. They stay permanent holes (never gap-filled or relabeled)."""
    return {ti for ti, t in enumerate(tris) if all(int(v) in aperture for v in t)}


def _per_triangle_labels(tris, vertex_label):
    """Region of each triangle's majority vertex (>=2 of 3), else None."""
    treg: list = [None] * len(tris)
    for ti, t in enumerate(tris):
        labs = [vertex_label.get(int(v)) for v in t]
        real = [x for x in labs if x is not None]
        if not real:
            continue
        win, cnt = Counter(real).most_common(1)[0]
        if cnt >= 2:
            treg[ti] = win
    return treg


def _gap_fill(treg, tri_adj, holes, max_iter=50):
    """Assign unlabeled triangles to the majority region of their labeled
    neighbours; iterate to convergence. Aperture-hole triangles stay None."""
    treg = list(treg)
    for _ in range(max_iter):
        changed = False
        for ti in range(len(treg)):
            if treg[ti] is not None or ti in holes:
                continue
            votes = Counter(treg[j] for j in tri_adj.get(ti, []) if treg[j] is not None)
            if votes:
                treg[ti] = votes.most_common(1)[0][0]
                changed = True
        if not changed:
            break
    return treg


def _perimeter_descent(treg, tri_adj, holes, cap=CAP, small=SMALL_REGION, max_iter=80):
    """Boundary-LENGTH (perimeter) descent: greedily flip each triangle to a
    neighbour's region when it strictly reduces the number of bi-colored edges
    around it. Shortens zigzag region↔region boundaries directly (an Ising /
    graph-cut energy), unlike majority vote which leaves sawtooth edges stable.

    Bounded so whole-triangle transfers can't push a region past ``cap`` of its
    starting triangle count, and regions of <= ``small`` triangles are frozen."""
    treg = list(treg)
    base = Counter(t for t in treg if t is not None)
    lo = {r: (c if c <= small else int(np.floor(c * (1 - cap)))) for r, c in base.items()}
    hi = {r: int(np.ceil(c * (1 + cap))) for r, c in base.items()}
    cur = Counter(treg)
    for _ in range(max_iter):
        changed = 0
        for ti in range(len(treg)):
            creg = treg[ti]
            if creg is None or ti in holes:
                continue
            nbrs = [treg[j] for j in tri_adj.get(ti, []) if treg[j] is not None]
            if not nbrs:
                continue
            cur_cost = sum(1 for x in nbrs if x != creg)
            best, best_cost = creg, cur_cost
            for cand in set(nbrs):
                if cand == creg:
                    continue
                cost = sum(1 for x in nbrs if x != cand)
                if cost < best_cost:
                    best, best_cost = cand, cost
            if best == creg:
                continue
            if cur[creg] - 1 < lo.get(creg, 0) or cur[best] + 1 > hi.get(best, len(treg)):
                continue
            treg[ti] = best
            cur[creg] -= 1
            cur[best] += 1
            changed += 1
        if not changed:
            break
    return treg


def _triangle_mirror_match(tris, xy, axis):
    """For each triangle, the index of the triangle physically across the
    midline (nearest mirrored centroid). A geometric mirror — never lands a
    triangle in a disconnected spot, so it can't manufacture island specks."""
    cen = np.array([xy[t].mean(0) for t in tris])
    mir = cen.copy()
    mir[:, 0] = 2 * axis - mir[:, 0]
    return cKDTree(cen).query(mir)[1], cen


def _symmetrize(treg, tris, xy, axis):
    """Force exact L/R symmetry: stamp the primary (subject-right) half's
    triangle labels onto the mirror half with ``Left``<->``Right`` swapped."""
    match, cen = _triangle_mirror_match(tris, xy, axis)
    primary_neg = bool(xy[107, 0] < axis) if len(xy) > 107 else True
    out = list(treg)
    for i in range(len(tris)):
        if bool(cen[i, 0] < axis) == primary_neg and treg[i] is not None:
            out[match[i]] = _mirror_name(treg[i])
        elif bool(cen[i, 0] < axis) == primary_neg:
            out[match[i]] = None
    return out


def _remove_islands(treg, tris, tri_adj, xy, axis, max_island=ISLAND_MAX, max_iter=10):
    """Reassign small DISCONNECTED specks of a region (a component of
    <= ``max_island`` triangles, dwarfed by a dominant main body) to the
    surrounding region, doing each speck and its mirror partner together so L/R
    symmetry is preserved. Run AFTER ``_symmetrize`` — symmetrization can
    manufacture mirror specks where the canonical triangulation differs slightly
    L/R (e.g. the noseSneer nose speck the geodesic frontier never produced)."""
    match, _ = _triangle_mirror_match(tris, xy, axis)
    treg = list(treg)
    for _ in range(max_iter):
        region_tris: dict[str, list] = defaultdict(list)
        for ti, r in enumerate(treg):
            if r is not None:
                region_tris[r].append(ti)
        changed = False
        for r, members in region_tris.items():
            seen, comps = set(), []
            for s in members:
                if s in seen:
                    continue
                stack, comp = [s], []
                while stack:
                    u = stack.pop()
                    if u in seen:
                        continue
                    seen.add(u)
                    comp.append(u)
                    for v in tri_adj.get(u, []):
                        if treg[v] == r and v not in seen:
                            stack.append(v)
                comps.append(comp)
            if len(comps) <= 1:
                continue
            comps.sort(key=len, reverse=True)
            largest = len(comps[0])
            for comp in comps[1:]:
                # only a true speck: small AND dwarfed by a dominant main body
                # (protects genuinely small regions from being erased)
                if len(comp) > max_island or largest < 3 * len(comp):
                    continue
                mirror_r = _mirror_name(r)
                for ti in comp:
                    votes = Counter(treg[j] for j in tri_adj.get(ti, [])
                                    if treg[j] is not None and treg[j] != r)
                    if not votes:
                        continue
                    x = votes.most_common(1)[0][0]
                    treg[ti] = x
                    # Mirror the fix, but only onto the genuine mirror speck (a
                    # triangle still holding this region's mirror): match[] is a
                    # geometric nearest-neighbour, not a strict bijection, so an
                    # unguarded write could clobber a hole or an adjacent region.
                    mj = match[ti]
                    if treg[mj] == mirror_r:
                        treg[mj] = _mirror_name(x)
                    changed = True
        if not changed:
            break
    return treg


# ---------------------------------------------------------------------------
# per-vertex seed labels (AU merges L/R; blendshape splits a feature L/R)
# ---------------------------------------------------------------------------
def _au_vertex_labels(adj, xy, aperture, max_geo):
    seeds = {m: s for m, (au, s) in MUSCLE_SEEDS.items()}
    muscle_au = {m: au for m, (au, s) in MUSCLE_SEEDS.items()}
    raw = geodesic_voronoi(seeds, adj, xy, aperture, max_geo)
    return {v: muscle_au[m] for v, m in raw.items()}


def _blendshape_vertex_labels(adj, xy, aperture, max_geo, V, axis):
    """Per-vertex FEATURE partition, then split each *sided* feature into
    ``...Left`` / ``...Right`` by which side of the midline the vertex is on.
    Center features keep the whole bilateral footprint. Midline-seam vertices
    are left unlabeled; gap-fill + symmetrize close the seam cleanly."""
    feat_label = geodesic_voronoi(_feature_seeds(), adj, xy, aperture, max_geo)
    sided = {_feature_of(bs) for bs in BLENDSHAPE_SEEDS if _feature_of(bs) != bs}
    eps = 0.01 * float(V[:, 0].max() - V[:, 0].min())
    out = {}
    for v, f in feat_label.items():
        if f not in sided:
            out[v] = f
        elif V[v, 0] > axis + eps:      # subject-left half (x > midline)
            out[v] = f + "Left"
        elif V[v, 0] < axis - eps:      # subject-right half (x < midline)
            out[v] = f + "Right"
    return out


# ---------------------------------------------------------------------------
# canonical partition — computed once, shared by build_* and render_assets
# ---------------------------------------------------------------------------
_PARTITION: dict[str, dict] = {}


def _triangle_partition(kind: str) -> dict:
    """Cleaned per-triangle region assignment on the canonical mesh for ``kind``
    in {"au", "blendshape"}. Returns ``{"treg": [region|None per triangle],
    "tris": int[M,3], "n_base": 468}``. The full cleanup pipeline (majority ->
    gap-fill -> perimeter descent -> symmetrize -> island removal)."""
    if kind not in ("au", "blendshape"):
        raise ValueError(f"kind must be 'au' or 'blendshape', got {kind!r}")
    if kind not in _PARTITION:
        V, tris = _canonical_geometry()
        xy = project_xy(V)
        adj = build_adjacency(tris, len(V))
        aperture = {a for a in APERTURE if a < len(V)}
        face = float(np.linalg.norm(xy.max(0) - xy.min(0)))
        max_geo = face * TIGHTNESS
        _, _, axis = _mirror_map(V)

        if kind == "au":
            vertex_label = _au_vertex_labels(adj, xy, aperture, max_geo)
        else:
            vertex_label = _blendshape_vertex_labels(adj, xy, aperture, max_geo, V, axis)

        tri_adj = triangle_adjacency(tris)
        holes = aperture_triangles(tris, aperture)
        treg = _per_triangle_labels(tris, vertex_label)
        treg = _gap_fill(treg, tri_adj, holes)
        treg = _perimeter_descent(treg, tri_adj, holes)
        treg = _symmetrize(treg, tris, xy, axis)
        treg = _remove_islands(treg, tris, tri_adj, xy, axis)
        _PARTITION[kind] = {"treg": treg, "tris": tris, "n_base": len(V)}
    p = _PARTITION[kind]
    return {"treg": list(p["treg"]), "tris": p["tris"], "n_base": p["n_base"]}


def _region_triangles(treg) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for ti, r in enumerate(treg):
        if r is not None:
            out[r].append(ti)
    return out


# ---------------------------------------------------------------------------
# map construction (used by scripts/build_region_maps.py + the loaders' fallback)
# ---------------------------------------------------------------------------
def build_au_region_map():
    """``{AU: {muscles, triangles, mp478_vertices, n_triangles, n_vertices}}`` —
    muscle partition merged to AU (left+right share an AU)."""
    p = _triangle_partition("au")
    treg, tris = p["treg"], p["tris"]
    region_tris = _region_triangles(treg)
    muscles_by_au: dict[str, set] = defaultdict(set)
    for muscle, (au, _seeds) in MUSCLE_SEEDS.items():
        muscles_by_au[au].add(muscle)
    out = {}
    for au in sorted(region_tris):
        ti = sorted(region_tris[au])
        verts = sorted({int(v) for t in ti for v in tris[t]})
        out[au] = dict(muscles=sorted(muscles_by_au[au]),
                       triangles=ti, mp478_vertices=verts,
                       n_triangles=len(ti), n_vertices=len(verts))
    return out


def build_blendshape_region_map():
    """``{blendshape: {au, muscle, side, triangles, mp478_vertices,
    n_triangles, n_vertices}}``.

    L/R pairs share one symmetric mesh FEATURE (their seeds merged) grown by the
    geodesic-Voronoi partition; the per-triangle cleanup divides each sided pair
    cleanly down the midline and forces exact L/R symmetry. Center shapes keep
    the whole bilateral feature. Non-overlapping by construction."""
    p = _triangle_partition("blendshape")
    treg, tris = p["treg"], p["tris"]
    region_tris = _region_triangles(treg)
    out = {}
    for bs, (au, muscle, side, _seeds) in BLENDSHAPE_SEEDS.items():
        ti = sorted(region_tris.get(bs, []))
        verts = sorted({int(v) for t in ti for v in tris[t]})
        out[bs] = dict(au=au, muscle=muscle, side=side,
                       triangles=ti, mp478_vertices=verts,
                       n_triangles=len(ti), n_vertices=len(verts))
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
    ``{"muscles", "triangles", "mp478_vertices", "n_triangles", "n_vertices"}``."""
    return _load(AU_MAP_FILENAME)


def load_blendshape_region_map() -> dict:
    """Bundled non-overlapping blendshape region map (L/R independent). Each
    value: ``{"au", "muscle", "side", "triangles", "mp478_vertices",
    "n_triangles", "n_vertices"}``."""
    return _load(BLENDSHAPE_MAP_FILENAME)


# ---------------------------------------------------------------------------
# render assets — raw tessellation + per-triangle region labels, computed once
# on the canonical mesh and cached. Reused for any 478-mesh (canonical or live):
# the triangle indices are fixed, only the vertex positions change per frame.
# ---------------------------------------------------------------------------
_RENDER_ASSETS: dict[str, dict] = {}


def render_assets(kind: str = "au") -> dict:
    """Cached rendering assets for ``kind`` in {"au", "blendshape"}:

    ``{"tris": int[M,3], "region_tris": {region: [triangle indices]},
       "n_base": 468}``

    ``tris`` is the bundled canonical tessellation; ``region_tris`` maps each
    final region name (an AU, or a sided/center blendshape) to the triangle
    indices it owns after the cleanup pipeline. Fixed in index space, so a
    deformed 478-mesh just supplies new vertex positions. See
    ``feat.plotting.plot_face_regions``."""
    if kind not in _RENDER_ASSETS:
        p = _triangle_partition(kind)
        _RENDER_ASSETS[kind] = {"tris": p["tris"],
                                "region_tris": _region_triangles(p["treg"]),
                                "n_base": p["n_base"]}
    a = _RENDER_ASSETS[kind]
    return {"tris": a["tris"].copy(),
            "region_tris": {k: list(v) for k, v in a["region_tris"].items()},
            "n_base": a["n_base"]}
