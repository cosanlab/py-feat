"""Build the facial-muscle -> MediaPipe-478 mesh / AU map, geodesic grow.

This is the generator for the map py-feat ships at
``feat/resources/muscle_to_mesh_map.json`` (loaded via
``feat.utils.muscle_to_landmark.load_muscle_to_landmark_map``). It is fully
self-contained on py-feat's own bundled resources — no external downloads.

Two design choices make this map better than the dlib-68-polygon translation it
replaces:
  1. Geometry source = py-feat's bundled MediaPipe canonical mesh
     (``feat/resources/canonical_face_model.pt``, 468 verts, the reference
     topology) + its triangle topology
     (``feat/resources/canonical_face_tessellation.json``, 898 tris). No
     frontalization hack — the canonical mesh is natively upright.
  2. Region growing is GEODESIC along the mesh surface (BFS over the triangle
     adjacency graph), not Euclidean 3D balls. A surface walk won't jump the lip
     gap or the eye opening, so muscle footprints respect the apertures by
     construction and follow the face's curved surface.

Muscle anchors + the muscle -> AU mapping come from facial myology + the Ozel
FACS cheat sheet + the pooyadeperson ARKit-blendshape guide (which confirmed the
blendshape -> AU -> muscle table; folded in mouthShrugUpper -> AU17 and
cheekPuff -> AD34).

  python scripts/map_muscles_canonical_geodesic.py
"""
import argparse
import json
import os
from collections import deque

import numpy as np

from feat.utils.io import get_resource_path
from feat.utils.face_pose import load_canonical_face_model
from feat.utils.mp_plotting import FaceLandmarksConnections as _F


def load_geometry():
    """Canonical mesh verts (468,3) + triangle topology (898,3) from the
    bundled py-feat resources — natively upright, no frontalization.

    NOTE the topology source is ``canonical_face_tessellation.json`` (the full
    898-triangle triangulation of MediaPipe's canonical_face_model.obj), NOT
    ``FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION`` — the latter is
    MediaPipe's curated *drawing* edge list and is an incomplete triangulation
    (it differs by ~250 edges), so it would give the wrong surface adjacency for
    a geodesic grow. The two are intentionally distinct artifacts."""
    V = load_canonical_face_model().detach().cpu().numpy()
    tess_path = os.path.join(get_resource_path(), "canonical_face_tessellation.json")
    with open(tess_path) as f:
        faces = np.array(json.load(f)["triangles"], dtype=int)
    return V, faces


def build_adj(faces, n):
    adj = [set() for _ in range(n)]
    for a, b, c in faces:
        for u, w in ((a, b), (b, c), (a, c)):
            adj[u].add(w); adj[w].add(u)
    return [sorted(s) for s in adj]


def geodesic_grow(seeds, adj, hops, blocked):
    """BFS over the mesh edge graph from seeds, up to `hops` rings, never
    entering `blocked` (aperture) vertices — so growth can't cross the eye/mouth
    openings (those rings wall it off)."""
    out = set(s for s in seeds if s not in blocked)
    frontier = deque((s, 0) for s in out)
    while frontier:
        v, d = frontier.popleft()
        if d >= hops:
            continue
        for w in adj[v]:
            if w in blocked or w in out:
                continue
            out.add(w); frontier.append((w, d + 1))
    return out


# Aperture rings (MP indices) — these WALL OFF growth and are excluded.
# Eye rings come straight from MediaPipe's canonical landmark groups (single
# source of truth). The inner-lip ring has no dedicated group (FACE_LANDMARKS_LIPS
# bundles inner+outer), so it stays explicit.
def _group_verts(name):
    return sorted({v for c in getattr(_F, name) for v in (c.start, c.end)})


RIGHT_EYE = _group_verts("FACE_LANDMARKS_RIGHT_EYE")
LEFT_EYE = _group_verts("FACE_LANDMARKS_LEFT_EYE")
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
             95, 88, 178, 87, 14, 317, 402, 318, 324]
IRIS = list(range(468, 478))   # not in 468-mesh, harmless
APERTURE = set(RIGHT_EYE) | set(LEFT_EYE) | set(INNER_LIP) | set(IRIS)

OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
             146, 91, 181, 84, 17, 314, 405, 321, 375]

# muscle -> (au, geodesic hop radius, anatomical seed vertices). AU validated
# against Ozel + pooyadeperson ARKit guide.
MUSCLES = {
    "frontalis_inner_R":   ("AU01", 3, [107, 66, 105, 69]),
    "frontalis_inner_L":   ("AU01", 3, [336, 296, 334, 299]),
    "frontalis_outer_R":   ("AU02", 3, [105, 63, 70, 71, 54, 68]),
    "frontalis_outer_L":   ("AU02", 3, [334, 293, 300, 301, 284, 298]),
    "corrugator_R":        ("AU04", 2, [107, 55, 65, 9]),
    "corrugator_L":        ("AU04", 2, [336, 285, 295, 8]),
    "procerus":            ("AU04", 2, [8, 9, 168, 6]),
    "orb_oculi_orbital_R": ("AU06", 2, [117, 118, 119, 100, 50, 36, 111, 31]),
    "orb_oculi_orbital_L": ("AU06", 2, [346, 347, 348, 329, 280, 266, 340, 261]),
    "orb_oculi_palpebral_R": ("AU07", 1, [226, 110, 24, 23, 22, 26, 112, 190]),
    "orb_oculi_palpebral_L": ("AU07", 1, [446, 339, 254, 253, 252, 256, 341, 414]),
    "levator_palpebrae_R": ("AU05", 1, [223, 222, 221, 189, 56]),
    "levator_palpebrae_L": ("AU05", 1, [443, 442, 441, 413, 286]),
    "llsan_R":             ("AU09", 2, [115, 48, 49, 64, 102, 219]),
    "llsan_L":             ("AU09", 2, [344, 278, 279, 294, 331, 439]),
    "nasalis_R":           ("AU38", 2, [48, 49, 64, 98, 240]),
    "nasalis_L":           ("AU38", 2, [278, 279, 294, 327, 460]),
    "levator_labii_R":     ("AU10", 2, [205, 36, 142, 203, 206, 207]),
    "levator_labii_L":     ("AU10", 2, [425, 266, 371, 423, 426, 427]),
    "zygomaticus_minor_R": ("AU11", 2, [116, 123, 117, 50]),
    "zygomaticus_minor_L": ("AU11", 2, [345, 352, 346, 280]),
    "zygomaticus_major_R": ("AU12", 2, [50, 101, 205, 187, 207, 61]),
    "zygomaticus_major_L": ("AU12", 2, [280, 330, 425, 411, 427, 291]),
    "levator_anguli_oris_R": ("AU13", 2, [205, 203, 206, 61]),
    "levator_anguli_oris_L": ("AU13", 2, [425, 423, 426, 291]),
    "buccinator_R":        ("AU14", 2, [212, 202, 57, 43, 204, 207]),
    "buccinator_L":        ("AU14", 2, [432, 422, 287, 273, 424, 427]),
    "risorius_R":          ("AU20", 2, [212, 202, 210, 169, 150]),
    "risorius_L":          ("AU20", 2, [432, 422, 430, 394, 379]),
    "depressor_anguli_oris_R": ("AU15", 2, [43, 204, 210, 135, 169, 61]),
    "depressor_anguli_oris_L": ("AU15", 2, [273, 424, 430, 364, 394, 291]),
    "depressor_labii_R":   ("AU16", 2, [84, 181, 91, 146, 61, 77]),
    "depressor_labii_L":   ("AU16", 2, [314, 405, 321, 375, 291, 307]),
    "mentalis":            ("AU17", 2, [175, 199, 200, 18, 83, 313, 152, 148, 377]),
    "orbicularis_oris":    ("AU24", 1, OUTER_LIP),
    "masseter_R":          ("AU26", 2, [58, 172, 136, 132, 215, 138]),
    "masseter_L":          ("AU26", 2, [288, 397, 365, 361, 435, 367]),
}


def build_map(V, faces):
    n = len(V)
    adj = build_adj(faces, n)
    blocked = {a for a in APERTURE if a < n}
    result = {}
    for muscle, (au, hops, seeds) in MUSCLES.items():
        seeds = [int(s) for s in seeds if s < n]
        # cast to plain int — adjacency carries numpy int64 which json can't serialize
        verts = sorted(int(v) for v in geodesic_grow(seeds, adj, hops, blocked))
        result[muscle] = dict(au=au, mp478_seed=seeds, mp478_vertices=verts,
                              n_vertices=len(verts), hops=int(hops))
    return result, blocked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=os.path.join(get_resource_path(), "muscle_to_mesh_map.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    V, faces = load_geometry()
    n = len(V)
    result, blocked = build_map(V, faces)

    covered = sorted(set(v for r in result.values() for v in r["mp478_vertices"]))
    leaks = set(covered) & blocked
    empty = [m for m, r in result.items() if r["n_vertices"] == 0]
    print(f"muscles: {len(result)}  covered: {len(covered)}/{n}  "
          f"aperture leaks: {len(leaks)}  "
          f"empty regions: {empty if empty else 'NONE'}", flush=True)
    bad = 0
    for m in result:
        if m.endswith("_L"):
            r = m[:-2] + "_R"
            if r in result:
                cl = V[result[m]["mp478_vertices"]].mean(0)
                cr = V[result[r]["mp478_vertices"]].mean(0)
                if abs(cl[0] + cr[0]) > 1.0 or abs(cl[1] - cr[1]) > 1.0:
                    bad += 1
    print(f"asymmetric L/R pairs: {bad}", flush=True)
    for m, r in result.items():
        print(f"  {m:24s} {r['au']}  {r['n_vertices']:3d}v", flush=True)

    payload = {
        "_meta": {
            "description": "Facial muscle -> MediaPipe-478 mesh vertex / FACS AU "
                           "map. Geodesic surface region-grow on the canonical "
                           "mesh, walled off by the eye/mouth aperture rings.",
            "n_muscles": len(result),
            "n_vertices_covered": len(covered),
            "mesh": "feat/resources/canonical_face_model.pt (468 verts)",
            "generator": "scripts/map_muscles_canonical_geodesic.py",
        },
        "muscles": result,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
