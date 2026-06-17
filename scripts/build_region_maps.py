"""Generate py-feat's AU + ARKit-blendshape region-overlay maps.

Writes two non-overlapping region maps (geodesic-Voronoi partition of the
bundled MediaPipe canonical mesh) consumed by ``feat.utils.region_maps`` and
``feat.plotting.plot_face_regions``:

    feat/resources/au_region_map.json          # 20 AUs, left+right merged
    feat/resources/blendshape_region_map.json  # ARKit blendshapes, L/R independent

All anatomy (seed verts, AU<->muscle<->blendshape correspondence) lives in
``feat.utils.region_maps`` so this script, the renderer, and the tests share one
source of truth. The mapping is grounded in FACS (Ekman & Friesen), Melinda
Ozel's ARKit-to-FACS cheat sheet, and pooyadeperson's ARKit-52 anatomy guide.

    python scripts/build_region_maps.py
"""
import argparse
import json
import os

from feat.utils import region_maps as rm
from feat.utils.io import get_resource_path


def _meta(kind, n):
    return {
        "description": f"Non-overlapping {kind} region overlay on the "
                       "MediaPipe-478 mesh. Winner-take-all geodesic-Voronoi "
                       "partition of the bundled canonical mesh, walled off by "
                       "the eye/mouth aperture rings.",
        "kind": kind,
        "n_regions": n,
        "tightness": rm.TIGHTNESS,
        "mesh": "feat/resources/canonical_face_model.pt (468 verts)",
        "generator": "scripts/build_region_maps.py",
        "sources": ["FACS (Ekman & Friesen)",
                    "melindaozel.com/arkit-to-facs-cheat-sheet",
                    "pooyadeperson.com ARKit-52 anatomy guide"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=get_resource_path())
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    au = rm.build_au_region_map()
    bs = rm.build_blendshape_region_map()

    for name, regions, kind in (
        (rm.AU_MAP_FILENAME, au, "AU"),
        (rm.BLENDSHAPE_MAP_FILENAME, bs, "blendshape"),
    ):
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            json.dump({"_meta": _meta(kind, len(regions)), "regions": regions},
                      f, indent=2)
        covered = sorted({v for r in regions.values() for v in r["mp478_vertices"]})
        empty = [k for k, r in regions.items() if r["n_vertices"] == 0]
        print(f"{kind:10s} regions={len(regions):3d}  verts_covered={len(covered)}/468"
              f"  empty={empty if empty else 'NONE'}", flush=True)
        for k, r in regions.items():
            tag = r["au"] if "au" in r else "+".join(r["muscles"])
            print(f"  {k:22s} {tag:10s} {r['n_vertices']:3d}v", flush=True)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
