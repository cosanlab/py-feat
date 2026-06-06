"""Export a DISFA+ evaluation manifest (image paths + AU intensity labels).

Competitor tools (OpenFace 3, LibreFace, PyAFAR) run in their OWN isolated
envs and must NOT import py-feat. This script — run in the py-feat env —
freezes the exact DISFA+ split + ground-truth labels feat.evaluation uses into
a plain CSV that any competitor runner can consume. That keeps the comparison
on identical frames/labels without the competitor envs depending on py-feat.

Output CSV columns: image_path, AU01, AU02, AU04, AU05, AU06, AU09, AU12,
AU15, AU17, AU20, AU25, AU26   (intensities 0..5; truth binarizes at >= 2).

Usage:
    PYFEAT_DATA_ROOT=/Storage/Data python scripts/competitors/export_disfaplus_manifest.py \
        [--subset 200] [--out scripts/competitors/disfaplus_manifest.csv]
"""

import argparse
from pathlib import Path

from feat.evaluation.datasets import load_disfaplus

AU_COLS = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU09",
           "AU12", "AU15", "AU17", "AU20", "AU25", "AU26"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None,
                    help="limit to N frames (smoke test); default = full split")
    ap.add_argument("--aligned", action="store_true",
                    help="emit DISFA+ Aligned/ face crops (competitors that "
                         "expect pre-aligned input, e.g. LibreFace)")
    ap.add_argument("--out", default="scripts/competitors/disfaplus_manifest.csv")
    args = ap.parse_args()

    split = load_disfaplus(subset_size=args.subset, use_aligned=args.aligned)
    if split is None:
        raise SystemExit("DISFA+ not found — set PYFEAT_DATA_ROOT")

    df = split.labels.copy()
    df["image_path"] = split.image_paths
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["image_path", *AU_COLS]].to_csv(out, index=False)
    print(f"wrote {len(df)} frames × {len(AU_COLS)} AUs -> {out}")


if __name__ == "__main__":
    main()
