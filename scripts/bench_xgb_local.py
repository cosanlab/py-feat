"""A/B bench a locally-trained xgb AU classifier against DISFA+ (or DISFA).

Loads a Detector with the v2 xgb baseline, optionally swaps the AU detector
with a local ``.skops`` file (so we don't have to upload to HF before
testing), and runs the same regression metrics ``bench_regression.py``
emits — but writes results into a dedicated ``bench-results/au_local/``
tree so the official run history isn't polluted.

Usage:
    # Baseline (v2 from HF)
    python scripts/bench_xgb_local.py --label v2_baseline

    # Local override (e.g. v3.2)
    python scripts/bench_xgb_local.py \\
        --au-skops models/xgb_au_v3_2/xgb_au_classifier_v3_2.skops \\
        --label v3_2

    # Pick the DISFA fold instead of DISFA+
    python scripts/bench_xgb_local.py --dataset disfa --label v2_disfa
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "bench-results" / "au_local"


def _device_default() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_au_from_skops(skops_path: Path):
    """Load a local ``XGBClassifier`` wrapper from a ``.skops`` file."""
    from feat.au_detectors.StatLearning.SL_test import XGBClassifier
    from skops.io import get_untrusted_types, load

    untrusted = get_untrusted_types(file=str(skops_path))
    loaded = load(str(skops_path), trusted=untrusted)
    wrapper = XGBClassifier()
    wrapper.load_weights(
        scaler_upper=loaded.scaler_upper,
        pca_model_upper=loaded.pca_model_upper,
        scaler_lower=loaded.scaler_lower,
        pca_model_lower=loaded.pca_model_lower,
        scaler_full=loaded.scaler_full,
        pca_model_full=loaded.pca_model_full,
        classifiers=loaded.classifiers,
    )
    return wrapper


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--au-skops", type=Path, default=None,
                   help="Local .skops to load in place of HF v2. If omitted, "
                        "use the default v2 from HuggingFace.")
    p.add_argument("--label", required=True,
                   help="Tag for output filenames, e.g. 'v3_2' or 'v2_baseline'.")
    p.add_argument("--dataset", choices=("disfa", "disfaplus"), default="disfaplus")
    p.add_argument("--subset-size", type=int, default=4500)
    p.add_argument("--device", default=_device_default())
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--face-model", default="img2pose")
    p.add_argument("--landmark-model", default="mobilefacenet")
    p.add_argument("--emotion-model", default=None,
                   help="Set to 'resmasknet' to mirror prod bench; None speeds up.")
    p.add_argument("--identity-model", default=None)
    args = p.parse_args(argv)

    from feat.detector import Detector
    from feat.evaluation import datasets, runner

    print(f"=== bench [{args.label}] on {args.dataset} ({args.device}) ===")
    print(f"    au-skops: {args.au_skops or '(default v2 from HF)'}")

    det = Detector(
        face_model=args.face_model,
        landmark_model=args.landmark_model,
        au_model="xgb",  # default; we'll swap if --au-skops provided
        emotion_model=args.emotion_model,
        identity_model=args.identity_model,
        device=args.device,
    )

    if args.au_skops is not None:
        print(f"    swapping au_detector weights from {args.au_skops}...")
        det.au_detector = _load_au_from_skops(args.au_skops)

    if args.dataset == "disfa":
        split = datasets.load_disfa(split="P3", subset_size=args.subset_size, seed=args.seed)
    else:
        split = datasets.load_disfaplus(subset_size=args.subset_size, seed=args.seed)
    if split is None:
        print(f"error: dataset {args.dataset} not available", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    result = runner.evaluate_dataset(
        det, split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n    elapsed: {elapsed:.1f}s")
    print(f"    AU F1 mean = {result['au_f1_mean']:.4f}")
    print(f"    AU ICC mean = {result['au_icc_mean']:.4f}")
    print(f"    n_evaluated AUs = {result['n_aus_evaluated']}")
    print(f"\n    per-AU:")
    for au in sorted(result["au_f1_per_au"]):
        print(f"      {au}: F1={result['au_f1_per_au'][au]:.4f}, ICC={result['au_icc_per_au'][au]:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
    out_path = RESULTS_DIR / f"{date}-{args.label}-{args.dataset}.json"
    payload = {
        "label": args.label,
        "date": date,
        "dataset": args.dataset,
        "au_skops": str(args.au_skops) if args.au_skops else None,
        "subset_size": args.subset_size,
        "host": platform.node(),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                else "CPU"),
        "device": args.device,
        "result": result,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n    JSON: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
