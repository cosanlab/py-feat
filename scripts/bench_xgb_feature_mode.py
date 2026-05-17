"""Bench v3.5 / v3.6 / v3.7 models against DISFA+.

These ablation variants don't use the standard SL_test.XGBClassifier
inference wrapper (no per-AU regional routing). They store scalers,
pcas, and classifiers as joblib + a ``feature_mode.json`` describing
which PCAs were fit and how to compose features.

This script loads those artifacts, swaps a custom inference path into
the Detector's AU detector slot, and runs the standard DISFA+ eval.

Usage:
    python scripts/bench_xgb_feature_mode.py \\
        --model-dir models/xgb_au_v35 \\
        --label v3_5_full \\
        --dataset disfaplus
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

import joblib
import numpy as np
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "bench-results" / "au_local"


def _device_default() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FeatureModeXGBClassifier:
    """Drop-in for SL_test.XGBClassifier but with no per-AU routing.

    All AUs share the same feature vector composed per the stored mode.
    Mirrors the calling convention `detect_au(frame, landmarks)`.
    """

    def __init__(self, scalers, pcas, classifiers, mode_info):
        self.scalers = scalers
        self.pcas = pcas
        self.classifiers = classifiers
        self.regions_used = mode_info["regions_used"]
        self._mode_info = mode_info
        self.au_keys = [
            "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU11",
            "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25",
            "AU26", "AU28", "AU43",
        ]
        # Map padded AU01 -> AU1 etc.
        self.classifiers = {f"AU{int(k[2:])}": v for k, v in classifiers.items()}
        self.weights_loaded = True

    def detect_au(self, frame, landmarks):
        # frame is the raw HOG (N, D). landmarks is per-face [68, 2] list.
        landmarks = np.concatenate(landmarks)
        landmarks = landmarks.reshape(-1, landmarks.shape[1] * landmarks.shape[2])
        # Optional: training-time transform info recorded in mode_info.
        # Apply the same spatial cell zeroing the PCA was fit on.
        transforms = self._mode_info.get("transforms", {}) if hasattr(self, "_mode_info") else {}
        feats = []
        for region in self.regions_used:
            hog_t = frame
            t = transforms.get(region)
            if t is not None and t[0] == "range":
                hog_t = frame.copy()
                _, start, end = t
                hog_t[:, start:end] = 0.0
            feats.append(self.pcas[region].transform(self.scalers[region].transform(hog_t)))
        feats.append(landmarks)
        X = np.concatenate(feats, axis=1)
        out = []
        for key in self.au_keys:
            clf = self.classifiers.get(key)
            if clf is None:
                out.append(np.full(X.shape[0], np.nan))
            else:
                out.append(clf.predict_proba(X)[:, 1])
        return np.array(out).T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--dataset", choices=("disfa", "disfaplus"), default="disfaplus")
    p.add_argument("--subset-size", type=int, default=4500)
    p.add_argument("--device", default=_device_default())
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--face-model", default="img2pose")  # match prior v3.x benches
    p.add_argument("--landmark-model", default="mobilefacenet")
    args = p.parse_args()

    print(f"=== bench [{args.label}] on {args.dataset} ({args.device}) ===")
    scalers = joblib.load(args.model_dir / "scalers.joblib")
    pcas = joblib.load(args.model_dir / "pcas.joblib")
    classifiers = joblib.load(args.model_dir / "classifiers.joblib")
    mode_info = json.loads((args.model_dir / "feature_mode.json").read_text())
    print(f"    feature mode: {mode_info['feature_mode']} "
          f"(regions: {mode_info['regions_used']})")

    from feat.detector import Detector
    from feat.evaluation import datasets, runner

    det = Detector(
        face_model=args.face_model,
        landmark_model=args.landmark_model,
        au_model="xgb",
        emotion_model=None,
        identity_model=None,
        gaze_model=None,
        device=args.device,
    )
    det.au_detector = FeatureModeXGBClassifier(scalers, pcas, classifiers, mode_info)

    if args.dataset == "disfa":
        split = datasets.load_disfa(split="P3", subset_size=args.subset_size, seed=args.seed)
    else:
        split = datasets.load_disfaplus(subset_size=args.subset_size, seed=args.seed)
    if split is None:
        print(f"error: dataset {args.dataset} not available", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    result = runner.evaluate_dataset(
        det, split, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n    elapsed: {elapsed:.1f}s")
    print(f"    AU F1 mean = {result['au_f1_mean']:.4f}")
    print(f"    AU ICC mean = {result['au_icc_mean']:.4f}")
    print(f"    per-AU:")
    for au in sorted(result["au_f1_per_au"]):
        print(f"      {au}: F1={result['au_f1_per_au'][au]:.4f}, ICC={result['au_icc_per_au'][au]:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
    out_path = RESULTS_DIR / f"{date}-{args.label}-{args.dataset}.json"
    payload = {
        "label": args.label,
        "date": date,
        "dataset": args.dataset,
        "model_dir": str(args.model_dir),
        "feature_mode": mode_info,
        "subset_size": args.subset_size,
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "device": args.device,
        "result": result,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n    JSON: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
