"""Post-hoc isotonic calibration for v3.2 (or any xgb_au model).

XGBoost binary:logistic with scale_pos_weight emits ranked probabilities
(good AUC) but they're shifted away from a calibrated 0..1 scale, so
thresholding at 0.5 produces poor F1. This script fits a per-AU
IsotonicRegression on the held-out CV fold (same seed/folds as training)
and embeds the calibrators into a new .skops bundle.

After this, ``predict_proba`` outputs from the wrapped XGBClassifier are
calibrated probabilities — threshold-0.5 means what users expect.

Usage:
    python scripts/calibrate_v3_2.py \\
        --input-dirs bench-cache/au_train_blackwell \\
                     bench-cache/au_train_3090 \\
                     bench-cache/au_train_pain \\
                     bench-cache/au_train_bp4d_male \\
        --model-dir models/xgb_au_v3_2 \\
        --output-skops models/xgb_au_v3_2/xgb_au_classifier_v3_2_calibrated.skops
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import joblib
import numpy as np

# Re-use training-script helpers so the CV split is bit-identical
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "trainmod", str(Path(__file__).resolve().parent / "train_au_xgb_v3_2.py")
)
trainmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainmod)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Directory containing scalers.joblib, pcas.joblib, "
                        "classifiers.joblib, hog_split.json")
    p.add_argument("--output-skops", type=Path, required=True)
    p.add_argument("--n-folds", type=int, default=3,
                   help="Must match the value used during training")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the value used during training")
    args = p.parse_args()

    print(f"loading artifacts from {args.model_dir}...")
    scalers = joblib.load(args.model_dir / "scalers.joblib")
    pcas = joblib.load(args.model_dir / "pcas.joblib")
    classifiers = joblib.load(args.model_dir / "classifiers.joblib")
    hog_split_meta = json.loads((args.model_dir / "hog_split.json").read_text())
    hog_split = (hog_split_meta["upper_end"], hog_split_meta["lower_start"])
    print(f"  classifiers: {sorted(classifiers.keys())}")
    print(f"  hog_split: {hog_split}")

    print(f"\nloading {len(args.input_dirs)} input dirs...")
    t0 = time.perf_counter()
    data = trainmod.load_chunks(args.input_dirs)
    n = len(data["hog"])
    print(f"  loaded {n} rows, HOG dim={data['hog'].shape[1]} ({time.perf_counter()-t0:.0f}s)")
    print(f"  per-source: {dict(zip(*np.unique(data['sources'], return_counts=True)))}")

    print(f"\ntransforming features per region (spatial-split PCA)...")
    X_feats = {}
    for region in trainmod.REGIONS:
        t0 = time.perf_counter()
        X_feats[region] = trainmod.transform_region(
            data["hog"], data["landmarks"], region, scalers, pcas, hog_split,
        )
        print(f"  {region}: {X_feats[region].shape} ({time.perf_counter()-t0:.0f}s)")

    print(f"\nfitting per-AU IsotonicRegression on holdout fold...")
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, f1_score

    calibrators = {}
    cal_log = {}
    subjects = data["subjects"]
    sources = data["sources"]
    labels = data["labels"]

    for j, au_key_padded in enumerate(trainmod.AU_KEYS):
        if j >= labels.shape[1]:
            continue
        au_key = f"AU{int(au_key_padded[2:])}"  # "AU01" -> "AU1"
        if au_key not in classifiers:
            print(f"  {au_key_padded}: no classifier; skipping")
            continue

        region = trainmod.AU_REGION.get(au_key_padded, "full")
        X = X_feats[region]

        # Reproduce the exact mask + CV split used during training so
        # the holdout fold is bit-identical to what train_one_au saw.
        allowed = trainmod.CHEONG_TRAIN_AU_DICT.get(
            au_key_padded, ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet", "pain"]
        )
        src_mask = np.isin(sources, allowed)
        label_mask = ~np.isnan(labels[:, j])
        mask = src_mask & label_mask
        yb = (labels[mask, j] >= trainmod.INTENSITY_BIN_THRESHOLD).astype(np.int8)
        subj = subjects[mask]
        X_au = X[mask]

        cv_splits = trainmod.subject_disjoint_kfold(subj, k=args.n_folds, seed=args.seed)
        _train_idx, holdout_idx = cv_splits[-1]

        # Get raw probabilities on the holdout fold
        raw_probs = classifiers[au_key].predict_proba(X_au[holdout_idx])[:, 1]
        y_hold = yb[holdout_idx]

        # Pre-calibration baseline
        auc_pre = float(roc_auc_score(y_hold, raw_probs))
        f1_pre = float(f1_score(y_hold, (raw_probs >= 0.5).astype(int), zero_division=0))

        # Fit isotonic. out_of_bounds='clip' is essential for inference-time
        # probs that may fall slightly outside the fit-time range.
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_probs, y_hold)
        cal_probs = iso.transform(raw_probs)

        # Post-calibration: AUC must be identical (isotonic is monotonic);
        # F1@0.5 should jump for shifted-probability AUs.
        auc_post = float(roc_auc_score(y_hold, cal_probs))
        f1_post = float(f1_score(y_hold, (cal_probs >= 0.5).astype(int), zero_division=0))

        calibrators[au_key] = iso
        cal_log[au_key] = {
            "n_holdout": int(len(holdout_idx)),
            "pos_rate_holdout": float(y_hold.mean()),
            "raw_prob_mean": float(raw_probs.mean()),
            "cal_prob_mean": float(cal_probs.mean()),
            "auc_pre": auc_pre,
            "auc_post": auc_post,
            "f1_at_0.5_pre": f1_pre,
            "f1_at_0.5_post": f1_post,
            "f1_gain": f1_post - f1_pre,
        }
        gain = f1_post - f1_pre
        sign = "+" if gain >= 0 else ""
        print(f"  {au_key:>5}: holdout n={len(holdout_idx):>6}, pos={y_hold.mean():.3f}, "
              f"raw_prob_mean={raw_probs.mean():.3f}, "
              f"AUC={auc_pre:.4f}, F1@0.5: {f1_pre:.4f} → {f1_post:.4f} ({sign}{gain:+.4f})")

    # Save calibrators alongside other artifacts
    joblib.dump(calibrators, args.model_dir / "calibrators.joblib")
    (args.model_dir / "calibration_log.json").write_text(
        json.dumps(cal_log, indent=2, default=str)
    )
    print(f"\nsaved calibrators.joblib + calibration_log.json to {args.model_dir}")

    # Re-export .skops with calibrators embedded
    print(f"\nrebuilding {args.output_skops} with calibrators...")
    from feat.au_detectors.StatLearning.SL_test import XGBClassifier as PyFeatXGBClassifier
    from skops.io import dump as skops_dump

    wrapper = PyFeatXGBClassifier()
    wrapper.load_weights(
        scaler_upper=scalers["upper"], pca_model_upper=pcas["upper"],
        scaler_lower=scalers["lower"], pca_model_lower=pcas["lower"],
        scaler_full=scalers["full"], pca_model_full=pcas["full"],
        classifiers=classifiers,
        calibrators=calibrators,
    )
    args.output_skops.parent.mkdir(parents=True, exist_ok=True)
    skops_dump(wrapper, args.output_skops)
    print(f"  wrote {args.output_skops}")

    # Summary
    f1_gains = [cal_log[k]["f1_gain"] for k in cal_log]
    print(f"\nsummary: {len(cal_log)} AUs calibrated")
    print(f"  mean F1@0.5 gain: {np.mean(f1_gains):+.4f}")
    print(f"  median F1@0.5 gain: {np.median(f1_gains):+.4f}")
    print(f"  per-AU gains > 0: {sum(1 for g in f1_gains if g > 0)} / {len(f1_gains)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
