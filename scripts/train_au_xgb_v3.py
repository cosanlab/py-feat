"""Train xgb_au_v3: one XGBoost binary classifier per AU.

Consumes the extracted HOG features cached by
``scripts/extract_au_training_features.py`` (chunked .npz under
``bench-cache/au_train_*``) and trains 20 binary classifiers
(AU01..AU43) matching the existing py-feat AU detector schema.

Pipeline mirrors the v0.6/v0.7 XGBClassifier:
  1. Concatenate per-region PCA features (`pca_full` shared by most AUs;
     `pca_upper` for AU1/2/7; `pca_lower` for AU11/14/17/23/24/26).
  2. Append landmarks (68 x 2 = 136 floats).
  3. predict_proba -> probability of AU presence.

For v3 we keep the same wrapping but RE-FIT the scaler + PCA + classifier
on the corrected HOG features. Subject-disjoint k-fold CV is used to
pick hyperparameters per AU; final classifier is refit on all training
data with the best HPs.

Usage:
    python scripts/train_au_xgb_v3.py \\
        --input-dirs bench-cache/au_train_blackwell bench-cache/au_train_3090 \\
        --output-dir models/xgb_au_v3 \\
        --device cuda \\
        --n-trials 30
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

# AU regional grouping (mirrors feat/au_detectors/StatLearning/SL_test.py)
AU_KEYS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09",
    "AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20",
    "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
]
AU_TO_REGION = {
    # Upper-face AUs
    "AU01": "upper", "AU02": "upper", "AU07": "upper",
    # Lower-face AUs
    "AU11": "lower", "AU14": "lower", "AU17": "lower",
    "AU23": "lower", "AU24": "lower", "AU26": "lower",
}
# All others default to "full"

# Threshold for binarizing intensity labels (DISFA convention).
INTENSITY_BIN_THRESHOLD = 1.0  # >=1 means AU present


def load_features(input_dirs: list[Path]) -> dict:
    """Load all .npz chunks from extraction output dirs into in-memory arrays.

    Returns dict with keys: hog, landmarks, labels, subjects, sources,
    image_paths. All as np.ndarray.
    """
    chunks = []
    for d in input_dirs:
        chunks.extend(sorted(d.glob("chunk_*.npz")))
    if not chunks:
        raise RuntimeError(f"no chunk_*.npz found under {[str(d) for d in input_dirs]}")
    print(f"loading {len(chunks)} chunks from {len(input_dirs)} dirs...")

    parts = {k: [] for k in ("hog", "landmarks", "labels", "subjects", "sources", "image_paths")}
    for c in chunks:
        with np.load(c, allow_pickle=True) as z:
            for k in parts:
                parts[k].append(z[k])
    out = {}
    for k, v in parts.items():
        out[k] = np.concatenate(v, axis=0)
    return out


def binarize_labels(labels: np.ndarray, threshold: float = INTENSITY_BIN_THRESHOLD) -> np.ndarray:
    """Map intensity labels in [0, 5] (or {0, 1, NaN}) to binary {0, 1, NaN}."""
    out = np.where(np.isnan(labels), np.nan, (labels >= threshold).astype(np.float32))
    return out


def fit_region_pca(
    hog: np.ndarray,
    landmarks: np.ndarray,
    n_components: int = 256,
    region: str = "full",
):
    """Fit StandardScaler + PCA on the training set. Returns (scaler, pca)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Region-specific HOG slicing happens at extraction time. Here we
    # apply scaler/PCA on the FULL HOG output and let each AU classifier
    # pick the appropriate region's features. For v0.7+ unified HOG, we
    # use a single PCA per region keyed off the SAME hog array — this
    # mirrors v2's design where scaler_upper/lower/full are different
    # PCAs of the same features, optimized per-region by training on
    # samples where region-specific AUs are active.
    scaler = StandardScaler()
    hog_scaled = scaler.fit_transform(hog)
    pca = PCA(n_components=min(n_components, hog.shape[1], hog.shape[0]))
    pca.fit(hog_scaled)
    return scaler, pca


def transform_features(
    hog: np.ndarray,
    landmarks: np.ndarray,
    scaler,
    pca,
) -> np.ndarray:
    """Apply scaler + PCA to HOG, then concatenate landmarks."""
    scaled = scaler.transform(hog)
    projected = pca.transform(scaled)
    return np.concatenate([projected, landmarks], axis=1)


def subject_disjoint_kfold(subjects: np.ndarray, k: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) tuples with subject-disjoint splits."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    out = []
    for i in range(k):
        val_subjects = set(folds[i])
        val_mask = np.array([s in val_subjects for s in subjects])
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        out.append((train_idx, val_idx))
    return out


def train_one_au(
    au_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    subjects_train: np.ndarray,
    device: str,
    n_trials: int = 20,
    n_folds: int = 3,
) -> tuple[object, dict]:
    """Train one XGBoost binary classifier with subject-disjoint CV + Optuna.

    Returns (best_classifier_fit_on_all_data, hp_log dict).
    """
    import xgboost as xgb

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        optuna = None

    # Drop rows where label is NaN (this AU not scored in the source)
    mask = ~np.isnan(y_train)
    if mask.sum() < 100:
        return None, {"error": f"too few labels: {mask.sum()}"}
    X = X_train[mask]
    y = y_train[mask].astype(np.int8)
    w = sample_weight[mask]
    subj = subjects_train[mask]

    pos_rate = float(y.mean())
    if pos_rate < 0.005 or pos_rate > 0.995:
        return None, {"error": f"degenerate class balance pos_rate={pos_rate:.4f}"}

    # Class imbalance handling
    pos_weight = (1.0 - pos_rate) / pos_rate

    cv_splits = subject_disjoint_kfold(subj, k=n_folds, seed=42)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": device,
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": pos_weight,
            "verbosity": 0,
        }
        f1s = []
        from sklearn.metrics import f1_score
        for train_idx, val_idx in cv_splits:
            clf = xgb.XGBClassifier(**params)
            clf.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx], verbose=False)
            p = clf.predict_proba(X[val_idx])[:, 1]
            pred = (p >= 0.5).astype(np.int8)
            f1s.append(f1_score(y[val_idx], pred, zero_division=0))
        return float(np.mean(f1s))

    log = {"n_samples": int(len(y)), "pos_rate": pos_rate}
    if optuna is None:
        # Fall back to default hyperparameters if optuna isn't installed
        best_params = {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
        }
        log["hp_method"] = "default (optuna missing)"
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        log["hp_method"] = "optuna"
        log["best_val_f1"] = float(study.best_value)
        log["n_trials"] = n_trials

    log["best_params"] = best_params

    # Final fit on all data with best params
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": device,
        "scale_pos_weight": pos_weight,
        "verbosity": 0,
        **best_params,
    }
    clf = xgb.XGBClassifier(**final_params)
    clf.fit(X, y, sample_weight=w, verbose=False)
    return clf, log


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("models/xgb_au_v3"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-trials", type=int, default=20,
                   help="optuna HP search trials per AU")
    p.add_argument("--n-folds", type=int, default=3,
                   help="subject-disjoint k-fold CV per Optuna trial")
    p.add_argument("--pca-components", type=int, default=256)
    p.add_argument("--emotionet-weight", type=float, default=0.3,
                   help="sample weight for EmotioNet rows (semi-automatic labels)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_features(args.input_dirs)
    n = len(data["hog"])
    print(f"loaded {n} examples, HOG dim={data['hog'].shape[1]}")
    print(f"per-source counts: {dict(zip(*np.unique(data['sources'], return_counts=True)))}")

    hog = data["hog"]
    landmarks = data["landmarks"]
    labels = data["labels"]
    subjects = data["subjects"]
    sources = data["sources"]

    # Sample weights: downweight EmotioNet (semi-automatic labels)
    sample_weight = np.ones(n, dtype=np.float32)
    sample_weight[sources == "emotionet"] = args.emotionet_weight
    print(f"sample weights: emotionet={args.emotionet_weight}, others=1.0")

    # Fit ONE shared scaler+PCA (we can switch to per-region later if F1
    # turns out region-dependent; the v2 design used 3 separate PCAs
    # but the same HOG inputs).
    print(f"fitting StandardScaler + PCA-{args.pca_components} on {n} samples...")
    t0 = time.perf_counter()
    scaler, pca = fit_region_pca(hog, landmarks, n_components=args.pca_components)
    print(f"  done in {time.perf_counter()-t0:.1f}s; PCA explained var = {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Transform once for all AUs (same features per row)
    print(f"transforming features...")
    t0 = time.perf_counter()
    X_all = transform_features(hog, landmarks, scaler, pca)
    print(f"  done in {time.perf_counter()-t0:.1f}s; X shape = {X_all.shape}")

    # Binarize labels
    y_all = binarize_labels(labels)

    # Train per-AU
    classifiers = {}
    hp_log = {}
    t0 = time.perf_counter()
    label_cols = [f"AU{n_:02d}" if False else c for c, n_ in zip(AU_KEYS, range(len(AU_KEYS)))]
    # labels columns correspond to AU_KEYS in order; verify and slice
    n_aus = y_all.shape[1] if y_all.ndim > 1 else 1
    for j, au_key in enumerate(AU_KEYS):
        if j >= n_aus:
            print(f"\n  {au_key}: label column missing -> skipped")
            continue
        print(f"\n  training {au_key}...", flush=True)
        clf, log = train_one_au(
            au_key=au_key,
            X_train=X_all,
            y_train=y_all[:, j],
            sample_weight=sample_weight,
            subjects_train=subjects,
            device=args.device,
            n_trials=args.n_trials,
            n_folds=args.n_folds,
        )
        hp_log[au_key] = log
        if clf is not None:
            classifiers[au_key] = clf
            print(f"  {au_key}: n={log['n_samples']}, pos_rate={log['pos_rate']:.3f}, "
                  f"best F1={log.get('best_val_f1', 'n/a'):.4f}")
        else:
            print(f"  {au_key}: SKIPPED — {log.get('error', 'unknown')}")

    elapsed = time.perf_counter() - t0
    print(f"\nall AUs trained in {elapsed:.0f}s")

    # Save raw artifacts via joblib (one-off reusable form)
    import joblib
    joblib.dump(scaler, args.output_dir / "scaler.joblib")
    joblib.dump(pca, args.output_dir / "pca.joblib")
    joblib.dump(classifiers, args.output_dir / "classifiers.joblib")
    (args.output_dir / "hp_log.json").write_text(json.dumps(hp_log, indent=2, default=str))
    print(f"saved scaler, pca, classifiers, hp_log to {args.output_dir}")

    # Also produce a v2-compatible .skops file at <output_dir>/xgb_au_classifier_v3.skops.
    # Format: XGBClassifier instance with scaler_upper/lower/full and
    # pca_model_upper/lower/full populated (we use the SAME fitted
    # scaler+pca for all three regions — the v2 model used 3 separate
    # PCAs trained on region-specific samples, but the corrected HOG
    # makes a unified PCA more appropriate. Inference code reads
    # whichever region the AU is assigned to and it's the same data).
    print("\nexporting v3 .skops file...")
    from feat.au_detectors.StatLearning.SL_test import XGBClassifier as PyFeatXGBClassifier
    from skops.io import dump as skops_dump

    # Convert classifier dict keys from "AU01" -> "AU1" to match the
    # non-zero-padded au_keys used by the inference path.
    classifiers_unpadded = {}
    for au_padded, clf in classifiers.items():
        au_n = int(au_padded[2:])
        classifiers_unpadded[f"AU{au_n}"] = clf
    # For AUs we didn't train (skipped due to missing labels), fall
    # back to the v2 classifier so we still produce all 20 AUs. Load
    # v2 once and pull out the missing keys.
    expected_keys = [
        "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9",
        "AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20",
        "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
    ]
    missing = [k for k in expected_keys if k not in classifiers_unpadded]
    if missing:
        print(f"  v3 missing {len(missing)} classifiers; loading v2 as fallback for: {missing}")
        from huggingface_hub import hf_hub_download
        from skops.io import load as skops_load, get_untrusted_types
        from feat.utils.io import get_resource_path
        v2_path = hf_hub_download(
            repo_id="py-feat/xgb_au",
            filename="xgb_au_classifier_v2.skops",
            cache_dir=get_resource_path(),
        )
        v2 = skops_load(v2_path, trusted=get_untrusted_types(file=v2_path))
        for k in missing:
            if k in v2.classifiers:
                classifiers_unpadded[k] = v2.classifiers[k]
            else:
                print(f"  warning: AU {k} not in v2 either; v3 won't have it")

    wrapper = PyFeatXGBClassifier()
    wrapper.load_weights(
        scaler_upper=scaler, pca_model_upper=pca,
        scaler_lower=scaler, pca_model_lower=pca,
        scaler_full=scaler, pca_model_full=pca,
        classifiers=classifiers_unpadded,
    )
    out_skops = args.output_dir / "xgb_au_classifier_v3.skops"
    skops_dump(wrapper, out_skops)
    print(f"  wrote {out_skops}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
