"""Train xgb_au_v3.9 = v3.1's winning design but upper+lower PCA only (no full).

Tests whether dropping the "full" PCA (and routing every AU through one of
upper/lower) preserves v3.1's DISFA+ F1 (0.462) or hurts it.

Diff vs v3.3:
  - REGIONS = ["upper", "lower"] — no full PCA fit
  - AU_REGION extended to cover ALL 20 AUs (every AU routed by anatomy)
  - Wider HP space (alpha/lambda 0.01-200, max_depth 4-10, lr 1e-4-1.0,
    n_estimators 80-300, min_child_weight 4-60, gamma 0-15, colsample 0.5-1)
  - Default n_trials = 120

Usage:
    python scripts/train_au_xgb_v39.py \\
        --input-dirs bench-cache/au_train_blackwell \\
                     bench-cache/au_train_3090 \\
                     bench-cache/au_train_pain \\
                     bench-cache/au_train_bp4d_male \\
        --output-dir models/xgb_au_v39 \\
        --device cuda \\
        --n-trials 120
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

# Region mapping per AU — extended so EVERY AU routes to upper or lower.
# No "full" region in v3.9. Assignments follow FACS anatomy:
#   upper: brows + eyes + nose-bridge (AU04, AU05, AU06, AU09, AU43 added)
#   lower: mouth + chin + lips (AU10, AU12, AU15, AU20, AU25, AU28 added)
AU_REGION = {
    "AU01": "upper", "AU02": "upper", "AU04": "upper", "AU05": "upper",
    "AU06": "upper", "AU07": "upper", "AU09": "upper", "AU43": "upper",
    "AU10": "lower", "AU11": "lower", "AU12": "lower", "AU14": "lower",
    "AU15": "lower", "AU17": "lower", "AU20": "lower", "AU23": "lower",
    "AU24": "lower", "AU25": "lower", "AU26": "lower", "AU28": "lower",
}
AU_KEYS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09",
    "AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20",
    "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
]
REGIONS = ["upper", "lower"]
INTENSITY_BIN_THRESHOLD = 1.0  # >=1 => active (Cheong-compatible)

# Cheong's per-AU dataset filter (from finetune_xgb.py:train_au_dict).
# Map their internal names to our `source` codes.
CHEONG_TRAIN_AU_DICT = {
    "AU01": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU02": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU04": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU05": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU06": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU07": ["ckplus", "pain", "bp4d", "bp4dplus"],
    "AU09": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU10": ["ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU11": ["bp4d", "bp4dplus"],  # rare AU, only BP4D family scored it
    "AU12": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU14": ["bp4d", "bp4dplus"],
    "AU15": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU17": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU20": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU23": ["bp4d", "bp4dplus"],
    "AU24": ["bp4d", "bp4dplus", "emotionet"],
    "AU25": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU26": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU28": ["bp4d", "bp4dplus", "emotionet"],
    "AU43": ["pain", "emotionet"],
}


def load_chunks(input_dirs: list[Path]) -> dict:
    chunks = []
    for d in input_dirs:
        chunks.extend(sorted(d.glob("chunk_*.npz")))
    if not chunks:
        raise RuntimeError(f"no chunk_*.npz in {[str(d) for d in input_dirs]}")
    print(f"loading {len(chunks)} chunks from {len(input_dirs)} dirs...")
    parts = {k: [] for k in ("hog", "landmarks", "labels", "subjects", "sources", "image_paths")}
    for c in chunks:
        with np.load(c, allow_pickle=True) as z:
            for k in parts:
                parts[k].append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def per_subject_normalize(
    hog: np.ndarray,
    subjects: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    """Subtract per-(subject, source) HOG mean.

    This removes inter-individual variability (face shape, skin texture)
    so the model focuses on within-subject expression deltas. Critical
    for AU classifiers — see Cheong et al. and OpenFace for prior art.

    (subject, source) grouping handles the case where the same subject
    ID appears in two datasets (none currently — but defensive).
    """
    out = hog.copy()
    # Composite key
    keys = np.array([f"{src}::{s}" for src, s in zip(sources, subjects)])
    uniq = np.unique(keys)
    print(f"  per-subject normalization across {len(uniq)} (source, subject) groups")
    for k in uniq:
        mask = keys == k
        if mask.sum() < 5:
            # Too few frames to estimate subject baseline reliably.
            continue
        mean_vec = out[mask].mean(axis=0)
        out[mask] -= mean_vec
    return out


def fit_region_pca(
    hog: np.ndarray,
    region: str,
    n_components: int = 256,
    subjects: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    region_au_mask: np.ndarray | None = None,
):
    """Fit StandardScaler + PCA on subset of rows relevant to this region.

    Cheong's original design fit `pca_upper` on samples where upper-face
    AUs (1, 2, 7) were active; same idea for lower (11, 14, 17, 23, 24, 26)
    and full (everything else). The PCA's principal components then capture
    the variance specific to that region's expressions.

    We approximate this by: PCA_upper fit on rows where ANY upper-face AU
    is active or all are zero (so neutral too); PCA_lower analogous; PCA_full
    on a uniform random subsample.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if region_au_mask is not None and labels is not None:
        # Use rows where the region's AUs have non-NaN scores AND at least
        # one positive, OR a neutral (all-zero) sample. Cap at 100k rows.
        active = np.any(np.nan_to_num(labels[:, region_au_mask]) >= INTENSITY_BIN_THRESHOLD, axis=1)
        neutral = np.all(np.nan_to_num(labels[:, region_au_mask]) == 0, axis=1)
        keep = active | neutral
        idx = np.where(keep)[0]
    else:
        idx = np.arange(hog.shape[0])
    if len(idx) > 100_000:
        rng = np.random.default_rng(42 + hash(region) % 1000)
        idx = rng.choice(idx, size=100_000, replace=False)
    sub_hog = hog[idx]
    scaler = StandardScaler()
    sub_hog_scaled = scaler.fit_transform(sub_hog)
    pca = PCA(n_components=min(n_components, sub_hog.shape[1], sub_hog.shape[0]))
    pca.fit(sub_hog_scaled)
    print(f"  PCA[{region}]: fit on {len(idx)} rows; explained variance "
          f"= {pca.explained_variance_ratio_.sum()*100:.1f}%")
    return scaler, pca


def transform_features(
    hog: np.ndarray, landmarks: np.ndarray, scaler, pca,
) -> np.ndarray:
    return np.concatenate([pca.transform(scaler.transform(hog)), landmarks], axis=1)


def subject_disjoint_kfold(subjects, k=3, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    out = []
    for i in range(k):
        val = set(folds[i])
        val_mask = np.array([s in val for s in subjects])
        out.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return out


def find_best_threshold(probs, y_true):
    """Sweep decision threshold 0.05..0.95 to maximize F1 on val set."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.025):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def train_one_au(
    au_key: str,
    X_feats_by_region: dict[str, np.ndarray],
    y: np.ndarray,
    subjects: np.ndarray,
    sources: np.ndarray,
    sample_weight: np.ndarray,
    device: str,
    n_trials: int = 50,
    n_folds: int = 3,
    use_oversample: bool = True,
) -> tuple[object, dict]:
    """Train one XGBoost binary classifier with Cheong-style methodology."""
    import xgboost as xgb
    from sklearn.metrics import f1_score
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        optuna = None
    try:
        from imblearn.over_sampling import RandomOverSampler
    except ImportError:
        RandomOverSampler = None

    region = AU_REGION.get(au_key, "full")
    X_all = X_feats_by_region[region]

    # Per-AU dataset filtering (Cheong's train_au_dict).
    allowed = CHEONG_TRAIN_AU_DICT.get(au_key, ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet", "pain"])
    src_mask = np.isin(sources, allowed)
    label_mask = ~np.isnan(y)
    mask = src_mask & label_mask
    if mask.sum() < 100:
        return None, {"error": f"too few labels for {au_key} (n={mask.sum()})"}

    X = X_all[mask]
    yb = (y[mask] >= INTENSITY_BIN_THRESHOLD).astype(np.int8)
    w = sample_weight[mask]
    subj = subjects[mask]
    pos_rate = float(yb.mean())
    if pos_rate < 0.005 or pos_rate > 0.995:
        return None, {"error": f"degenerate pos_rate={pos_rate:.4f}", "n_samples": int(len(yb)),
                      "datasets_used": allowed}
    pos_weight = (1.0 - pos_rate) / pos_rate

    cv_splits = subject_disjoint_kfold(subj, k=n_folds, seed=42)

    def objective(trial):
        params = {
            "objective": trial.suggest_categorical(
                "objective", ["binary:logistic", "binary:hinge"]
            ),
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": device,
            # Cheong-faithful HP ranges (from /Storage/Projects/pyfeat_testing/tune_results
            # Cheong's wider ranges: alpha picks up to 70, lambda to 42, lr to 0.001).
            # n_estimators kept at Cheong's fixed 120 ± a small range.
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
            "subsample": 0.8,  # fixed like Cheong
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 60, step=4),
            "gamma": trial.suggest_float("gamma", 0.0, 15.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 200.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 200.0, log=True),
            "scale_pos_weight": pos_weight,
            "verbosity": 0,
        }
        # binary:hinge needs error metric, not logloss
        if params["objective"] == "binary:hinge":
            params["eval_metric"] = "error"

        oversample = use_oversample and trial.suggest_categorical(
            "oversample", [False, True]
        )

        f1s = []
        for train_idx, val_idx in cv_splits:
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = yb[train_idx], yb[val_idx]
            w_tr = w[train_idx]

            if oversample and RandomOverSampler is not None and yb[train_idx].mean() < 0.3:
                ros = RandomOverSampler(random_state=42)
                X_tr_r, y_tr_r = ros.fit_resample(X_tr, y_tr)
                # sample_weight is dropped when oversampling (imblearn doesn't propagate)
                w_tr_r = None
            else:
                X_tr_r, y_tr_r, w_tr_r = X_tr, y_tr, w_tr

            clf = xgb.XGBClassifier(**params)
            if params["objective"] == "binary:hinge":
                clf.fit(X_tr_r, y_tr_r, sample_weight=w_tr_r, verbose=False)
                preds = clf.predict(X_va)
            else:
                clf.fit(X_tr_r, y_tr_r, sample_weight=w_tr_r, verbose=False)
                probs = clf.predict_proba(X_va)[:, 1]
                preds = (probs >= 0.5).astype(np.int8)
            f1s.append(f1_score(y_va, preds, zero_division=0))
        return float(np.mean(f1s))

    log = {
        "n_samples": int(len(yb)),
        "pos_rate": pos_rate,
        "region": region,
        "datasets_used": allowed,
    }
    if optuna is None:
        best_params = {
            "n_estimators": 120, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 4,
            "gamma": 1.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
            "objective": "binary:logistic",
        }
        oversample = False
        log["hp_method"] = "default"
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = dict(study.best_params)
        oversample = best_params.pop("oversample", False)
        log["hp_method"] = "optuna"
        log["best_val_f1"] = float(study.best_value)
        log["n_trials"] = n_trials
    log["best_params"] = best_params
    log["oversample_used"] = oversample

    # Final fit on all data with best params
    final_params = {
        "objective": "binary:logistic",  # always use logistic for final so predict_proba works
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": device,
        "subsample": 0.8,
        "scale_pos_weight": pos_weight,
        "verbosity": 0,
        **{k: v for k, v in best_params.items() if k != "objective"},
    }
    if oversample and RandomOverSampler is not None and yb.mean() < 0.3:
        ros = RandomOverSampler(random_state=42)
        X_r, y_r = ros.fit_resample(X, yb)
        w_r = None
    else:
        X_r, y_r, w_r = X, yb, w
    clf = xgb.XGBClassifier(**final_params)
    clf.fit(X_r, y_r, sample_weight=w_r, verbose=False)

    # Per-AU threshold sweep on a held-out 20% val (subject-disjoint)
    train_idx, val_idx = cv_splits[0]
    probs_val = clf.predict_proba(X[val_idx])[:, 1]
    best_t, best_f1_thr = find_best_threshold(probs_val, yb[val_idx])
    log["best_threshold"] = best_t
    log["val_f1_at_best_threshold"] = float(best_f1_thr)
    log["val_f1_at_0.5"] = float(
        __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
            yb[val_idx], (probs_val >= 0.5).astype(int), zero_division=0
        )
    )
    return clf, log


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("models/xgb_au_v39"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-trials", type=int, default=120)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--pca-components", type=int, default=256)
    p.add_argument("--emotionet-weight", type=float, default=0.3)
    p.add_argument("--no-subject-norm", action="store_true",
                   help="Disable per-subject HOG normalization")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_chunks(args.input_dirs)
    n = len(data["hog"])
    print(f"loaded {n} examples, HOG dim={data['hog'].shape[1]}")
    print(f"per-source counts: {dict(zip(*np.unique(data['sources'], return_counts=True)))}")
    print(f"unique (source, subject) pairs: "
          f"{len(set(zip(data['sources'].tolist(), data['subjects'].tolist())))}")

    hog = data["hog"]
    landmarks = data["landmarks"]
    labels = data["labels"]
    subjects = data["subjects"]
    sources = data["sources"]

    # Step 1: per-subject HOG normalization (your suggestion)
    if not args.no_subject_norm:
        print("\nstep 1: per-subject HOG normalization")
        t0 = time.perf_counter()
        hog = per_subject_normalize(hog, subjects, sources)
        print(f"  done in {time.perf_counter()-t0:.0f}s")

    # Step 2: sample weights — downweight EmotioNet
    sample_weight = np.ones(n, dtype=np.float32)
    sample_weight[sources == "emotionet"] = args.emotionet_weight
    print(f"sample weights: emotionet={args.emotionet_weight}, others=1.0")

    # Step 3: fit 2 region-specific PCAs (upper + lower only, no full)
    print(f"\nstep 3: fitting 2 region-specific PCAs ({args.pca_components} comps)")
    region_au_masks = {}
    upper_aus = ["AU01", "AU02", "AU07"]
    lower_aus = ["AU11", "AU14", "AU17", "AU23", "AU24", "AU26"]
    region_au_masks["upper"] = np.isin(AU_KEYS, upper_aus)
    region_au_masks["lower"] = np.isin(AU_KEYS, lower_aus)

    scalers, pcas = {}, {}
    for region in REGIONS:
        scalers[region], pcas[region] = fit_region_pca(
            hog, region,
            n_components=args.pca_components,
            labels=labels,
            region_au_mask=region_au_masks[region],
        )

    # Step 4: transform features once per region
    print("\nstep 4: transforming features for each region")
    X_feats = {}
    for region in REGIONS:
        t0 = time.perf_counter()
        X_feats[region] = transform_features(hog, landmarks, scalers[region], pcas[region])
        print(f"  {region}: {X_feats[region].shape} ({time.perf_counter()-t0:.0f}s)")

    # Step 5: per-AU training
    print(f"\nstep 5: per-AU training (Optuna {args.n_trials} trials × {args.n_folds}-fold CV)")
    classifiers = {}
    hp_log = {}
    thresholds = {}
    t0 = time.perf_counter()
    for j, au_key in enumerate(AU_KEYS):
        if j >= labels.shape[1]:
            continue
        print(f"\n  training {au_key} (region={AU_REGION.get(au_key, 'full')})...", flush=True)
        clf, log = train_one_au(
            au_key=au_key,
            X_feats_by_region=X_feats,
            y=labels[:, j],
            subjects=subjects,
            sources=sources,
            sample_weight=sample_weight,
            device=args.device,
            n_trials=args.n_trials,
            n_folds=args.n_folds,
        )
        hp_log[au_key] = log
        if clf is not None:
            classifiers[au_key] = clf
            thresholds[au_key] = log["best_threshold"]
            print(f"  {au_key}: n={log['n_samples']}, pos={log['pos_rate']:.3f}, "
                  f"sources={','.join(log['datasets_used'])}, "
                  f"best_val_F1={log.get('best_val_f1', 0):.4f}, "
                  f"thr={log['best_threshold']:.3f} → F1@thr={log['val_f1_at_best_threshold']:.4f}")
        else:
            print(f"  {au_key}: SKIPPED — {log.get('error')}")

    print(f"\nall AUs trained in {time.perf_counter()-t0:.0f}s")

    # Save artifacts
    import joblib
    joblib.dump(scalers, args.output_dir / "scalers.joblib")
    joblib.dump(pcas, args.output_dir / "pcas.joblib")
    joblib.dump(classifiers, args.output_dir / "classifiers.joblib")
    joblib.dump(thresholds, args.output_dir / "thresholds.joblib")
    (args.output_dir / "hp_log.json").write_text(json.dumps(hp_log, indent=2, default=str))

    # v3.9 doesn't have a "full" PCA so the SL_test wrapper (which assumes
    # 3 PCAs + hardcoded routing) doesn't fit. Save a feature_mode.json
    # manifest compatible with bench_xgb_feature_mode.py's "per_au" mode:
    # bench reads au_region to pick which region's transform to apply per AU.
    feature_mode = {
        "feature_mode": "per_au",
        "regions_used": REGIONS,
        "au_region": AU_REGION,
        "per_subject_norm": True,
    }
    (args.output_dir / "feature_mode.json").write_text(json.dumps(feature_mode, indent=2))
    (args.output_dir / "thresholds.json").write_text(json.dumps({
        f"AU{int(k[2:])}": v for k, v in thresholds.items()
    }, indent=2))
    print(f"saved artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
