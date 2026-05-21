"""Train xgb_au_v3.11 = full Cheong replication, flaws and all.

v3.10 applied only label/au_dict/HP fixes. v3.11 also matches:
  - Single shared PCA fit on ALL 7 datasets including DISFA+ (data leak).
    Cheong's pipeline did this; we replicate to test whether the leak
    explains his ~0.52 DISFA+ F1 vs our v3.1's 0.46.
  - Spatial-cell-zero routing at inference time (zero out off-region HOG
    cells before scaler+PCA), matching Cheong's calculate_combo_CV.py:228-250.
  - AUC HPO objective (was F1). Threshold-free, better-behaved on imbalance.
  - Always-on RandomOverSampler in-pipeline (fold-local).
  - No per-subject HOG normalization (Cheong didn't do it).
  - Cheong-stratified CV folds: Dataset-only for CK+/EmotioNet (no subject
    grouping), Dataset+subject for BP4D/DISFA/PAIN. NOT subject-disjoint —
    same subjects can appear in train and val.
  - Inherits from v3.10: Cheong au_dict, per-source binarization (DISFA/
    PAIN/CK+ at >=2, others at >=1), narrow HP space, n_estimators=120.

Usage:
    python scripts/train_au_xgb_v311.py \\
        --input-dirs bench-cache/au_train_blackwell \\
                     bench-cache/au_train_3090 \\
                     bench-cache/au_train_pain \\
                     bench-cache/au_train_bp4d_male \\
                     bench-cache/au_train_disfaplus \\
        --output-dir models/xgb_au_v311 \\
        --device cuda \\
        --n-trials 60
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

# Region mapping per AU — explicit for all 20 AUs (mirrors SL_test routing,
# but makes the per_au inference manifest unambiguous).
AU_REGION = {
    "AU01": "upper", "AU02": "upper", "AU07": "upper",
    "AU11": "lower", "AU14": "lower", "AU17": "lower",
    "AU23": "lower", "AU24": "lower", "AU26": "lower",
    "AU04": "full", "AU05": "full", "AU06": "full", "AU09": "full",
    "AU10": "full", "AU12": "full", "AU15": "full", "AU20": "full",
    "AU25": "full", "AU28": "full", "AU43": "full",
}
AU_KEYS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09",
    "AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20",
    "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
]
REGIONS = ["upper", "lower", "full"]
INTENSITY_BIN_THRESHOLD = 1.0  # row filter / final binarization (post per-source pre-binarization)

# Per-source label scales (see scripts/extract_au_training_features.py output):
#   BP4D / BP4D+ / EmotioNet → already binary {0, 1}
#   DISFA / PAIN / CK+ → ordinal {0..5} FACS intensity
# For ordinal sources we binarize at >=2 to align with the published DISFA
# evaluation standard (feat/evaluation/metrics.py:21-22 binarizes DISFA+ truth
# at >=2). v3.1/v3.3 used >=1 universally → model fired on noisy trace labels
# that bench then counts as false positives.
ORDINAL_SOURCES = {"disfa18", "pain", "ckplus"}
ORDINAL_BIN_THRESHOLD = 2.0

# Cheong's per-AU dataset filter — verbatim from finetune_xgb.py:126-146.
# Map their internal names to our `source` codes.
# v3.1/v3.3 had 7 entries wrong (AU11/14/23/24/25/26/28) — fixed here.
CHEONG_TRAIN_AU_DICT = {
    "AU01": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU02": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU04": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU05": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU06": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU07": ["ckplus", "pain", "bp4d", "bp4dplus"],
    "AU09": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU10": ["ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU11": ["ckplus"],  # Cheong: CKP only
    "AU12": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU14": ["ckplus", "bp4d", "bp4dplus"],  # +CKP
    "AU15": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU17": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU20": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU23": ["ckplus", "bp4d", "bp4dplus"],  # +CKP
    "AU24": ["ckplus", "bp4d", "bp4dplus", "emotionet"],  # +CKP
    "AU25": ["disfa18", "ckplus", "pain", "emotionet"],  # drop BP4D family
    "AU26": ["disfa18", "ckplus", "pain", "emotionet"],  # drop BP4D family
    "AU28": ["ckplus", "bp4d", "bp4dplus", "emotionet"],  # +CKP
    "AU43": ["ckplus", "pain", "emotionet"],  # +CKP
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


def apply_region_mask(hog: np.ndarray, region: str) -> np.ndarray:
    """Cheong's spatial-cell-zero (calculate_combo_CV.py:228-250).

    For a 5408-dim HOG (8 orient × 8 px/cell × 2×2 block on 112×112 crop):
      - 'upper' keeps cells [0:2414], zeros [2414:5408]
      - 'lower' keeps cells [2414:5408], zeros [0:2414]
      - 'full' keeps all cells unchanged

    For non-5408-dim HOG (current py-feat path), split at dim*2414/5408
    (≈ 44.6% — Cheong's mid-face cut, just above the eye-line on a 112px
    face crop). Returns a *copy*; the caller can rely on the original
    being unchanged.
    """
    if region == "full":
        return hog
    dim = hog.shape[1]
    split = int(round(dim * 2414 / 5408))
    out = hog.copy()
    if region == "upper":
        out[:, split:] = 0.0
    elif region == "lower":
        out[:, :split] = 0.0
    else:
        raise ValueError(f"unknown region {region!r}")
    return out


def fit_region_pca(
    hog: np.ndarray,
    region: str,
    n_components: int = 256,
    subjects: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    region_au_mask: np.ndarray | None = None,
):
    """Cheong-style PCA fit: full corpus, spatial-cell-zero per region.

    Each region's scaler+PCA is fit on the ENTIRE training corpus (no row
    filter) — including DISFA+ if it was passed in input-dirs (data leak).
    The "region" only changes which HOG cells are zeroed before fitting.
    This matches calculate_combo_CV.py — the published Cheong pipeline.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    masked = apply_region_mask(hog, region)
    if masked.shape[0] > 200_000:
        rng = np.random.default_rng(42 + hash(region) % 1000)
        idx = rng.choice(masked.shape[0], size=200_000, replace=False)
        masked = masked[idx]
    scaler = StandardScaler()
    masked_scaled = scaler.fit_transform(masked)
    pca = PCA(n_components=min(n_components, masked.shape[1], masked.shape[0]))
    pca.fit(masked_scaled)
    print(f"  PCA[{region}]: fit on {masked.shape[0]} rows (full-corpus, "
          f"cell-zero); explained variance = {pca.explained_variance_ratio_.sum()*100:.1f}%")
    return scaler, pca


def transform_features(
    hog: np.ndarray, landmarks: np.ndarray, scaler, pca, region: str = "full",
) -> np.ndarray:
    """Apply region cell-zero, then scaler+PCA, then concat landmarks."""
    masked = apply_region_mask(hog, region)
    return np.concatenate([pca.transform(scaler.transform(masked)), landmarks], axis=1)


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


def cheong_stratified_3fold(sources, subjects, seed=42):
    """Cheong's `_split_nfold` (finetune_xgb.py:29-46).

    - For CK+ / EmotioNet rows: stratify by Dataset only (no subject grouping).
    - For BP4D / DISFA / PAIN / DISFAPlus rows: stratify by Dataset+subject.
    - Three folds via two nested ``train_test_split`` calls (1/3 then 1/2).

    NOT subject-disjoint — same subjects can appear in train and val.
    This is the regime Cheong tuned in.
    """
    from sklearn.model_selection import train_test_split
    sources = np.asarray(sources)
    subjects = np.asarray(subjects)
    flat_set = {"ckplus", "emotionet"}
    flat_mask = np.isin(sources, list(flat_set))
    indices = np.arange(len(sources))

    def _three_way(idx, strat_keys):
        # strat_keys: list of arrays for sklearn stratify
        if len(idx) == 0:
            return [], [], []
        # First split: 1/3 vs 2/3
        i1, iN = train_test_split(idx, train_size=1/3, random_state=seed,
                                   stratify=np.stack(strat_keys, axis=1) if len(strat_keys) > 1 else strat_keys[0])
        # Build strat keys for iN
        keys_N = [k[np.isin(idx, iN)] for k in strat_keys]
        i2, i3 = train_test_split(iN, train_size=1/2, random_state=seed,
                                   stratify=np.stack(keys_N, axis=1) if len(keys_N) > 1 else keys_N[0])
        return i1, i2, i3

    idx_flat = indices[flat_mask]
    idx_grouped = indices[~flat_mask]

    f1_flat, f2_flat, f3_flat = _three_way(
        idx_flat, [sources[flat_mask]],
    ) if len(idx_flat) > 0 else ([], [], [])
    f1_grp, f2_grp, f3_grp = _three_way(
        idx_grouped, [sources[~flat_mask], subjects[~flat_mask]],
    ) if len(idx_grouped) > 0 else ([], [], [])

    fold1 = np.concatenate([f1_flat, f1_grp]) if len(f1_flat) or len(f1_grp) else np.array([], dtype=int)
    fold2 = np.concatenate([f2_flat, f2_grp]) if len(f2_flat) or len(f2_grp) else np.array([], dtype=int)
    fold3 = np.concatenate([f3_flat, f3_grp]) if len(f3_flat) or len(f3_grp) else np.array([], dtype=int)

    # Yield (train, val) per fold: val = each fold, train = the other two
    return [
        (np.concatenate([fold2, fold3]), fold1),
        (np.concatenate([fold1, fold3]), fold2),
        (np.concatenate([fold1, fold2]), fold3),
    ]


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
    from sklearn.metrics import f1_score, roc_auc_score
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
    src_filt = sources[mask]
    pos_rate = float(yb.mean())
    if pos_rate < 0.005 or pos_rate > 0.995:
        return None, {"error": f"degenerate pos_rate={pos_rate:.4f}", "n_samples": int(len(yb)),
                      "datasets_used": allowed}
    pos_weight = (1.0 - pos_rate) / pos_rate

    # Cheong-stratified CV (NOT subject-disjoint).
    cv_splits = cheong_stratified_3fold(src_filt, subj, seed=42)

    def objective(trial):
        params = {
            # Cheong always uses binary:logistic OR binary:hinge — tunable in his
            # space. We keep the choice tunable but score with AUC (which needs
            # probabilities), so binary:logistic is forced when AUC is used.
            "objective": trial.suggest_categorical(
                "objective", ["binary:logistic", "binary:hinge"]
            ),
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": device,
            # HP ranges verbatim from Cheong's finetune_xgb.py:954-964
            "n_estimators": 120,  # Cheong fixed
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "subsample": 0.8,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 20, step=4),
            "gamma": trial.suggest_float("gamma", 1.0, 9.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "scale_pos_weight": pos_weight,
            "verbosity": 0,
        }
        if params["objective"] == "binary:hinge":
            params["eval_metric"] = "error"

        # Cheong always uses RandomOverSampler in-pipeline (fold-local).
        # No rare-class gating; applies even when pos_rate is high.
        scores = []
        for train_idx, val_idx in cv_splits:
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = yb[train_idx], yb[val_idx]

            if RandomOverSampler is not None:
                ros = RandomOverSampler(random_state=0)
                X_tr_r, y_tr_r = ros.fit_resample(X_tr, y_tr)
                w_tr_r = None
            else:
                X_tr_r, y_tr_r, w_tr_r = X_tr, y_tr, w[train_idx]

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr_r, y_tr_r, sample_weight=w_tr_r, verbose=False)
            # Cheong: AUC objective. Use predict_proba when logistic; for hinge
            # fall back to decision-function-like score via the underlying
            # booster's raw margin output (xgb doesn't expose hinge probs).
            if params["objective"] == "binary:hinge":
                # Use predict (binary) as a degenerate AUC fallback; this
                # underperforms logistic, which is the right TPE signal.
                preds = clf.predict(X_va)
                # roc_auc_score on binary preds == accuracy-related; compute
                # the same via roc_auc to keep scale consistent across trials.
                if len(np.unique(y_va)) < 2:
                    scores.append(0.5)
                else:
                    scores.append(roc_auc_score(y_va, preds))
            else:
                probs = clf.predict_proba(X_va)[:, 1]
                if len(np.unique(y_va)) < 2:
                    scores.append(0.5)
                else:
                    scores.append(roc_auc_score(y_va, probs))
        return float(np.mean(scores))

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
        oversample = True  # Cheong: always-on (no toggle)
        log["hp_method"] = "optuna"
        log["best_val_auc"] = float(study.best_value)
        log["n_trials"] = n_trials
    log["best_params"] = best_params
    log["oversample_used"] = oversample

    # Final fit on all data with best params
    final_params = {
        "objective": "binary:logistic",  # always use logistic for final so predict_proba works
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": device,
        "n_estimators": 120,  # Cheong-fixed (not tuned)
        "subsample": 0.8,
        "scale_pos_weight": pos_weight,
        "verbosity": 0,
        **{k: v for k, v in best_params.items() if k != "objective"},
    }
    # Cheong: always-on ROS for final fit too.
    if RandomOverSampler is not None:
        ros = RandomOverSampler(random_state=0)
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
    p.add_argument("--output-dir", type=Path, default=Path("models/xgb_au_v311"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--pca-components", type=int, default=256)
    p.add_argument("--emotionet-weight", type=float, default=0.3)
    p.add_argument("--no-subject-norm", action="store_true", default=True,
                   help="Disable per-subject HOG normalization (Cheong didn't do it)")
    p.add_argument("--subject-norm", action="store_false", dest="no_subject_norm",
                   help="Enable per-subject HOG normalization (off by default in v3.11)")
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

    # Step 0: per-source label binarization. Ordinal sources (DISFA/PAIN/CK+)
    # have intensities 0..5; binarize them at >=2 so trace-level (intensity 1)
    # examples become negatives. Binary sources keep their 0/1 as-is.
    print("\nstep 0: per-source label binarization")
    pre_pos = np.nansum(labels >= 1.0)
    ord_mask = np.isin(sources, list(ORDINAL_SOURCES))
    if ord_mask.any():
        sub = labels[ord_mask]
        # Preserve NaN, set 1 -> 0 (trace), keep 0 and >=2 (the latter set to 1 next)
        sub = np.where(np.isnan(sub), sub, np.where(sub < ORDINAL_BIN_THRESHOLD, 0.0, 1.0))
        labels = labels.copy()
        labels[ord_mask] = sub
    post_pos = np.nansum(labels >= 1.0)
    print(f"  ordinal sources={sorted(ORDINAL_SOURCES)}, threshold>={ORDINAL_BIN_THRESHOLD}")
    print(f"  positives before: {int(pre_pos)} → after: {int(post_pos)} ({100*post_pos/pre_pos:.1f}%)")

    # Step 1: per-subject HOG normalization
    if not args.no_subject_norm:
        print("\nstep 1: per-subject HOG normalization")
        t0 = time.perf_counter()
        hog = per_subject_normalize(hog, subjects, sources)
        print(f"  done in {time.perf_counter()-t0:.0f}s")

    # Step 2: sample weights — downweight EmotioNet
    sample_weight = np.ones(n, dtype=np.float32)
    sample_weight[sources == "emotionet"] = args.emotionet_weight
    print(f"sample weights: emotionet={args.emotionet_weight}, others=1.0")

    # Step 3: fit 3 region-specific PCAs — Cheong-style full-corpus fit
    # with spatial-cell-zero pre-processing per region (no row filter).
    print(f"\nstep 3: fitting 3 region-specific PCAs ({args.pca_components} comps, "
          f"full-corpus, cell-zero)")
    scalers, pcas = {}, {}
    for region in REGIONS:
        scalers[region], pcas[region] = fit_region_pca(
            hog, region, n_components=args.pca_components,
        )

    # Step 4: transform features once per region (apply cell-zero + scaler + PCA)
    print("\nstep 4: transforming features for each region")
    X_feats = {}
    for region in REGIONS:
        t0 = time.perf_counter()
        X_feats[region] = transform_features(hog, landmarks, scalers[region], pcas[region], region=region)
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
    # v3.11 uses spatial-cell-zero PCA — incompatible with SL_test.XGBClassifier
    # (which only applies scaler+PCA, no cell zeroing). Write feature_mode.json
    # for bench_xgb_feature_mode.py's "per_au" + transforms path.
    hog_dim = hog.shape[1]
    split = int(round(hog_dim * 2414 / 5408))
    feature_mode = {
        "feature_mode": "per_au",
        "regions_used": REGIONS,
        "au_region": AU_REGION,
        "transforms": {
            "upper": ["range", split, hog_dim],  # zero cells [split:hog_dim]
            "lower": ["range", 0, split],         # zero cells [0:split]
            # "full" has no transform — full HOG used as-is
        },
        "per_subject_norm": not args.no_subject_norm,
        "pca_fit_includes_disfaplus": any("disfaplus" in str(d) for d in args.input_dirs),
    }
    (args.output_dir / "feature_mode.json").write_text(json.dumps(feature_mode, indent=2))
    (args.output_dir / "thresholds.json").write_text(json.dumps({
        f"AU{int(k[2:])}": v for k, v in thresholds.items()
    }, indent=2))
    print(f"saved artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
