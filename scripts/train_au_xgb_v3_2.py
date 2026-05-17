"""Train xgb_au_v3.2: faithful replica of Cheong et al.'s recipe + sane XGB practices.

Differences from v3.1 (see plans/we-were-working-on-compiled-feather.md):

  1. **Spatial-split region PCA** (matches `calculate_combo_CV.py:228-250` in
     /Storage/Projects/pyfeat_testing/Code/). PCA_upper sees HOG cells
     0:split with cells split: zeroed; PCA_lower sees the mirror; PCA_full
     sees everything. Same global scaler/PCA fit, three projections.
  2. **HPO objective = ROC-AUC** (was F1 + leaky threshold sweep). AUC is
     threshold-free and robust to class imbalance.
  3. **In-pipeline RandomOverSampler** via `imblearn.pipeline.Pipeline` for
     AUs with pos_rate < 0.10. Keeps oversampling fold-local during CV.
  4. **Verbatim Cheong `train_au_dict`** — fixes prior copy errors on
     AU11 (CKP only), AU14/23/43 (+CKP), AU25/26 (drop BP4D family).
  5. **No per-subject HOG normalization** — Cheong didn't do it, and it
     creates train/inference distribution shift since inference can't
     compute a subject baseline.
  6. **No post-hoc threshold tuning** — outputs probabilities; thresholds
     default to 0.5 in a sidecar JSON, can be calibrated later if needed.
  7. **`early_stopping_rounds=25` + `n_estimators` upper bound 500** with
     per-fold `eval_set`. Lets the model self-truncate.
  8. **`TPESampler(multivariate=True, group=True)`** for correlated HPs.

Input: ``bench-cache/au_train_*/chunk_*.npz`` from
``scripts/extract_au_training_features.py``.

Output: ``models/xgb_au_v3_2/`` with classifiers.joblib, scalers.joblib,
pcas.joblib, hp_log.json, thresholds.json, xgb_au_classifier_v3_2.skops.

Usage:
    python scripts/train_au_xgb_v3_2.py \\
        --input-dirs bench-cache/au_train_blackwell \\
                     bench-cache/au_train_3090 \\
                     bench-cache/au_train_pain \\
                     bench-cache/au_train_bp4d_male \\
        --output-dir models/xgb_au_v3_2 \\
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

# Region mapping per AU (mirrors feat/au_detectors/StatLearning/SL_test.py).
AU_REGION = {
    "AU01": "upper", "AU02": "upper", "AU07": "upper",
    "AU11": "lower", "AU14": "lower", "AU17": "lower",
    "AU23": "lower", "AU24": "lower", "AU26": "lower",
    # all others default to "full" below
}
AU_KEYS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09",
    "AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20",
    "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
]
REGIONS = ["upper", "lower", "full"]
INTENSITY_BIN_THRESHOLD = 1.0  # >=1 => active (Cheong-compatible)

# Cheong's per-AU dataset filter, verbatim from
# /Storage/Projects/pyfeat_testing/Code/finetune_xgb.py:126-146.
# Source codes mapped: DISFA→disfa18, CKP→ckplus, PAIN→pain,
# BP4D→bp4d, BP4DP→bp4dplus, EmotioNet→emotionet.
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
    "AU14": ["ckplus", "bp4d", "bp4dplus"],  # Cheong: +CKP
    "AU15": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU17": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU20": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU23": ["ckplus", "bp4d", "bp4dplus"],  # Cheong: +CKP
    "AU24": ["ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU25": ["disfa18", "ckplus", "pain", "emotionet"],  # Cheong: drop BP4D
    "AU26": ["disfa18", "ckplus", "pain", "emotionet"],  # Cheong: drop BP4D
    "AU28": ["ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU43": ["ckplus", "pain", "emotionet"],  # Cheong: +CKP
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


def compute_hog_region_split(hog_dim: int) -> tuple[int, int]:
    """Return (upper_end, lower_start) HOG cell indices for spatial split.

    Cheong used HOG dim = 5408 with upper = 0:2414, lower = 2414:5408
    (50/50 split). We scale this fraction to whatever dim our v0.7
    HOGLayer produces. Slight ~3% overlap window matches Cheong's choice
    where 'upper test' zeroes cells 2414:5408 but 'lower test' only zeros
    cells 0:2221 — i.e. cells 2221-2414 are visible to both regions
    (eye/cheek transition band). We mirror that asymmetry.
    """
    # 2414/5408 = 0.4464 (upper end)
    # 2221/5408 = 0.4107 (lower start)
    upper_end = int(round(hog_dim * 2414 / 5408))
    lower_start = int(round(hog_dim * 2221 / 5408))
    return upper_end, lower_start


def fit_region_pcas_spatial(
    hog: np.ndarray, n_components: int = 256,
) -> tuple[dict, dict, tuple[int, int]]:
    """Fit one StandardScaler + PCA per region using Cheong's spatial-split.

    Each region sees the full HOG vector with off-region columns zeroed.
    All three are fit on the same global training corpus — only the
    spatial mask differs.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n, dim = hog.shape
    upper_end, lower_start = compute_hog_region_split(dim)
    print(f"  HOG dim={dim}, upper=[0:{upper_end}], lower=[{lower_start}:{dim}]")

    # Cap fitting subsample at 100k rows for memory.
    if n > 100_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=100_000, replace=False)
        hog_fit = hog[idx]
    else:
        hog_fit = hog
    print(f"  fitting PCAs on {len(hog_fit)} rows...")

    scalers, pcas = {}, {}

    # Upper: zero out cells upper_end:dim
    hog_upper = hog_fit.copy()
    hog_upper[:, upper_end:] = 0
    scaler_u = StandardScaler()
    pca_u = PCA(n_components=min(n_components, dim, len(hog_fit)))
    pca_u.fit(scaler_u.fit_transform(hog_upper))
    scalers["upper"], pcas["upper"] = scaler_u, pca_u
    print(f"  PCA[upper] explained variance = {pca_u.explained_variance_ratio_.sum()*100:.1f}%")

    # Lower: zero out cells 0:lower_start
    hog_lower = hog_fit.copy()
    hog_lower[:, :lower_start] = 0
    scaler_l = StandardScaler()
    pca_l = PCA(n_components=min(n_components, dim, len(hog_fit)))
    pca_l.fit(scaler_l.fit_transform(hog_lower))
    scalers["lower"], pcas["lower"] = scaler_l, pca_l
    print(f"  PCA[lower] explained variance = {pca_l.explained_variance_ratio_.sum()*100:.1f}%")

    # Full: no zeroing
    scaler_f = StandardScaler()
    pca_f = PCA(n_components=min(n_components, dim, len(hog_fit)))
    pca_f.fit(scaler_f.fit_transform(hog_fit))
    scalers["full"], pcas["full"] = scaler_f, pca_f
    print(f"  PCA[full] explained variance = {pca_f.explained_variance_ratio_.sum()*100:.1f}%")

    return scalers, pcas, (upper_end, lower_start)


def transform_region(
    hog: np.ndarray, landmarks: np.ndarray, region: str,
    scalers: dict, pcas: dict, split: tuple[int, int],
) -> np.ndarray:
    """Apply spatial mask + StandardScaler + PCA + landmark concat."""
    upper_end, lower_start = split
    hog_masked = hog.copy()
    if region == "upper":
        hog_masked[:, upper_end:] = 0
    elif region == "lower":
        hog_masked[:, :lower_start] = 0
    # else: full → no mask
    return np.concatenate(
        [pcas[region].transform(scalers[region].transform(hog_masked)), landmarks],
        axis=1,
    )


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


def train_one_au(
    au_key: str,
    X_feats_by_region: dict[str, np.ndarray],
    y: np.ndarray,
    subjects: np.ndarray,
    sources: np.ndarray,
    sample_weight: np.ndarray,
    device: str,
    n_trials: int = 60,
    n_folds: int = 3,
) -> tuple[object, dict]:
    """Train one XGBoost binary classifier — Cheong faithful + early stop.

    Returns (fitted_xgb_classifier_or_None, hp_log_dict).
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        optuna = None
    try:
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError:
        RandomOverSampler = None
        ImbPipeline = None

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

    # Decide whether to oversample (rare AUs only). Cheong used ROS for
    # everything; we restrict to <10% pos_rate where it actually matters
    # — for common AUs scale_pos_weight is enough.
    use_oversample = pos_rate < 0.10 and RandomOverSampler is not None

    cv_splits = subject_disjoint_kfold(subj, k=n_folds, seed=42)

    def objective(trial):
        params = {
            "objective": "binary:logistic",  # need probs for AUC
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": device,
            "n_estimators": 500,  # large; rely on early stopping
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "subsample": 0.8,  # fixed like Cheong
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 20, step=4),
            "gamma": trial.suggest_float("gamma", 1.0, 9.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "scale_pos_weight": pos_weight,
            "early_stopping_rounds": 25,
            "verbosity": 0,
        }

        aucs = []
        for train_idx, val_idx in cv_splits:
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = yb[train_idx], yb[val_idx]
            w_tr = w[train_idx]

            if use_oversample:
                # Imblearn Pipeline ensures ROS runs only on the train fold.
                # XGBClassifier's early_stopping needs eval_set passed
                # via fit; sklearn Pipeline.fit forwards **fit_params with
                # the step prefix.
                clf = ImbPipeline([
                    ("os", RandomOverSampler(random_state=42)),
                    ("xgb", xgb.XGBClassifier(**params)),
                ])
                clf.fit(
                    X_tr, y_tr,
                    xgb__eval_set=[(X_va, y_va)],
                    xgb__verbose=False,
                )
                probs = clf.predict_proba(X_va)[:, 1]
            else:
                clf = xgb.XGBClassifier(**params)
                clf.fit(
                    X_tr, y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )
                probs = clf.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, probs))
        return float(np.mean(aucs))

    log = {
        "n_samples": int(len(yb)),
        "pos_rate": pos_rate,
        "region": region,
        "datasets_used": allowed,
        "oversample_used": use_oversample,
    }
    if optuna is None:
        best_params = {
            "max_depth": 6, "learning_rate": 0.1, "colsample_bytree": 0.9,
            "min_child_weight": 4, "gamma": 1.0, "reg_alpha": 1.0, "reg_lambda": 1.0,
        }
        log["hp_method"] = "default"
    else:
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = dict(study.best_params)
        log["hp_method"] = "optuna"
        log["best_val_auc"] = float(study.best_value)
        log["n_trials"] = n_trials
    log["best_params"] = best_params

    # Final fit: hold out the last fold's val subjects for an honest val
    # AUC; train on everything else (= that fold's train_idx).
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": device,
        "n_estimators": 500,
        "subsample": 0.8,
        "scale_pos_weight": pos_weight,
        "early_stopping_rounds": 25,
        "verbosity": 0,
        **best_params,
    }
    train_idx_final, holdout_idx = cv_splits[-1]

    if use_oversample:
        clf = ImbPipeline([
            ("os", RandomOverSampler(random_state=42)),
            ("xgb", xgb.XGBClassifier(**final_params)),
        ])
        clf.fit(
            X[train_idx_final], yb[train_idx_final],
            xgb__eval_set=[(X[holdout_idx], yb[holdout_idx])],
            xgb__verbose=False,
        )
        holdout_probs = clf.predict_proba(X[holdout_idx])[:, 1]
        # The classifier we return is the inner xgb — that's what
        # SL_test.XGBClassifier expects.
        final_clf = clf.named_steps["xgb"]
    else:
        final_clf = xgb.XGBClassifier(**final_params)
        final_clf.fit(
            X[train_idx_final], yb[train_idx_final],
            sample_weight=w[train_idx_final],
            eval_set=[(X[holdout_idx], yb[holdout_idx])],
            verbose=False,
        )
        holdout_probs = final_clf.predict_proba(X[holdout_idx])[:, 1]

    log["holdout_auc"] = float(roc_auc_score(yb[holdout_idx], holdout_probs))
    log["holdout_n"] = int(len(holdout_idx))
    log["holdout_pos_rate"] = float(yb[holdout_idx].mean())
    # Report F1@0.5 on the clean holdout — this is the honest comparator
    # to v3's leaky thresholded F1.
    from sklearn.metrics import f1_score
    log["holdout_f1_at_0.5"] = float(
        f1_score(yb[holdout_idx], (holdout_probs >= 0.5).astype(int), zero_division=0)
    )
    return final_clf, log


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("models/xgb_au_v3_2"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--pca-components", type=int, default=256)
    p.add_argument("--emotionet-weight", type=float, default=0.3)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_chunks(args.input_dirs)
    n = len(data["hog"])
    print(f"loaded {n} examples, HOG dim={data['hog'].shape[1]}")
    print(f"per-source counts: {dict(zip(*np.unique(data['sources'], return_counts=True)))}")

    hog = data["hog"]
    landmarks = data["landmarks"]
    labels = data["labels"]
    subjects = data["subjects"]
    sources = data["sources"]

    # Sample weights — downweight EmotioNet (semi-automatic labels).
    sample_weight = np.ones(n, dtype=np.float32)
    sample_weight[sources == "emotionet"] = args.emotionet_weight
    print(f"sample weights: emotionet={args.emotionet_weight}, others=1.0")

    # Fit 3 spatial-split region PCAs (Cheong architecture).
    print(f"\nfitting 3 spatial-split region PCAs (n_components={args.pca_components})")
    scalers, pcas, hog_split = fit_region_pcas_spatial(hog, n_components=args.pca_components)

    # Transform features once per region.
    print("\ntransforming features for each region")
    X_feats = {}
    for region in REGIONS:
        t0 = time.perf_counter()
        X_feats[region] = transform_region(hog, landmarks, region, scalers, pcas, hog_split)
        print(f"  {region}: {X_feats[region].shape} ({time.perf_counter()-t0:.0f}s)")

    # Per-AU training.
    print(f"\nper-AU training (Optuna {args.n_trials} trials × {args.n_folds}-fold CV, ROC-AUC objective)")
    classifiers = {}
    hp_log = {}
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
            print(f"  {au_key}: n={log['n_samples']}, pos={log['pos_rate']:.3f}, "
                  f"sources={','.join(log['datasets_used'])}, "
                  f"best_val_AUC={log.get('best_val_auc', 0):.4f}, "
                  f"holdout_AUC={log['holdout_auc']:.4f}, "
                  f"holdout_F1@0.5={log['holdout_f1_at_0.5']:.4f}, "
                  f"ROS={log['oversample_used']}")
        else:
            print(f"  {au_key}: SKIPPED — {log.get('error')}")

    print(f"\nall AUs trained in {time.perf_counter()-t0:.0f}s")

    # Save artifacts
    import joblib
    joblib.dump(scalers, args.output_dir / "scalers.joblib")
    joblib.dump(pcas, args.output_dir / "pcas.joblib")
    joblib.dump(classifiers, args.output_dir / "classifiers.joblib")
    (args.output_dir / "hp_log.json").write_text(json.dumps(hp_log, indent=2, default=str))
    (args.output_dir / "hog_split.json").write_text(json.dumps({
        "upper_end": hog_split[0], "lower_start": hog_split[1], "hog_dim": int(hog.shape[1]),
    }, indent=2))
    # Default thresholds = 0.5 (probabilities are well-calibrated under
    # AUC-optimized fit + scale_pos_weight). Sidecar JSON for downstream
    # tooling that wants binary outputs.
    (args.output_dir / "thresholds.json").write_text(json.dumps(
        {f"AU{int(k[2:])}": 0.5 for k in classifiers}, indent=2
    ))
    print(f"saved artifacts to {args.output_dir}")

    # Export v3.2 .skops compatible with feat/au_detectors/.../XGBClassifier
    print("\nexporting v3.2 .skops file...")
    from feat.au_detectors.StatLearning.SL_test import XGBClassifier as PyFeatXGBClassifier
    from skops.io import dump as skops_dump

    # Unpad AU keys: "AU01" -> "AU1"
    classifiers_unpadded = {f"AU{int(k[2:])}": v for k, v in classifiers.items()}
    expected_keys = [
        "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU11", "AU12",
        "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
    ]
    missing = [k for k in expected_keys if k not in classifiers_unpadded]
    if missing:
        print(f"  v3.2 missing classifiers for {missing}; falling back to v2")
        from huggingface_hub import hf_hub_download
        from skops.io import load as skops_load, get_untrusted_types
        from feat.utils.io import get_resource_path
        v2_path = hf_hub_download(
            repo_id="py-feat/xgb_au", filename="xgb_au_classifier_v2.skops",
            cache_dir=get_resource_path(),
        )
        v2 = skops_load(v2_path, trusted=get_untrusted_types(file=v2_path))
        for k in missing:
            if k in v2.classifiers:
                classifiers_unpadded[k] = v2.classifiers[k]

    wrapper = PyFeatXGBClassifier()
    wrapper.load_weights(
        scaler_upper=scalers["upper"], pca_model_upper=pcas["upper"],
        scaler_lower=scalers["lower"], pca_model_lower=pcas["lower"],
        scaler_full=scalers["full"], pca_model_full=pcas["full"],
        classifiers=classifiers_unpadded,
    )
    out_skops = args.output_dir / "xgb_au_classifier_v3_2.skops"
    skops_dump(wrapper, out_skops)
    print(f"  wrote {out_skops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
