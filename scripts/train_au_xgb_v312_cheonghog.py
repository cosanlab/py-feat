"""xgb_au_v3.12 — train + bench on Cheong's stored HOG features directly.

Diagnostic only — measures the upper bound on F1 we can achieve through our
training stack if we feed it Cheong's own HOG features. This is NOT a model
we will ship: the HOG path is incompatible with v0.7's corrected face/landmark
pipeline. The point is to isolate the HOG-feature contribution to Cheong's
~0.520 DISFA+ F1.

Inputs (read-only from Cheong's storage):
  /Storage/Projects/pyfeat_testing/HOGFeatures/{ckp,pain,emotionet,bp4d,bp4dp,disfa,disfap}_hogs.pkl
  /Storage/Projects/pyfeat_testing/HOGFeatures/{same}_hog_2022.csv

Methodology mirrors v3.11 (full Cheong replication):
  - 3 PCAs fit on full-corpus including DISFA+ (data leak)
  - Spatial-cell-zero per region (cells [0:2414] for upper, [2414:5408] for lower)
  - Per-AU routing via AU_REGION
  - AUC HPO objective, in-pipeline RandomOverSampler always on
  - Cheong-stratified CV (dataset for CK+/EmotioNet, dataset+subject for others)
  - No subject normalization
  - Cheong-faithful au_dict
  - Per-source binarization (ordinal sources at >=2, binary at >=1)
  - Narrow HP space, n_estimators=120

Bench: predict on Cheong's stored DISFA+ HOGs (NOT through Detector pipeline),
score F1 vs binarized DISFA+ labels for the 12 AUs DISFA+ ships.
"""
from __future__ import annotations
import argparse, json, pickle, time, warnings
from pathlib import Path
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd

CHEONG_ROOT = Path("/Storage/Projects/pyfeat_testing/HOGFeatures")

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
ORDINAL_SOURCES = {"disfa18", "pain", "ckplus"}
ORDINAL_BIN_THRESHOLD = 2.0

# Cheong's verbatim au_dict (finetune_xgb.py:126-146).
CHEONG_TRAIN_AU_DICT = {
    "AU01": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU02": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU04": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU05": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU06": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU07": ["ckplus", "pain", "bp4d", "bp4dplus"],
    "AU09": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU10": ["ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU11": ["ckplus"],
    "AU12": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU14": ["ckplus", "bp4d", "bp4dplus"],
    "AU15": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU17": ["disfa18", "ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU20": ["disfa18", "ckplus", "pain", "bp4d", "bp4dplus", "emotionet"],
    "AU23": ["ckplus", "bp4d", "bp4dplus"],
    "AU24": ["ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU25": ["disfa18", "ckplus", "pain", "emotionet"],
    "AU26": ["disfa18", "ckplus", "pain", "emotionet"],
    "AU28": ["ckplus", "bp4d", "bp4dplus", "emotionet"],
    "AU43": ["ckplus", "pain", "emotionet"],
}


def _norm_au_col(c: str) -> str:
    """Cheong used 'AU1', 'AU1.0', etc. Normalize to 'AU01'.. style."""
    if not c.startswith("AU"):
        return c
    body = c[2:]
    # Drop trailing '.0' if present
    if body.endswith(".0"):
        body = body[:-2]
    if not body.replace(".", "").isdigit():
        return c
    try:
        n = int(float(body))
    except Exception:
        return c
    return f"AU{n:02d}"


def load_cheong_data(include_disfaplus: bool):
    """Concat all Cheong's HOGs + labels into one corpus."""
    NAME_TO_SOURCE = {
        "ckp": "ckplus", "pain": "pain", "emotionet": "emotionet",
        "bp4d": "bp4d", "bp4dp": "bp4dplus", "disfa": "disfa18", "disfap": "disfaplus",
    }
    names = ["ckp", "pain", "emotionet", "bp4d", "bp4dp", "disfa"]
    if include_disfaplus:
        names.append("disfap")

    hogs_parts, label_parts, subj_parts, src_parts = [], [], [], []
    for name in names:
        src = NAME_TO_SOURCE[name]
        with open(CHEONG_ROOT / f"{name}_hogs.pkl", "rb") as fp:
            h = pickle.load(fp)
        df = pd.read_csv(CHEONG_ROOT / f"{name}_hog_2022.csv", index_col=0, low_memory=False)
        df = df.dropna(subset=["land_locs"], how="all").reset_index(drop=True)
        assert len(df) == len(h), f"{name}: hog {len(h)} != master {len(df)}"

        # Normalize AU column names
        au_cols_orig = [c for c in df.columns if c.startswith("AU") and any(ch.isdigit() for ch in c)]
        rename_map = {c: _norm_au_col(c) for c in au_cols_orig}
        df = df.rename(columns=rename_map)

        # Build per-AU label matrix (NaN for missing AUs in this source)
        labels = np.full((len(df), len(AU_KEYS)), np.nan, dtype=np.float32)
        for j, au in enumerate(AU_KEYS):
            if au in df.columns:
                labels[:, j] = pd.to_numeric(df[au], errors="coerce").astype(np.float32).values
        # Replace Cheong's sentinel codes (999, 9 = not-coded)
        labels = np.where(np.isin(labels, [999, 9]), np.nan, labels)

        subjects = df["subject"].astype(str).values if "subject" in df.columns \
            else np.array([f"{name}_{i}" for i in range(len(df))], dtype=object)

        hogs_parts.append(h)
        label_parts.append(labels)
        subj_parts.append(subjects)
        src_parts.append(np.array([src] * len(df), dtype=object))
        print(f"  {src:12} n={len(df):>7}  AUs={sum(au in df.columns for au in AU_KEYS):>2}")

    hogs = np.concatenate(hogs_parts, axis=0).astype(np.float32)
    labels = np.concatenate(label_parts, axis=0)
    subjects = np.concatenate(subj_parts, axis=0)
    sources = np.concatenate(src_parts, axis=0)
    return hogs, labels, subjects, sources


def apply_region_mask(hog: np.ndarray, region: str) -> np.ndarray:
    if region == "full":
        return hog
    dim = hog.shape[1]
    split = int(round(dim * 2414 / 5408))  # = 2414 for 5408-dim
    out = hog.copy()
    if region == "upper":
        out[:, split:] = 0.0
    elif region == "lower":
        out[:, :split] = 0.0
    return out


def fit_region_pca(hog, region, n_components=256):
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
    print(f"  PCA[{region}]: fit on {masked.shape[0]} rows; "
          f"explained variance = {pca.explained_variance_ratio_.sum()*100:.1f}%")
    return scaler, pca


def transform_features(hog, scaler, pca, region):
    masked = apply_region_mask(hog, region)
    return pca.transform(scaler.transform(masked))


def cheong_stratified_3fold(sources, subjects, seed=42):
    from sklearn.model_selection import train_test_split
    sources = np.asarray(sources)
    subjects = np.asarray(subjects)
    flat_set = {"ckplus", "emotionet"}
    flat_mask = np.isin(sources, list(flat_set))
    indices = np.arange(len(sources))

    def _three_way(idx, strat_keys):
        if len(idx) == 0:
            return [], [], []
        if len(strat_keys) > 1:
            strat1 = np.stack(strat_keys, axis=1)
        else:
            strat1 = strat_keys[0]
        i1, iN = train_test_split(idx, train_size=1/3, random_state=seed, stratify=strat1)
        keys_N = [k[np.isin(idx, iN)] for k in strat_keys]
        if len(keys_N) > 1:
            strat2 = np.stack(keys_N, axis=1)
        else:
            strat2 = keys_N[0]
        i2, i3 = train_test_split(iN, train_size=1/2, random_state=seed, stratify=strat2)
        return i1, i2, i3

    idx_flat = indices[flat_mask]
    idx_grouped = indices[~flat_mask]
    f1f, f2f, f3f = _three_way(idx_flat, [sources[flat_mask]]) if len(idx_flat) else ([], [], [])
    f1g, f2g, f3g = _three_way(idx_grouped, [sources[~flat_mask], subjects[~flat_mask]]) if len(idx_grouped) else ([], [], [])
    fold1 = np.concatenate([f1f, f1g]) if len(f1f) or len(f1g) else np.array([], dtype=int)
    fold2 = np.concatenate([f2f, f2g]) if len(f2f) or len(f2g) else np.array([], dtype=int)
    fold3 = np.concatenate([f3f, f3g]) if len(f3f) or len(f3g) else np.array([], dtype=int)
    return [
        (np.concatenate([fold2, fold3]), fold1),
        (np.concatenate([fold1, fold3]), fold2),
        (np.concatenate([fold1, fold2]), fold3),
    ]


def train_one_au(au_key, X_feats_by_region, y, subjects, sources, n_trials, device):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        optuna = None
    from imblearn.over_sampling import RandomOverSampler

    region = AU_REGION.get(au_key, "full")
    X_all = X_feats_by_region[region]
    allowed = CHEONG_TRAIN_AU_DICT.get(au_key, list(set(sources)))
    src_mask = np.isin(sources, allowed)
    label_mask = ~np.isnan(y)
    mask = src_mask & label_mask
    if mask.sum() < 100:
        return None, {"error": f"too few labels (n={mask.sum()})"}
    X = X_all[mask]; yb = (y[mask] >= 1.0).astype(np.int8)  # already binarized upstream
    subj = subjects[mask]; src_filt = sources[mask]
    pos_rate = float(yb.mean())
    if pos_rate < 0.005 or pos_rate > 0.995:
        return None, {"error": f"degenerate pos_rate={pos_rate:.4f}", "n_samples": int(len(yb))}
    pos_weight = (1.0 - pos_rate) / pos_rate
    cv_splits = cheong_stratified_3fold(src_filt, subj, seed=42)

    def objective(trial):
        params = {
            "objective": trial.suggest_categorical("objective", ["binary:logistic", "binary:hinge"]),
            "eval_metric": "logloss", "tree_method": "hist", "device": device,
            "n_estimators": 120,
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "subsample": 0.8,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 20, step=4),
            "gamma": trial.suggest_float("gamma", 1.0, 9.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "scale_pos_weight": pos_weight, "verbosity": 0,
        }
        if params["objective"] == "binary:hinge":
            params["eval_metric"] = "error"
        scores = []
        for tr, va in cv_splits:
            X_tr, X_va = X[tr], X[va]; y_tr, y_va = yb[tr], yb[va]
            ros = RandomOverSampler(random_state=0)
            X_tr_r, y_tr_r = ros.fit_resample(X_tr, y_tr)
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr_r, y_tr_r, verbose=False)
            if params["objective"] == "binary:hinge":
                preds = clf.predict(X_va)
                scores.append(0.5 if len(np.unique(y_va)) < 2 else roc_auc_score(y_va, preds))
            else:
                probs = clf.predict_proba(X_va)[:, 1]
                scores.append(0.5 if len(np.unique(y_va)) < 2 else roc_auc_score(y_va, probs))
        return float(np.mean(scores))

    log = {"n_samples": int(len(yb)), "pos_rate": pos_rate, "region": region, "datasets_used": allowed}
    if optuna is None:
        best_params = {"max_depth": 6, "learning_rate": 0.1, "colsample_bytree": 0.9,
                       "min_child_weight": 4, "gamma": 1.0, "reg_alpha": 1.0, "reg_lambda": 1.0,
                       "objective": "binary:logistic"}
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = dict(study.best_params)
        log["best_val_auc"] = float(study.best_value)
        log["n_trials"] = n_trials
    log["best_params"] = best_params

    final_params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "tree_method": "hist", "device": device,
        "n_estimators": 120, "subsample": 0.8, "scale_pos_weight": pos_weight,
        "verbosity": 0,
        **{k: v for k, v in best_params.items() if k != "objective"},
    }
    ros = RandomOverSampler(random_state=0)
    X_r, y_r = ros.fit_resample(X, yb)
    clf = xgb.XGBClassifier(**final_params)
    clf.fit(X_r, y_r, verbose=False)
    return clf, log


def bench_on_cheong_disfap(scalers, pcas, classifiers, threshold=0.5):
    """Bench: predict on Cheong's stored DISFA+ HOGs.

    Cheong's master labels are 0..5 intensity; binarize at >=2 (matches our protocol).
    Returns per-AU F1 over the 12 AUs DISFA+ scores.
    """
    from sklearn.metrics import f1_score
    with open(CHEONG_ROOT / "disfap_hogs.pkl", "rb") as fp:
        disfap_hog = pickle.load(fp).astype(np.float32)
    df = pd.read_csv(CHEONG_ROOT / "disfap_hog_2022.csv", index_col=0, low_memory=False)
    df = df.dropna(subset=["land_locs"], how="all").reset_index(drop=True)
    # Rename to AU01.. format
    rename = {c: _norm_au_col(c) for c in df.columns if c.startswith("AU")}
    df = df.rename(columns=rename)

    print(f"\nbench on Cheong DISFA+ HOG: n={len(disfap_hog)}")
    disfap_aus = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU09",
                  "AU12", "AU15", "AU17", "AU20", "AU25", "AU26"]
    feats_by_region = {}
    for region in REGIONS:
        feats_by_region[region] = transform_features(disfap_hog, scalers[region], pcas[region], region)

    per_au = {}
    for au in disfap_aus:
        if au not in classifiers:
            per_au[au] = float('nan'); continue
        if au not in df.columns:
            per_au[au] = float('nan'); continue
        region = AU_REGION.get(au, "full")
        # NO landmarks concat — Cheong used HOG-PCA + landmarks but here we test
        # pure HOG-PCA→XGB to isolate HOG-feature contribution. (We can add
        # landmarks too via Cheong's stored *_lands.pkl if needed; the current
        # bench focuses on HOG.)
        probs = classifiers[au].predict_proba(feats_by_region[region])[:, 1]
        preds = (probs >= threshold).astype(np.int8)
        y_int = pd.to_numeric(df[au], errors="coerce").fillna(-1).astype(int).values
        valid = (y_int >= 0)
        y_bin = (y_int[valid] >= 2).astype(np.int8)
        p_bin = preds[valid]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_au[au] = float('nan')
        else:
            per_au[au] = float(f1_score(y_bin, p_bin, zero_division=0))
    return per_au


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-disfaplus-in-pca", action="store_true", default=True,
                   help="Include DISFA+ in PCA fit (Cheong's data leak — default ON)")
    p.add_argument("--exclude-disfaplus-from-pca", action="store_false",
                   dest="include_disfaplus_in_pca")
    p.add_argument("--output-dir", type=Path, default=Path("models/xgb_au_v312"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--pca-components", type=int, default=256)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== v3.12: train on Cheong's stored HOG features ===")
    print(f"\nstep 0: load Cheong's HOGs + labels (DISFA+ in PCA: {args.include_disfaplus_in_pca})")
    hog, labels, subjects, sources = load_cheong_data(include_disfaplus=args.include_disfaplus_in_pca)
    print(f"total: hog={hog.shape}  labels={labels.shape}  n_subjects={len(np.unique(subjects))}")

    # Per-source binarization (ordinal → >=2, binary kept)
    print("\nstep 1: per-source label binarization")
    pre = np.nansum(labels >= 1.0)
    ord_mask = np.isin(sources, list(ORDINAL_SOURCES))
    if ord_mask.any():
        sub = labels[ord_mask]
        sub = np.where(np.isnan(sub), sub, np.where(sub < ORDINAL_BIN_THRESHOLD, 0.0, 1.0))
        labels = labels.copy()
        labels[ord_mask] = sub
    post = np.nansum(labels >= 1.0)
    print(f"  ordinal {sorted(ORDINAL_SOURCES)} → >={ORDINAL_BIN_THRESHOLD}")
    print(f"  positives: {int(pre)} → {int(post)} ({100*post/pre:.1f}%)")

    # Step 2: fit 3 region PCAs (full-corpus, spatial-cell-zero)
    print("\nstep 2: fit region PCAs")
    scalers, pcas = {}, {}
    for region in REGIONS:
        scalers[region], pcas[region] = fit_region_pca(hog, region, n_components=args.pca_components)

    # Step 3: transform features
    print("\nstep 3: transform features per region")
    X_feats = {}
    for region in REGIONS:
        t0 = time.perf_counter()
        X_feats[region] = transform_features(hog, scalers[region], pcas[region], region)
        print(f"  {region}: {X_feats[region].shape} ({time.perf_counter()-t0:.0f}s)")

    # Step 4: per-AU training
    print(f"\nstep 4: per-AU training (Optuna {args.n_trials} trials × 3-fold Cheong-stratified CV)")
    classifiers = {}; hp_log = {}
    t0 = time.perf_counter()
    for j, au_key in enumerate(AU_KEYS):
        print(f"\n  training {au_key} (region={AU_REGION.get(au_key)})...", flush=True)
        clf, log = train_one_au(au_key, X_feats, labels[:, j], subjects, sources,
                                n_trials=args.n_trials, device=args.device)
        hp_log[au_key] = log
        if clf is not None:
            classifiers[au_key] = clf
            print(f"  {au_key}: n={log['n_samples']}, pos={log['pos_rate']:.3f}, "
                  f"best_val_AUC={log.get('best_val_auc', 0):.4f}")
        else:
            print(f"  {au_key}: SKIPPED — {log.get('error')}")
    print(f"\nall AUs trained in {time.perf_counter()-t0:.0f}s")

    # Step 5: bench on Cheong's DISFA+ HOG
    per_au = bench_on_cheong_disfap(scalers, pcas, classifiers)
    print(f"\n=== v3.12 on Cheong's DISFA+ HOG (12 AUs) ===")
    for au in sorted(per_au):
        v = per_au[au]
        print(f"  {au}: F1={v:.4f}" if v == v else f"  {au}: F1=nan")
    vals = [v for v in per_au.values() if v == v]
    print(f"\n  mean F1 = {sum(vals)/len(vals):.4f}  (n={len(vals)})")

    # Save artifacts (diagnostic only — won't ship)
    import joblib
    joblib.dump(scalers, args.output_dir / "scalers.joblib")
    joblib.dump(pcas, args.output_dir / "pcas.joblib")
    joblib.dump(classifiers, args.output_dir / "classifiers.joblib")
    (args.output_dir / "hp_log.json").write_text(json.dumps(hp_log, indent=2, default=str))
    (args.output_dir / "bench_results.json").write_text(json.dumps({
        "per_au": per_au,
        "mean": sum(vals)/len(vals) if vals else None,
        "n_aus": len(vals),
    }, indent=2))
    print(f"\nsaved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
