"""Train xgb_au_v3.14 = v3.6 architecture + expanded BP4D+ corpus (197K rows).

v3.6 was the "honest" upper+lower regional-PCA model — no subject
normalization (so no train-only transform), no full PCA, but every AU
classifier sees [upper_pca + lower_pca + landmarks] = 648-dim features
without per-AU routing. v3.6 hit 0.4468 mean F1 on DISFA+ (just 0.015
behind v3.1's 0.4618), and won AU06 outright (0.635). Worth scaling.

v3.14 retrains v3.6 on the expanded corpus after the BP4D+ padding-bug
fix recovered 47,878 rows (149,960 → 197,838). Tests whether the simpler
"no-train-only-transform" architecture scales better — i.e. whether the
gap to v3.1 closes as data grows.

This script preserves v35.py's --feature-mode flag (full / upperlower /
all). Defaults to `upperlower` (= v3.6). Includes the load_chunks dedup
fix from v3.13 so old + new BP4D+ chunks combine cleanly.

Saves artifacts in a form the bench script (--feature-mode flag) can read.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

# Reuse the v3.1 helpers — they're the row-filtered PCA we want.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "v31", str(Path(__file__).resolve().parent / "train_au_xgb_v3_1.py")
)
v31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v31)


def _hog_region_split(hog_dim: int) -> tuple[int, int]:
    """Mirror v3.2's spatial-split boundary: upper_end=2414, lower_start=2221
    on Cheong's 5408-dim HOG. Scale proportionally for different HOG dims.
    """
    upper_end = int(round(hog_dim * 2414 / 5408))
    lower_start = int(round(hog_dim * 2221 / 5408))
    return upper_end, lower_start


def _fit_pca_no_filter(
    hog: np.ndarray, n_components: int, zero_mask: tuple[int, int] | None = None,
    label: str = "pca",
):
    """Fit StandardScaler + PCA on all rows (no label filter).

    If ``zero_mask`` is (start, end), those HOG cell columns are zeroed
    before fitting (matches v3.2's spatial-split convention). When the
    full PCA is desired, pass zero_mask=None.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    n = hog.shape[0]
    if n > 100_000:
        rng = np.random.default_rng(42 + hash(label) % 1000)
        idx = rng.choice(n, size=100_000, replace=False)
        sub = hog[idx]
    else:
        sub = hog
    sub = sub.copy()
    if zero_mask is not None:
        start, end = zero_mask
        sub[:, start:end] = 0.0
    scaler = StandardScaler()
    pca = PCA(n_components=min(n_components, sub.shape[1], sub.shape[0]))
    pca.fit(scaler.fit_transform(sub))
    print(f"  PCA[{label}]: fit on {len(sub)} rows; explained variance = "
          f"{pca.explained_variance_ratio_.sum() * 100:.1f}%")
    return scaler, pca


def build_features(
    hog: np.ndarray, landmarks: np.ndarray, labels: np.ndarray, mode: str,
    n_components: int = 256,
) -> tuple[np.ndarray, dict, dict, dict]:
    """Fit PCAs per `mode` and return (X, scalers, pcas, transforms).

    All PCAs are fit on ALL rows (no label filter, matching Cheong's
    approach in calculate_combo_CV.py). For multi-PCA modes the PCAs are
    differentiated by **spatial cell zeroing**, again matching Cheong.

    Modes:
      full        — one full-HOG PCA. X = [full_pca + landmarks].
      upperlower  — two PCAs (upper-cells-only, lower-cells-only),
                    NO full PCA. X = [upper_pca + lower_pca + landmarks].
      all         — three PCAs. X = [upper + lower + full + landmarks].
    """
    hog_dim = hog.shape[1]
    upper_end, lower_start = _hog_region_split(hog_dim)
    # Zero masks: which cells to zero out before fitting that PCA.
    # "upper" PCA sees only the upper-face cells → zero everything from upper_end onward.
    # "lower" PCA sees only the lower-face cells → zero everything before lower_start.
    transforms = {  # records what to zero at *inference* time too
        "upper": ("range", upper_end, hog_dim),
        "lower": ("range", 0, lower_start),
        "full":  None,  # no zeroing
    }
    print(f"  HOG dim={hog_dim}, upper sees [0:{upper_end}], lower sees [{lower_start}:{hog_dim}]")

    needed = {
        "full": ["full"],
        "upperlower": ["upper", "lower"],
        "all": ["upper", "lower", "full"],
    }[mode]

    scalers, pcas, feats = {}, {}, []
    for region in needed:
        zero_mask = None
        if transforms[region] is not None:
            _, start, end = transforms[region]
            zero_mask = (start, end)
        scaler, pca = _fit_pca_no_filter(
            hog, n_components=n_components, zero_mask=zero_mask, label=region,
        )
        scalers[region], pcas[region] = scaler, pca
        # Apply same zero-mask to transform the full HOG into this PCA basis.
        hog_t = hog.copy()
        if zero_mask is not None:
            hog_t[:, zero_mask[0]:zero_mask[1]] = 0.0
        feats.append(pca.transform(scaler.transform(hog_t)))
    # Stack PCA outputs + landmarks
    feats.append(landmarks)
    X = np.concatenate(feats, axis=1)
    print(f"  feature matrix [{mode}]: {X.shape} "
          f"({len(needed)} PCA(s) + landmarks)")
    used_transforms = {r: transforms[r] for r in needed}
    return X, scalers, pcas, used_transforms


def train_one_au_shared(
    au_key: str,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    sources: np.ndarray,
    sample_weight: np.ndarray,
    device: str,
    n_trials: int,
    n_folds: int,
) -> tuple[object, dict]:
    """Run v3.1's per-AU training but with a fixed feature matrix X.

    Reuses v31.train_one_au by passing X under all three region keys
    (it routes by AU_REGION but since every region maps to the same X,
    routing is a no-op).
    """
    X_feats = {"upper": X, "lower": X, "full": X}
    return v31.train_one_au(
        au_key=au_key,
        X_feats_by_region=X_feats,
        y=y,
        subjects=subjects,
        sources=sources,
        sample_weight=sample_weight,
        device=device,
        n_trials=n_trials,
        n_folds=n_folds,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--feature-mode", required=True,
                   choices=("full", "upperlower", "all"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--pca-components", type=int, default=256)
    p.add_argument("--emotionet-weight", type=float, default=0.3)
    p.add_argument("--subject-normalize", action="store_true",
                   help="Subtract per-(source,subject) mean HOG before PCA fit. "
                   "Match v3.1's design choice. Disabled by default to match "
                   "Cheong's approach.")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = v31.load_chunks(args.input_dirs)
    # Dedup by image_path: old BP4D+ chunks (in au_train_blackwell/3090) are
    # proper subsets of the new au_train_bp4dplus_v2 — keep first-seen path.
    paths = np.asarray(data["image_paths"], dtype=str)
    _, first_idx = np.unique(paths, return_index=True)
    first_idx.sort()
    if len(first_idx) < len(paths):
        n_before = len(paths)
        for k in data:
            data[k] = data[k][first_idx]
        print(f"deduped {n_before} → {len(first_idx)} ({n_before - len(first_idx)} duplicate paths dropped)")
    n = len(data["hog"])
    print(f"loaded {n} examples, HOG dim={data['hog'].shape[1]}")
    print(f"per-source: {dict(zip(*np.unique(data['sources'], return_counts=True)))}")

    hog = data["hog"]
    landmarks = data["landmarks"]
    labels = data["labels"]
    subjects = data["subjects"]
    sources = data["sources"]

    # Per-subject HOG normalization is OPTIONAL per --subject-normalize.
    # Match Cheong (no norm) by default. Set the flag to test v3.1's
    # design choice with the cleaner v3.5/v3.6/v3.7 PCA architectures.
    if args.subject_normalize:
        print("\nper-subject HOG normalization (matches v3.1)")
        t0 = time.perf_counter()
        hog = v31.per_subject_normalize(hog, subjects, sources)
        print(f"  done in {time.perf_counter()-t0:.0f}s")

    # Sample weights.
    sample_weight = np.ones(n, dtype=np.float32)
    sample_weight[sources == "emotionet"] = args.emotionet_weight

    # Build PCA + feature matrix per mode (no row filter).
    print(f"\nbuilding features [mode={args.feature_mode}, n_components={args.pca_components}]")
    X, scalers, pcas, transforms = build_features(
        hog, landmarks, labels, args.feature_mode,
        n_components=args.pca_components,
    )

    print(f"\nper-AU training (Optuna {args.n_trials} trials × {args.n_folds}-fold CV)")
    classifiers = {}
    hp_log = {}
    t0 = time.perf_counter()
    for j, au_key in enumerate(v31.AU_KEYS):
        if j >= labels.shape[1]:
            continue
        print(f"\n  training {au_key}...", flush=True)
        clf, log = train_one_au_shared(
            au_key=au_key,
            X=X,
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
                  f"best_val_F1={log.get('best_val_f1', 0):.4f}, "
                  f"thr={log['best_threshold']:.3f} → F1@thr={log['val_f1_at_best_threshold']:.4f}")
        else:
            print(f"  {au_key}: SKIPPED — {log.get('error')}")

    print(f"\nall AUs trained in {time.perf_counter()-t0:.0f}s")

    import joblib
    joblib.dump(scalers, args.output_dir / "scalers.joblib")
    joblib.dump(pcas, args.output_dir / "pcas.joblib")
    joblib.dump(classifiers, args.output_dir / "classifiers.joblib")
    (args.output_dir / "hp_log.json").write_text(json.dumps(hp_log, indent=2, default=str))
    (args.output_dir / "feature_mode.json").write_text(json.dumps({
        "feature_mode": args.feature_mode,
        "pca_components": args.pca_components,
        "regions_used": list(scalers.keys()),
        "transforms": transforms,
        "per_subject_norm": bool(args.subject_normalize),
        "row_filter": "none",
    }, indent=2))
    print(f"saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
