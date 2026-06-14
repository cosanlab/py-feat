"""Bench v3.5 / v3.6 / v3.7 models against DISFA+.

These ablation variants don't use the standard SL_test.XGBClassifier
inference wrapper (no per-AU regional routing). They store scalers,
pcas, and classifiers as joblib + a ``feature_mode.json`` describing
which PCAs were fit and how to compose features.

This script loads those artifacts, swaps a custom inference path into
the Detectorv1's AU detector slot, and runs the standard DISFA+ eval.

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
        feature_mode = self._mode_info.get("feature_mode")
        au_region = self._mode_info.get("au_region", {})

        # Per-region transformed feature matrices [N, D_region+136]
        feats_by_region = {}
        for region in self.regions_used:
            hog_t = frame
            t = transforms.get(region)
            if t is not None and t[0] == "range":
                hog_t = frame.copy()
                _, start, end = t
                hog_t[:, start:end] = 0.0
            pca_feat = self.pcas[region].transform(self.scalers[region].transform(hog_t))
            feats_by_region[region] = np.concatenate([pca_feat, landmarks], axis=1)

        # Default concat for non-per-AU modes
        if feature_mode != "per_au":
            X_concat = np.concatenate(
                [feats_by_region[r][:, :-landmarks.shape[1]] for r in self.regions_used]
                + [landmarks], axis=1,
            )

        out = []
        for key in self.au_keys:
            clf = self.classifiers.get(key)
            if clf is None:
                out.append(np.full(landmarks.shape[0], np.nan))
                continue
            if feature_mode == "per_au":
                # Per-AU routing: pick the single region this AU was trained on.
                # au_region keys are padded ("AU01"); self.au_keys are unpadded ("AU1").
                padded = f"AU{int(key[2:]):02d}"
                region = au_region.get(padded, self.regions_used[0])
                X_au = feats_by_region[region]
            else:
                X_au = X_concat
            out.append(clf.predict_proba(X_au)[:, 1])
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
    p.add_argument("--align-faces", action="store_true",
                   help="Procrustes-warp face crops to canonical template before HOG "
                        "(required for v3.8+ models trained on aligned features).")
    p.add_argument("--use-aligned", action=argparse.BooleanOptionalAction, default=False,
                   help="DISFA+ only: load pre-aligned crops (Aligned/) instead of "
                        "originals (Images/). Default False runs the full detection "
                        "pipeline on original images; pass --use-aligned to match the "
                        "older v3.x benches that scored pre-aligned crops.")
    args = p.parse_args()

    print(f"=== bench [{args.label}] on {args.dataset} ({args.device}) ===")
    scalers = joblib.load(args.model_dir / "scalers.joblib")
    pcas = joblib.load(args.model_dir / "pcas.joblib")
    classifiers = joblib.load(args.model_dir / "classifiers.joblib")
    mode_info = json.loads((args.model_dir / "feature_mode.json").read_text())
    print(f"    feature mode: {mode_info['feature_mode']} "
          f"(regions: {mode_info['regions_used']})")

    from feat.detector import Detectorv1
    from feat.evaluation import datasets, runner

    det = Detectorv1(
        face_model=args.face_model,
        landmark_model=args.landmark_model,
        au_model="xgb",
        emotion_model=None,
        identity_model=None,
        gaze_model=None,
        device=args.device,
    )
    det.au_detector = FeatureModeXGBClassifier(scalers, pcas, classifiers, mode_info)

    if args.align_faces:
        import pandas as _pd
        from feat.utils import face_mask
        from feat.utils.image_operations import procrustes_similarity_torch
        from feat.utils.geometry import warp_affine

        tpl_path = REPO_ROOT / "feat" / "resources" / "neutral_face_coordinates.csv"
        tpl_df = _pd.read_csv(tpl_path)
        aligned_template = torch.tensor(
            tpl_df[["x", "y"]].to_numpy(), dtype=torch.float32, device=args.device,
        )
        print(f"    Procrustes alignment ON (template: {tpl_path.name})")

        orig_extract = face_mask.extract_hog_features_batched

        def aligned_extract(extracted_faces, landmarks, hog_layer=None):
            ef = extracted_faces
            lm = landmarks
            if ef.shape[0] > 0:
                face_size = ef.shape[-1]
                N = ef.shape[0]
                lm_pix = lm.view(N, 68, 2).to(ef.device, dtype=ef.dtype) * face_size
                target_template = aligned_template.to(ef.device) * (face_size / 256.0)
                M = procrustes_similarity_torch(lm_pix, target_template)
                ef = warp_affine(ef, M, dsize=(face_size, face_size), mode="bilinear")
                ones = torch.ones(N, 68, 1, device=lm_pix.device, dtype=lm_pix.dtype)
                lm_hom = torch.cat([lm_pix, ones], dim=-1)
                lm_warped = torch.einsum("bij,bkj->bki", M, lm_hom)
                lm = (lm_warped / face_size).reshape(N, 136)
            return orig_extract(ef, lm, hog_layer=hog_layer)

        face_mask.extract_hog_features_batched = aligned_extract
        import feat.detector as _det_mod
        _det_mod.extract_hog_features_batched = aligned_extract

    if args.dataset == "disfa":
        split = datasets.load_disfa(split="P3", subset_size=args.subset_size, seed=args.seed)
    else:
        split = datasets.load_disfaplus(
            subset_size=args.subset_size, seed=args.seed, use_aligned=args.use_aligned,
        )
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
