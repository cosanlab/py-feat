"""Glue between a Detectorv1 instance and ground-truth labels.

``evaluate_dataset(detector, dataset_split)`` runs the detector over the
images in a :class:`DatasetSplit`, aligns the per-image top face back to
the row order of the ground-truth DataFrame, and returns a metrics dict.

Top-face selection: each input here is already a face-aligned crop (see
``feat/evaluation/datasets.py``), so ~1 face is expected per image. When
the detector returns 0 faces we record NaN and that sample is excluded
from metric aggregation. When the detector returns >1 face we keep the
one with the highest ``FaceScore`` — this happens on aligned crops only
when a spurious detection fires (rare).
"""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from feat.evaluation.datasets import (
    AFFECTNET_EMOTION_MAP,
    DISFA_AU_RENAME,
    DatasetSplit,
)
from feat.evaluation.metrics import (
    cosine_similarity_pairs,
    emotion_accuracy,
    emotion_f1_macro,
    rank_k_identification,
    summarize_au_metrics,
    verification_accuracy_lfw_10fold,
)

DISFA_AU_OUTPUT_COLUMNS = list(DISFA_AU_RENAME.values())  # ["AU01", ..., "AU26"]
EMOTION_COLS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


def _detect(
    detector,
    image_paths: list[str],
    batch_size: int,
    num_workers: int,
    output_size: int | None = None,
) -> pd.DataFrame:
    """Run ``detector.detect`` and return the raw Fex DataFrame.

    ``output_size`` pads varied-size inputs to a fixed dimension so
    ``batch_size > 1`` works on heterogeneous datasets (DISFA aligned
    crops are not uniform; AffectNet originals are varying resolutions).
    """
    return detector.detect(
        image_paths,
        data_type="image",
        output_size=output_size,
        batch_size=batch_size,
        num_workers=num_workers,
        progress_bar=True,
    )


def _top_face_per_frame(fex: pd.DataFrame) -> pd.DataFrame:
    """Keep only the highest-scoring face per ``frame`` row.

    ``Detectorv1.detect`` returns one row per detected face. For aligned-crop
    eval we expect one face per image and want a per-image row to align
    against the ground-truth DataFrame. Sort by FaceScore desc, then drop
    duplicate frames. Frames with zero detections will simply be absent
    from ``fex`` and reappear here as NaN-filled rows after the reindex.
    """
    if "FaceScore" in fex.columns:
        fex = fex.sort_values("FaceScore", ascending=False)
    fex = fex.drop_duplicates(subset="frame", keep="first")
    return fex.sort_values("frame").reset_index(drop=True)


def evaluate_dataset(
    detector,
    split: DatasetSplit,
    batch_size: int = 1,
    num_workers: int = 0,
    output_size: int | None = 512,
) -> dict:
    """Score ``detector`` on a single :class:`DatasetSplit`. Returns metrics dict."""
    t0 = time.perf_counter()
    fex = _detect(detector, split.image_paths, batch_size, num_workers, output_size=output_size)
    elapsed = time.perf_counter() - t0
    fex = _top_face_per_frame(fex)

    n_total = len(split.image_paths)
    fex = fex.set_index("frame").reindex(range(n_total)).reset_index(drop=True)

    out: dict = {
        "n_samples": n_total,
        "n_faces_detected": int(fex["FaceScore"].notna().sum()) if "FaceScore" in fex else n_total,
        "elapsed_s": round(elapsed, 2),
        "fps": round(n_total / elapsed, 2) if elapsed > 0 else None,
    }

    if split.metric_kind == "au_intensity":
        out.update(_score_au(fex, split.labels))
    elif split.metric_kind == "emotion_class":
        out.update(_score_emotion(fex, split.labels))
    elif split.metric_kind == "gaze_angular":
        out.update(_score_gaze(fex, split.labels))
    elif split.metric_kind in ("identity_pairs", "identity_search"):
        raise ValueError(
            f"{split.metric_kind!r} uses evaluate_identity(), not evaluate_dataset()"
        )
    else:
        raise ValueError(f"Unknown metric_kind: {split.metric_kind!r}")
    return out


# ---------------------------------------------------------------------------
# Identity evaluation — bypasses face detection.
#
# CALFW / CPLFW / TinyFace crops are already face-aligned (224x224 RGB or
# tiny surveillance crops). Running them through Detectorv1.detect would
# waste cycles on a redundant retinaface pass *and* conflate face-detection
# changes with identity-model changes. We load ArcFace directly and feed
# the aligned crops in. The ArcFace wrapper auto-resizes to 112x112 and
# rescales [0, 1] -> [-1, 1] internally.
# ---------------------------------------------------------------------------


def _load_arcface_direct(device: str = "cpu"):
    """Load ArcFace and its weights without spinning up the full Detectorv1."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from feat.identity_detectors.arcface.arcface_model import ArcFace
    from feat.utils.io import get_resource_path

    model = ArcFace(backbone="r50")
    weights_path = hf_hub_download(
        repo_id="py-feat/arcface_r50",
        filename="arcface_r50.safetensors",
        cache_dir=get_resource_path(),
    )
    model.net.load_state_dict(load_file(weights_path), strict=False)
    model.eval()
    model.to(device)
    return model


# InsightFace's reference 5-point template for 112x112 ArcFace input.
# Order: left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner.
# Source: deepinsight/insightface arcface_torch.utils.face_align.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def _read_5pt_landmarks(path: str) -> np.ndarray:
    """Parse a 5-line ``<x> <y>`` landmark text file -> [5, 2] float array."""
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = line.split()
            pts.append([float(x), float(y)])
            if len(pts) == 5:
                break
    return np.asarray(pts, dtype=np.float64)


def _arcface_align(image_pil, landmarks: np.ndarray):
    """Apply similarity transform so 5 landmarks land on the ArcFace template.

    Returns a 112x112 RGB PIL image.
    """
    from PIL import Image
    from skimage.transform import SimilarityTransform, warp

    # Newer skimage prefers the class-method constructor; fall back for older.
    if hasattr(SimilarityTransform, "from_estimate"):
        tform = SimilarityTransform.from_estimate(landmarks, ARCFACE_TEMPLATE_112)
    else:
        tform = SimilarityTransform()
        tform.estimate(landmarks, ARCFACE_TEMPLATE_112)
    arr = np.asarray(image_pil, dtype=np.float32) / 255.0
    aligned = warp(arr, tform.inverse, output_shape=(112, 112), order=1, mode="constant")
    return Image.fromarray(np.clip(aligned * 255.0, 0, 255).astype(np.uint8))


def _embed_paths(
    model,
    paths: list[str],
    device: str,
    batch_size: int = 64,
    landmark_paths: list[str] | None = None,
) -> np.ndarray:
    """Read PIL images, stack into [N,3,112,112] in [0,1], embed with ArcFace.

    When ``landmark_paths`` is provided, each image is aligned to the
    InsightFace 5-point template before embedding. Otherwise images are
    bilinearly resized to 112x112 (TinyFace path — no landmarks available).
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    to_tensor = transforms.ToTensor()  # PIL -> [3,H,W] in [0,1]

    embeds = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors = []
            for j, p in enumerate(batch_paths):
                img = Image.open(p).convert("RGB")
                if landmark_paths is not None:
                    lm = _read_5pt_landmarks(landmark_paths[start + j])
                    img = _arcface_align(img, lm)
                tensors.append(to_tensor(img))
            tensors = [
                t if t.shape[-1] == 112 and t.shape[-2] == 112
                else torch.nn.functional.interpolate(
                    t.unsqueeze(0), size=(112, 112),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
                for t in tensors
            ]
            batch = torch.stack(tensors).to(device)
            out = model(batch).cpu().numpy()
            embeds.append(out)
    return np.concatenate(embeds, axis=0)


def evaluate_identity(
    split: DatasetSplit,
    device: str = "cpu",
    batch_size: int = 64,
) -> dict:
    """Score a CALFW/CPLFW/TinyFace split. Loads ArcFace internally."""
    t0 = time.perf_counter()
    model = _load_arcface_direct(device=device)

    if split.metric_kind == "identity_pairs":
        # Unique-path embedding: each image often appears in multiple
        # pairs (LFW protocol reuses images). Dedupe before embedding.
        # Preserve the path -> landmark mapping for alignment.
        path_to_lm: dict[str, str] = {}
        if "landmark_a" in split.labels.columns:
            for _, row in split.labels.iterrows():
                path_to_lm[row["path_a"]] = row["landmark_a"]
                path_to_lm[row["path_b"]] = row["landmark_b"]
        all_paths = sorted(
            set(split.labels["path_a"].tolist()) | set(split.labels["path_b"].tolist())
        )
        lm_paths = [path_to_lm[p] for p in all_paths] if path_to_lm else None
        emb = _embed_paths(
            model, all_paths, device=device, batch_size=batch_size,
            landmark_paths=lm_paths,
        )
        index = {p: i for i, p in enumerate(all_paths)}
        idx_a = split.labels["path_a"].map(index).values
        idx_b = split.labels["path_b"].map(index).values
        sim = cosine_similarity_pairs(emb[idx_a], emb[idx_b])
        result = verification_accuracy_lfw_10fold(
            similarity=sim,
            same_id=split.labels["same_id"].values,
            fold=split.labels["fold"].values,
        )
        result["elapsed_s"] = round(time.perf_counter() - t0, 2)
        result["n_unique_images"] = len(all_paths)
        return result

    if split.metric_kind == "identity_search":
        gallery_paths = list(split.extras["gallery_paths"])
        gallery_ids = list(split.extras["gallery_ids"])
        distractor_paths = list(split.extras.get("distractor_paths", []))
        all_gallery_paths = gallery_paths + distractor_paths
        all_gallery_ids = gallery_ids + [-1] * len(distractor_paths)

        probe_emb = _embed_paths(model, split.image_paths, device=device, batch_size=batch_size)
        gallery_emb = _embed_paths(model, all_gallery_paths, device=device, batch_size=batch_size)
        result = rank_k_identification(
            probe_emb=probe_emb,
            probe_ids=np.asarray(split.labels["probe_id"].values),
            gallery_emb=gallery_emb,
            gallery_ids=np.asarray(all_gallery_ids),
            ks=(1, 5, 10, 20),
        )
        result["elapsed_s"] = round(time.perf_counter() - t0, 2)
        result["n_gallery"] = len(all_gallery_paths)
        result["n_distractors"] = len(distractor_paths)
        return result

    raise ValueError(f"evaluate_identity got non-identity split {split.metric_kind!r}")


def _score_au(fex: pd.DataFrame, labels: pd.DataFrame) -> dict:
    truth = {au: labels[au].to_numpy() for au in DISFA_AU_OUTPUT_COLUMNS if au in labels}
    pred = {au: fex[au].to_numpy() for au in DISFA_AU_OUTPUT_COLUMNS if au in fex}
    summary = summarize_au_metrics(truth, pred)
    return {
        "au_f1_mean": round(summary["mean_f1"], 4),
        "au_icc_mean": round(summary["mean_icc"], 4),
        "au_f1_per_au": {au: round(v["f1"], 4) for au, v in summary["per_au"].items()},
        "au_icc_per_au": {au: round(v["icc"], 4) for au, v in summary["per_au"].items()},
        "n_aus_evaluated": summary["n_aus"],
    }


def _score_gaze(fex: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Score gaze predictions against ground-truth (pitch, yaw).

    Both predicted and truth gaze are in head-centric radians. The standard
    metric in the gaze literature is mean angular error (degrees) between
    the two unit gaze vectors. We also break out per-axis MAE for diagnostics.
    """
    from feat.evaluation.metrics import angular_error_degrees

    valid = fex["gaze_pitch"].notna().values & fex["gaze_yaw"].notna().values
    pitch_pred = fex["gaze_pitch"].to_numpy()
    yaw_pred = fex["gaze_yaw"].to_numpy()
    pitch_true = labels["gaze_pitch_rad"].to_numpy()
    yaw_true = labels["gaze_yaw_rad"].to_numpy()

    ang_err = angular_error_degrees(
        pitch_true[valid], yaw_true[valid],
        pitch_pred[valid], yaw_pred[valid],
    )
    pitch_mae_deg = float(np.rad2deg(np.abs(pitch_true[valid] - pitch_pred[valid])).mean())
    yaw_mae_deg = float(np.rad2deg(np.abs(yaw_true[valid] - yaw_pred[valid])).mean())

    return {
        "n_scored": int(valid.sum()),
        "gaze_angular_mae_deg": round(float(ang_err.mean()), 3),
        "gaze_angular_mae_std": round(float(ang_err.std()), 3),
        "gaze_angular_mae_median": round(float(np.median(ang_err)), 3),
        "gaze_pitch_mae_deg": round(pitch_mae_deg, 3),
        "gaze_yaw_mae_deg": round(yaw_mae_deg, 3),
    }


def _score_emotion(fex: pd.DataFrame, labels: pd.DataFrame) -> dict:
    # py-feat emits one probability per emotion; take argmax for top-1.
    valid_mask = fex[EMOTION_COLS].notna().all(axis=1).values
    y_pred_int = np.full(len(fex), -1, dtype=int)
    if valid_mask.any():
        argmax_idx = fex.loc[valid_mask, EMOTION_COLS].values.argmax(axis=1)
        y_pred_emotion = [EMOTION_COLS[i] for i in argmax_idx]
        reverse_map = {v: k for k, v in AFFECTNET_EMOTION_MAP.items()}
        y_pred_int_valid = np.array([reverse_map[e] for e in y_pred_emotion])
        y_pred_int[valid_mask] = y_pred_int_valid

    y_true_int = labels["expression_int"].to_numpy()
    score_mask = (y_pred_int >= 0)
    out = {
        "n_scored": int(score_mask.sum()),
        "emotion_accuracy": round(
            emotion_accuracy(y_true_int[score_mask], y_pred_int[score_mask]), 4
        ) if score_mask.any() else None,
        "emotion_f1_macro": round(
            emotion_f1_macro(y_true_int[score_mask], y_pred_int[score_mask]), 4
        ) if score_mask.any() else None,
    }

    # Valence / arousal CCC (AffectNet only).
    if "valence" in labels and "arousal" in labels:
        # py-feat doesn't predict V/A directly; report intermediate
        # signals if a future detector emits them. For now we emit the
        # ground-truth ranges so the schema is fixed.
        out["valence_arousal_supported"] = False
    return out


def evaluate_all(
    detector,
    splits: Iterable[DatasetSplit],
    batch_size: int = 1,
    num_workers: int = 0,
) -> dict[str, dict]:
    return {
        s.name: evaluate_dataset(detector, s, batch_size=batch_size, num_workers=num_workers)
        for s in splits
    }
