"""Map MediaPipe / ARKit blendshapes to FACS Action Unit intensities.

Provides a learned PLS regression (``pls_predict_batch``) trained on paired
Detectorv1 (xgb AUs) + MPDetector (52 blendshapes) outputs from ~10K CelebV-HQ
celebrity videos (~350K frames). Cheong / Py-Feat-tutorial-06 style: linear
features only, no pairwise interactions, no clipping at training. 3-fold
GroupKFold (by video_id) OOS variance-weighted R² = 0.236 ± 0.008.

Weights are downloaded from HuggingFace Hub at first use:
    https://huggingface.co/py-feat/bs_to_au → ``bs_to_au_pls_v2.npz``

Also defines a **dlib-68 → MediaPipe-478 landmark index mapping** so the
existing dlib-based AU muscle-polygon heatmap drawing can be reused on
MPDetector output.
"""

from __future__ import annotations

import numpy as np

from huggingface_hub import hf_hub_download

from feat.utils.io import get_resource_path

# ---------------------------------------------------------------------
# Learned PLS regression: 52 MP blendshapes → 20 AU intensities
# ---------------------------------------------------------------------
_PLS_WEIGHTS = None
_PLS_REPO_ID = "py-feat/bs_to_au"
_PLS_FILENAME = "bs_to_au_pls_v2.npz"


def _load_pls_weights():
    """Lazy-load PLS weights from HuggingFace Hub. Cached after first call."""
    global _PLS_WEIGHTS
    if _PLS_WEIGHTS is not None:
        return _PLS_WEIGHTS

    path = hf_hub_download(
        repo_id=_PLS_REPO_ID,
        filename=_PLS_FILENAME,
        cache_dir=get_resource_path(),
    )
    z = np.load(path, allow_pickle=False)
    _PLS_WEIGHTS = {
        "coef": z["coef"].astype(np.float32),                    # (52, 20)
        "intercept": z["intercept"].astype(np.float32),          # (20,)
        "blendshape_columns": [str(s) for s in z["bs_columns"]],
        "au_columns": [str(s) for s in z["au_columns"]],
    }
    return _PLS_WEIGHTS


def pls_predict_batch(
    blendshape_array: np.ndarray, clip: bool = True,
) -> np.ndarray:
    """(N, 52) MP blendshapes → (N, 20) AU intensities via Cheong-style PLS.

    Trained on ~350K frames from 10K CelebV-HQ wild-celebrity videos, paired
    (xgb AU intensity, MP blendshape coefficient) per frame, pose-filtered to
    |yaw| ≤ 40° and |pitch| ≤ 30°. PLS-2 with n_components=20 (full rank),
    linear features only — pairwise BS interactions were tested and degraded
    out-of-sample R², so they are NOT used.

    3-fold GroupKFold (by video_id) variance-weighted R² = 0.236 ± 0.008.
    Per-AU R² is strongest on AU06/12/43 (~0.50), weakest on AU11/15/28 (<0.10).

    See `https://huggingface.co/py-feat/bs_to_au` for the model card.

    Args:
        blendshape_array: blendshape coefficients in MediaPipe FaceLandmarker
            order (matches MPDetector output). Either a 1-D ``(52,)`` vector
            for a single face or a 2-D ``(N, 52)`` batch. See
            ``_PLS_WEIGHTS["blendshape_columns"]`` after load for the exact
            column names.
        clip: if True (default), output is clipped to [0, 1] for display
            consistency with FACS intensity convention.

    Returns:
        AU intensities in py-feat's standard order. Shape matches input batching:
        ``(20,)`` for a 1-D input, ``(N, 20)`` for a 2-D input.
    """
    w = _load_pls_weights()
    bs = np.asarray(blendshape_array, dtype=np.float32)
    n_features = w["coef"].shape[0]
    if bs.ndim == 1:
        if bs.shape[0] != n_features:
            raise ValueError(
                f"Expected 1-D input of length {n_features}, got {bs.shape[0]}."
            )
        bs = bs.reshape(1, -1)
        squeeze_out = True
    elif bs.ndim == 2:
        if bs.shape[1] != n_features:
            raise ValueError(
                f"Expected 2-D input with {n_features} columns, got shape {bs.shape}."
            )
        squeeze_out = False
    else:
        raise ValueError(
            f"blendshape_array must be 1-D ({n_features},) or 2-D (N, {n_features}); "
            f"got ndim={bs.ndim}, shape={bs.shape}."
        )
    out = bs @ w["coef"] + w["intercept"]
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out[0] if squeeze_out else out


# ---------------------------------------------------------------------
# dlib-68 → MediaPipe Face Mesh-478 index correspondence
# ---------------------------------------------------------------------
DLIB68_FROM_MP478: list[int] = [
    # Jaw 0-16
    127, 234, 93, 132, 58, 172, 136, 150, 176, 148, 152, 377, 400, 379, 365, 397, 356,
    # Left eyebrow 17-21
    70, 63, 105, 66, 107,
    # Right eyebrow 22-26
    336, 296, 334, 293, 300,
    # Nose 27-30 bridge
    168, 6, 195, 4,
    # Nose tip 31-35
    240, 75, 1, 305, 460,
    # Left eye 36-41
    33, 160, 158, 133, 153, 144,
    # Right eye 42-47
    362, 385, 387, 263, 373, 380,
    # Outer lip 48-59
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Inner lip 60-67
    78, 82, 13, 312, 308, 317, 14, 87,
]
assert len(DLIB68_FROM_MP478) == 68


def mp478_row_to_dlib68_view(row) -> dict:
    """Build a dict with x_0..x_67 / y_0..y_67 keys by sampling
    the matching MediaPipe-478 landmarks."""
    view: dict = {}
    for dlib_idx, mp_idx in enumerate(DLIB68_FROM_MP478):
        view[f"x_{dlib_idx}"] = row.get(f"x_{mp_idx}", np.nan)
        view[f"y_{dlib_idx}"] = row.get(f"y_{mp_idx}", np.nan)
    for k in (
        "FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight",
        "Pitch", "Roll", "Yaw",
    ):
        if k in row.index if hasattr(row, "index") else k in row:
            view[k] = row[k]
    return view
