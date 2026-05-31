"""Multitask v2 model (ConvNeXt-V2 backbone; AU + emotion + V/A + gaze + landmark + pose).

Vendored from the au_deep training repo for inference inside py-feat's Detectorv2.
"""
from feat.multitask.model_v2 import (
    AU_NAMES,
    N_AU,
    N_EMOTION,
    N_MESH,
    MEGraphAUv2,
    ModelV2Config,
)

# Emotion class order emitted by the v2.3 head (NOT the v1 lowercase order).
EMOTION_NAMES = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt",
]

# --- Native-v2 Fex column schema ---
# AUs: 24 columns, names already "AU01".."AU45".
AU_COLUMNS_V2 = list(AU_NAMES)
# Emotion: 8 columns (includes Contempt, unlike v1's 7).
EMOTION_COLUMNS_V2 = list(EMOTION_NAMES)
# Valence / arousal regression (continuous, [-1, 1]).
VA_COLUMNS_V2 = ["valence", "arousal"]
# 478-point MediaPipe mesh in original-frame coords (x,y) + relative depth (z).
MESH_COLUMNS_V2 = (
    [f"mesh_x_{i}" for i in range(N_MESH)]
    + [f"mesh_y_{i}" for i in range(N_MESH)]
    + [f"mesh_z_{i}" for i in range(N_MESH)]
)

__all__ = [
    "AU_NAMES",
    "EMOTION_NAMES",
    "N_AU",
    "N_EMOTION",
    "N_MESH",
    "MEGraphAUv2",
    "ModelV2Config",
    "AU_COLUMNS_V2",
    "EMOTION_COLUMNS_V2",
    "VA_COLUMNS_V2",
    "MESH_COLUMNS_V2",
]
