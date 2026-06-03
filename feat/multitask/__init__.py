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

# --- v2.4 head dims ---
# v2.4 drops 4 poorly-represented AUs (AU16/18/27/45 -> back to v1's 20-AU set)
# and drops Contempt from emotion (7-class). The model emits AU probs / emotion
# logits in exactly these orders.
DROP_AU_V24 = {"AU16", "AU18", "AU27", "AU45"}
AU_NAMES_V24 = [a for a in AU_NAMES if a not in DROP_AU_V24]   # 20
EMOTION_NAMES_V24 = EMOTION_NAMES[:7]                          # drop Contempt

# --- Native-v2 Fex column schema (v2.4 = shipped model: 20 AU / 7 emotion) ---
AU_COLUMNS_V2 = list(AU_NAMES_V24)
EMOTION_COLUMNS_V2 = list(EMOTION_NAMES_V24)
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
    "AU_NAMES_V24",
    "EMOTION_NAMES",
    "EMOTION_NAMES_V24",
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
