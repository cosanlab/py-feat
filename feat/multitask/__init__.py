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

EMOTION_NAMES = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt",
]

__all__ = [
    "AU_NAMES",
    "EMOTION_NAMES",
    "N_AU",
    "N_EMOTION",
    "N_MESH",
    "MEGraphAUv2",
    "ModelV2Config",
]
