"""Tests for Detectorv2 (RetinaFace + v2.3 multitask model + ArcFace).

These hit the network on first run (downloads py-feat/face_multitask_v1 +
arcface_r50 + the timm backbone). Kept lightweight: one single-face and one
multi-face image, plus schema + value-range + batch-consistency checks.
"""
import numpy as np
import pytest
import torch

from feat import Detectorv2
from feat.data import Fex
from feat.multitask import (
    AU_COLUMNS_V2,
    EMOTION_COLUMNS_V2,
    VA_COLUMNS_V2,
    MESH_COLUMNS_V2,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def detector():
    return Detectorv2(device=_DEVICE)


def test_init(detector):
    assert detector.info["face_model"] == "retinaface"
    assert detector.info["identity_model"] == "arcface"
    assert detector.multitask is not None


def test_single_face_schema(detector, single_face_img):
    fex = detector.detect(single_face_img, progress_bar=False)
    assert isinstance(fex, Fex)
    assert len(fex) == 1
    assert fex.columns.duplicated().sum() == 0
    for col in AU_COLUMNS_V2 + EMOTION_COLUMNS_V2 + VA_COLUMNS_V2:
        assert col in fex.columns
    assert all(c in fex.columns for c in ["gaze_pitch", "gaze_yaw", "gaze_angle"])
    assert all(c in fex.columns for c in ["Pitch", "Roll", "Yaw", "X", "Y", "Z"])
    assert sum(c.startswith("Identity_") for c in fex.columns) == 512
    assert sum(c in MESH_COLUMNS_V2 for c in fex.columns) == len(MESH_COLUMNS_V2)


def test_single_face_values(detector, single_face_img):
    fex = detector.detect(single_face_img, progress_bar=False)
    emo = fex[EMOTION_COLUMNS_V2].iloc[0].to_numpy(dtype=float)
    assert np.isclose(emo.sum(), 1.0, atol=1e-4)
    aus = fex[AU_COLUMNS_V2].iloc[0].to_numpy(dtype=float)
    assert aus.min() >= 0.0 and aus.max() <= 1.0
    assert -1.0 <= float(fex["valence"].iloc[0]) <= 1.0
    assert -1.0 <= float(fex["arousal"].iloc[0]) <= 1.0
    assert fex["Identity_1"].notna().iloc[0]


def test_landmarks_inside_facebox(detector, single_face_img):
    """The dlib-68 block (derived from the 478 mesh) must land within the
    detected facebox — guards the mesh->original-frame coordinate transform.

    Tolerance is tight (strictly inside, +/-5% margin): a looser bound hid a
    real axis-major/interleaved scramble bug that put landmarks ~48px off on
    non-square crops while still landing inside a 1.3x-padded box.
    """
    fex = detector.detect(single_face_img, progress_bar=False)
    xs = fex[[f"x_{i}" for i in range(68)]].iloc[0].to_numpy(dtype=float)
    ys = fex[[f"y_{i}" for i in range(68)]].iloc[0].to_numpy(dtype=float)
    x0 = float(fex["FaceRectX"].iloc[0]); y0 = float(fex["FaceRectY"].iloc[0])
    w = float(fex["FaceRectWidth"].iloc[0]); h = float(fex["FaceRectHeight"].iloc[0])
    inside = (
        (xs >= x0 - 0.05 * w) & (xs <= x0 + 1.05 * w)
        & (ys >= y0 - 0.05 * h) & (ys <= y0 + 1.05 * h)
    )
    assert inside.mean() > 0.9


def test_no_face_returns_nan_predictions(detector):
    """A face-less image yields one row with NaN facebox AND NaN predictions
    (AU/emotion/identity/landmarks) — not fabricated values from the zeroed
    placeholder crop."""
    import os
    from feat.utils.io import get_test_data_path
    no_face = os.path.join(get_test_data_path(), "free-mountain-vector-01.jpg")
    fex = detector.detect(no_face, progress_bar=False)
    assert fex["FaceRectX"].isna().all()
    assert fex[AU_COLUMNS_V2].isna().all().all()
    assert fex[EMOTION_COLUMNS_V2].isna().all().all()
    assert fex["Identity_1"].isna().all()
    assert fex["x_0"].isna().all()


def test_multi_face(detector, multi_face_img):
    fex = detector.detect(multi_face_img, progress_bar=False)
    assert len(fex) == 5
    assert fex["Identity_1"].notna().all()


def test_batch_size_consistency(detector, single_face_img):
    """Same image twice in one batch should give identical predictions."""
    fex = detector.detect([single_face_img, single_face_img],
                          batch_size=2, output_size=512, progress_bar=False)
    assert len(fex) == 2
    a = fex[AU_COLUMNS_V2].iloc[0].to_numpy(dtype=float)
    b = fex[AU_COLUMNS_V2].iloc[1].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, atol=1e-4)
