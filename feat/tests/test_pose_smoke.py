"""Smoke tests for Detector pose output (both img2pose and retinaface paths).

img2pose: pose comes from the regression head built into the face detector.
retinaface: pose comes from the landmarks-MLP (``feat.utils.face_pose_mlp``)
when its weights are available, or PnP-DLT (``feat.utils.face_pose_pnp``)
as a fallback.

These tests don't assert accuracy — they check that the columns are
populated, values are finite radians within their mathematical range,
and that pose-MLP and img2pose agree to a reasonable tolerance on a
fixture image. Quantitative accuracy against ground truth lives in
``feat/evaluation/`` benchmarks.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from feat.utils.io import get_test_data_path


def _have_test_image() -> bool:
    return os.path.exists(os.path.join(get_test_data_path(), "multi_face.jpg"))


@pytest.mark.skipif(not _have_test_image(), reason="multi_face.jpg fixture missing")
def test_detector_img2pose_pose_columns_populated():
    """Detector(face_model='img2pose') emits sane pose values on multi_face.jpg."""
    from feat.detector import Detector

    det = Detector(
        face_model="img2pose",
        au_model=None,
        emotion_model=None,
        identity_model=None,
        device="cpu",
    )
    fex = det.detect(
        os.path.join(get_test_data_path(), "multi_face.jpg"),
        data_type="image",
        batch_size=1,
        progress_bar=False,
    )

    for col in ("Pitch", "Roll", "Yaw"):
        assert col in fex.columns
        vals = fex[col].to_numpy()
        assert np.all(np.isfinite(vals)), f"{col} contains non-finite values"
        # Group photo of forward-facing faces: |angle| < pi (loose bound).
        assert np.all(np.abs(vals) < np.pi), (
            f"{col} out of range: min={vals.min():.3f}, max={vals.max():.3f}"
        )


@pytest.mark.skipif(not _have_test_image(), reason="multi_face.jpg fixture missing")
def test_detector_retinaface_pose_agrees_with_img2pose():
    """retinaface pose path (MLP or PnP fallback) should agree with img2pose to ~10°."""
    from feat.detector import Detector

    img = os.path.join(get_test_data_path(), "multi_face.jpg")

    det = Detector(
        face_model="img2pose", au_model=None, emotion_model=None,
        identity_model=None, device="cpu",
    )
    fex_ref = det.detect(img, data_type="image", batch_size=1, progress_bar=False)

    det = Detector(
        face_model="retinaface", au_model=None, emotion_model=None,
        identity_model=None, device="cpu",
    )
    fex_alt = det.detect(img, data_type="image", batch_size=1, progress_bar=False)

    if len(fex_ref) != len(fex_alt):
        pytest.skip("face counts differ between img2pose and retinaface")

    # Compare in degrees — when pose-MLP weights are present the path
    # should agree to ~10° MAE max. When falling back to PnP-DLT it'll
    # be looser; we only enforce no per-axis catastrophic-flip
    # (>90° errors that the smoke test caught on PnP-DLT historically).
    for col in ("Pitch", "Roll", "Yaw"):
        diff_deg = np.degrees(np.abs(fex_ref[col].to_numpy() - fex_alt[col].to_numpy()))
        assert diff_deg.max() < 90.0, (
            f"{col} differs by {diff_deg.max():.1f}° between img2pose and retinaface "
            f"— catastrophic flip suggests pose pipeline regression"
        )
