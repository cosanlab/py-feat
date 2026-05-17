"""Smoke test for MPDetector gaze output.

Verifies the gaze pipeline doesn't silently break: columns are emitted,
values are finite, and within the mathematically valid range for the
atan2-based conventions (pitch in [-pi/2, pi/2], yaw in (-pi, pi]).

This is a structural smoke test, **not** an accuracy check. Empirical
inspection of ``multi_face.jpg`` shows gaze yaw values up to ~150°
absolute on faces whose head pose is only ~5° off-frontal — a known
limitation of the geometric ``normalize(iris_center - eye_center)``
approach when "eye_center" is approximated from contour landmarks
rather than eyeball-center. Quantitative validation against
ground-truth gaze (e.g., MPIIFaceGaze, Columbia Gaze) is tracked
separately under the regression-benchmark project.

The correctness of ``estimate_gaze()`` math on synthetic canonical
input is covered by ``test_face_pose.py``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from feat.utils import FEAT_GAZE_COLUMNS
from feat.utils.io import get_test_data_path


def _have_test_image() -> bool:
    return os.path.exists(os.path.join(get_test_data_path(), "multi_face.jpg"))


@pytest.mark.skipif(not _have_test_image(), reason="multi_face.jpg fixture missing")
def test_mpdetector_emits_gaze_columns():
    """Run MPDetector with the geometric gaze path and verify gaze columns
    are populated. L2CS path is tested separately (requires HF weights)."""
    from feat.MPDetector import MPDetector

    det = MPDetector(
        face_model="retinaface",
        landmark_model="mp_facemesh_v2",
        au_model="mp_blendshapes",
        emotion_model=None,
        identity_model=None,
        facepose_model=None,
        gaze_model="geometric",
        device="cpu",
    )
    img = os.path.join(get_test_data_path(), "multi_face.jpg")
    fex = det.detect(img, data_type="image", batch_size=1, progress_bar=False)

    # All three gaze columns must exist.
    for col in FEAT_GAZE_COLUMNS:
        assert col in fex.columns, f"missing gaze column {col!r}"

    # At least one face was detected (multi_face.jpg has 5).
    assert len(fex) > 0, "expected at least one detected face"

    # Values must be finite and within their mathematical ranges.
    # atan2: pitch is in [-pi/2, pi/2]; yaw is in (-pi, pi]. We don't
    # assert "near zero" here — see the module docstring: gaze accuracy
    # on real images is a known weakness of the current geometric
    # approach and is validated against ground-truth separately.
    pitch = fex["gaze_pitch"].to_numpy()
    yaw = fex["gaze_yaw"].to_numpy()
    assert np.all(np.isfinite(pitch)), "gaze_pitch contains non-finite values"
    assert np.all(np.isfinite(yaw)), "gaze_yaw contains non-finite values"
    assert np.all(np.abs(pitch) <= np.pi / 2 + 1e-3), (
        f"gaze_pitch out of mathematical range [-pi/2, pi/2]: "
        f"min={pitch.min():.3f}, max={pitch.max():.3f}"
    )
    assert np.all(np.abs(yaw) <= np.pi + 1e-3), (
        f"gaze_yaw out of mathematical range [-pi, pi]: "
        f"min={yaw.min():.3f}, max={yaw.max():.3f}"
    )

    # gaze_angle is the absolute angle from head-forward; non-negative.
    angle = fex["gaze_angle"].to_numpy()
    assert np.all(np.isfinite(angle)), "gaze_angle contains non-finite values"
    assert np.all(angle >= 0), "gaze_angle must be non-negative"
    assert np.all(angle < np.pi), f"gaze_angle out of range [0, pi): max={angle.max():.3f}"


def _l2cs_weights_reachable() -> bool:
    """Whether the py-feat/l2cs HF repo is up + weights cached or downloadable."""
    try:
        from huggingface_hub import hf_hub_download
        from feat.utils.io import get_resource_path
        hf_hub_download(
            repo_id="py-feat/l2cs",
            filename="l2cs_gaze360_resnet50.safetensors",
            cache_dir=get_resource_path(),
        )
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_test_image(), reason="multi_face.jpg fixture missing")
@pytest.mark.skipif(
    not _l2cs_weights_reachable(),
    reason="py-feat/l2cs weights not on HF yet — upload via scripts/convert_l2cs_pickle_to_safetensors.py",
)
def test_mpdetector_l2cs_gaze_smoke():
    """L2CS gaze path on MPDetector: columns exist, values finite + in range."""
    from feat.MPDetector import MPDetector

    det = MPDetector(
        face_model="retinaface",
        landmark_model="mp_facemesh_v2",
        au_model="mp_blendshapes",
        emotion_model=None,
        identity_model=None,
        facepose_model=None,
        gaze_model="l2cs",
        device="cpu",
    )
    img = os.path.join(get_test_data_path(), "multi_face.jpg")
    fex = det.detect(img, data_type="image", batch_size=1, progress_bar=False)

    for col in FEAT_GAZE_COLUMNS:
        assert col in fex.columns, f"missing gaze column {col!r}"
    assert len(fex) > 0
    pitch = fex["gaze_pitch"].to_numpy()
    yaw = fex["gaze_yaw"].to_numpy()
    assert np.all(np.isfinite(pitch))
    assert np.all(np.isfinite(yaw))
    # L2CS bin range is [-pi, pi]; in practice |angle| should be well below pi.
    assert np.all(np.abs(pitch) <= np.pi + 1e-3)
    assert np.all(np.abs(yaw) <= np.pi + 1e-3)


@pytest.mark.skipif(not _have_test_image(), reason="multi_face.jpg fixture missing")
@pytest.mark.skipif(
    not _l2cs_weights_reachable(),
    reason="py-feat/l2cs weights not on HF yet",
)
def test_detector_l2cs_gaze_smoke():
    """L2CS gaze path on classic Detector: columns exist, values finite."""
    from feat.detector import Detector

    det = Detector(
        face_model="img2pose",
        landmark_model="mobilefacenet",
        au_model=None,
        emotion_model=None,
        identity_model=None,
        gaze_model="l2cs",
        device="cpu",
    )
    img = os.path.join(get_test_data_path(), "multi_face.jpg")
    fex = det.detect(img, data_type="image", batch_size=1, progress_bar=False)

    for col in FEAT_GAZE_COLUMNS:
        assert col in fex.columns, f"missing gaze column {col!r}"
    assert len(fex) > 0
    assert np.all(np.isfinite(fex["gaze_pitch"].to_numpy()))
    assert np.all(np.isfinite(fex["gaze_yaw"].to_numpy()))
