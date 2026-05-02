"""Regression test for the MPDetector retinaface migration.

Before this PR landed, MPDetector(face_model='retinaface') raised a
NotImplementedError because the v0.7 RetinaFace rebuild left MPDetector's
face-detection path pointing at the deleted MobileNet0.25 class. Now it
uses the same ResNet34 wrapper that Detector(face_model='retinaface_r34')
uses, so end-to-end MPDetector with retinaface works again.

Pin the contract:
1. Construction succeeds with face_model='retinaface' (the legacy MPDetector
   default) AND face_model='retinaface_r34' (the canonical name in
   Detector). Both resolve to the same wrapper.
2. detect() on multi_face.jpg returns a Fex with one row per detected face,
   real bbox coords, and finite pose values (no NaN, no inf).
"""

from __future__ import annotations

import math
import os

import pytest


def _have_test_image() -> bool:
    from feat.utils.io import get_test_data_path

    return os.path.exists(os.path.join(get_test_data_path(), "multi_face.jpg"))


@pytest.mark.skipif(not _have_test_image(), reason="multi_face.jpg missing")
@pytest.mark.parametrize("face_model_name", ["retinaface", "retinaface_r34"])
def test_mpdetector_retinaface_constructs_and_detects(face_model_name):
    """End-to-end: MPDetector with face_model='retinaface' (or its alias
    'retinaface_r34') builds, detects faces in a real image, and returns
    a Fex with finite pose values. Specifically pins down that the
    pre-existing dtype mismatch in convert_landmarks_3d (Python `float`
    -> float64 vs canonical face model's float32) is fixed alongside the
    migration."""
    from feat.MPDetector import MPDetector
    from feat.utils.io import get_test_data_path

    mp = MPDetector(
        device="cpu",
        face_model=face_model_name,
        landmark_model="mp_facemesh_v2",
        au_model=None,
        emotion_model=None,
        identity_model=None,
        facepose_model=None,
    )
    assert mp.face_detector is not None
    # The wrapper class name is `Retinaface` regardless of which alias was used.
    assert type(mp.face_detector).__name__ == "Retinaface"

    img = os.path.join(get_test_data_path(), "multi_face.jpg")
    fex = mp.detect(img)

    # multi_face.jpg has 5 detectable faces at default thresholds.
    assert len(fex) >= 4, f"expected >=4 faces, got {len(fex)}"

    # Pose is computed via MPDetector's mesh-based path (Umeyama alignment
    # of the 478-pt mesh against MediaPipe's canonical model). It must be
    # finite — the dtype-mismatch bug previously made this whole stage
    # crash before the dataframe was filled.
    for col in ("Pitch", "Roll", "Yaw"):
        vals = fex[col]
        assert not vals.isna().any(), f"{col} has NaN values"
        assert vals.abs().max() <= math.pi, f"{col} out of [-pi, pi]"

    # Sanity bound: multi_face.jpg is 5 forward-facing upright portraits.
    # Pre-fix Pitch clustered at +/- pi because the MediaPipe-pixel and
    # canonical-mesh coordinate systems disagreed on Y and Z axis sign;
    # convert_landmarks_3d now translates between them. After the fix,
    # |Pitch| should be modest (< 30 degrees) for every detected face.
    assert fex.Pitch.abs().max() < math.radians(30), (
        f"max |Pitch| = {math.degrees(fex.Pitch.abs().max()):.1f} deg "
        "on a forward-facing group is suspicious - check the MediaPipe "
        "Y/Z-axis convention flip in convert_landmarks_3d"
    )


@pytest.mark.skipif(not _have_test_image(), reason="multi_face.jpg missing")
def test_mpdetector_retinaface_with_resmasknet_emotion():
    """Regression test for a NameError that lurked in detect_faces' resmasknet
    branch. The old retinaface block defined `single_frame` (mean-subtracted
    copy of `frame`) and the resmasknet branch downstream re-used that
    variable. When the migration deleted the mean-subtract block, the
    `single_frame` reference stayed but the variable no longer existed -
    so MPDetector(face_model='retinaface', emotion_model='resmasknet')
    would raise NameError at the first detected face. This test exercises
    that exact combination."""
    from feat.MPDetector import MPDetector
    from feat.utils.io import get_test_data_path

    mp = MPDetector(
        device="cpu",
        face_model="retinaface",
        landmark_model="mp_facemesh_v2",
        au_model=None,
        emotion_model="resmasknet",  # the path that used to NameError
        identity_model=None,
        facepose_model=None,
    )
    img = os.path.join(get_test_data_path(), "multi_face.jpg")
    # Just need to confirm detect runs to completion without NameError.
    fex = mp.detect(img)
    assert len(fex) >= 4
    # Resmasknet emotion outputs must be finite.
    for col in ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"):
        if col in fex.columns:
            assert not fex[col].isna().any(), f"{col} has NaN"
