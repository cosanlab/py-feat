"""Tests for Detectorv2 (RetinaFace + v2.5 multitask model + ArcFace).

These hit the network on first run (downloads py-feat/face_multitask_v2 +
arcface_r50 + the timm backbone). Kept lightweight: one single-face and one
multi-face image, plus schema + value-range + batch-consistency checks.
"""
import numpy as np
import pytest
import torch

from feat import Detectorv2
from feat.data import Fex
from feat.utils import MP_BLENDSHAPE_NAMES
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
    # v2.5: 52 MediaPipe/ARKit blendshape columns + the .blendshapes accessor
    assert all(c in fex.columns for c in MP_BLENDSHAPE_NAMES)
    assert fex.blendshapes.shape == (1, 52)


def test_single_face_values(detector, single_face_img):
    fex = detector.detect(single_face_img, progress_bar=False)
    emo = fex[EMOTION_COLUMNS_V2].iloc[0].to_numpy(dtype=float)
    # Softmax runs under bf16 autocast by default, so the sum is 1.0 only to
    # bf16 precision (~1e-2), not fp32.
    assert np.isclose(emo.sum(), 1.0, atol=2e-2)
    aus = fex[AU_COLUMNS_V2].iloc[0].to_numpy(dtype=float)
    assert aus.min() >= 0.0 and aus.max() <= 1.0
    assert -1.0 <= float(fex["valence"].iloc[0]) <= 1.0
    assert -1.0 <= float(fex["arousal"].iloc[0]) <= 1.0
    assert fex["Identity_1"].notna().iloc[0]
    bs = fex.blendshapes.iloc[0].to_numpy(dtype=float)
    assert bs.min() >= 0.0 and bs.max() <= 1.0
    # blendshape_columns must survive extract_* metadata rewrite (like AUs)
    assert fex.extract_mean().blendshapes.shape == (1, 52)


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
    x0 = float(fex["FaceRectX"].iloc[0])
    y0 = float(fex["FaceRectY"].iloc[0])
    w = float(fex["FaceRectWidth"].iloc[0])
    h = float(fex["FaceRectHeight"].iloc[0])
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


def test_arcface_weights_loaded_not_random():
    """Guard the trained-weights load: a bare ``ArcFace(backbone='r50')`` (the
    bug this fixes) ships *random* init, so every construction would embed the
    same face differently. Loaded weights are deterministic — two independent
    loads must agree — and carry real signal (embeddings aren't ~0)."""
    from feat.identity_detectors.arcface.arcface_model import (
        load_arcface_identity_detector,
    )
    torch.manual_seed(0)
    x = torch.rand(2, 3, 112, 112)
    a = load_arcface_identity_detector("cpu")
    b = load_arcface_identity_detector("cpu")
    with torch.no_grad():
        ea = a(x).numpy()
        eb = b(x).numpy()
    np.testing.assert_allclose(ea, eb, atol=1e-5)
    assert np.abs(ea).mean() > 1e-3


def test_crop_faces_from_boxes_shape_and_forward(detector, single_face_img):
    """crop_faces_from_boxes mirrors detect_faces' structure without
    RetinaFace, and forward() consumes it to a populated mesh."""
    from torchvision.io import read_image

    # Get a real face box from a normal detect, then re-crop from it.
    fex = detector.detect(single_face_img, progress_bar=False)
    x = float(fex["FaceRectX"].iloc[0])
    y = float(fex["FaceRectY"].iloc[0])
    w = float(fex["FaceRectWidth"].iloc[0])
    h = float(fex["FaceRectHeight"].iloc[0])
    box = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)

    img_t = read_image(single_face_img).unsqueeze(0).float()  # [1,3,H,W], 0-255

    faces_data = detector.crop_faces_from_boxes(img_t, box)
    assert isinstance(faces_data, list) and len(faces_data) == 1
    d = faces_data[0]
    assert set(d) == {"face_id", "faces", "boxes", "new_boxes", "scores", "image_size"}
    assert d["faces"].shape == (1, 3, detector.face_size, detector.face_size)
    assert d["new_boxes"].shape == (1, 4)
    assert d["scores"].shape == (1,)
    assert float(d["scores"][0]) == 1.0  # placeholder confidence

    batch_data = {
        "Image": img_t,
        "Scale": torch.ones(1),
        "Padding": {"Left": torch.zeros(1), "Top": torch.zeros(1),
                    "Right": torch.zeros(1), "Bottom": torch.zeros(1)},
        "FileName": ["x"],
    }
    df = detector.forward(faces_data, batch_data)
    mesh_cols = [c for c in df.columns if c.startswith("mesh_x_")]
    assert len(mesh_cols) == 478
    assert df["mesh_x_0"].notna().all()

    # Box came from detect(), so the re-cropped mesh should land on the same
    # face: centroid within a few px of the detect() mesh centroid. The exact
    # tolerance is model-dependent (crop-jitter sensitivity of the mesh head);
    # ~25px keeps "same face" meaningful for the v2.5 model.
    cx_detect = fex[[f"mesh_x_{i}" for i in range(478)]].iloc[0].to_numpy(float).mean()
    cx_track = df[[f"mesh_x_{i}" for i in range(478)]].iloc[0].to_numpy(float).mean()
    assert abs(cx_detect - cx_track) < 25.0


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="bf16 autocast default is CUDA-only")
def test_bf16_matches_fp32(single_face_img):
    """bf16 autocast is the default on CUDA, but the CPU test run never
    exercises it. Pin the bf16-vs-fp32 agreement (AU within model noise, emotion
    still a valid simplex) so a real bf16 regression can't pass silently."""
    fex16 = Detectorv2(device="cuda", amp=True).detect(
        single_face_img, progress_bar=False)
    fex32 = Detectorv2(device="cuda", amp=False).detect(
        single_face_img, progress_bar=False)
    au16 = fex16[AU_COLUMNS_V2].iloc[0].to_numpy(dtype=float)
    au32 = fex32[AU_COLUMNS_V2].iloc[0].to_numpy(dtype=float)
    np.testing.assert_allclose(au16, au32, atol=0.05)
    emo16 = fex16[EMOTION_COLUMNS_V2].iloc[0].to_numpy(dtype=float)
    assert np.isclose(emo16.sum(), 1.0, atol=2e-2)
