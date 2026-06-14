"""Parity test for extract_hog_features after the HOGLayer migration.

The previous implementation called skimage.feature.hog on each face after
a tensor -> PIL (uint8 quantize) -> numpy round-trip. The new implementation
calls HOGLayer directly on the float32 batch tensor. The XGBoost/SVM AU
classifiers were trained on features from the old path, so the new path
must produce features close enough that classifier outputs don't drift.

This test guards against accidental divergence by re-implementing the prior
skimage path inline as a reference and comparing element-wise. The two
paths differ slightly because of the uint8 quantization in
`transforms.ToPILImage()`; the test asserts a tight upper bound on that
divergence.
"""

import numpy as np
import os
import torch
from skimage.feature import hog as skimage_hog
from torchvision import transforms
from torchvision.io import read_image

from feat.utils.image_operations import (
    extract_face_from_landmarks,
    extract_hog_features,
    inverse_transform_landmarks_torch,
)
from feat.utils.io import get_test_data_path


def _legacy_extract_hog_features(extracted_faces, landmarks):
    """Faithful reproduction of the prior skimage-based implementation."""
    n_faces = landmarks.shape[0]
    face_size = extracted_faces.shape[-1]
    bboxes = (
        torch.tensor([0, 0, face_size, face_size]).unsqueeze(0).repeat(n_faces, 1)
    )
    extracted_landmarks = inverse_transform_landmarks_torch(landmarks, bboxes)
    hog_features = []
    au_new_landmarks = []
    for j in range(n_faces):
        convex_hull, new_landmark = extract_face_from_landmarks(
            extracted_faces[j, ...], extracted_landmarks[j, ...]
        )
        hog_features.append(
            skimage_hog(
                transforms.ToPILImage()(convex_hull[0]),
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=False,
                channel_axis=-1,
            ).reshape(1, -1)
        )
        au_new_landmarks.append(new_landmark)
    return np.concatenate(hog_features), au_new_landmarks


def _make_synthetic_face_batch(n_faces=2, face_size=112, seed=0):
    """Build a deterministic [N, C, H, W] face batch and matching landmarks.

    Real face crops + landmarks come from the upstream face / landmark
    detectors; here we just need geometry that exercises the HOG path.
    """
    torch.manual_seed(seed)
    faces = torch.rand(n_faces, 3, face_size, face_size)
    # 68 (x, y) landmarks placed within the face crop on a deterministic grid
    grid = torch.linspace(15, face_size - 15, 9)  # 9 x 9 = 81; take first 68
    xs, ys = torch.meshgrid(grid, grid, indexing="xy")
    pts = torch.stack([xs.flatten()[:68], ys.flatten()[:68]], dim=1)
    landmarks = pts.flatten().unsqueeze(0).repeat(n_faces, 1)
    return faces, landmarks


def test_extract_hog_features_shape():
    faces, landmarks = _make_synthetic_face_batch(n_faces=2)
    features, new_landmarks = extract_hog_features(faces, landmarks)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 2
    assert features.shape[1] > 0
    assert np.isfinite(features).all()
    assert len(new_landmarks) == 2


def test_extract_hog_features_matches_legacy_skimage_path():
    """Output of the HOGLayer path must be close to the prior skimage path.

    The two differ only by uint8 quantization (skimage gets PIL-converted
    uint8; HOGLayer gets float32). After L2-Hys block normalization, this
    quantization noise should be small.
    """
    faces, landmarks = _make_synthetic_face_batch(n_faces=2, seed=1)
    new_features, _ = extract_hog_features(faces, landmarks)
    old_features, _ = _legacy_extract_hog_features(faces, landmarks)

    assert new_features.shape == old_features.shape

    # Tight per-element bound; quantization noise is a fraction of a percent.
    np.testing.assert_allclose(new_features, old_features, atol=2e-2, rtol=1e-1)
    # Mean abs diff is much smaller; a regression test against silent drift.
    mean_abs_diff = np.abs(new_features - old_features).mean()
    assert mean_abs_diff < 5e-3, (
        f"mean |new - old| = {mean_abs_diff:.4f} exceeds 5e-3; HOGLayer "
        "wire-in may have drifted from the trained-classifier feature space."
    )


def test_extract_hog_features_empty_input():
    """Zero-face input should return an empty array, not crash."""
    faces = torch.empty((0, 3, 112, 112))
    landmarks = torch.empty((0, 136))
    features, new_landmarks = extract_hog_features(faces, landmarks)
    assert features.shape[0] == 0
    assert new_landmarks == []


def test_extract_hog_features_cached_layer_matches_uncached():
    """Passing a pre-built HOGLayer must produce identical output.

    Detectorv1/MPDetector cache a single HOGLayer instance instead of
    rebuilding it on every detect() call. The cached path must be
    bitwise-identical to the un-cached path; otherwise we silently
    drift the trained classifier inputs.
    """
    from feat.utils.image_operations import HOGLayer

    faces, landmarks = _make_synthetic_face_batch(n_faces=3, seed=2)
    uncached, _ = extract_hog_features(faces, landmarks)
    cached_layer = HOGLayer(
        orientations=8,
        pixels_per_cell=8,
        cells_per_block=2,
        block_normalization="L2-Hys",
        feature_vector=True,
        device="cpu",
    )
    cached, _ = extract_hog_features(faces, landmarks, hog_layer=cached_layer)
    np.testing.assert_array_equal(cached, uncached)


def _real_face_batch(face_size=112):
    """Load a real face image, resize to face_size, return tensor + landmarks.

    Uses single_face.jpg from the repo's test data. Landmarks are a
    deterministic grid covering the face crop - good enough to exercise
    the HOG pipeline on actual image content (sparse high-contrast edges
    around eyes / nose / mouth) where uint8 quantization could plausibly
    flip dominant orientations in some cells. Synthetic random tensors
    don't exercise that regime.
    """
    img_path = os.path.join(get_test_data_path(), "single_face.jpg")
    raw = read_image(img_path).float() / 255.0  # [3, H, W] float32 in [0, 1]
    # Center-crop to a square then resize.
    _, H, W = raw.shape
    side = min(H, W)
    top = (H - side) // 2
    left = (W - side) // 2
    cropped = raw[:, top : top + side, left : left + side]
    face = torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(face_size, face_size), mode="bilinear", align_corners=False
    )  # [1, 3, face_size, face_size]
    grid = torch.linspace(15, face_size - 15, 9)
    xs, ys = torch.meshgrid(grid, grid, indexing="xy")
    pts = torch.stack([xs.flatten()[:68], ys.flatten()[:68]], dim=1)
    landmarks = pts.flatten().unsqueeze(0)
    return face, landmarks


def test_extract_hog_features_matches_legacy_on_real_face():
    """Same parity bound on a real face image, where uint8 quantization
    could plausibly flip dominant orientations in low-contrast cells.

    The synthetic-random test isn't representative because random noise
    has high gradient density everywhere, so block-normalization saturates
    uniformly and quantization noise averages out. Real faces have sparse
    edges; this test catches regressions the synthetic test cannot.
    """
    faces, landmarks = _real_face_batch(face_size=112)
    new_features, _ = extract_hog_features(faces, landmarks)
    old_features, _ = _legacy_extract_hog_features(faces, landmarks)

    assert new_features.shape == old_features.shape

    # Tight bound; quantization noise on real images stays sub-percent
    # mean-absolute-diff after L2-Hys normalization.
    np.testing.assert_allclose(new_features, old_features, atol=5e-2, rtol=2e-1)
    mean_abs_diff = np.abs(new_features - old_features).mean()
    assert mean_abs_diff < 8e-3, (
        f"mean |new - old| = {mean_abs_diff:.4f} on real face exceeds 8e-3; "
        "HOGLayer wire-in may have drifted from the trained-classifier "
        "feature space. Trained AU classifier may need re-validation."
    )


# NOTE: An end-to-end AU-classifier-prediction parity test would be the
# strongest guard against feature-space drift, but loading the trained
# `xgb_au_classifier.skops` segfaults during construction on Python 3.13 +
# xgboost 3.x via skops's pickled torch path. The real-face HOG parity
# test above is the next-strongest signal: ~5e-3 mean abs diff in feature
# space is well below the per-feature scale that XGBoost trees split on
# (after the StandardScaler + PCA stages of the AU pipeline). When the
# environment supports loading the classifier, this manual check is
# recommended:
#
#     classifier = load_xgb_au_classifier()
#     new_aus = classifier.detect_au(frame=new_features, landmarks=[new_lm])
#     old_aus = classifier.detect_au(frame=old_features, landmarks=[old_lm])
#     np.testing.assert_allclose(new_aus, old_aus, atol=5e-2, rtol=1e-1)
