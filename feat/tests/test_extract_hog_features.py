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
import pytest
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
