"""Tests for feat.utils.landmark_symmetry.landmark_asymmetry.

Uses the canonical mesh (symmetric by construction) as ground truth
instead of a real detector.
"""


import numpy as np
import pytest

from feat.utils.landmark_symmetry import landmark_asymmetry
from feat.utils.face_pose import load_canonical_face_model


def _canonical_xy():
    V = load_canonical_face_model().detach().cpu().numpy().astype(np.float64)
    return V[:, 0], V[:, 1]


def test_canonical_mesh_is_symmetric():
    x, y = _canonical_xy()
    assert landmark_asymmetry(x, y) < 0.01


def test_one_sided_perturbation_increases_score():
    x, y = _canonical_xy()
    baseline = landmark_asymmetry(x, y)
    x_perturbed = x.copy()
    midline = (x[263] + x[33]) / 2.0
    x_perturbed[x_perturbed < midline] += 5.0
    assert landmark_asymmetry(x_perturbed, y) > baseline


def test_wrong_point_count_raises():
    with pytest.raises(ValueError):
        landmark_asymmetry(np.zeros(68), np.zeros(68))