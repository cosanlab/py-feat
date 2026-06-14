"""Tests for ``Fex.landmarks_dlib68_xy``.

Pins the dispatch contract for the unified dlib-68 landmark accessor that
makes ``plot_detections(faces='landmarks')`` work on both Detectorv1 (136-d)
and MPDetector (1434-d) Fex objects.

All tests are offline — synthetic Fex objects with known landmark values
are sufficient since the method is pure DataFrame indexing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feat.data import Fex
from feat.utils import openface_2d_landmark_columns, MP_LANDMARK_COLUMNS
from feat.utils.blendshape_to_au import DLIB68_FROM_MP478


# ---------------------------------------------------------------------
# Synthetic Fex fixtures
# ---------------------------------------------------------------------

def _make_dlib68_fex(n_rows: int = 3) -> Fex:
    """Detectorv1-style Fex: 136 landmark columns in axis-major layout
    [x_0..x_67, y_0..y_67]. Encodes ``x_i = i`` and ``y_i = 100 + i`` per row
    so tests can verify positional indexing is correct."""
    data = {}
    for i in range(68):
        data[f"x_{i}"] = np.full(n_rows, float(i))
        data[f"y_{i}"] = np.full(n_rows, 100.0 + i)
    df = pd.DataFrame(data, columns=openface_2d_landmark_columns)
    return Fex(df, landmark_columns=openface_2d_landmark_columns)


def _make_mp478_fex(n_rows: int = 3) -> Fex:
    """MPDetector-style Fex: 1434 landmark columns named x_i / y_i / z_i for
    i in 0..477 (interleaved per-vertex layout). Encodes ``x_i = i``,
    ``y_i = 1000 + i``, ``z_i = 2000 + i`` so tests can confirm the
    ``DLIB68_FROM_MP478`` index map is honored."""
    data = {}
    for i in range(478):
        data[f"x_{i}"] = np.full(n_rows, float(i))
        data[f"y_{i}"] = np.full(n_rows, 1000.0 + i)
        data[f"z_{i}"] = np.full(n_rows, 2000.0 + i)
    df = pd.DataFrame(data, columns=MP_LANDMARK_COLUMNS)
    return Fex(df, landmark_columns=MP_LANDMARK_COLUMNS)


# ---------------------------------------------------------------------
# Detectorv1 path (136-d, axis-major)
# ---------------------------------------------------------------------

class TestDlib68Path:
    def test_dataframe_form_returns_2d(self):
        fex = _make_dlib68_fex(n_rows=4)
        x, y = fex.landmarks_dlib68_xy()
        assert x.shape == (4, 68)
        assert y.shape == (4, 68)

    def test_dataframe_form_values_axis_major_split(self):
        """x[:, i] == i and y[:, i] == 100 + i for the canonical Detectorv1 layout."""
        fex = _make_dlib68_fex(n_rows=2)
        x, y = fex.landmarks_dlib68_xy()
        for i in range(68):
            assert (x[:, i] == float(i)).all(), f"x[{i}] mismatch"
            assert (y[:, i] == 100.0 + i).all(), f"y[{i}] mismatch"

    def test_row_form_returns_1d(self):
        fex = _make_dlib68_fex(n_rows=3)
        x, y = fex.landmarks_dlib68_xy(row=fex.iloc[1])
        assert x.shape == (68,)
        assert y.shape == (68,)
        assert (x == np.arange(68, dtype=np.float64)).all()
        assert (y == 100.0 + np.arange(68, dtype=np.float64)).all()


# ---------------------------------------------------------------------
# MPDetector path (1434-d, named-column lookup)
# ---------------------------------------------------------------------

class TestMp478Path:
    def test_dataframe_form_returns_2d(self):
        fex = _make_mp478_fex(n_rows=2)
        x, y = fex.landmarks_dlib68_xy()
        assert x.shape == (2, 68)
        assert y.shape == (2, 68)

    def test_dataframe_form_samples_dlib68_indices(self):
        """For dlib slot j, the returned x value must equal MP vertex
        ``DLIB68_FROM_MP478[j]`` (encoded as float). Same for y, offset by 1000."""
        fex = _make_mp478_fex(n_rows=2)
        x, y = fex.landmarks_dlib68_xy()
        for j, mp_idx in enumerate(DLIB68_FROM_MP478):
            assert (x[:, j] == float(mp_idx)).all(), \
                f"dlib idx {j} -> MP {mp_idx} mismatch on x"
            assert (y[:, j] == 1000.0 + mp_idx).all(), \
                f"dlib idx {j} -> MP {mp_idx} mismatch on y"

    def test_row_form_returns_1d_correct_indices(self):
        fex = _make_mp478_fex(n_rows=3)
        x, y = fex.landmarks_dlib68_xy(row=fex.iloc[2])
        assert x.shape == (68,)
        assert y.shape == (68,)
        # Spot-check: dlib jaw center idx 8 maps to MP 176 (per DLIB68_FROM_MP478),
        # idx 10 maps to MP 152.
        assert x[8] == 176.0
        assert y[8] == 1176.0
        assert x[10] == 152.0
        assert y[10] == 1152.0


# ---------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------

class TestErrorPaths:
    def test_none_landmark_columns_raises(self):
        """A Fex without landmark_columns set should give a clear error,
        not a TypeError from len(None)."""
        fex = Fex(pd.DataFrame({"a": [1, 2]}), landmark_columns=None)
        with pytest.raises(ValueError, match=r"landmark_columns is not set"):
            fex.landmarks_dlib68_xy()

    def test_unexpected_length_raises(self):
        """Anything that isn't 136 (dlib) or 1434 (MP-478) is rejected."""
        cols = [f"x_{i}" for i in range(50)] + [f"y_{i}" for i in range(50)]
        df = pd.DataFrame({c: [0.0, 0.0] for c in cols}, columns=cols)
        fex = Fex(df, landmark_columns=cols)
        with pytest.raises(ValueError, match=r"Unexpected landmark_columns length"):
            fex.landmarks_dlib68_xy()
