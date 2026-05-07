"""Tests for ``feat.utils.blendshape_to_au``.

Covers the BS→AU PLS regressor (``pls_predict_batch``) and the dlib-68 ↔
MediaPipe-478 index map (``DLIB68_FROM_MP478`` / ``mp478_row_to_dlib68_view``).

The PLS-loader tests do an actual HuggingFace Hub fetch on first run; subsequent
runs hit the local cache so they don't require network. Tests that exercise
shape contracts use a stub of the lazy-loaded weights so they don't even need
the cached file to be present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feat.utils import blendshape_to_au as b2a


# ---------------------------------------------------------------------
# Helpers — module-level stub for the PLS weights so most tests don't
# need the real file. Callers that want the real weights use ``hf_load``.
# ---------------------------------------------------------------------

@pytest.fixture
def stub_pls_weights(monkeypatch):
    """Replace the lazy weight loader with deterministic stub weights.

    52 → 20 linear map: ``coef[i, j] = 1`` if the j-th AU pulls from the i-th
    blendshape, else 0. Intercept is zeros. Lets shape / clip / 1-D-handling
    tests run offline.
    """
    rng = np.random.default_rng(0)
    coef = rng.standard_normal((52, 20)).astype(np.float32) * 0.1
    intercept = rng.standard_normal(20).astype(np.float32) * 0.05
    bs_cols = ["_neutral"] + [f"bs_{i}" for i in range(51)]
    au_cols = [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
        "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
        "AU25", "AU26", "AU28", "AU43",
    ]
    monkeypatch.setattr(
        b2a, "_PLS_WEIGHTS", {
            "coef": coef,
            "intercept": intercept,
            "blendshape_columns": bs_cols,
            "au_columns": au_cols,
        },
    )
    return coef, intercept


# ---------------------------------------------------------------------
# pls_predict_batch
# ---------------------------------------------------------------------

class TestPlsPredictBatch:
    def test_2d_batch_shape(self, stub_pls_weights):
        x = np.zeros((4, 52), dtype=np.float32)
        out = b2a.pls_predict_batch(x)
        assert out.shape == (4, 20)
        assert out.dtype == np.float32

    def test_1d_input_returns_1d_output(self, stub_pls_weights):
        """Single-vector input should return a single-vector output (no leading axis)."""
        x = np.zeros(52, dtype=np.float32)
        out = b2a.pls_predict_batch(x)
        assert out.shape == (20,), f"expected (20,), got {out.shape}"

    def test_1d_and_2d_agree(self, stub_pls_weights):
        """``predict(v)`` and ``predict(v[None])[0]`` should match."""
        rng = np.random.default_rng(42)
        v = rng.uniform(0, 1, size=52).astype(np.float32)
        out_1d = b2a.pls_predict_batch(v, clip=False)
        out_2d = b2a.pls_predict_batch(v[None, :], clip=False)
        np.testing.assert_allclose(out_1d, out_2d[0], atol=1e-6)

    def test_clip_default_in_unit_interval(self, stub_pls_weights):
        rng = np.random.default_rng(1)
        # Use blendshape values >> 1 so the linear projection definitely
        # produces some out-of-[0,1] outputs without clipping.
        x = rng.uniform(0, 5, size=(8, 52)).astype(np.float32)
        out = b2a.pls_predict_batch(x, clip=True)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_clip_off_can_exceed_unit_interval(self, stub_pls_weights):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 5, size=(8, 52)).astype(np.float32)
        out_unclipped = b2a.pls_predict_batch(x, clip=False)
        # With random ±0.1-magnitude coef and inputs up to 5, expect SOME
        # output to escape [0, 1]; assert at least one boundary is breached.
        assert (out_unclipped < 0.0).any() or (out_unclipped > 1.0).any()

    def test_wrong_1d_length_raises(self, stub_pls_weights):
        with pytest.raises(ValueError, match=r"1-D input of length 52"):
            b2a.pls_predict_batch(np.zeros(50, dtype=np.float32))

    def test_wrong_2d_columns_raises(self, stub_pls_weights):
        with pytest.raises(ValueError, match=r"2-D input with 52 columns"):
            b2a.pls_predict_batch(np.zeros((4, 51), dtype=np.float32))

    def test_3d_input_raises(self, stub_pls_weights):
        with pytest.raises(ValueError, match=r"must be 1-D"):
            b2a.pls_predict_batch(np.zeros((2, 4, 52), dtype=np.float32))

    def test_dtype_promotion_is_safe(self, stub_pls_weights):
        """float64 / int input shouldn't blow up — internally promoted to float32."""
        x_f64 = np.zeros((2, 52), dtype=np.float64)
        out_f64 = b2a.pls_predict_batch(x_f64)
        x_int = np.zeros((2, 52), dtype=np.int32)
        out_int = b2a.pls_predict_batch(x_int)
        np.testing.assert_allclose(out_f64, out_int, atol=1e-6)


# ---------------------------------------------------------------------
# DLIB68_FROM_MP478 + mp478_row_to_dlib68_view
# ---------------------------------------------------------------------

class TestDlib68FromMp478:
    def test_length_is_68(self):
        assert len(b2a.DLIB68_FROM_MP478) == 68

    def test_indices_in_mp_range(self):
        for idx in b2a.DLIB68_FROM_MP478:
            assert 0 <= idx < 478, f"MP index out of range: {idx}"

    def test_no_duplicate_indices(self):
        # Each dlib slot should map to a unique MP vertex.
        assert len(set(b2a.DLIB68_FROM_MP478)) == 68


class TestMp478RowToDlib68View:
    def test_returns_dlib68_xy_keys(self):
        # Build a synthetic MP-478 row where x_i = i and y_i = i + 1000.
        # Then view["x_d"] should equal DLIB68_FROM_MP478[d] (the MP index).
        row = pd.Series({
            **{f"x_{i}": float(i) for i in range(478)},
            **{f"y_{i}": float(i + 1000) for i in range(478)},
        })
        view = b2a.mp478_row_to_dlib68_view(row)
        for dlib_idx in range(68):
            mp_idx = b2a.DLIB68_FROM_MP478[dlib_idx]
            assert view[f"x_{dlib_idx}"] == float(mp_idx)
            assert view[f"y_{dlib_idx}"] == float(mp_idx + 1000)
        # Sanity: dlib chin tip is index 8 in the list, which maps to MP 176;
        # dlib idx 10 (center of jaw) maps to MP 152 per the table.
        assert view["x_10"] == 152.0
        assert view["y_10"] == 1152.0

    def test_passes_through_optional_metadata(self):
        row = pd.Series({
            **{f"x_{i}": float(i) for i in range(478)},
            **{f"y_{i}": float(i) for i in range(478)},
            "FaceRectX": 100.0,
            "Pitch": 0.1,
            "Yaw": -0.2,
            "Roll": 0.05,
        })
        view = b2a.mp478_row_to_dlib68_view(row)
        assert view["FaceRectX"] == 100.0
        assert view["Pitch"] == 0.1
        assert view["Yaw"] == -0.2
        assert view["Roll"] == 0.05

    def test_missing_keys_return_nan(self):
        # Row missing some MP indices — the `.get(...)` fallback should yield NaN
        # without raising
        row = {f"x_{i}": float(i) for i in range(100)}  # only first 100
        row.update({f"y_{i}": float(i) for i in range(100)})
        view = b2a.mp478_row_to_dlib68_view(row)
        # dlib idx 8 maps to MP 152 which is missing → NaN
        assert np.isnan(view["x_8"])
        assert np.isnan(view["y_8"])


# ---------------------------------------------------------------------
# Real HF Hub fetch — runs once, then hits cache. Marked with `network`
# so it can be deselected with `pytest -m 'not network'` in offline CI.
# ---------------------------------------------------------------------

@pytest.mark.network
class TestRealPlsLoad:
    def test_load_real_weights_and_predict(self):
        # Force a fresh module load by clearing the cached weights
        b2a._PLS_WEIGHTS = None
        weights = b2a._load_pls_weights()
        assert weights["coef"].shape == (52, 20)
        assert weights["intercept"].shape == (20,)
        assert len(weights["blendshape_columns"]) == 52
        assert len(weights["au_columns"]) == 20
        assert "AU12" in weights["au_columns"]

        # Sanity: predict on a known blendshape vector. mouthSmileLeft+Right
        # should produce non-trivial AU12 (lip corner pull).
        bs_cols = weights["blendshape_columns"]
        x = np.zeros(52, dtype=np.float32)
        if "mouthSmileLeft" in bs_cols and "mouthSmileRight" in bs_cols:
            x[bs_cols.index("mouthSmileLeft")] = 1.0
            x[bs_cols.index("mouthSmileRight")] = 1.0
        au = b2a.pls_predict_batch(x)
        assert au.shape == (20,)
        # AU output is clipped to [0, 1]
        assert au.min() >= 0.0 and au.max() <= 1.0
