"""Tests for ``feat.plotting.PLSAULandmarkModel`` and ``load_viz_model``.

Covers the v2 PLS landmark wrapper (default since the v0.7-pls-au-to-landmarks
swap) and the legacy v1 ``PLSRegression`` path.

The wrapper-shape and ``predict`` contract tests run offline by stubbing the
module-level cached model. Real HuggingFace Hub fetches are marked
``@pytest.mark.network`` so offline CI can deselect with ``-m "not network"``.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from feat import plotting as plt_mod
from feat.plotting import (
    PLSAULandmarkModel,
    load_viz_model,
    predict,
)


# ---------------------------------------------------------------------
# Helpers — install a stub PLSAULandmarkModel into the module-level cache
# so loader / predict / plot_face tests run offline without an HF fetch.
# ---------------------------------------------------------------------

@pytest.fixture
def stub_v2_model(monkeypatch):
    """Inject a deterministic PLSAULandmarkModel into the module cache."""
    rng = np.random.default_rng(0)
    coef = rng.standard_normal((23, 136)).astype(np.float32) * 0.05
    intercept = rng.standard_normal(136).astype(np.float32) * 100  # face-pixel scale
    au_cols = [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
        "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
        "AU25", "AU26", "AU28", "AU43",
    ]
    model = PLSAULandmarkModel(coef, intercept, au_cols)
    # Loader caches by version in a dict now; inject the stub under "v2".
    monkeypatch.setattr(plt_mod, "_PLS_LMK_MODELS", {"v2": model})
    return model


# ---------------------------------------------------------------------
# PLSAULandmarkModel
# ---------------------------------------------------------------------

class TestPLSAULandmarkModel:
    def test_predict_2d_batch_shape(self, stub_v2_model):
        au = np.zeros((3, 20), dtype=np.float32)
        out = stub_v2_model.predict(au)
        assert out.shape == (3, 136)

    def test_predict_1d_input_returns_2d(self, stub_v2_model):
        """1-D input is reshaped to (1, n_components) — output stays 2-D so
        downstream ``np.reshape(..., (2, 68))`` in ``feat.plotting.predict``
        works regardless of input shape."""
        au = np.zeros(20, dtype=np.float32)
        out = stub_v2_model.predict(au)
        assert out.shape == (1, 136)

    def test_predict_pose_padding_is_zero(self, stub_v2_model):
        """Wrapper zero-pads the 3 pose features. AU vector + zero pose ==
        same as 23-d explicitly-zero-pose call."""
        au = np.array([0.5, 0.3, 0.0] + [0.0] * 17, dtype=np.float32)
        out_wrapper = stub_v2_model.predict(au)
        # Replicate the wrapper math directly with explicit zero pose
        x_explicit = np.concatenate([au, np.zeros(3, dtype=np.float32)])
        out_explicit = (x_explicit[None, :] @ stub_v2_model._coef
                        + stub_v2_model._intercept)
        np.testing.assert_allclose(out_wrapper, out_explicit, atol=1e-6)

    def test_n_components_is_20(self, stub_v2_model):
        """The exposed n_components is the AU-input dim users pass, NOT the
        underlying coef rank (which is 23 with pose absorbed)."""
        assert stub_v2_model.n_components == 20

    def test_repr_includes_model_name(self, stub_v2_model):
        r = repr(stub_v2_model)
        assert "au_to_landmarks_pls_v2" in r
        assert "n_components=20" in r

    def test_au_columns_preserved(self, stub_v2_model):
        assert len(stub_v2_model.au_columns) == 20
        assert "AU12" in stub_v2_model.au_columns


# ---------------------------------------------------------------------
# load_viz_model + predict integration
# ---------------------------------------------------------------------

class TestLoadVizModelDefault:
    def test_default_returns_wrapper(self, stub_v2_model):
        m = load_viz_model()
        assert isinstance(m, PLSAULandmarkModel)
        # Cached: same instance on a second call
        assert load_viz_model() is m

    def test_predict_with_wrapper_returns_2x68(self, stub_v2_model):
        au = np.zeros(20, dtype=np.float32)
        landmarks = predict(au)
        assert landmarks.shape == (2, 68)

    def test_predict_rejects_non_pls_model(self, stub_v2_model):
        class NotAModel:
            n_components = 20
            def predict(self, x): return np.zeros((1, 136))
        with pytest.raises(ValueError, match=r"PLSRegression instance or PLSAULandmarkModel"):
            predict(np.zeros(20), model=NotAModel())

    def test_predict_validates_au_length(self, stub_v2_model):
        with pytest.raises(ValueError, match=r"au vector must be length"):
            predict(np.zeros(15))

    def test_predict_accepts_explicit_legacy_model(self, stub_v2_model, monkeypatch):
        """A v1 PLSRegression instance should still be accepted."""
        # Build a minimal PLSRegression object and stuff in the attributes that
        # ``predict()`` reads. We don't need it to actually predict — just to
        # not raise the isinstance check. The ``model.predict(au)`` call inside
        # plotting.predict() requires a working .predict, so we monkeypatch.
        legacy = PLSRegression(n_components=2)
        legacy.n_components = 20  # match input shape py-feat passes
        monkeypatch.setattr(legacy, "predict", lambda x: np.zeros((x.shape[0], 136)))
        out = predict(np.zeros(20), model=legacy)
        assert out.shape == (2, 68)


# ---------------------------------------------------------------------
# Real HF Hub fetch — runs once, then hits cache.
# ---------------------------------------------------------------------

@pytest.mark.network
class TestRealVizLoad:
    def test_load_real_v2_model(self, monkeypatch):
        # Force a fresh load by clearing the module-level cache
        monkeypatch.setattr(plt_mod, "_PLS_LMK_MODELS", {})
        m = load_viz_model()
        assert isinstance(m, PLSAULandmarkModel)
        assert m.n_components == 20
        # Quick sanity round-trip: AU=0 should yield a coherent face shape
        landmarks = predict(np.zeros(20), model=m)
        assert landmarks.shape == (2, 68)
        # Reasonable face dimensions in image-space pixels (~100-500 px range).
        assert landmarks.min() > 0
        assert landmarks.max() < 1000

    def test_legacy_v1_load_via_hf_hub(self, monkeypatch):
        """Legacy ``pyfeat_aus_to_landmarks`` filename should fetch from
        py-feat/au_to_landmarks and return a PLSRegression instance."""
        monkeypatch.setattr(plt_mod, "_PLS_LMK_MODELS", {})
        m = load_viz_model(file_name="pyfeat_aus_to_landmarks")
        assert isinstance(m, PLSRegression)
        assert m.n_components == 20
