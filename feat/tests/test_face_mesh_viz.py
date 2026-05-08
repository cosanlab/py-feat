"""Tests for the AU → 478-pt MediaPipe FaceMesh PLS visualization (PR5).

Covers ``PLSAUMeshModel``, ``predict_face_mesh``, and ``plot_face_mesh``.
Shape and reshape contracts run offline by stubbing the module-level cached
model. One ``@pytest.mark.network`` test loads the real npz from HF Hub.
"""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # headless for test environments

from feat import plotting as plt_mod
from feat.plotting import (
    PLSAUMeshModel,
    load_face_mesh_viz_model,
    plot_face_mesh,
    predict_face_mesh,
)


# ---------------------------------------------------------------------
# Fixture: install a stub PLSAUMeshModel into the module-level cache so
# loader / predict / plot tests run offline without an HF fetch.
# ---------------------------------------------------------------------

@pytest.fixture
def stub_mesh_model(monkeypatch):
    """Inject a deterministic PLSAUMeshModel into the module cache.

    Uses small per-axis values so the equal-aspect bbox math doesn't NaN,
    and a non-zero coef so AU activation produces a measurable mesh delta.
    """
    rng = np.random.default_rng(0)
    coef = (rng.standard_normal((23, 1434)).astype(np.float32) * 0.05)
    # Intercept lays out a plausible canonical face mesh: x ∈ [-7, 7], y ∈
    # [-9, 8], z ∈ [-1, 7] — matches the real model's ranges so the plot
    # path's equal-aspect logic exercises a face-shaped bbox.
    intercept = np.empty(1434, dtype=np.float32)
    intercept[:478] = rng.uniform(-7, 7, 478)            # x
    intercept[478:956] = rng.uniform(-9, 8, 478)         # y
    intercept[956:] = rng.uniform(-1, 7, 478)            # z
    au_cols = [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
        "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
        "AU25", "AU26", "AU28", "AU43",
    ]
    pose_cols = ["Pitch", "Yaw", "Roll"]
    mean_mesh = np.column_stack([
        intercept[:478], intercept[478:956], intercept[956:],
    ])
    model = PLSAUMeshModel(
        coef=coef, intercept=intercept,
        au_columns=au_cols, pose_columns=pose_cols,
        mean_aligned_mesh=mean_mesh,
    )
    monkeypatch.setattr(plt_mod, "_PLS_V2_MESH_MODEL", model)
    return model


# ---------------------------------------------------------------------
# PLSAUMeshModel
# ---------------------------------------------------------------------

class TestPLSAUMeshModel:
    def test_predict_2d_batch_returns_flat(self, stub_mesh_model):
        au = np.zeros((3, 20), dtype=np.float32)
        out = stub_mesh_model.predict(au)
        assert out.shape == (3, 1434)

    def test_predict_1d_returns_2d_one_row(self, stub_mesh_model):
        au = np.zeros(20, dtype=np.float32)
        out = stub_mesh_model.predict(au)
        assert out.shape == (1, 1434)

    def test_predict_pose_padding_is_zero(self, stub_mesh_model):
        """Wrapper zero-pads 3 pose features. AU + zero pose ≡ explicit
        23-d call with pose channels set to 0."""
        au = np.array([0.5, 0.3, 0.0] + [0.0] * 17, dtype=np.float32)
        out_wrapper = stub_mesh_model.predict(au)
        x_explicit = np.concatenate([au, np.zeros(3, dtype=np.float32)])
        out_explicit = (x_explicit[None, :] @ stub_mesh_model._coef
                        + stub_mesh_model._intercept)
        np.testing.assert_allclose(out_wrapper, out_explicit, atol=1e-6)

    def test_n_components_is_20(self, stub_mesh_model):
        assert stub_mesh_model.n_components == 20

    def test_repr_includes_model_name(self, stub_mesh_model):
        r = repr(stub_mesh_model)
        assert "au_to_mesh_pls_v2" in r
        assert "n_components=20" in r

    def test_au_and_pose_columns_preserved(self, stub_mesh_model):
        assert len(stub_mesh_model.au_columns) == 20
        assert "AU12" in stub_mesh_model.au_columns
        assert stub_mesh_model.pose_columns == ["Pitch", "Yaw", "Roll"]

    def test_mean_aligned_mesh_shape(self, stub_mesh_model):
        assert stub_mesh_model.mean_aligned_mesh.shape == (478, 3)


# ---------------------------------------------------------------------
# predict_face_mesh — axis-major reshape contract
# ---------------------------------------------------------------------

class TestPredictFaceMesh:
    def test_default_loads_cached_model(self, stub_mesh_model):
        out = predict_face_mesh(np.zeros(20))
        assert out.shape == (478, 3)

    def test_batched_input_returns_3d(self, stub_mesh_model):
        out = predict_face_mesh(np.zeros((4, 20)))
        assert out.shape == (4, 478, 3)

    def test_axis_major_reshape_separates_x_y_z(self, stub_mesh_model):
        """Confirm we slice the flat 1434-d output as ``[x | y | z]``,
        not the wrong ``(478, 3)`` reshape that would interleave per-vertex.

        The stub fixture explicitly packs three different ranges into the
        x / y / z slabs of the intercept; the predicted mesh at AU=0 must
        recover those exact per-axis ranges.
        """
        out = predict_face_mesh(np.zeros(20), model=stub_mesh_model)
        # x slab from intercept was uniform(-7, 7); y was uniform(-9, 8);
        # z was uniform(-1, 7). With AU=0, predict == intercept exactly,
        # so slabs reappear column-by-column.
        np.testing.assert_array_equal(out[:, 0], stub_mesh_model._intercept[:478])
        np.testing.assert_array_equal(out[:, 1], stub_mesh_model._intercept[478:956])
        np.testing.assert_array_equal(out[:, 2], stub_mesh_model._intercept[956:])

    def test_au_activation_changes_mesh(self, stub_mesh_model):
        rest = predict_face_mesh(np.zeros(20), model=stub_mesh_model)
        au12 = np.zeros(20)
        au12[stub_mesh_model.au_columns.index("AU12")] = 3.0
        active = predict_face_mesh(au12, model=stub_mesh_model)
        assert active.shape == rest.shape
        # Some non-trivial mesh delta for a moderately strong activation
        assert np.linalg.norm(active - rest) > 0.0

    def test_rejects_wrong_au_length(self, stub_mesh_model):
        with pytest.raises(ValueError, match=r"au vector must be length 20"):
            predict_face_mesh(np.zeros(15), model=stub_mesh_model)

    def test_rejects_non_mesh_model(self, stub_mesh_model):
        class NotAModel:
            n_components = 20
            def predict(self, x): return np.zeros((1, 1434))
        with pytest.raises(ValueError, match=r"PLSAUMeshModel"):
            predict_face_mesh(np.zeros(20), model=NotAModel())


# ---------------------------------------------------------------------
# plot_face_mesh — smoke (matplotlib renders without raising)
# ---------------------------------------------------------------------

class TestPlotFaceMesh:
    def test_plots_rest_mesh_without_au(self, stub_mesh_model):
        ax = plot_face_mesh(au=None, model=stub_mesh_model)
        assert ax is not None
        # Lines drawn for each canonical contour connection
        assert len(ax.get_lines()) > 0

    def test_plots_with_au_activation(self, stub_mesh_model):
        au = np.zeros(20)
        au[stub_mesh_model.au_columns.index("AU12")] = 2.0
        ax = plot_face_mesh(au=au, model=stub_mesh_model)
        assert len(ax.get_lines()) > 0

    def test_rejects_batched_au(self, stub_mesh_model):
        with pytest.raises(ValueError, match=r"single AU vector"):
            plot_face_mesh(au=np.zeros((2, 20)), model=stub_mesh_model)


# ---------------------------------------------------------------------
# Real HF Hub fetch — runs once, then hits cache.
# ---------------------------------------------------------------------

@pytest.mark.network
class TestRealMeshLoad:
    def test_load_real_v2_model(self, monkeypatch):
        # Force a fresh load by clearing the module-level cache
        monkeypatch.setattr(plt_mod, "_PLS_V2_MESH_MODEL", None)
        m = load_face_mesh_viz_model()
        assert isinstance(m, PLSAUMeshModel)
        assert m.n_components == 20
        assert m.mean_aligned_mesh.shape == (478, 3)
        # Sanity: a smile activation should move mesh vertices noticeably
        au = np.zeros(20)
        au[m.au_columns.index("AU12")] = 3.0
        rest = predict_face_mesh(np.zeros(20), model=m)
        active = predict_face_mesh(au, model=m)
        per_vertex_disp = np.linalg.norm(active - rest, axis=1)
        # Lip corner pull should produce a measurable max displacement.
        # Threshold is intentionally loose: just confirm AU has *some* effect.
        assert per_vertex_disp.max() > 0.05
