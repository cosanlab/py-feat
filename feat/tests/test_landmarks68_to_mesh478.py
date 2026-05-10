"""Tests for the dlib-68 → MP-478 mesh bridge (PR6).

Covers ``PCALandmarks68ToMeshModel``, ``predict_mesh_from_dlib68``, the
private ``_procrustes_align_2d_batched`` helper, the ``mesh=`` extension to
``plot_face_mesh``, and the column-order drift guard. Offline tests use
synthetic stub models; one ``@pytest.mark.network`` test loads the real npz.
"""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # headless

from feat import plotting as plt_mod
from feat.plotting import (
    PCALandmarks68ToMeshModel,
    _procrustes_align_2d_batched,
    load_landmarks68_to_mesh478_model,
    plot_face_mesh,
    predict_mesh_from_dlib68,
)


# ---------------------------------------------------------------------
# Stub fixture — synthetic model whose intercept slabs encode distinct
# per-axis ranges so axis-major reshape can be pinned end-to-end.
# ---------------------------------------------------------------------

@pytest.fixture
def stub_bridge_model(monkeypatch):
    rng = np.random.default_rng(0)
    coef = rng.standard_normal((136, 1434)).astype(np.float32) * 0.001
    intercept = np.empty(1434, dtype=np.float32)
    intercept[:478] = rng.uniform(-7, 7, 478)            # x slab
    intercept[478:956] = rng.uniform(-9, 8, 478)         # y slab
    intercept[956:] = rng.uniform(-1, 7, 478)            # z slab
    input_columns = ([f"lm_x_{i}" for i in range(68)]
                     + [f"lm_y_{i}" for i in range(68)])
    anchor_indices = np.array([27, 28, 29, 30, 36, 39, 42, 45], dtype=np.int64)
    # Reference: a plausible canonical dlib face shape.
    ref_anchors = np.array([
        [320.0, 250.0],   # 27: nose root top
        [322.0, 266.0],   # 28
        [323.0, 281.0],   # 29
        [324.0, 295.0],   # 30: nose tip
        [270.0, 260.0],   # 36: outer right canthus
        [299.0, 258.0],   # 39: inner right canthus
        [343.0, 254.0],   # 42: inner left canthus
        [372.0, 250.0],   # 45: outer left canthus
    ], dtype=np.float32)
    mean_dlib = rng.uniform(200, 400, (68, 2)).astype(np.float32)
    mean_mesh = np.column_stack([
        intercept[:478], intercept[478:956], intercept[956:],
    ])
    model = PCALandmarks68ToMeshModel(
        coef=coef, intercept=intercept, input_columns=input_columns,
        anchor_indices_dlib68=anchor_indices,
        reference_dlib_anchors=ref_anchors,
        mean_aligned_dlib_landmarks=mean_dlib,
        mean_predicted_mesh=mean_mesh,
    )
    monkeypatch.setattr(plt_mod, "_LM68_TO_MESH478_MODEL", model)
    return model


# ---------------------------------------------------------------------
# _procrustes_align_2d_batched
# ---------------------------------------------------------------------

class TestProcrustes2D:
    def test_identity_when_anchors_already_match(self, stub_bridge_model):
        """If a face's anchors already equal the reference, alignment is a no-op."""
        landmarks = stub_bridge_model.mean_aligned_dlib_landmarks.copy()
        # Place the reference anchors at the canonical positions
        landmarks[stub_bridge_model.anchor_indices_dlib68] = (
            stub_bridge_model.reference_dlib_anchors
        )
        aligned = _procrustes_align_2d_batched(
            landmarks[None],
            stub_bridge_model.anchor_indices_dlib68,
            stub_bridge_model.reference_dlib_anchors,
        )
        np.testing.assert_allclose(
            aligned[0, stub_bridge_model.anchor_indices_dlib68],
            stub_bridge_model.reference_dlib_anchors,
            atol=1e-3,
        )

    def test_recovers_reference_after_rotation_scale(self, stub_bridge_model):
        """Apply a rotation + scale + translation to a face that already
        matches the reference; alignment should invert it."""
        landmarks = np.zeros((68, 2), dtype=np.float32)
        landmarks[stub_bridge_model.anchor_indices_dlib68] = (
            stub_bridge_model.reference_dlib_anchors
        )
        # Apply a known similarity: rotate by 30°, scale 1.5x, translate by (50, -20)
        theta = np.deg2rad(30.0)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
        s = 1.5
        t = np.array([50.0, -20.0], dtype=np.float32)
        transformed = (s * landmarks @ R.T + t).astype(np.float32)
        aligned = _procrustes_align_2d_batched(
            transformed[None],
            stub_bridge_model.anchor_indices_dlib68,
            stub_bridge_model.reference_dlib_anchors,
        )
        np.testing.assert_allclose(
            aligned[0, stub_bridge_model.anchor_indices_dlib68],
            stub_bridge_model.reference_dlib_anchors,
            atol=1e-2,
        )

    def test_rejects_wrong_shape(self, stub_bridge_model):
        with pytest.raises(ValueError, match=r"\(n, 68, 2\)"):
            _procrustes_align_2d_batched(
                np.zeros((10, 2), dtype=np.float32),
                stub_bridge_model.anchor_indices_dlib68,
                stub_bridge_model.reference_dlib_anchors,
            )


# ---------------------------------------------------------------------
# PCALandmarks68ToMeshModel
# ---------------------------------------------------------------------

class TestPCALandmarks68ToMeshModel:
    def test_predict_2d_batch_returns_flat(self, stub_bridge_model):
        x = np.zeros((3, 136), dtype=np.float32)
        out = stub_bridge_model.predict(x)
        assert out.shape == (3, 1434)

    def test_predict_1d_returns_2d_one_row(self, stub_bridge_model):
        x = np.zeros(136, dtype=np.float32)
        out = stub_bridge_model.predict(x)
        assert out.shape == (1, 1434)

    def test_predict_rejects_wrong_input_width(self, stub_bridge_model):
        with pytest.raises(ValueError, match=r"must have 136 columns"):
            stub_bridge_model.predict(np.zeros((4, 130), dtype=np.float32))
        with pytest.raises(ValueError, match=r"must have 136 columns"):
            stub_bridge_model.predict(np.zeros(130, dtype=np.float32))

    def test_repr_includes_model_name(self, stub_bridge_model):
        r = repr(stub_bridge_model)
        assert "landmarks68_to_mesh478_pca_v2" in r
        assert "input_dim=136" in r


# ---------------------------------------------------------------------
# predict_mesh_from_dlib68 — end-to-end with axis-major reshape pinning
# ---------------------------------------------------------------------

class TestPredictMeshFromDlib68:
    def test_single_face_returns_478_3(self, stub_bridge_model):
        landmarks = stub_bridge_model.mean_aligned_dlib_landmarks
        mesh = predict_mesh_from_dlib68(landmarks)
        assert mesh.shape == (478, 3)

    def test_batched_returns_n_478_3(self, stub_bridge_model):
        batch = np.stack([stub_bridge_model.mean_aligned_dlib_landmarks] * 4)
        mesh = predict_mesh_from_dlib68(batch)
        assert mesh.shape == (4, 478, 3)

    def test_axis_major_reshape_recovers_per_axis_ranges(self, stub_bridge_model):
        """Pin axis-major reshape end-to-end: with coef set to zero, predict
        is exactly the intercept regardless of input. Each intercept slab's
        distinct range must reappear on the matching axis of the (478, 3)
        output. Catches a future swap to per-vertex (478, 3) reshape that
        would scramble vertex ordering.
        """
        # Use the existing stub but null out the coef so the regression
        # contribution drops to zero — the output IS the intercept slabs.
        zero_coef_model = PCALandmarks68ToMeshModel(
            coef=np.zeros_like(stub_bridge_model._coef),
            intercept=stub_bridge_model._intercept,
            input_columns=stub_bridge_model.input_columns,
            anchor_indices_dlib68=stub_bridge_model.anchor_indices_dlib68,
            reference_dlib_anchors=stub_bridge_model.reference_dlib_anchors,
            mean_aligned_dlib_landmarks=stub_bridge_model.mean_aligned_dlib_landmarks,
            mean_predicted_mesh=stub_bridge_model.mean_predicted_mesh,
        )
        mesh = predict_mesh_from_dlib68(
            stub_bridge_model.mean_aligned_dlib_landmarks, model=zero_coef_model,
        )
        np.testing.assert_array_equal(mesh[:, 0], zero_coef_model._intercept[:478])
        np.testing.assert_array_equal(mesh[:, 1], zero_coef_model._intercept[478:956])
        np.testing.assert_array_equal(mesh[:, 2], zero_coef_model._intercept[956:])

    def test_empty_batch_passthrough(self, stub_bridge_model):
        """A zero-row batch (e.g. when no faces detected upstream) should
        produce an empty (0, 478, 3) result without raising."""
        empty = np.zeros((0, 68, 2), dtype=np.float32)
        out = predict_mesh_from_dlib68(empty, model=stub_bridge_model)
        assert out.shape == (0, 478, 3)

    def test_rejects_wrong_landmark_shape(self, stub_bridge_model):
        with pytest.raises(ValueError, match=r"\(68, 2\) or \(n, 68, 2\)"):
            predict_mesh_from_dlib68(np.zeros((50, 2)), model=stub_bridge_model)

    def test_rejects_non_bridge_model(self, stub_bridge_model):
        class NotAModel:
            def predict(self, x): return np.zeros((1, 1434))
        with pytest.raises(ValueError, match=r"PCALandmarks68ToMeshModel"):
            predict_mesh_from_dlib68(
                np.zeros((68, 2)), model=NotAModel(),
            )


# ---------------------------------------------------------------------
# plot_face_mesh — new mesh= path
# ---------------------------------------------------------------------

class TestPlotFaceMeshWithMesh:
    def test_accepts_precomputed_mesh(self, stub_bridge_model):
        mesh = stub_bridge_model.mean_predicted_mesh
        ax = plot_face_mesh(mesh=mesh)
        assert ax is not None
        assert len(ax.get_lines()) > 0

    def test_rejects_both_au_and_mesh(self, stub_bridge_model):
        mesh = stub_bridge_model.mean_predicted_mesh
        with pytest.raises(ValueError, match=r"either `au` or `mesh`"):
            plot_face_mesh(au=np.zeros(20), mesh=mesh)

    def test_rejects_wrong_mesh_shape(self):
        with pytest.raises(ValueError, match=r"shape \(478, 3\)"):
            plot_face_mesh(mesh=np.zeros((100, 3)))


# ---------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------

class TestInputColumnDriftGuard:
    def test_drift_raises_at_load(self, monkeypatch, tmp_path):
        """If a future re-trained npz uses a non-axis-major layout (e.g.,
        interleaved per-vertex), the loader must refuse rather than silently
        producing garbage meshes."""
        # Build a fake npz with interleaved column names instead of axis-major
        bad_path = tmp_path / "lm68_to_mesh_bad.npz"
        bad_columns = []
        for i in range(68):
            bad_columns.extend([f"lm_x_{i}", f"lm_y_{i}"])  # interleaved
        np.savez(
            bad_path,
            coef=np.zeros((136, 1434), dtype=np.float32),
            intercept=np.zeros(1434, dtype=np.float32),
            input_columns=np.array(bad_columns),
            anchor_indices_dlib68=np.array([27, 28, 29, 30, 36, 39, 42, 45], dtype=np.int32),
            reference_dlib_anchors=np.zeros((8, 2), dtype=np.float32),
            mean_aligned_dlib_landmarks=np.zeros((68, 2), dtype=np.float32),
            mean_predicted_mesh=np.zeros((478, 3), dtype=np.float32),
        )

        monkeypatch.setattr(plt_mod, "_LM68_TO_MESH478_MODEL", None)
        monkeypatch.setattr(plt_mod, "hf_hub_download",
                            lambda **kwargs: str(bad_path))

        with pytest.raises(RuntimeError, match=r"input_columns drifted"):
            plt_mod._load_landmarks68_to_mesh478_v2_from_hub()


# ---------------------------------------------------------------------
# Real HF Hub fetch — runs once, then hits cache.
# ---------------------------------------------------------------------

@pytest.mark.network
class TestRealBridgeLoad:
    def test_load_real_v2_model(self, monkeypatch):
        monkeypatch.setattr(plt_mod, "_LM68_TO_MESH478_MODEL", None)
        m = load_landmarks68_to_mesh478_model()
        assert isinstance(m, PCALandmarks68ToMeshModel)
        assert m._coef.shape == (136, 1434)
        assert m.mean_aligned_dlib_landmarks.shape == (68, 2)
        assert m.mean_predicted_mesh.shape == (478, 3)

    def test_mean_landmarks_reconstruct_close_to_mean_mesh(self, monkeypatch):
        """Feeding the saved mean dlib landmarks back through the bridge
        should approximately reproduce the saved mean predicted mesh.
        """
        monkeypatch.setattr(plt_mod, "_LM68_TO_MESH478_MODEL", None)
        m = load_landmarks68_to_mesh478_model()
        mesh = predict_mesh_from_dlib68(m.mean_aligned_dlib_landmarks, model=m)
        # Threshold is loose: alignment of the mean landmarks vs. the
        # population mean isn't a perfect round-trip, but should be within
        # 1 cm per axis (mesh is in cm units).
        max_diff = float(np.abs(mesh - m.mean_predicted_mesh).max())
        assert max_diff < 1.0, f"Reconstruction diff too high: {max_diff:.3f} cm"
