"""Tests for the emotion / blendshape → 478-pt MediaPipe FaceMesh PLS models.

Covers ``PLSFeatMeshModel``, ``predict_face_mesh_from_features``, the
``load_emotion_face_mesh_model`` / ``load_blendshape_face_mesh_model`` loaders,
and the ``emotion=`` / ``blendshapes=`` paths of ``plot_face_mesh``. Shape
contracts run offline by stubbing the module-level cache; two
``@pytest.mark.network`` tests load the real npz from HF Hub.
"""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # headless

from feat import plotting as plt_mod
from feat.plotting import (
    PLSFeatMeshModel,
    load_emotion_face_mesh_model,
    load_blendshape_face_mesh_model,
    predict_face_mesh_from_features,
    plot_face_mesh,
)
from feat.utils import FEAT_EMOTION_COLUMNS, MP_BLENDSHAPE_NAMES


def _make_stub(feature, cols):
    rng = np.random.default_rng(0)
    nfeat = len(cols)
    coef = rng.standard_normal((nfeat + 3, 1434)).astype(np.float32) * 0.05
    intercept = np.empty(1434, dtype=np.float32)
    intercept[:478] = rng.uniform(20, 195, 478)       # x  (v5 pixel frame)
    intercept[478:956] = rng.uniform(-247, -40, 478)  # y
    intercept[956:] = rng.uniform(-72, 51, 478)       # z
    mean_mesh = np.column_stack([intercept[:478], intercept[478:956], intercept[956:]])
    return PLSFeatMeshModel(
        coef=coef, intercept=intercept, feature_columns=cols,
        pose_columns=["Pitch", "Yaw", "Roll"], mean_aligned_mesh=mean_mesh,
        feature_name=feature, model_name=f"{feature}_to_mesh_pls_stub",
    )


@pytest.fixture
def stub_emotion_model(monkeypatch):
    m = _make_stub("emotion", FEAT_EMOTION_COLUMNS)
    monkeypatch.setattr(plt_mod, "_PLS_FEAT_MESH_MODELS", {("emotion", "v5"): m})
    return m


@pytest.fixture
def stub_blendshape_model(monkeypatch):
    m = _make_stub("blendshape", MP_BLENDSHAPE_NAMES)
    monkeypatch.setattr(plt_mod, "_PLS_FEAT_MESH_MODELS", {("blendshape", "v5"): m})
    return m


class TestPLSFeatMeshModel:
    def test_emotion_predict_shape(self, stub_emotion_model):
        flat = stub_emotion_model.predict(np.zeros((4, 7)))
        assert flat.shape == (4, 1434)

    def test_blendshape_predict_shape(self, stub_blendshape_model):
        flat = stub_blendshape_model.predict(np.zeros((3, 52)))
        assert flat.shape == (3, 1434)

    def test_predict_1d_promotes(self, stub_emotion_model):
        assert stub_emotion_model.predict(np.zeros(7)).shape == (1, 1434)

    def test_wrong_width_raises(self, stub_emotion_model):
        with pytest.raises(ValueError, match="7 columns"):
            stub_emotion_model.predict(np.zeros((2, 5)))

    def test_pose_is_implicit_zero(self, stub_emotion_model):
        # deployed coef has nfeat + 3 rows; predict pads the pose channels to 0
        out0 = stub_emotion_model.predict(np.zeros(7))
        assert np.allclose(out0, stub_emotion_model._intercept)


class TestPredictFromFeatures:
    def test_single_returns_478x3(self, stub_emotion_model):
        mesh = predict_face_mesh_from_features(np.zeros(7), model=stub_emotion_model)
        assert mesh.shape == (478, 3)

    def test_batch_returns_n478x3(self, stub_blendshape_model):
        mesh = predict_face_mesh_from_features(np.zeros((5, 52)), model=stub_blendshape_model)
        assert mesh.shape == (5, 478, 3)

    def test_axis_major_reshape(self, stub_emotion_model):
        flat = stub_emotion_model.predict(np.zeros(7))[0]
        mesh = predict_face_mesh_from_features(np.zeros(7), model=stub_emotion_model)
        assert np.allclose(mesh[:, 0], flat[:478])
        assert np.allclose(mesh[:, 1], flat[478:956])
        assert np.allclose(mesh[:, 2], flat[956:])

    def test_rejects_wrong_model_type(self):
        with pytest.raises(ValueError, match="PLSFeatMeshModel"):
            predict_face_mesh_from_features(np.zeros(7), model="not a model")


class TestPlotFaceMeshFeatures:
    def test_plot_emotion(self, stub_emotion_model):
        emo = np.zeros(7); emo[FEAT_EMOTION_COLUMNS.index("happiness")] = 1.0
        ax = plot_face_mesh(emotion=emo)
        assert ax is not None

    def test_plot_blendshapes(self, stub_blendshape_model):
        bs = np.zeros(52); bs[MP_BLENDSHAPE_NAMES.index("jawOpen")] = 1.0
        ax = plot_face_mesh(blendshapes=bs)
        assert ax is not None

    def test_mutually_exclusive_inputs(self, stub_emotion_model):
        with pytest.raises(ValueError, match="at most one"):
            plot_face_mesh(emotion=np.zeros(7), blendshapes=np.zeros(52))


class TestLoaderValidation:
    def test_bad_feature_name(self):
        with pytest.raises(ValueError, match="feature must be one of"):
            plt_mod._load_pls_feat_to_mesh_from_hub("gaze")


@pytest.mark.network
def test_real_emotion_model_from_hub():
    m = load_emotion_face_mesh_model()
    assert isinstance(m, PLSFeatMeshModel)
    assert m.feature_columns == FEAT_EMOTION_COLUMNS
    mesh = predict_face_mesh_from_features(np.zeros(7), model=m)
    assert mesh.shape == (478, 3)
    # happiness should move the mesh away from neutral
    neu = np.zeros(7); neu[FEAT_EMOTION_COLUMNS.index("neutral")] = 1.0
    hap = np.zeros(7); hap[FEAT_EMOTION_COLUMNS.index("happiness")] = 1.0
    d = predict_face_mesh_from_features(hap, model=m) - predict_face_mesh_from_features(neu, model=m)
    assert np.abs(d).max() > 1.0  # pixel-frame units


@pytest.mark.network
def test_real_blendshape_model_from_hub():
    m = load_blendshape_face_mesh_model()
    assert isinstance(m, PLSFeatMeshModel)
    assert m.feature_columns == MP_BLENDSHAPE_NAMES
    mesh = predict_face_mesh_from_features(np.zeros(52), model=m)
    assert mesh.shape == (478, 3)
