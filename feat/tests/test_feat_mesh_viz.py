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
    monkeypatch.setattr(plt_mod, "_PLS_FEAT_MESH_MODELS", {("emotion", plt_mod._FEAT_MESH_DEFAULT_VERSION): m})
    return m


@pytest.fixture
def stub_blendshape_model(monkeypatch):
    m = _make_stub("blendshape", MP_BLENDSHAPE_NAMES)
    monkeypatch.setattr(plt_mod, "_PLS_FEAT_MESH_MODELS", {("blendshape", plt_mod._FEAT_MESH_DEFAULT_VERSION): m})
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
        with pytest.raises(ValueError, match="length-7 vector or"):
            stub_emotion_model.predict(np.zeros((2, 5)))

    def test_scalar_input_raises_clear_error(self, stub_emotion_model):
        # 0-d / scalar must give a clear ValueError, not a cryptic IndexError
        with pytest.raises(ValueError, match="length-7 vector"):
            stub_emotion_model.predict(np.float32(0.5))

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

    def test_single_face_2d_row_kept_as_batch(self, stub_emotion_model):
        # (1, 7) is a batch of one -> (1, 478, 3) from the predict helper
        mesh = predict_face_mesh_from_features(np.zeros((1, 7)), model=stub_emotion_model)
        assert mesh.shape == (1, 478, 3)

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
        emo = np.zeros(7)
        emo[FEAT_EMOTION_COLUMNS.index("happiness")] = 1.0
        ax = plot_face_mesh(emotion=emo)
        assert ax is not None

    def test_plot_blendshapes(self, stub_blendshape_model):
        bs = np.zeros(52)
        bs[MP_BLENDSHAPE_NAMES.index("jawOpen")] = 1.0
        ax = plot_face_mesh(blendshapes=bs)
        assert ax is not None

    def test_plot_accepts_single_face_2d_row(self, stub_emotion_model):
        # plot_face_mesh should accept a (1, 7) single-face row, not reject it
        ax = plot_face_mesh(emotion=np.zeros((1, 7)))
        assert ax is not None

    def test_mutually_exclusive_inputs(self, stub_emotion_model):
        with pytest.raises(ValueError, match="at most one"):
            plot_face_mesh(emotion=np.zeros(7), blendshapes=np.zeros(52))


class TestLoaderValidation:
    def test_bad_feature_name(self):
        with pytest.raises(ValueError, match="feature must be one of"):
            plt_mod._load_pls_feat_to_mesh_from_hub("gaze")


class TestNamedInputs:
    def test_detectorv2_emotion_names_match_array(self, stub_emotion_model):
        import pandas as pd
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(7))
        arr = np.zeros(7)
        # Detectorv2 Fex order and capitalisation
        names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
        feat = ["neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger"]
        for n, f, v in zip(names, feat, p):
            arr[FEAT_EMOTION_COLUMNS.index(f)] = v
        s = pd.Series(dict(zip(names, p)))
        a = predict_face_mesh_from_features(arr, model=stub_emotion_model)
        b = predict_face_mesh_from_features(s, model=stub_emotion_model)
        assert np.allclose(a, b, atol=1e-4)

    def test_permuting_named_input_is_invariant(self, stub_blendshape_model):
        import pandas as pd
        rng = np.random.default_rng(2)
        s = pd.Series(rng.random(52) * 0.3, index=MP_BLENDSHAPE_NAMES)
        a = predict_face_mesh_from_features(s, model=stub_blendshape_model)
        b = predict_face_mesh_from_features(s.sample(frac=1, random_state=0), model=stub_blendshape_model)
        assert np.allclose(a, b, atol=1e-4)

    def test_partial_dict(self, stub_blendshape_model):
        x = np.zeros(52)
        x[MP_BLENDSHAPE_NAMES.index("jawOpen")] = 0.5
        a = predict_face_mesh_from_features(x, model=stub_blendshape_model)
        b = predict_face_mesh_from_features({"jawOpen": 0.5}, model=stub_blendshape_model)
        assert np.allclose(a, b, atol=1e-4)

    def test_dataframe_extra_columns_ignored_and_batched(self, stub_blendshape_model):
        import pandas as pd
        df = pd.DataFrame(np.zeros((3, 52)), columns=MP_BLENDSHAPE_NAMES)
        df["FaceScore"] = 0.9
        assert predict_face_mesh_from_features(df, model=stub_blendshape_model).shape == (3, 478, 3)

    def test_missing_columns_raise(self, stub_emotion_model):
        import pandas as pd
        with pytest.raises(ValueError, match="missing columns"):
            predict_face_mesh_from_features(pd.Series({"Happy": 1.0}), model=stub_emotion_model)

    def test_unknown_dict_key_raises(self, stub_emotion_model):
        with pytest.raises(ValueError, match="unknown emotion column"):
            predict_face_mesh_from_features({"joy": 1.0}, model=stub_emotion_model)

    def test_plot_accepts_dict(self, stub_emotion_model):
        assert plot_face_mesh(emotion={"Happy": 1.0}) is not None


class TestClipping:
    def test_clip_to_feature_max_and_warn(self, stub_blendshape_model):
        m = stub_blendshape_model
        m.feature_max = np.full(52, 0.5, dtype=np.float32)
        m.unsupported_features = ["noseSneerLeft"]
        a = predict_face_mesh_from_features({"jawOpen": 0.9}, model=m)
        b = predict_face_mesh_from_features({"jawOpen": 0.5}, model=m)
        assert np.allclose(a, b, atol=1e-4)
        c = predict_face_mesh_from_features({"jawOpen": 0.9}, model=m, clip=False)
        assert not np.allclose(a, c)
        with pytest.warns(UserWarning, match="never"):
            predict_face_mesh_from_features({"noseSneerLeft": 0.3}, model=m)


def _lm(mesh):
    """Landmark geometry (MediaPipe indices). +y is up in the canonical frame;
    'L' = subject's left (left-eye vertex 386 side)."""
    y, x = mesh[:, 1], mesh[:, 0]
    return dict(eyeL=y[386] - y[374], eyeR=y[159] - y[145], mouth_open=y[13] - y[14],
                mouth_w=abs(x[291] - x[61]), cornerL=y[291] - y[13], cornerR=y[61] - y[13],
                browL=y[334] - y[386], browR=y[105] - y[159], chin=y[152] - y[1],
                innerbrow_gap=abs(x[336] - x[107]))


def _delta(m, act, base):
    a = _lm(predict_face_mesh_from_features(act, model=m))
    b = _lm(predict_face_mesh_from_features(base, model=m))
    return {k: a[k] - b[k] for k in a}


@pytest.mark.network
def test_real_emotion_v6_semantics():
    """Regression test for the v5 label permutation: each emotion must move the
    landmarks it should, relative to one-hot neutral."""
    m = load_emotion_face_mesh_model()
    neu = {"neutral": 1.0}
    d = _delta(m, {"happiness": 1.0}, neu)
    assert d["cornerL"] > 0 and d["cornerR"] > 0 and d["mouth_w"] > 0
    d = _delta(m, {"sadness": 1.0}, neu)
    assert d["cornerL"] < 0 and d["cornerR"] < 0
    d = _delta(m, {"surprise": 1.0}, neu)
    assert d["browL"] > 0 and d["browR"] > 0 and d["mouth_open"] > 0
    d = _delta(m, {"fear": 1.0}, neu)
    assert d["eyeL"] > 0 and d["eyeR"] > 0
    d = _delta(m, {"anger": 1.0}, neu)
    assert d["browL"] < 0 and d["browR"] < 0 and d["innerbrow_gap"] < 0


@pytest.mark.network
def test_real_blendshape_v6_semantics():
    m = load_blendshape_face_mesh_model()
    assert m.feature_max is not None and "noseSneerLeft" in m.unsupported_features
    d = _delta(m, {"jawOpen": 0.6}, {})
    assert d["mouth_open"] > 0 and d["chin"] < 0
    dl, dr = _delta(m, {"eyeBlinkLeft": 0.9}, {}), _delta(m, {"eyeBlinkRight": 0.9}, {})
    assert dl["eyeL"] < 0 and dr["eyeR"] < 0
    d = _delta(m, {"mouthSmileLeft": 0.9}, {})
    assert d["cornerL"] > 0          # same 'Left' side convention as eyeBlinkLeft
    d = _delta(m, {"browDownLeft": 0.9}, {})
    assert d["browL"] < 0


@pytest.mark.network
def test_real_emotion_model_from_hub():
    m = load_emotion_face_mesh_model()
    assert isinstance(m, PLSFeatMeshModel)
    assert m.feature_columns == FEAT_EMOTION_COLUMNS
    mesh = predict_face_mesh_from_features(np.zeros(7), model=m)
    assert mesh.shape == (478, 3)
    # happiness should move the mesh away from neutral
    neu = np.zeros(7)
    neu[FEAT_EMOTION_COLUMNS.index("neutral")] = 1.0
    hap = np.zeros(7)
    hap[FEAT_EMOTION_COLUMNS.index("happiness")] = 1.0
    d = predict_face_mesh_from_features(hap, model=m) - predict_face_mesh_from_features(neu, model=m)
    assert np.abs(d).max() > 1.0  # pixel-frame units


@pytest.mark.network
def test_real_blendshape_model_from_hub():
    m = load_blendshape_face_mesh_model()
    assert isinstance(m, PLSFeatMeshModel)
    assert m.feature_columns == MP_BLENDSHAPE_NAMES
    mesh = predict_face_mesh_from_features(np.zeros(52), model=m)
    assert mesh.shape == (478, 3)
