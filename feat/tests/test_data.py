import pytest
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
from feat.data import Fex
from feat.utils.io import read_openface, get_test_data_path, read_feat, read_fex
from nltools.data import Adjacency


def test_info(capsys):
    importantstring = "ThisStringMustBeIncluded"
    fex = Fex(filename=importantstring)
    fex.info
    captured = capsys.readouterr()
    assert importantstring in captured.out


def test_fex_new(data_path):
    fex = pd.concat(
        map(lambda f: read_feat(os.path.join(data_path, f)), ["001.csv", "002.csv"])
    )
    assert fex.shape == (68, 173)

    assert "AU01" in fex.au_columns

    # Update sessions (grouping factor) to video ids to group rows (frames) by video
    by_video = fex.update_sessions(fex["input"])
    # Compute the mean per video
    video_means = by_video.extract_mean()

    # one row per video
    assert video_means.shape == (2, 172)

    # we rename columns when using extract methods
    # test that attribute renames have also propagated correctly
    hasprefix = lambda col: col.startswith("mean")
    assert all(map(hasprefix, video_means.au_columns))
    assert all(map(hasprefix, video_means.emotion_columns))
    assert all(map(hasprefix, video_means.facebox_columns))
    assert all(map(hasprefix, video_means.landmark_columns))
    assert all(map(hasprefix, video_means.facepose_columns))
    assert all(map(hasprefix, video_means.time_columns))


def test_fex_old(imotions_data):
    # Dropped support in >= 0.4.0
    with pytest.raises(Exception):
        Fex().read_facet()
    with pytest.raises(Exception):
        Fex().read_affectiva()

    df = imotions_data

    # Test slicing functions
    assert df.aus.shape == (519, 20)
    assert df.emotions.shape == (519, 12)
    assert df.facebox.shape == (519, 4)
    assert df.time.shape[-1] == 4
    assert df.design.shape[-1] == 4

    # Test metadata propagation to sliced series
    assert df.iloc[0].aus.shape == (20,)
    assert df.iloc[0].emotions.shape == (12,)
    assert df.iloc[0].facebox.shape == (4,)
    assert df.iloc[0].time.shape == (4,)
    assert df.iloc[0].design.shape == (4,)

    sessions = np.array([[x] * 10 for x in range(1 + int(len(df) / 10))]).flatten()[:-1]
    dat = Fex(
        df,
        sampling_freq=30,
        sessions=sessions,
        emotion_columns=[
            "Joy",
            "Anger",
            "Surprise",
            "Fear",
            "Contempt",
            "Disgust",
            "Sadness",
            "Confusion",
            "Frustration",
            "Neutral",
            "Positive",
            "Negative",
        ],
    )
    dat = dat[
        [
            "Joy",
            "Anger",
            "Surprise",
            "Fear",
            "Contempt",
            "Disgust",
            "Sadness",
            "Confusion",
            "Frustration",
            "Neutral",
            "Positive",
            "Negative",
        ]
    ]

    # Test Session ValueError
    with pytest.raises(ValueError):
        Fex(df, sampling_freq=30, sessions=sessions[:10])

    # Test length
    assert len(dat) == 519

    # Test sessions generator
    assert len(np.unique(dat.sessions)) == len([x for x in dat.itersessions()])

    # Test metadata propagation
    assert dat[["Joy"]].sampling_freq == dat.sampling_freq
    assert dat.iloc[:, 0].sampling_freq == dat.sampling_freq
    assert dat.iloc[0, :].sampling_freq == dat.sampling_freq
    assert dat.loc[[0], :].sampling_freq == dat.sampling_freq
    assert dat.loc[:, ["Joy"]].sampling_freq == dat.sampling_freq

    # Test Downsample
    assert len(dat.downsample(target=10)) == 52

    # Test upsample
    # Commenting out because of a bug in nltools: https://github.com/cosanlab/nltools/issues/418
    # assert len(dat.upsample(target=60, target_type="hz")) == (len(dat) - 1) * 2

    # Test interpolation
    assert (
        dat.interpolate(method="linear").isnull().sum()["Positive"]
        < dat.isnull().sum()["Positive"]
    )
    dat = dat.interpolate(method="linear")

    # Test distance
    d = dat[["Positive"]].distance()
    assert isinstance(d, Adjacency)
    assert d.square_shape()[0] == len(dat)

    # Test Copy
    assert isinstance(dat.copy(), Fex)
    assert dat.copy().sampling_freq == dat.sampling_freq

    # Test rectification
    rectified = df.rectification()
    assert (
        df[df.au_columns].isna().sum()[0]
        < rectified[rectified.au_columns].isna().sum()[0]
    )

    # Test baseline
    assert isinstance(dat.baseline(baseline="median"), Fex)
    assert isinstance(dat.baseline(baseline="mean"), Fex)
    assert isinstance(dat.baseline(baseline="begin"), Fex)
    assert isinstance(dat.baseline(baseline=dat.mean()), Fex)
    assert isinstance(dat.baseline(baseline="median", ignore_sessions=True), Fex)
    assert isinstance(dat.baseline(baseline="mean", ignore_sessions=True), Fex)
    assert isinstance(dat.baseline(baseline=dat.mean(), ignore_sessions=True), Fex)
    assert isinstance(dat.baseline(baseline="median", normalize="pct"), Fex)
    assert isinstance(dat.baseline(baseline="mean", normalize="pct"), Fex)
    assert isinstance(dat.baseline(baseline=dat.mean(), normalize="pct"), Fex)
    assert isinstance(
        dat.baseline(baseline="median", ignore_sessions=True, normalize="pct"), Fex
    )
    assert isinstance(
        dat.baseline(baseline="mean", ignore_sessions=True, normalize="pct"), Fex
    )
    assert isinstance(
        dat.baseline(baseline=dat.mean(), ignore_sessions=True, normalize="pct"), Fex
    )
    # Test ValueError
    with pytest.raises(ValueError):
        dat.baseline(baseline="BadValue")

    # Test summary
    dat2 = dat.loc[:, ["Positive", "Negative"]].interpolate()
    out = dat2.extract_summary(min=True, max=True, mean=True)
    assert len(out) == len(np.unique(dat2.sessions))
    assert np.array_equal(out.sessions, np.unique(dat2.sessions))
    assert out.sampling_freq == dat2.sampling_freq
    assert dat2.shape[1] * 5 == out.shape[1]
    out = dat2.extract_summary(min=True, max=True, mean=True, ignore_sessions=True)
    assert len(out) == 1
    assert dat2.shape[1] * 5 == out.shape[1]

    # Test clean
    assert isinstance(dat.clean(), Fex)
    assert dat.clean().columns is dat.columns
    assert dat.clean().sampling_freq == dat.sampling_freq

    # Test Decompose
    n_components = 3
    stats = dat.decompose(algorithm="pca", axis=1, n_components=n_components)
    assert n_components == stats["components"].shape[1]
    assert n_components == stats["weights"].shape[1]

    stats = dat.decompose(algorithm="ica", axis=1, n_components=n_components)
    assert n_components == stats["components"].shape[1]
    assert n_components == stats["weights"].shape[1]

    new_dat = dat + 100
    stats = new_dat.decompose(algorithm="nnmf", axis=1, n_components=n_components)
    assert n_components == stats["components"].shape[1]
    assert n_components == stats["weights"].shape[1]

    stats = dat.decompose(algorithm="fa", axis=1, n_components=n_components)
    assert n_components == stats["components"].shape[1]
    assert n_components == stats["weights"].shape[1]

    stats = dat.decompose(algorithm="pca", axis=0, n_components=n_components)
    assert n_components == stats["components"].shape[1]
    assert n_components == stats["weights"].shape[1]

    stats = dat.decompose(algorithm="ica", axis=0, n_components=n_components)
    assert n_components == stats["components"].shape[1]
    assert n_components == stats["weights"].shape[1]

    new_dat = dat + 100
    stats = new_dat.decompose(algorithm="nnmf", axis=0, n_components=n_components)
    assert n_components == stats["components"].shape[1]
    assert n_components == stats["weights"].shape[1]

    stats = dat.decompose(algorithm="fa", axis=0, n_components=n_components)
    assert n_components == stats["components"].shape[1]
    assert n_components == stats["weights"].shape[1]


def test_openface():
    # For OpenFace data file
    filename = os.path.join(get_test_data_path(), "OpenFace_Test.csv")
    openface = Fex(read_openface(filename), sampling_freq=30)

    # Test KeyError
    with pytest.raises(KeyError):
        Fex(read_openface(filename, features=["NotHere"]), sampling_freq=30)

    # Test length
    assert len(openface) == 100

    # Test loading from filename
    openface = Fex(filename=filename, sampling_freq=30, detector="OpenFace")
    openface = openface.read_file()

    # Test length?
    assert len(openface) == 100

    # Test landmark methods
    assert openface.landmark.shape[1] == 136
    assert openface.iloc[0].landmark.shape[0] == 136
    assert openface.landmark_x.shape[1] == openface.landmark_y.shape[1]
    assert openface.iloc[0].landmark_x.shape[0] == openface.iloc[0].landmark_y.shape[0]


def test_feat():
    filename = os.path.join(get_test_data_path(), "Feat_Test.csv")
    fex = Fex(filename=filename, detector="Feat")
    fex = fex.read_file()
    # test input property
    assert fex.input.values[0] == fex.iloc[0].input

def test_feat_io(default_detector, single_face_img):
    fex1 = default_detector.detect_image(single_face_img)
    fex1.write('Feat_Test_With_Metadata.csv')
    
    fex2 = read_fex('Feat_Test_With_Metadata.csv')
    assert fex1.face_model == fex2.face_model
    assert fex1.landmark_model == fex2.landmark_model
    assert fex1.facepose_model == fex2.facepose_model
    assert fex1.au_model == fex2.au_model
    assert fex1.identity_model == fex2.identity_model
    assert fex1.inputs.values[0] == fex2.inputs.values[0]
    assert len(fex1.facebox_columns) == len(fex2.facebox_columns)
    assert len(fex1.emotion_columns) == len(fex2.emotion_columns)
    assert len(fex1.au_columns) == len(fex2.au_columns)
    assert len(fex1.landmark_columns) == len(fex2.landmark_columns)
    assert len(fex1.identity_columns) == len(fex2.identity_columns)
    assert fex1.aus['AU01'].values[0] == fex2.aus['AU01'].values[0]
    
    
def test_stats():
    filename = os.path.join(get_test_data_path(), "OpenFace_Test.csv")
    openface = Fex(filename=filename, sampling_freq=30, detector="OpenFace")
    openface = openface.read_file()

    aus = openface.aus
    aus.sessions = range(len(aus))
    y = aus[[i for i in aus.columns if "_r" in i]]
    X = pd.DataFrame(aus.sessions)
    b, se, t, p, df, res = aus.regress(X, y, mode="ols", fit_intercept=True)
    assert b.shape == (2, 17)
    assert res.mean().mean() < 1

    clf, scores = openface.predict(X=["AU02_c"], y=["AU04_c"])
    assert clf.coef_ < 0

    clf, scores = openface.predict(X=openface[["AU02_c"]], y=openface["AU04_c"])
    assert clf.coef_ < 0

    t, p = openface[["AU02_c"]].ttest_1samp()
    assert t > 0

    a = openface.aus.assign(input="0")
    b = openface.aus.apply(lambda x: x + np.random.rand(100)).assign(input="1")
    doubled = pd.concat([a, b])
    doubled.sessions = doubled["input"]
    t, p = doubled.ttest_ind(col="AU12_r", sessions=("0", "1"))
    assert t < 0

    frame = np.concatenate(
        [np.array(range(int(len(doubled) / 2))), np.array(range(int(len(doubled) / 2)))]
    )
    assert doubled.assign(frame=frame).isc(col="AU04_r").iloc[0, 0] == 1
