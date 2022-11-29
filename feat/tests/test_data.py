import pytest
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
from feat.data import Fex, Fextractor
from feat.utils.io import read_openface, get_test_data_path
from nltools.data import Adjacency


def test_info(capsys):
    importantstring = "ThisStringMustBeIncluded"
    fex = Fex(filename=importantstring)
    fex.info
    captured = capsys.readouterr()
    assert importantstring in captured.out


def test_fex(imotions_data):

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
    dat = Fex(df, sampling_freq=30, sessions=sessions)
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
    assert len(dat.upsample(target=60, target_type="hz")) == (len(dat) - 1) * 2

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
    assert dat2.shape[1] * 3 == out.shape[1]
    out = dat2.extract_summary(min=True, max=True, mean=True, ignore_sessions=True)
    assert len(out) == 1
    assert dat2.shape[1] * 3 == out.shape[1]

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


def test_fextractor(imotions_data):
    df = imotions_data
    sessions = np.array([[x] * 10 for x in range(1 + int(len(df) / 10))]).flatten()[:-1]
    dat = Fex(df, sampling_freq=30, sessions=sessions)
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

    # Test Fextractor class
    extractor = Fextractor()
    dat = dat.interpolate()  # interpolate data to get rid of NAs
    f = 0.5
    num_cyc = 3  # for wavelet extraction
    # Test each extraction method
    extractor.mean(fex_object=dat)
    extractor.max(fex_object=dat)
    extractor.min(fex_object=dat)
    # boft needs a groupby function.
    extractor.multi_wavelet(fex_object=dat)
    extractor.wavelet(fex_object=dat, freq=f, num_cyc=num_cyc)
    # Test ValueError
    with pytest.raises(ValueError):
        extractor.wavelet(fex_object=dat, freq=f, num_cyc=num_cyc, mode="BadValue")
    # Test Fextracor merge method
    newdat = extractor.merge(out_format="long")
    assert newdat["sessions"].nunique() == 52
    assert isinstance(newdat, DataFrame)
    assert len(extractor.merge(out_format="long")) == 7488
    assert len(extractor.merge(out_format="wide")) == 52

    # Test summary method
    extractor = Fextractor()
    dat2 = dat.loc[:, ["Positive", "Negative"]].interpolate()
    extractor.summary(fex_object=dat2, min=True, max=True, mean=True)
    assert extractor.merge(out_format="wide").shape[1] == dat2.shape[1] * 3 + 1

    # Test wavelet extraction
    extractor = Fextractor()
    extractor.wavelet(fex_object=dat, freq=f, num_cyc=num_cyc, ignore_sessions=False)
    extractor.wavelet(fex_object=dat, freq=f, num_cyc=num_cyc, ignore_sessions=True)
    wavelet = extractor.extracted_features[0]  # ignore_sessions = False
    assert wavelet.sampling_freq == dat.sampling_freq
    assert len(wavelet) == len(dat)
    wavelet = extractor.extracted_features[1]  # ignore_sessions = True
    assert wavelet.sampling_freq == dat.sampling_freq
    assert len(wavelet) == len(dat)
    assert np.array_equal(wavelet.sessions, dat.sessions)
    for i in ["filtered", "phase", "magnitude", "power"]:
        extractor = Fextractor()
        extractor.wavelet(
            fex_object=dat, freq=f, num_cyc=num_cyc, ignore_sessions=True, mode=i
        )
        wavelet = extractor.extracted_features[0]
        assert wavelet.sampling_freq == dat.sampling_freq
        assert len(wavelet) == len(dat)

    # Test multi wavelet
    dat2 = dat.loc[:, ["Positive", "Negative"]].interpolate()
    n_bank = 4
    extractor = Fextractor()
    extractor.multi_wavelet(
        fex_object=dat2,
        min_freq=0.1,
        max_freq=2,
        bank=n_bank,
        mode="power",
        ignore_sessions=False,
    )
    out = extractor.extracted_features[0]
    assert n_bank * dat2.shape[1] == out.shape[1]
    assert len(out) == len(dat2)
    assert np.array_equal(out.sessions, dat2.sessions)
    assert out.sampling_freq == dat2.sampling_freq

    # Test Bag Of Temporal Features Extraction
    facet = Fex(df, sampling_freq=30, detector="FACET")
    facet_filled = facet.fillna(0)
    facet_filled = facet_filled[
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
    extractor = Fextractor()
    extractor.boft(facet_filled)
    assert isinstance(extractor.extracted_features[0], DataFrame)
    filters, histograms = 8, 12
    assert (
        extractor.extracted_features[0].shape[1]
        == facet_filled.columns.shape[0] * filters * histograms
    )


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

    clf = openface.predict(X=["AU02_c"], y="AU04_c")
    assert clf.coef_ < 0

    clf = openface.predict(X=openface[["AU02_c"]], y=openface["AU04_c"])
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
