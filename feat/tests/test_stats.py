"""Numerical equivalence tests for the in-house regress / downsample /
upsample / set_decomposition_algorithm helpers (drop-in replacements for
the nltools functions formerly used in py-feat).

The regress test compares against statsmodels.OLS - an independent and
well-tested reference. Other helpers are tested for shape and edge cases
since their behavior is straightforward.
"""

import numpy as np
import pandas as pd
import pytest

from feat.utils.stats import (
    regress,
    downsample,
    upsample,
    set_decomposition_algorithm,
)


# ----------------------------- regress vs statsmodels ----------------------------


def test_regress_matches_statsmodels_single_y():
    """In-house OLS must produce identical betas, SEs, t, p, residuals
    to statsmodels.OLS on a single response variable."""
    sm = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(0)
    n, p = 200, 4
    X = np.hstack([rng.normal(size=(n, p - 1)), np.ones((n, 1))])  # incl intercept
    y = X @ rng.normal(size=p) + rng.normal(size=n) * 0.5

    b, se, t, pv, df, res = regress(X, y, mode="ols")

    sm_model = sm.OLS(y, X).fit()
    np.testing.assert_allclose(b, sm_model.params, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(se, sm_model.bse, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(t, sm_model.tvalues, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(pv, sm_model.pvalues, atol=1e-9, rtol=1e-9)
    assert df == sm_model.df_resid
    np.testing.assert_allclose(res, sm_model.resid, atol=1e-9, rtol=1e-9)


def test_regress_matches_statsmodels_multi_y():
    """Multi-output OLS: betas/SEs/t/p must match statsmodels per column."""
    sm = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(1)
    n, p, k = 150, 3, 4
    X = np.hstack([rng.normal(size=(n, p - 1)), np.ones((n, 1))])
    Y = X @ rng.normal(size=(p, k)) + rng.normal(size=(n, k)) * 0.5

    b, se, t, pv, df, res = regress(X, Y, mode="ols")

    for col in range(k):
        sm_model = sm.OLS(Y[:, col], X).fit()
        np.testing.assert_allclose(b[:, col], sm_model.params, atol=1e-9)
        np.testing.assert_allclose(se[:, col], sm_model.bse, atol=1e-9)
        np.testing.assert_allclose(t[:, col], sm_model.tvalues, atol=1e-9)
        np.testing.assert_allclose(pv[:, col], sm_model.pvalues, atol=1e-9)
        np.testing.assert_allclose(res[:, col], sm_model.resid, atol=1e-9)


def test_regress_pvalue_precision_for_moderately_large_t():
    """The new `t.sf()` recipe must not collapse to 0 for |t| ~ 10
    where `1 - t.cdf()` already underflows to exactly 0. Catches the
    prior precision-loss bug."""
    rng = np.random.default_rng(2)
    n = 200
    # Tune SNR so |t| ends up around 10-15.
    X = np.column_stack([rng.normal(size=n), np.ones(n)])
    y = X @ np.array([1.0, 0.0]) + rng.normal(size=n) * 0.5

    _, _, t, pv, _, _ = regress(X, y)
    assert np.all(np.isfinite(pv))
    assert np.all(pv >= 0)
    # With sf(), p-value for slope (|t|~15-20) should be ~1e-30, not 0.
    # The 1-cdf() recipe underflows above |t|=8 for double precision.
    slope_p = pv[0]
    assert slope_p > 0, "p-value collapsed to exactly 0 (precision-loss bug)"
    # Sanity: slope p-value should be tiny (highly significant).
    assert slope_p < 1e-20


# ----------------------------- regress error paths -------------------------------


def test_regress_unsupported_mode_raises():
    with pytest.raises(NotImplementedError, match="mode='robust'"):
        regress(np.ones((5, 2)), np.ones(5), mode="robust")


def test_regress_unexpected_kwargs_raises():
    with pytest.raises(TypeError, match="unexpected keyword"):
        regress(np.ones((5, 2)), np.ones(5), bogus_kwarg=42)


# ----------------------------- downsample ---------------------------------------


def test_downsample_dataframe_block_mean():
    """Block-mean over groups of `factor` rows."""
    df = pd.DataFrame(np.arange(60).reshape(60, 1), columns=["x"])
    out = downsample(df, sampling_freq=30, target=10)
    assert len(out) == 20
    # First output row should be mean of input rows [0, 1, 2] = 1.0
    assert out.iloc[0]["x"] == 1.0
    # Last output row should be mean of [57, 58, 59] = 58.0
    assert out.iloc[-1]["x"] == 58.0


def test_downsample_target_equals_source_returns_copy():
    df = pd.DataFrame(np.arange(20).reshape(20, 1), columns=["x"])
    out = downsample(df, sampling_freq=30, target=30)
    pd.testing.assert_frame_equal(out, df)
    # Mutating the output must not mutate input.
    out.iloc[0, 0] = 999
    assert df.iloc[0, 0] == 0


def test_downsample_rejects_target_above_source():
    df = pd.DataFrame(np.arange(20).reshape(20, 1), columns=["x"])
    with pytest.raises(ValueError, match="must be <="):
        downsample(df, sampling_freq=10, target=30)


# ----------------------------- upsample -----------------------------------------


def test_upsample_doubles_length_at_2x_target():
    df = pd.DataFrame(np.arange(50).reshape(50, 1).astype(float), columns=["x"])
    out = upsample(df, sampling_freq=30, target=60)
    assert len(out) == 100


def test_upsample_target_type_samples():
    df = pd.DataFrame(np.arange(50).reshape(50, 1).astype(float), columns=["x"])
    out = upsample(df, sampling_freq=30, target=200, target_type="samples")
    assert len(out) == 200


def test_upsample_invalid_target_type_raises():
    df = pd.DataFrame(np.arange(50).reshape(50, 1).astype(float), columns=["x"])
    with pytest.raises(ValueError, match="target_type"):
        upsample(df, sampling_freq=30, target=60, target_type="bogus")


# ----------------------------- set_decomposition_algorithm ----------------------


def test_decomposition_factory_returns_correct_classes():
    pca = set_decomposition_algorithm("pca", n_components=2)
    ica = set_decomposition_algorithm("ica", n_components=2)
    nmf = set_decomposition_algorithm("nnmf", n_components=2)
    fa = set_decomposition_algorithm("fa", n_components=2)
    assert type(pca).__name__ == "PCA"
    assert type(ica).__name__ == "FastICA"
    assert type(nmf).__name__ == "NMF"
    assert type(fa).__name__ == "FactorAnalysis"


def test_decomposition_factory_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        set_decomposition_algorithm("rsvd", n_components=2)
