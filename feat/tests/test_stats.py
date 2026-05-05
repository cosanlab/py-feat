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
    cluster_identities,
)
import torch


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


def test_downsample_default_samples_block_mean():
    """Default target_type='samples': target is rows per output bin
    (matches legacy nltools.stats.downsample default)."""
    df = pd.DataFrame(np.arange(60).reshape(60, 1), columns=["x"])
    out = downsample(df, sampling_freq=30, target=10)
    assert len(out) == 6
    # First output row = mean of input rows [0..9] = 4.5
    assert out.iloc[0]["x"] == 4.5
    # Last output row = mean of [50..59] = 54.5
    assert out.iloc[-1]["x"] == 54.5


def test_downsample_samples_uneven_last_bin():
    """When n_input is not a multiple of bin size, the last bin is
    shorter (matches nltools' ceil-grouping behavior, e.g. 519 -> 52
    output rows for bin size 10)."""
    df = pd.DataFrame(np.arange(519).reshape(519, 1).astype(float), columns=["x"])
    out = downsample(df, sampling_freq=30, target=10)
    assert len(out) == 52
    # Last bin has 9 rows: mean of 510..518 = 514.0
    assert out.iloc[-1]["x"] == 514.0


def test_downsample_target_type_hz():
    """target_type='hz': target is the output Hz."""
    df = pd.DataFrame(np.arange(60).reshape(60, 1), columns=["x"])
    out = downsample(df, sampling_freq=30, target=10, target_type="hz")
    # bin size = round(30/10) = 3 -> 60/3 = 20 output rows
    assert len(out) == 20
    assert out.iloc[0]["x"] == 1.0  # mean of [0,1,2]


def test_downsample_target_type_seconds():
    """target_type='seconds': target is the output bin duration."""
    df = pd.DataFrame(np.arange(60).reshape(60, 1).astype(float), columns=["x"])
    out = downsample(df, sampling_freq=30, target=0.5, target_type="seconds")
    # bin size = round(0.5 * 30) = 15 -> 60/15 = 4 output rows
    assert len(out) == 4


def test_downsample_target_equals_one_returns_copy():
    """target=1 with target_type='samples' is a no-op."""
    df = pd.DataFrame(np.arange(20).reshape(20, 1), columns=["x"])
    out = downsample(df, sampling_freq=30, target=1)
    pd.testing.assert_frame_equal(out, df)
    out.iloc[0, 0] = 999
    assert df.iloc[0, 0] == 0


def test_downsample_target_type_hz_rejects_target_above_source():
    df = pd.DataFrame(np.arange(20).reshape(20, 1), columns=["x"])
    with pytest.raises(ValueError, match="must be <="):
        downsample(df, sampling_freq=10, target=30, target_type="hz")


def test_downsample_method_median():
    """method='median' aggregates with median per bin."""
    df = pd.DataFrame([[0], [10], [100]] * 4 + [[0], [10], [100]],
                      columns=["x"]).astype(float)
    out = downsample(df, sampling_freq=30, target=3, method="median")
    # Each bin = [0, 10, 100] -> median = 10
    assert (out["x"] == 10.0).all()


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


# ----------------------------- cluster_identities ----------------------------


def test_cluster_identities_single_embedding():
    emb = torch.tensor([[1.0, 0.0, 0.0]])
    assert cluster_identities(emb, threshold=0.5) == ["Person_0"]


def test_cluster_identities_two_identical_one_unique():
    """Two identical embeddings cluster together; the third clusters alone."""
    emb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    out = cluster_identities(emb, threshold=0.5)
    # First two share cluster, third is its own.
    assert out[0] == out[1]
    assert out[0] != out[2]


def test_cluster_identities_three_distinct():
    """Orthogonal embeddings each form their own cluster."""
    emb = torch.eye(3)
    out = cluster_identities(emb, threshold=0.5)
    assert len(set(out)) == 3


def test_cluster_identities_transitive_chain():
    """A-B and B-C similarity above threshold; A-C below. All three should
    still cluster together via transitivity (B bridges them)."""
    # cos(A,B) = cos(B,C) ~ 0.93; cos(A,C) ~ 0.74
    emb = torch.tensor(
        [
            [1.0, 0.0, 0.0],   # A
            [0.7, 0.7, 0.0],   # B (close to A and C)
            [0.0, 1.0, 0.0],   # C
        ]
    )
    # Threshold 0.8: A-B (0.7) below, B-C (0.7) below — actually all separate.
    # Use a lower threshold to test transitivity.
    out2 = cluster_identities(emb, threshold=0.6)
    # cos(A,B) = 0.7, cos(B,C) = 0.7, cos(A,C) = 0
    # At 0.6: A connects to B, B connects to C. A-C transitively.
    assert out2[0] == out2[1] == out2[2]


def test_cluster_identities_format():
    emb = torch.eye(2)
    out = cluster_identities(emb, threshold=0.5)
    assert all(p.startswith("Person_") for p in out)
    assert all(p.split("_")[1].isdigit() for p in out)


def test_cluster_identities_large_input_completes():
    """Regression: prior implementation was O(N^3) due to a Python list
    comprehension on every BFS pop. 200 embeddings finishes in well under
    a second on the new path."""
    import time
    torch.manual_seed(0)
    emb = torch.randn(200, 64)
    emb = torch.nn.functional.normalize(emb, dim=1)
    t = time.perf_counter()
    out = cluster_identities(emb, threshold=0.3)
    elapsed = time.perf_counter() - t
    assert len(out) == 200
    assert elapsed < 1.0, f"cluster_identities took {elapsed:.2f}s on 200 embeddings"
