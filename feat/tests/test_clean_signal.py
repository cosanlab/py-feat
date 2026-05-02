"""Tests for the in-house clean_signal helper that replaced nilearn.signal.clean."""

import numpy as np
import pytest

from feat.utils.stats import clean_signal


def test_detrend_removes_linear_trend():
    t = np.arange(100)
    # Pure linear trend, no noise
    x = (2.0 * t + 5.0).reshape(-1, 1).astype(np.float64)
    out = clean_signal(x, detrend=True, standardize=False)
    # After linear detrend the residuals should be ~0
    assert np.abs(out).max() < 1e-9


def test_standardize_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=10.0, scale=3.0, size=(200, 4))
    out = clean_signal(x, detrend=False, standardize=True)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-9)
    # ddof=1 sample std
    np.testing.assert_allclose(out.std(axis=0, ddof=1), 1.0, atol=1e-9)


def test_confound_regression_removes_confound_variance():
    rng = np.random.default_rng(1)
    n = 500
    confound = rng.normal(size=n)
    # signal = 3 * confound + noise
    noise = rng.normal(size=n)
    signal = 3.0 * confound + noise
    out = clean_signal(
        signal[:, None],
        confounds=confound[:, None],
        detrend=False,
        standardize=False,
    )
    # After regressing out confound, correlation with confound should be ~0
    correlation = np.corrcoef(out[:, 0], confound)[0, 1]
    assert abs(correlation) < 1e-10


def test_low_pass_filter_attenuates_high_frequency():
    t = np.linspace(0, 1, 1000, endpoint=False)
    fs = 1000.0
    # 5 Hz (low) + 200 Hz (high) signal
    x = (np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 200 * t))[:, None]
    out = clean_signal(
        x,
        detrend=False,
        standardize=False,
        low_pass=20.0,  # cut off the 200 Hz component
        sampling_freq=fs,
    )
    # After low-pass at 20 Hz, the 200 Hz component should be heavily
    # attenuated. Compare power: input power ~= 1.0 (each sine);
    # output power from the 200 Hz sine should be << 1e-2.
    fft = np.fft.rfft(out[:, 0])
    freqs = np.fft.rfftfreq(out.shape[0], d=1 / fs)
    high_band = (freqs > 100) & (freqs < 300)
    assert np.abs(fft[high_band]).max() < 5  # << 200 Hz input amplitude (~500)


def test_high_pass_filter_attenuates_low_frequency():
    t = np.linspace(0, 1, 1000, endpoint=False)
    fs = 1000.0
    x = (np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 200 * t))[:, None]
    out = clean_signal(
        x,
        detrend=False,
        standardize=False,
        high_pass=50.0,
        sampling_freq=fs,
    )
    fft = np.fft.rfft(out[:, 0])
    freqs = np.fft.rfftfreq(out.shape[0], d=1 / fs)
    low_band = (freqs > 0) & (freqs < 20)
    assert np.abs(fft[low_band]).max() < 5  # 5 Hz attenuated


def test_ensure_finite_replaces_nan_inf():
    x = np.array([[1.0, np.nan], [np.inf, -np.inf], [3.0, 4.0]])
    out = clean_signal(
        x, detrend=False, standardize=False, ensure_finite=True
    )
    assert np.isfinite(out).all()


def test_runs_apply_per_session():
    """Each unique label in `runs` should be cleaned independently."""
    rng = np.random.default_rng(2)
    n_per = 100
    a = rng.normal(loc=0, scale=1, size=n_per)
    b = rng.normal(loc=10, scale=2, size=n_per)
    x = np.concatenate([a, b])[:, None]
    runs = np.array(["A"] * n_per + ["B"] * n_per)
    out = clean_signal(x, detrend=False, standardize=True, runs=runs)
    # Each run's standardized output should be ~zero mean unit variance
    assert abs(out[runs == "A"].mean()) < 1e-9
    assert abs(out[runs == "B"].mean()) < 1e-9
    np.testing.assert_allclose(out[runs == "A"].std(ddof=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(out[runs == "B"].std(ddof=1), 1.0, atol=1e-9)


def test_constant_column_doesnt_divide_by_zero():
    """A column with zero variance should pass through standardize without NaN."""
    x = np.column_stack([np.full(100, 5.0), np.arange(100, dtype=float)])
    out = clean_signal(x, detrend=False, standardize=True)
    assert np.isfinite(out).all()


def test_one_d_input_returns_one_d():
    rng = np.random.default_rng(3)
    x = rng.normal(size=100)
    out = clean_signal(x, detrend=True, standardize=True)
    assert out.shape == (100,)


def test_confound_length_mismatch_raises():
    x = np.ones((10, 1))
    confounds = np.ones((5, 1))
    with pytest.raises(ValueError, match="confounds length"):
        clean_signal(x, confounds=confounds)


def test_runs_length_mismatch_raises():
    x = np.ones((10, 1))
    runs = np.array([0, 1])
    with pytest.raises(ValueError, match="runs length"):
        clean_signal(x, runs=runs)


def test_matches_nilearn_when_available():
    """End-to-end equivalence with nilearn.signal.clean. Skipped if
    nilearn is not installed."""
    nilearn_signal = pytest.importorskip("nilearn.signal")

    rng = np.random.default_rng(42)
    n, n_signals, n_conf = 200, 3, 2
    fs = 30.0
    x = rng.normal(size=(n, n_signals))
    confounds = rng.normal(size=(n, n_conf))

    ours = clean_signal(
        x,
        detrend=True,
        standardize=True,
        confounds=confounds,
        low_pass=5.0,
        high_pass=0.1,
        sampling_freq=fs,
    )
    theirs = nilearn_signal.clean(
        x,
        detrend=True,
        standardize="zscore_sample",  # nilearn's match for our ddof=1 std
        confounds=confounds,
        low_pass=5.0,
        high_pass=0.1,
        t_r=1.0 / fs,
        standardize_confounds=False,  # we don't standardize confounds
    )
    # Loose tolerance: nilearn uses pivoted-QR with rank-tolerance for
    # confound regression while we use plain lstsq; some divergence is
    # expected for non-orthogonal confounds. The point of the test is to
    # catch regression in operation order, not bit-exact equivalence.
    np.testing.assert_allclose(ours, theirs, atol=0.3, rtol=0.3)
