"""Unit tests for feat.evaluation.metrics."""
import numpy as np
import pytest

from feat.evaluation import metrics


def test_binarize_au_truth_threshold():
    intensities = np.array([0, 1, 2, 3, 4, 5])
    assert metrics.binarize_au_truth(intensities, threshold=2).tolist() == [0, 0, 1, 1, 1, 1]
    assert metrics.binarize_au_truth(intensities, threshold=3).tolist() == [0, 0, 0, 1, 1, 1]


def test_au_f1_binary_perfect():
    y = np.array([0, 0, 1, 1, 0, 1])
    assert metrics.au_f1_binary(y, y) == 1.0


def test_au_f1_binary_all_zero_returns_zero_not_nan():
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    assert metrics.au_f1_binary(y_true, y_pred) == 0.0


def test_au_icc_identical_inputs_is_one():
    y = np.array([0.1, 0.4, 0.9, 0.5, 0.7])
    assert metrics.au_icc(y, y) == pytest.approx(1.0, abs=1e-6)


def test_au_icc_anti_correlated_below_zero():
    y_true = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_pred = y_true[::-1]
    assert metrics.au_icc(y_true, y_pred) < 0.0


def test_ccc_identical_is_one():
    y = np.array([-0.5, -0.1, 0.0, 0.2, 0.4, 0.9])
    assert metrics.concordance_correlation_coefficient(y, y) == pytest.approx(1.0, abs=1e-6)


def test_ccc_constant_shift_penalized():
    """Lin's CCC penalizes constant shifts; Pearson would not."""
    y_true = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    y_pred = y_true + 0.5
    assert metrics.concordance_correlation_coefficient(y_true, y_pred) < 0.7


def test_cosine_similarity_pairs_orthogonal():
    a = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sim = metrics.cosine_similarity_pairs(a, b)
    assert sim[0] == pytest.approx(1.0)
    assert sim[1] == pytest.approx(0.0)


def test_verification_accuracy_lfw_10fold_perfectly_separable():
    """When same-pair sims are always above neg-pair sims, expect 100%."""
    rng = np.random.default_rng(0)
    sims_pos = rng.uniform(0.7, 0.9, 1000)
    sims_neg = rng.uniform(0.1, 0.3, 1000)
    sim = np.concatenate([sims_pos, sims_neg])
    label = np.array([1] * 1000 + [0] * 1000)
    fold = np.concatenate(
        [np.repeat(np.arange(10), 100), np.repeat(np.arange(10), 100)]
    )
    r = metrics.verification_accuracy_lfw_10fold(sim, label, fold)
    assert r["accuracy_mean"] == pytest.approx(1.0)
    assert r["auc"] == pytest.approx(1.0)
    assert r["n_folds"] == 10


def test_rank_k_identification_known_matches():
    """10 probes have exact gallery matches; other 10 do not."""
    rng = np.random.default_rng(1)
    probe = rng.standard_normal((20, 32)).astype(np.float32)
    gallery = rng.standard_normal((100, 32)).astype(np.float32)
    gallery[:10] = probe[:10] + 1e-3 * rng.standard_normal((10, 32))
    probe_ids = np.arange(20)
    gallery_ids = np.concatenate([np.arange(10), 100 + np.arange(90)])
    r = metrics.rank_k_identification(
        probe, probe_ids, gallery, gallery_ids, ks=(1, 5)
    )
    # 10/20 probes have their identity present in gallery.
    assert r["rank_1"] == pytest.approx(0.5)
    assert r["rank_5"] == pytest.approx(0.5)
    assert r["n_probes"] == 20


def test_summarize_au_metrics_drops_nan():
    truth = {
        "AU01": np.array([0, 0, 2, 3]),
        "AU02": np.array([np.nan, 1, 2, 3]),
    }
    pred = {
        "AU01": np.array([0.1, 0.2, 0.9, 0.8]),
        "AU02": np.array([0.1, 0.6, 0.7, 0.9]),
    }
    summary = metrics.summarize_au_metrics(truth, pred)
    assert summary["n_aus"] == 2
    # AU02 should have only 3 effective samples after dropping NaN.
    assert summary["per_au"]["AU02"]["n"] == 3
