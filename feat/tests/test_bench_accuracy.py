"""Tests for scripts/bench_accuracy.py — the evaluate_all_datasets → records flatten.

Loads the script by path and exercises ``flatten_eval`` against a synthetic
``evaluate_all_datasets`` output (no datasets / detector needed).
"""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench_accuracy.py"


@pytest.fixture(scope="module")
def acc():
    spec = importlib.util.spec_from_file_location("bench_accuracy", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample_results():
    return {
        "DISFAPlus": {
            "n_samples": 1000,
            "au_f1_mean": 0.61,
            "au_icc_mean": 0.55,
            "au_f1_per_au": {"AU01": 0.5, "AU12": 0.8},
            "au_icc_per_au": {"AU01": 0.4, "AU12": 0.7},
            "n_aus_evaluated": 2,
        },
        "AffectNet-val": {
            "n_samples": 500,
            "n_scored": 480,
            "emotion_accuracy": 0.62,
            "emotion_f1_macro": 0.58,
        },
        "Columbia": {
            "n_samples": 200,
            "pitch_mae_deg": 5.1,
            "yaw_mae_deg": 6.3,
        },
    }


def test_flatten_au_records(acc):
    recs = acc.flatten_eval(_sample_results())
    au_f1 = [r for r in recs if r["dataset"] == "DISFAPlus" and r["metric_kind"] == "au_f1"]
    names = {r["metric_name"] for r in au_f1}
    assert names == {"mean", "AU01", "AU12"}
    mean = next(r for r in au_f1 if r["metric_name"] == "mean")
    assert mean["value"] == 0.61 and mean["n"] == 1000
    # ICC records present too
    assert any(r["metric_kind"] == "au_icc" and r["metric_name"] == "AU12" for r in recs)


def test_flatten_emotion_uses_n_scored(acc):
    recs = acc.flatten_eval(_sample_results())
    emo = [r for r in recs if r["metric_kind"] == "emotion"]
    assert {r["metric_name"] for r in emo} == {"accuracy", "f1_macro"}
    assert all(r["n"] == 480 for r in emo)  # n_scored, not n_samples


def test_flatten_gaze_records(acc):
    recs = acc.flatten_eval(_sample_results())
    gaze = {r["metric_name"]: r["value"] for r in recs if r["metric_kind"] == "gaze"}
    assert gaze == {"pitch_mae_deg": 5.1, "yaw_mae_deg": 6.3}


def test_flatten_skips_none_values(acc):
    res = {"D": {"n_samples": 10, "au_f1_per_au": {"AU01": 0.5},
                 "au_f1_mean": None, "au_icc_mean": 0.3, "au_icc_per_au": {}}}
    recs = acc.flatten_eval(res)
    # au_f1_mean is None → dropped; per-AU and icc_mean kept
    assert not any(r["metric_kind"] == "au_f1" and r["metric_name"] == "mean" for r in recs)
    assert any(r["metric_kind"] == "au_f1" and r["metric_name"] == "AU01" for r in recs)
    assert any(r["metric_kind"] == "au_icc" and r["metric_name"] == "mean" for r in recs)


def test_records_have_required_columns(acc):
    recs = acc.flatten_eval(_sample_results())
    assert recs and all(
        set(r) == {"dataset", "metric_kind", "metric_name", "value", "n"} for r in recs
    )
