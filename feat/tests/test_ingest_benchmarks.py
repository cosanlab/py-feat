"""Tests for scripts/ingest_benchmarks.py — the --json → CSV flatten/append.

Loads the script by path (it's not a package module) and exercises the pure
flatten/append functions against a sample matching bench_detectors.py --json.
No network / HF access.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ingest_benchmarks.py"


@pytest.fixture(scope="module")
def ingest():
    spec = importlib.util.spec_from_file_location("ingest_benchmarks", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample_payload():
    return {
        "schema_version": 1,
        "metadata": {
            "date": "2026-06-05T10:00:00-04:00",
            "feat_version": "0.7.0",
            "git_commit": "abc1234",
            "host": "liquidswords2",
            "machine": "x86_64",
            "cpu_count": 32,
            "python": "3.12.3",
            "pytorch": "2.11.0+cu128",
            "gpu": "CUDA 12.8, NVIDIA RTX",
            "omp_num_threads": "1",
            # list-valued sweep descriptors — must be dropped from per-row output
            "devices_swept": ["cuda"],
            "batch_sizes": [1, 16],
            "workers": [0],
        },
        "records": [
            {
                "section_kind": "video", "section_label": "short", "config": "Detectorv2 multitask",
                "device": "cuda", "batch": 1, "workers": 0, "sec": 2.0,
                "n_rows": 72, "n_units": 72, "per_unit_ms": 27.78, "fps": 36.0,
            },
            {
                "section_kind": "video", "section_label": "short", "config": "Detectorv2 multitask",
                "device": "cuda", "batch": 16, "workers": 0, "sec": 1.0,
                "n_rows": 72, "n_units": 72, "per_unit_ms": 13.89, "fps": 72.0,
            },
        ],
    }


def test_flatten_run_one_row_per_record_with_metadata(ingest):
    df = ingest.flatten_run(_sample_payload())
    assert len(df) == 2
    # scalar metadata is repeated on every row
    assert (df["git_commit"] == "abc1234").all()
    assert (df["host"] == "liquidswords2").all()
    assert (df["feat_version"] == "0.7.0").all()
    # per-record measurement columns present
    assert set(["config", "device", "batch", "fps", "per_unit_ms"]).issubset(df.columns)
    assert sorted(df["fps"].tolist()) == [36.0, 72.0]


def test_flatten_run_drops_list_valued_metadata(ingest):
    df = ingest.flatten_run(_sample_payload())
    for col in ("devices_swept", "batch_sizes", "workers_swept"):
        assert col not in df.columns
    # NB: per-record 'workers' (scalar) is kept; the list 'workers' metadata is dropped
    assert "workers" in df.columns and set(df["workers"]) == {0}


def test_flatten_run_empty_records(ingest):
    payload = _sample_payload()
    payload["records"] = []
    assert ingest.flatten_run(payload).empty


def test_append_rows_to_empty_and_existing(ingest):
    df = ingest.flatten_run(_sample_payload())
    assert len(ingest.append_rows(df, None)) == 2
    assert ingest.append_rows(df, pd.DataFrame()).shape[0] == 2
    combined = ingest.append_rows(df, df)
    assert len(combined) == 4  # appends, doesn't dedupe


def test_local_ingest_roundtrip(ingest, tmp_path):
    """--local path: write a run, then append a second run, CSV grows."""
    import json

    run = tmp_path / "run.json"
    run.write_text(json.dumps(_sample_payload()))
    csv = tmp_path / "throughput.csv"

    df1 = ingest.flatten_run(json.loads(run.read_text()))
    ingest.append_rows(df1, ingest._read_local(csv)).to_csv(csv, index=False)
    assert pd.read_csv(csv).shape[0] == 2

    ingest.append_rows(df1, ingest._read_local(csv)).to_csv(csv, index=False)
    assert pd.read_csv(csv).shape[0] == 4
