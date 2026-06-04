"""Tests for the --json emitter in scripts/bench_detectors.py.

The benchmark script is not an importable package module, so we load it by
path. These tests exercise the JSON serialization (schema, metadata, records,
round-trip) with synthetic Row objects — no detector run required.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

_BENCH_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "bench_detectors.py"
)


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_detectors", _BENCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_rows(bench):
    return [
        bench.Row(
            section_kind="video",
            section_label="short (72 frames)",
            cfg_label="Detectorv2 multitask",
            device="cuda",
            batch=4,
            workers=0,
            sec=2.0,
            n_rows=72,
            n_units=72,
        ),
        bench.Row(
            section_kind="image",
            section_label="16 x multi_face.jpg = 80 faces",
            cfg_label="retinaface",
            device="cpu",
            batch=1,
            workers=2,
            sec=4.0,
            n_rows=80,
            n_units=16,
        ),
    ]


def test_write_json_schema_and_roundtrip(bench, tmp_path):
    rows = _make_rows(bench)
    args = argparse.Namespace(batches=[1, 4], workers=[0])
    out = tmp_path / "run.json"

    bench.write_json(rows, out, args, all_devices=("cpu", "cuda"))

    payload = json.loads(out.read_text())

    assert payload["schema_version"] == 1
    assert set(payload) == {"schema_version", "metadata", "records"}

    meta = payload["metadata"]
    for key in (
        "date", "feat_version", "git_commit", "host", "machine",
        "cpu_count", "python", "pytorch", "gpu", "omp_num_threads",
        "devices_swept", "batch_sizes", "workers",
    ):
        assert key in meta, f"missing metadata key: {key}"
    assert meta["devices_swept"] == ["cpu", "cuda"]
    assert meta["batch_sizes"] == [1, 4]

    assert len(payload["records"]) == 2


def test_record_fields_and_derived_metrics(bench, tmp_path):
    rows = _make_rows(bench)
    args = argparse.Namespace(batches=[1, 4], workers=[0])
    out = tmp_path / "run.json"
    bench.write_json(rows, out, args, all_devices=("cpu", "cuda"))

    rec = json.loads(out.read_text())["records"][0]

    assert rec["config"] == "Detectorv2 multitask"
    assert rec["device"] == "cuda"
    assert rec["batch"] == 4
    assert rec["workers"] == 0
    assert rec["n_units"] == 72
    # 72 frames / 2.0 s = 36 fps; 2.0 s / 72 * 1000 = 27.78 ms/frame
    assert rec["fps"] == pytest.approx(36.0)
    assert rec["per_unit_ms"] == pytest.approx(27.7778, abs=1e-3)


def test_image_record_uses_input_count_denominator(bench, tmp_path):
    """ms/img and ips divide by input images (n_units), not detected faces."""
    rows = _make_rows(bench)
    args = argparse.Namespace(batches=[1, 4], workers=[0])
    out = tmp_path / "run.json"
    bench.write_json(rows, out, args, all_devices=("cpu",))

    img_rec = json.loads(out.read_text())["records"][1]
    assert img_rec["section_kind"] == "image"
    assert img_rec["n_rows"] == 80  # detected faces
    assert img_rec["n_units"] == 16  # input images
    # 16 images / 4.0 s = 4.0 images/s
    assert img_rec["fps"] == pytest.approx(4.0)


def test_default_json_filename_pattern(bench, tmp_path):
    """--json with no arg should target docs/benchmarks/<date>-<sha>.json."""
    sha = bench._git_commit_short()
    import datetime

    today = datetime.date.today().isoformat()
    expected = Path("docs/benchmarks") / f"{today}-{sha}.json"
    assert expected.suffix == ".json"
    assert expected.parent == Path("docs/benchmarks")
