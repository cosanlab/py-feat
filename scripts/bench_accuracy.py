"""Emit py-feat accuracy benchmarks in the live-benchmarks ``--json`` schema.

Wraps ``feat.evaluation.evaluate_all_datasets`` and flattens the per-dataset
metric dicts into ``{schema_version, metadata, records[]}`` — the same shape
``bench_detectors.py --json`` produces — so throughput and accuracy feed one store.

Each record is one metric: ``dataset, metric_kind, metric_name, value, n``.
``metric_kind`` ∈ {au_f1, au_icc, emotion, gaze}; ``metric_name`` is an AU id
(``AU01``), ``mean``, ``accuracy``/``f1_macro``, or a gaze error name.

Datasets are discovered from ``feat.evaluation.datasets`` loaders and require the
data to be present (``PYFEAT_DATA_ROOT``); missing datasets are skipped with a note.
Evaluation uses **validation/held-out** splits (e.g. ``load_affectnet_val``).

Usage:
    PYFEAT_DATA_ROOT=/Storage/Data python scripts/bench_accuracy.py --detector v2 --json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import torch

from feat.version import __version__ as _feat_version


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def _gpu_summary() -> str:
    if torch.cuda.is_available():
        return f"CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}"
    if torch.backends.mps.is_available():
        return "MPS available"
    return "no GPU"


def _run_metadata(tool: str, tool_version: str, device: str) -> dict:
    return {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": tool,
        "tool_version": tool_version,
        "feat_version": _feat_version,
        "git_commit": _git_commit_short(),
        "host": platform.node(),
        "machine": platform.machine(),
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "pytorch": torch.__version__,
        "gpu": _gpu_summary(),
        "device": device,
    }


def flatten_eval(results: dict) -> list[dict]:
    """Flatten ``evaluate_all_datasets`` output → one record per metric.

    ``results`` maps dataset name → the metric dict from ``evaluate_dataset``.
    Returns records ``{dataset, metric_kind, metric_name, value, n}``; run-level
    metadata (tool, host, ...) is added separately and joined at ingest time.
    """
    records: list[dict] = []

    def add(dataset, kind, name, value, n):
        if value is not None:
            records.append(
                {
                    "dataset": dataset,
                    "metric_kind": kind,
                    "metric_name": name,
                    "value": round(float(value), 4),
                    "n": n,
                }
            )

    for dataset, res in results.items():
        n = res.get("n_samples")
        # Action Units: mean + per-AU F1 and ICC
        if "au_f1_per_au" in res:
            add(dataset, "au_f1", "mean", res.get("au_f1_mean"), n)
            add(dataset, "au_icc", "mean", res.get("au_icc_mean"), n)
            for au, v in res["au_f1_per_au"].items():
                add(dataset, "au_f1", au, v, n)
            for au, v in res.get("au_icc_per_au", {}).items():
                add(dataset, "au_icc", au, v, n)
        # Emotion classification
        if "emotion_accuracy" in res:
            n_emo = res.get("n_scored", n)
            add(dataset, "emotion", "accuracy", res.get("emotion_accuracy"), n_emo)
            add(dataset, "emotion", "f1_macro", res.get("emotion_f1_macro"), n_emo)
        # Gaze angular error
        for name in ("pitch_mae_deg", "yaw_mae_deg", "angular_error_deg"):
            if name in res:
                add(dataset, "gaze", name, res[name], n)
    return records


def _build_detector(which: str, device: str):
    if which == "v2":
        from feat.detector_v2 import Detectorv2

        return Detectorv2(device=device, identity_model=None), "py-feat-v2"
    from feat.detector import Detector

    return Detector(device=device), "py-feat-v1"


def _discover_splits():
    """Build the AU/emotion/gaze splits whose data is present; skip the rest."""
    from feat.evaluation import datasets as ds

    loaders = [
        ds.load_disfaplus,
        ds.load_disfa,
        ds.load_affectnet_val,  # validation split (held-out)
        ds.load_columbia_gaze,
    ]
    splits = []
    for fn in loaders:
        try:
            s = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fn.__name__}: {exc.__class__.__name__}", file=sys.stderr)
            continue
        if s is not None:
            splits.append(s)
    return splits


def main() -> None:
    p = argparse.ArgumentParser(description="py-feat accuracy benchmark → --json")
    p.add_argument("--detector", choices=["v1", "v2"], default="v2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--json",
        nargs="?",
        const="__default__",
        default=None,
        help="Write JSON report (default docs/benchmarks/accuracy-<date>-<sha>.json).",
    )
    args = p.parse_args()

    from feat.evaluation.runner import evaluate_all as evaluate_all_datasets

    detector, tool = _build_detector(args.detector, args.device)
    splits = _discover_splits()
    if not splits:
        raise SystemExit(
            "no evaluation datasets found — set PYFEAT_DATA_ROOT to the data dir"
        )
    print(f"evaluating {tool} on: {[s.name for s in splits]}", file=sys.stderr)
    results = evaluate_all_datasets(detector, splits, batch_size=args.batch_size)

    payload = {
        "schema_version": 1,
        "metadata": _run_metadata(tool, _feat_version, args.device),
        "records": flatten_eval(results),
    }
    if args.json is not None:
        if args.json == "__default__":
            today = datetime.date.today().isoformat()
            out = Path("docs/benchmarks") / f"accuracy-{today}-{_git_commit_short()}.json"
        else:
            out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(payload['records'])} records → {out}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
