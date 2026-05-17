"""Regression / accuracy benchmarking for py-feat detectors.

Compares Detector outputs against held-out labeled benchmarks (DISFA,
AffectNet, ...) and emits a metrics dict suitable for time-series tracking
across py-feat versions. Throughput numbers live in
``scripts/bench_detectors.py``; this module is the accuracy counterpart.

Data root resolution: each dataset loader reads ``$PYFEAT_DATA_ROOT``
(default ``/Storage/Data``). Loaders return ``None`` cleanly when their
dataset directory is absent, so the same script runs on any machine — it
just reports fewer datasets.
"""

from feat.evaluation import datasets, metrics, runner, subsets

__all__ = ["datasets", "metrics", "runner", "subsets"]
