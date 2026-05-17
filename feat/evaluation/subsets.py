"""Frozen subset generation + loading helpers.

The first time a benchmark is run, the per-dataset loaders downsample
deterministically with seed=42. To guarantee accuracy numbers stay
comparable across releases (and across machines that may add/remove
images), call :func:`freeze` once to pin the chosen IDs into
``feat/evaluation/subsets/<dataset>.json``. After that, the loaders
read from the frozen list instead of resampling.

Each JSON has shape:

    {"dataset": "<name>", "seed": 42, "ids": ["<id1>", "<id2>", ...]}

ID formats:
- DISFA: "<subject>/<frame_int>"  e.g. "SN005/2085"
- AffectNet: "<subDirectory_filePath>"  e.g. "459/abc...jpg"
"""
from __future__ import annotations

import json
from pathlib import Path

from feat.evaluation.datasets import (
    DatasetSplit,
    load_affectnet_val,
    load_disfa,
    _subset_json_path,
)


def _ids_for(dataset: DatasetSplit) -> list[str]:
    if dataset.name.startswith("disfa"):
        return [
            f"{row.subject}/{int(row.frame)}"
            for row in dataset.labels.itertuples()
        ]
    if dataset.name == "affectnet_val":
        return dataset.labels["subDirectory_filePath"].astype(str).tolist()
    raise ValueError(f"No id-extraction rule for {dataset.name!r}")


def freeze(dataset_name: str, **loader_kwargs) -> Path:
    """Generate (or regenerate) a frozen subset JSON. Returns the file path."""
    if dataset_name.startswith("disfa"):
        split = "P3"
        if "_" in dataset_name and dataset_name.split("_")[-1].upper().startswith("P"):
            split = dataset_name.split("_")[-1].upper()
        ds = load_disfa(split=split, **loader_kwargs)
    elif dataset_name == "affectnet_val":
        ds = load_affectnet_val(**loader_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")
    if ds is None:
        raise RuntimeError(
            f"Dataset {dataset_name!r} not present at $PYFEAT_DATA_ROOT — cannot freeze"
        )
    path = _subset_json_path(dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "seed": loader_kwargs.get("seed", 42),
        "n": len(ds),
        "ids": _ids_for(ds),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
