"""Flatten a ``bench_detectors.py --json`` run and append it to the benchmark
results store (the ``py-feat/benchmarks`` HF Dataset, or a local CSV for testing).

The ``--json`` emitter writes ``{schema_version, metadata, records[]}``. This
flattens it to one CSV row per measurement, repeating the run-level metadata on
each row so the benchmark dashboard can plot without a join. Scalar metadata only
(the list-valued sweep descriptors — ``devices_swept`` etc. — are dropped, since
they describe the run, not a single measurement).

Usage:
    # Append a throughput run to the HF Dataset (needs HF_TOKEN with write access)
    python scripts/ingest_benchmarks.py run.json --repo py-feat/benchmarks --file throughput.csv

    # Dry-run against a local CSV (no network) — used by tests and for inspection
    python scripts/ingest_benchmarks.py run.json --local /tmp/throughput.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_run(payload: dict) -> pd.DataFrame:
    """Flatten one ``--json`` run into a per-measurement DataFrame.

    Each ``records[]`` entry becomes a row carrying that run's scalar metadata
    (date, version, git_commit, host, gpu, ...). List/dict metadata are dropped.
    Returns an empty DataFrame when the run has no records.
    """
    meta = {
        k: v
        for k, v in dict(payload.get("metadata", {})).items()
        if not isinstance(v, (list, dict))
    }
    records = payload.get("records", [])
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([{**meta, **rec} for rec in records])


def append_rows(new: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    """Append ``new`` rows to ``existing`` (None/empty → just ``new``)."""
    if existing is None or existing.empty:
        return new.reset_index(drop=True)
    return pd.concat([existing, new], ignore_index=True)


def _read_local(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _read_hf(repo: str, fname: str) -> pd.DataFrame | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        return pd.read_csv(hf_hub_download(repo, fname, repo_type="dataset"))
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None  # first write to a fresh dataset / new file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json_file", help="A run.json produced by bench_detectors.py --json")
    p.add_argument("--repo", default="py-feat/benchmarks", help="HF Dataset repo id")
    p.add_argument("--file", default="throughput.csv", help="CSV name within the dataset")
    p.add_argument(
        "--local",
        default=None,
        help="Append to this local CSV instead of HF (no network; for testing).",
    )
    args = p.parse_args()

    payload = json.loads(Path(args.json_file).read_text())
    new = flatten_run(payload)
    if new.empty:
        raise SystemExit(f"{args.json_file}: no records to ingest")

    if args.local:
        path = Path(args.local)
        merged = append_rows(new, _read_local(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)
        print(f"appended {len(new)} rows → {path} ({len(merged)} total)")
        return

    import tempfile

    from huggingface_hub import upload_file

    merged = append_rows(new, _read_hf(args.repo, args.file))
    sha = payload.get("metadata", {}).get("git_commit", "?")
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        merged.to_csv(fh.name, index=False)
        tmp = fh.name
    upload_file(
        path_or_fileobj=tmp,
        path_in_repo=args.file,
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"benchmarks: append {len(new)} rows (git {sha})",
    )
    print(f"appended {len(new)} rows → {args.repo}/{args.file} ({len(merged)} total)")


if __name__ == "__main__":
    main()
