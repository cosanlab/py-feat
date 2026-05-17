"""Accuracy / regression benchmark for py-feat detectors.

Counterpart to ``scripts/bench_detectors.py`` (which measures throughput).
Runs the Detector + ArcFace stack against held-out labeled datasets and
emits per-metric numbers suitable for time-series tracking across
releases.

Datasets auto-discovered from ``PYFEAT_DATA_ROOT`` (default
``/Storage/Data``) and ``PYFEAT_IDENTITY_ROOT`` (default
``/Storage/IdentityDatasets``):

    AU / intensity:        disfa
    Emotion class + V/A:   affectnet_val
    Identity 1:1:          calfw, cplfw
    Identity 1:N:          tinyface

Usage:

    # Default: run every dataset that's present, write markdown + JSON
    python scripts/bench_regression.py --markdown

    # Subset selection
    python scripts/bench_regression.py --datasets disfa,affectnet_val --markdown

    # Custom Detector config (else uses v0.7 defaults)
    python scripts/bench_regression.py --au-model xgb --emotion-model resmasknet

    # Skip the huge TinyFace distractor gallery (closed-set rank-K only)
    python scripts/bench_regression.py --tinyface-no-distractors

The script writes:
- ``docs/benchmarks/<YYYY-MM-DD>-<sha>-accuracy.md`` — per-run markdown
- ``docs/benchmarks/accuracy.md`` — rolling index page (jupyter-book)
- ``bench-results/<YYYY-MM-DD>-<sha>.json`` — machine-readable record
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

import torch  # noqa: E402

from feat.evaluation import datasets, runner  # noqa: E402
from feat.version import __version__ as _feat_version  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_BENCH_DIR = REPO_ROOT / "docs" / "benchmarks"
RESULTS_DIR = REPO_ROOT / "bench-results"

AU_EMOTION_DATASETS = {"disfa", "disfaplus", "affectnet_val"}
IDENTITY_DATASETS = {"calfw", "cplfw", "tinyface"}
GAZE_DATASETS = {"columbia_gaze"}
ALL_DATASETS = AU_EMOTION_DATASETS | IDENTITY_DATASETS | GAZE_DATASETS


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT)
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "Apple Silicon (MPS)"
    return "CPU"


def _device_default() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _meta(args) -> dict:
    return {
        "date": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "py_feat_version": _feat_version,
        "git_sha": _git_sha(),
        "host": platform.node(),
        "gpu": _gpu_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": args.device,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
        "args": {
            "datasets": args.datasets,
            "disfa_subset": args.disfa_subset,
            "affectnet_subset": args.affectnet_subset,
            "tinyface_no_distractors": args.tinyface_no_distractors,
            "columbia_subset": args.columbia_subset,
            "columbia_head_pose": args.columbia_head_pose,
            "face_model": args.face_model,
            "landmark_model": args.landmark_model,
            "au_model": args.au_model,
            "emotion_model": args.emotion_model,
            "facepose_model": args.facepose_model,
            "identity_model": args.identity_model,
            "gaze_model": args.gaze_model,
        },
    }


def _build_detector(args):
    """Build a Detector.

    Note: facepose model is implied by ``face_model`` (img2pose regresses
    pose natively; retinaface uses PnP-DLT from landmarks). Detector
    doesn't take a separate ``facepose_model`` kwarg, but we keep one in
    the metadata for clarity.
    """
    from feat.detector import Detector

    return Detector(
        face_model=args.face_model,
        landmark_model=args.landmark_model,
        au_model=args.au_model,
        emotion_model=args.emotion_model,
        identity_model=args.identity_model,
        gaze_model=args.gaze_model,
        device=args.device,
    )


def _run_gaze(args, name: str) -> dict | None:
    if name == "columbia_gaze":
        split = datasets.load_columbia_gaze(
            head_pose_filter=args.columbia_head_pose,
            subset_size=args.columbia_subset,
            seed=args.seed,
        )
    else:
        raise ValueError(f"unknown gaze dataset {name!r}")
    if split is None:
        return None
    det = _build_detector(args)
    return runner.evaluate_dataset(
        det,
        split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def _run_au_emotion(args, name: str) -> dict | None:
    if name == "disfa":
        split = datasets.load_disfa(split="P3", subset_size=args.disfa_subset, seed=args.seed)
    elif name == "disfaplus":
        split = datasets.load_disfaplus(subset_size=args.disfa_subset, seed=args.seed)
    elif name == "affectnet_val":
        split = datasets.load_affectnet_val(subset_size=args.affectnet_subset, seed=args.seed)
    else:
        raise ValueError(f"unknown AU/emotion dataset {name!r}")
    if split is None:
        return None
    det = _build_detector(args)
    return runner.evaluate_dataset(
        det,
        split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def _run_identity(args, name: str) -> dict | None:
    if name == "calfw":
        split = datasets.load_calfw()
    elif name == "cplfw":
        split = datasets.load_cplfw()
    elif name == "tinyface":
        split = datasets.load_tinyface(use_distractors=not args.tinyface_no_distractors)
    else:
        raise ValueError(f"unknown identity dataset {name!r}")
    if split is None:
        return None
    return runner.evaluate_identity(
        split, device=args.device, batch_size=args.identity_batch_size
    )


def _format_markdown(meta: dict, results: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Accuracy benchmark — {meta['date']}")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    for k, v in meta.items():
        if k == "args":
            continue
        lines.append(f"| {k} | `{v}` |")
    lines.append("")
    lines.append("**Detector config**")
    lines.append("")
    lines.append("| stage | model |")
    lines.append("|---|---|")
    for k in ("face_model", "landmark_model", "au_model", "emotion_model",
              "facepose_model", "identity_model"):
        lines.append(f"| {k} | `{meta['args'][k]}` |")
    lines.append("")

    for disfa_key, header in (("disfa", "DISFA (P3 fold)"), ("disfaplus", "DISFA+ (posed peak)")):
        r = results.get(disfa_key)
        if r is not None and "error" not in r:
            lines.append(f"## {header} — AU intensity")
            lines.append("")
            lines.append(f"- samples: **{r['n_samples']}**, faces detected: {r['n_faces_detected']}")
            lines.append(f"- elapsed: {r['elapsed_s']}s ({r.get('fps')} fps)")
            lines.append(f"- **AU F1 mean: {r['au_f1_mean']:.4f}**")
            lines.append(f"- **AU ICC mean: {r['au_icc_mean']:.4f}**")
            lines.append("")
            lines.append("| AU | F1 | ICC |")
            lines.append("|---|---|---|")
            for au in sorted(r["au_f1_per_au"]):
                lines.append(f"| {au} | {r['au_f1_per_au'][au]:.4f} | {r['au_icc_per_au'][au]:.4f} |")
            lines.append("")

    if "affectnet_val" in results and results["affectnet_val"] is not None and "error" not in results["affectnet_val"]:
        r = results["affectnet_val"]
        lines.append("## AffectNet validation — 7-class emotion")
        lines.append("")
        lines.append(f"- samples: **{r['n_samples']}**, scored: {r['n_scored']}")
        lines.append(f"- elapsed: {r['elapsed_s']}s")
        lines.append(f"- **accuracy: {r['emotion_accuracy']:.4f}**")
        lines.append(f"- **macro F1: {r['emotion_f1_macro']:.4f}**")
        lines.append("")

    if "calfw" in results and results["calfw"] is not None and "error" not in results["calfw"]:
        r = results["calfw"]
        lines.append("## CALFW — cross-age 1:1 verification")
        lines.append("")
        lines.append(f"- pairs: **{r['n_pairs']}**, unique images: {r['n_unique_images']}")
        lines.append(f"- elapsed: {r['elapsed_s']}s")
        lines.append(f"- **accuracy: {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}** (10-fold CV)")
        lines.append(f"- AUC: {r['auc']:.4f}, mean threshold: {r['threshold_mean']:.4f}")
        lines.append("")

    if "cplfw" in results and results["cplfw"] is not None and "error" not in results["cplfw"]:
        r = results["cplfw"]
        lines.append("## CPLFW — cross-pose 1:1 verification")
        lines.append("")
        lines.append(f"- pairs: **{r['n_pairs']}**, unique images: {r['n_unique_images']}")
        lines.append(f"- elapsed: {r['elapsed_s']}s")
        lines.append(f"- **accuracy: {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}** (10-fold CV)")
        lines.append(f"- AUC: {r['auc']:.4f}, mean threshold: {r['threshold_mean']:.4f}")
        lines.append("")

    if "tinyface" in results and results["tinyface"] is not None and "error" not in results["tinyface"]:
        r = results["tinyface"]
        lines.append("## TinyFace — low-resolution 1:N identification")
        lines.append("")
        lines.append(f"- probes: **{r['n_probes']}**, gallery: {r['n_gallery']} (distractors: {r['n_distractors']})")
        lines.append(f"- elapsed: {r['elapsed_s']}s")
        lines.append("")
        lines.append("| rank | accuracy |")
        lines.append("|---|---|")
        for k_str, v in r.items():
            if isinstance(k_str, str) and k_str.startswith("rank_"):
                lines.append(f"| {k_str} | {v:.4f} |")
        lines.append("")

    return "\n".join(lines)


def _update_accuracy_index(latest_md_path: Path):
    """Maintain ``docs/benchmarks/accuracy.md`` as the jupyter-book entry.

    Rebuild from scratch each time — small enough that listing all
    historical runs is fine, and avoids drift from append-only logic.
    """
    runs = sorted(DOCS_BENCH_DIR.glob("*-accuracy.md"), reverse=True)
    out: list[str] = []
    out.append("# Accuracy benchmarks")
    out.append("")
    out.append(
        "Tracks per-release accuracy of py-feat detectors against held-out "
        "labeled datasets. Each entry is a single run produced by "
        "`python scripts/bench_regression.py --markdown`. "
        "Throughput benchmarks live in [throughput.md](throughput.md)."
    )
    out.append("")
    out.append("## Latest")
    out.append("")
    out.append(f"See [{latest_md_path.name}]({latest_md_path.name}).")
    out.append("")
    out.append("## Methodology")
    out.append("")
    out.append(
        "- **DISFA** P3 fold, ArcFace-aligned crops, AU intensity binarized "
        "at >=2 for F1; ICC(3,1) on continuous intensity vs. py-feat probability."
    )
    out.append(
        "- **AffectNet** validation set, classes 0..6 mapped to the 7 py-feat "
        "emotion columns; top-1 emotion accuracy and macro F1."
    )
    out.append(
        "- **CALFW / CPLFW** 6000 pairs, LFW 10-fold CV protocol, "
        "InsightFace 5-landmark template alignment before ArcFace embedding."
    )
    out.append(
        "- **TinyFace** closed-set + open-set rank-K identification with the "
        "Gallery_Distractor set (153k images) when not disabled."
    )
    out.append("")
    out.append("## History")
    out.append("")
    out.append("| date | run |")
    out.append("|---|---|")
    for r in runs:
        date = r.name.split("-accuracy.md")[0]
        out.append(f"| {date} | [{r.name}]({r.name}) |")
    out.append("")
    (DOCS_BENCH_DIR / "accuracy.md").write_text("\n".join(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="auto",
        help="Comma-separated dataset names or 'auto' (default: every "
        "dataset found on disk)",
    )
    parser.add_argument("--disfa-subset", type=int, default=4500,
                        help="Frames per DISFA P3 fold (default: 4500 ~= 500 frames x 9 subjects)")
    parser.add_argument("--affectnet-subset", type=int, default=1000,
                        help="Stratified subset size for AffectNet val (default: 1000)")
    parser.add_argument("--tinyface-no-distractors", action="store_true",
                        help="Skip the 153k-image distractor gallery for fast TinyFace runs")
    parser.add_argument("--columbia-subset", type=int, default=None,
                        help="Limit Columbia Gaze to N images (default: all)")
    parser.add_argument("--columbia-head-pose", default="0P",
                        help="Filter Columbia Gaze to head pose '0P' (default; head-frontal "
                        "subset where camera-frame == head-frame). Pass empty string to load all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--identity-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=_device_default())
    parser.add_argument("--face-model", default="retinaface")
    parser.add_argument("--landmark-model", default="mobilefacenet")
    parser.add_argument("--au-model", default="xgb")
    parser.add_argument("--emotion-model", default="resmasknet")
    parser.add_argument("--facepose-model", default="img2pose")
    parser.add_argument("--identity-model", default="arcface")
    parser.add_argument("--gaze-model", default="l2cs")
    parser.add_argument("--markdown", action="store_true",
                        help="Write a markdown file under docs/benchmarks/ "
                        "and refresh the jupyter-book accuracy index page")
    parser.add_argument("--json-output", type=Path, default=None,
                        help="Path for the JSON results file "
                        "(default: bench-results/<date>-<sha>.json)")
    args = parser.parse_args(argv)

    available = set(datasets.available())
    if args.datasets == "auto":
        selected = sorted(available & ALL_DATASETS)
    else:
        selected = [s.strip() for s in args.datasets.split(",") if s.strip()]
        unknown = set(selected) - ALL_DATASETS
        if unknown:
            print(f"error: unknown datasets {unknown}", file=sys.stderr)
            return 2
        missing = set(selected) - available
        if missing:
            print(f"warning: requested but not on disk: {missing}", file=sys.stderr)
            selected = [s for s in selected if s in available]

    if not selected:
        print("error: no datasets to run (nothing on disk?)", file=sys.stderr)
        return 1

    print(f"=== running on {len(selected)} dataset(s): {selected} ===")
    print(f"    device={args.device} | gpu={_gpu_name()} | py-feat={_feat_version}")

    # Allow --columbia-head-pose '' to mean None (load all head poses).
    if args.columbia_head_pose == "":
        args.columbia_head_pose = None

    results: dict = {}
    for name in selected:
        print(f"\n--- {name} ---")
        try:
            if name in AU_EMOTION_DATASETS:
                results[name] = _run_au_emotion(args, name)
            elif name in GAZE_DATASETS:
                results[name] = _run_gaze(args, name)
            else:
                results[name] = _run_identity(args, name)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        r = results[name]
        if r is None:
            print("  skipped (dataset not present)")
            continue
        # Print a one-line summary
        if name == "disfa" or name == "disfaplus":
            print(f"  AU F1 mean = {r['au_f1_mean']:.4f}, ICC mean = {r['au_icc_mean']:.4f} ({r['elapsed_s']}s)")
        elif name == "affectnet_val":
            print(f"  emotion acc = {r['emotion_accuracy']:.4f}, F1 macro = {r['emotion_f1_macro']:.4f} ({r['elapsed_s']}s)")
        elif name in ("calfw", "cplfw"):
            print(f"  acc = {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}, AUC = {r['auc']:.4f} ({r['elapsed_s']}s)")
        elif name == "tinyface":
            print(f"  rank-1 = {r['rank_1']:.4f}, rank-5 = {r['rank_5']:.4f} ({r['elapsed_s']}s)")
        elif name == "columbia_gaze":
            print(f"  gaze angular MAE = {r['gaze_angular_mae_deg']:.2f}° (median {r['gaze_angular_mae_median']:.2f}°, std {r['gaze_angular_mae_std']:.2f}°) on {r['n_scored']} faces ({r['elapsed_s']}s)")

    meta = _meta(args)
    payload = {"run": meta, "results": results}

    date = meta["date"].split("T")[0]
    sha = meta["git_sha"]
    if args.json_output is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        args.json_output = RESULTS_DIR / f"{date}-{sha}.json"
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON written: {args.json_output}")

    if args.markdown:
        DOCS_BENCH_DIR.mkdir(parents=True, exist_ok=True)
        md_path = DOCS_BENCH_DIR / f"{date}-{sha}-accuracy.md"
        md_path.write_text(_format_markdown(meta, results))
        print(f"Markdown written: {md_path}")
        _update_accuracy_index(md_path)
        print(f"Index refreshed: {DOCS_BENCH_DIR / 'accuracy.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
