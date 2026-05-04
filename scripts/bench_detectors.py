"""Apples-to-apples benchmark for Detector + MPDetector.

Sweeps face_model x device x batch_size (and optionally num_workers) on
a fixed real video and a fixed image set. Prints per-frame ms and
frames-per-second to stdout. Reproducible via the test fixtures in
``feat/tests/data``; no external assets required.

Usage:
    # Default sweep (device x batch, num_workers=0 only)
    python scripts/bench_detectors.py

    # Add num_workers axis to compare DataLoader parallelism
    python scripts/bench_detectors.py --workers 0 2

    # Custom batch sizes
    python scripts/bench_detectors.py --batches 1 8 32

    # Restrict to specific devices
    python scripts/bench_detectors.py --devices mps
    python scripts/bench_detectors.py --devices cpu cuda

    # Also write a markdown report (good for tracking results over time)
    python scripts/bench_detectors.py --workers 0 2 --markdown

    # Custom markdown path
    python scripts/bench_detectors.py --markdown docs/benchmarks/my-run.md

What it covers (all three configurations process the same input):
    1. Detector(face_model='img2pose', au_model='svm')         (default path)
    2. Detector(face_model='retinaface', au_model='svm')       (new fast path)
    3. MPDetector(face_model='retinaface',
                  landmark_model='mp_facemesh_v2',
                  au_model='mp_blendshapes')                    (mediapipe path)

The Detector svm AU classifier is held constant in the first two so
varying face_model is the only variable. MPDetector uses its native
mp_blendshapes stage. Emotion / identity / facepose disabled across
the board so we measure detection + landmark + AU only.

Notes:
- Skips CPU+img2pose on the long-video path (~10 min/run on M-series CPUs;
  the CPU+img2pose number is implied by the short-video baseline).
- xgb AU is not benchmarked here because it segfaults on Python 3.13 +
  skops on some configurations. svm AU is the apples-to-apples constant.
- Each timed call is preceded by one untimed warmup. Reports the wall
  time of the timed call.
- num_workers > 0 has been measured slower than num_workers=0 in every
  cell on M-series + Python 3.13 with the default OMP_NUM_THREADS=1
  (per ``feat/__init__.py``); the option exists so the regression
  baseline is reproducible. See py-feat #288 for context.
"""
from __future__ import annotations

import argparse
import datetime
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import torch

# Pull just the version string. `import feat` works too but pulls in
# torch and the whole detector graph just for a string lookup.
from feat.version import __version__ as _feat_version
from feat.utils.io import get_test_data_path

VIDEO_LONG = os.path.join(get_test_data_path(), "WolfgangLanger_Pexels.mp4")
VIDEO_SHORT = os.path.join(get_test_data_path(), "single_face.mp4")
IMG_MULTI = os.path.join(get_test_data_path(), "multi_face.jpg")


def time_one(detector, inputs, data_type: str, batch_size: int, num_workers: int):
    """Run one warmup call + one timed call.

    Returns ``(seconds, n_rows, n_units)`` where ``n_units`` is the
    denominator the per-unit metric divides by:

    - For ``data_type='video'``: number of unique frames processed
      (``fex['frame'].nunique()``). For multi-face video this is
      smaller than ``n_rows = len(fex)``, so ms/frame is computed
      correctly.
    - For ``data_type='image'``: number of input images (``len(inputs)``),
      independent of how many faces were detected per image.
    """
    detector.detect(
        inputs, data_type=data_type, batch_size=batch_size, num_workers=num_workers
    )
    t = time.perf_counter()
    fex = detector.detect(
        inputs, data_type=data_type, batch_size=batch_size, num_workers=num_workers
    )
    sec = time.perf_counter() - t
    n_rows = len(fex)
    if data_type == "video":
        n_units = int(fex["frame"].nunique())
    else:  # image
        n_units = len(inputs)
    return sec, n_rows, n_units


def banner(s: str) -> None:
    print(f"\n=== {s} ===", flush=True)


def _build_d_img2pose(device):
    from feat.detector import Detector
    return Detector(face_model="img2pose", au_model="svm", device=device)


def _build_d_retinaface(device):
    from feat.detector import Detector
    return Detector(face_model="retinaface", au_model="svm", device=device)


def _build_mp_detector(device):
    from feat.MPDetector import MPDetector
    return MPDetector(
        face_model="retinaface",
        landmark_model="mp_facemesh_v2",
        au_model="mp_blendshapes",
        emotion_model=None,
        identity_model=None,
        facepose_model=None,
        device=device,
    )


CONFIGS = [
    ("img2pose", _build_d_img2pose),
    ("retinaface", _build_d_retinaface),
    ("MPDetector retinaface", _build_mp_detector),
]


# ---------------------------------------------------------------------------
# Result collection — every cell appends a Row, then we render to stdout (live)
# and optionally to a single markdown file at the end.
# ---------------------------------------------------------------------------


class Row:
    __slots__ = (
        "section_kind",  # 'video' or 'image'
        "section_label",  # e.g. 'short (72 frames)' or '16 x multi_face.jpg = 80 faces'
        "cfg_label",  # one of CONFIGS' first elements
        "device",
        "batch",
        "workers",
        "sec",
        "n_rows",  # output rows = detected faces (sum across frames/images)
        "n_units",  # frames (for video) or input images (for image kind)
    )

    def __init__(
        self, section_kind, section_label, cfg_label,
        device, batch, workers, sec, n_rows, n_units,
    ):
        self.section_kind = section_kind
        self.section_label = section_label
        self.cfg_label = cfg_label
        self.device = device
        self.batch = batch
        self.workers = workers
        self.sec = sec
        self.n_rows = n_rows
        self.n_units = n_units

    @property
    def per_unit_ms(self):
        """ms per frame (video) or per image (image kind), correct for
        multi-face inputs."""
        return self.sec / self.n_units * 1000

    @property
    def fps(self):
        """frames per second processed (video) or images per second (image)."""
        return self.n_units / self.sec


def _live_print_video(row: Row, has_workers_axis: bool):
    """Print one row inline to stdout in the same format as the prior bench."""
    if has_workers_axis:
        print(
            f"{row.device:>8} {row.batch:>6} {row.workers:>8} {row.sec:>8.2f} "
            f"{row.per_unit_ms:>10.1f} {row.fps:>8.1f}",
            flush=True,
        )
    else:
        print(
            f"{row.device:>8} {row.batch:>6} {row.sec:>8.2f} "
            f"{row.per_unit_ms:>10.1f} {row.fps:>8.1f}",
            flush=True,
        )


def _live_print_image(row: Row, has_workers_axis: bool):
    if has_workers_axis:
        print(
            f"{row.device:>8} {row.batch:>6} {row.workers:>8} {row.sec:>8.2f} "
            f"{row.per_unit_ms:>10.1f} {row.n_rows:>6}",
            flush=True,
        )
    else:
        print(
            f"{row.device:>8} {row.batch:>6} {row.sec:>8.2f} "
            f"{row.per_unit_ms:>10.1f} {row.n_rows:>6}",
            flush=True,
        )


def _video_header(has_workers_axis: bool) -> str:
    if has_workers_axis:
        return (
            f"{'device':>8} {'batch':>6} {'workers':>8} "
            f"{'sec':>8} {'ms/frame':>10} {'fps':>8}"
        )
    return f"{'device':>8} {'batch':>6} {'sec':>8} {'ms/frame':>10} {'fps':>8}"


def _image_header(has_workers_axis: bool) -> str:
    if has_workers_axis:
        return (
            f"{'device':>8} {'batch':>6} {'workers':>8} "
            f"{'sec':>8} {'ms/img':>10} {'rows':>6}"
        )
    return f"{'device':>8} {'batch':>6} {'sec':>8} {'ms/img':>10} {'rows':>6}"


def run_video_sweep(
    video_path: str,
    label: str,
    devices_per_config,
    batch_sizes,
    num_workers_options,
    rows_out,
) -> None:
    print(f"\n# Video: {os.path.basename(video_path)} ({label})", flush=True)
    has_workers_axis = num_workers_options != (0,)
    for cfg_label, build in CONFIGS:
        banner(f"VIDEO {label}: {cfg_label}")
        print(_video_header(has_workers_axis), flush=True)
        for device in devices_per_config[cfg_label]:
            det = build(device)
            for bs in batch_sizes:
                for nw in num_workers_options:
                    sec, n_rows, n_units = time_one(det, video_path, "video", bs, nw)
                    row = Row(
                        section_kind="video",
                        section_label=label,
                        cfg_label=cfg_label,
                        device=device,
                        batch=bs,
                        workers=nw,
                        sec=sec,
                        n_rows=n_rows,
                        n_units=n_units,
                    )
                    rows_out.append(row)
                    _live_print_video(row, has_workers_axis)
            del det
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()


def run_image_sweep(
    image_paths: list[str],
    label: str,
    devices,
    batch_sizes,
    num_workers_options,
    rows_out,
) -> None:
    print(f"\n# Images: {label} ({len(image_paths)} images)", flush=True)
    has_workers_axis = num_workers_options != (0,)
    for cfg_label, build in CONFIGS:
        banner(f"IMAGES {label}: {cfg_label}")
        print(_image_header(has_workers_axis), flush=True)
        for device in devices:
            det = build(device)
            for bs in batch_sizes:
                for nw in num_workers_options:
                    sec, n_rows, n_units = time_one(
                        det, image_paths, "image", bs, nw
                    )
                    row = Row(
                        section_kind="image",
                        section_label=label,
                        cfg_label=cfg_label,
                        device=device,
                        batch=bs,
                        workers=nw,
                        sec=sec,
                        n_rows=n_rows,
                        n_units=n_units,
                    )
                    rows_out.append(row)
                    _live_print_image(row, has_workers_axis)
            del det
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _git_commit_short() -> str:
    """Return the short SHA, or 'nogit' if git or repo is unavailable.

    Used both as run metadata and as part of the default markdown filename;
    avoid characters like parentheses that would make the resulting path
    awkward to type or grep for.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def _hardware_summary() -> str:
    try:
        return f"{platform.node()} ({platform.machine()}, {os.cpu_count()} CPUs)"
    except Exception:
        return "unknown"


def _md_table_rows(rows: list, has_workers_axis: bool, kind: str) -> str:
    """Render rows for one (section, config) tile as a markdown table body."""
    if has_workers_axis:
        if kind == "video":
            head = "| device | batch | workers | sec | ms/frame | fps |\n"
            sep = "|---|---|---|---|---|---|\n"
            lines = [
                f"| {r.device} | {r.batch} | {r.workers} | {r.sec:.2f} | "
                f"{r.per_unit_ms:.1f} | {r.fps:.1f} |"
                for r in rows
            ]
        else:
            head = "| device | batch | workers | sec | ms/img | rows |\n"
            sep = "|---|---|---|---|---|---|\n"
            lines = [
                f"| {r.device} | {r.batch} | {r.workers} | {r.sec:.2f} | "
                f"{r.per_unit_ms:.1f} | {r.n_rows} |"
                for r in rows
            ]
    else:
        if kind == "video":
            head = "| device | batch | sec | ms/frame | fps |\n"
            sep = "|---|---|---|---|---|\n"
            lines = [
                f"| {r.device} | {r.batch} | {r.sec:.2f} | "
                f"{r.per_unit_ms:.1f} | {r.fps:.1f} |"
                for r in rows
            ]
        else:
            head = "| device | batch | sec | ms/img | rows |\n"
            sep = "|---|---|---|---|---|\n"
            lines = [
                f"| {r.device} | {r.batch} | {r.sec:.2f} | "
                f"{r.per_unit_ms:.1f} | {r.n_rows} |"
                for r in rows
            ]
    return head + sep + "\n".join(lines) + "\n"


def write_markdown(
    rows: list,
    out_path: Path,
    args,
    all_devices,
    has_workers_axis: bool,
) -> None:
    """Render the run as a single markdown file with one table per (section, cfg)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sections = {}  # (section_kind, section_label, cfg_label) -> [rows]
    for r in rows:
        sections.setdefault((r.section_kind, r.section_label, r.cfg_label), []).append(r)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    pyver = ".".join(str(x) for x in __import__("sys").version_info[:3])
    omp = os.environ.get("OMP_NUM_THREADS", "unset")
    git_sha = _git_commit_short()
    hw = _hardware_summary()

    md = []
    md.append(f"# py-feat detector benchmark — {now}\n")
    md.append("## Run metadata\n")
    md.append(f"- **Date:** {now}")
    md.append(f"- **py-feat version:** {_feat_version}")
    md.append(f"- **Git commit:** {git_sha}")
    md.append(f"- **Host:** {hw}")
    md.append(f"- **Python:** {pyver}")
    md.append(f"- **PyTorch:** {torch.__version__}")
    md.append(f"- **GPU:** {_gpu_summary()}")
    md.append(f"- **OMP_NUM_THREADS:** `{omp}`")
    md.append(f"- **Devices swept:** {list(all_devices)}")
    md.append(f"- **Batch sizes:** {args.batches}")
    md.append(f"- **DataLoader workers:** {args.workers}")
    md.append("")
    md.append("Each timed call is preceded by one untimed warmup; the timed-call wall time is reported.\n")

    # Group output: short video, long video, images.
    section_order = []
    seen = set()
    for r in rows:
        key = (r.section_kind, r.section_label)
        if key not in seen:
            seen.add(key)
            section_order.append(key)

    for section_kind, section_label in section_order:
        if section_kind == "video":
            md.append(f"## Video: {section_label}\n")
        else:
            md.append(f"## Images: {section_label}\n")
        for cfg_label, _build in CONFIGS:
            cfg_rows = sections.get((section_kind, section_label, cfg_label))
            if not cfg_rows:
                continue
            md.append(f"### {cfg_label}\n")
            md.append(_md_table_rows(cfg_rows, has_workers_axis, section_kind))
            md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n# Markdown report written to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="py-feat detector benchmark sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="Batch sizes to sweep.",
    )
    p.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "DataLoader num_workers values to sweep. Default [0] (current "
            "behavior). Pass `--workers 0 2` to compare parallelism; "
            "be aware num_workers>0 is typically slower on M-series + "
            "Python 3.13 with OMP_NUM_THREADS=1."
        ),
    )
    p.add_argument(
        "--devices",
        nargs="+",
        default=["auto"],
        help=(
            "Devices to test. Pass one or more of: cpu, mps, cuda, auto. "
            "'auto' (default) picks cpu + the best available accelerator."
        ),
    )
    p.add_argument(
        "--skip-long-video",
        action="store_true",
        help="Skip the 472-frame video sweep (cuts runtime ~5x).",
    )
    p.add_argument(
        "--skip-short-video",
        action="store_true",
        help="Skip the 72-frame video sweep.",
    )
    p.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip the image-batch sweep.",
    )
    p.add_argument(
        "--markdown",
        nargs="?",
        const="__default__",
        default=None,
        help=(
            "Write a markdown report. Pass with no argument for the default "
            "path (`docs/benchmarks/<YYYY-MM-DD>-<git-sha>.md`), or supply "
            "an explicit path."
        ),
    )
    return p.parse_args()


def _resolve_devices(requested: list[str]) -> tuple[str, ...]:
    """Turn the --devices list into concrete device strings."""
    mps_ok = torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()

    # Handle legacy "both" and "auto"
    resolved = []
    for d in requested:
        if d in ("both", "auto"):
            resolved.append("cpu")
            if cuda_ok:
                resolved.append("cuda")
            elif mps_ok:
                resolved.append("mps")
        elif d == "mps":
            if not mps_ok:
                raise SystemExit("mps requested but torch.backends.mps.is_available() is False")
            resolved.append("mps")
        elif d == "cuda":
            if not cuda_ok:
                raise SystemExit("cuda requested but torch.cuda.is_available() is False")
            resolved.append("cuda")
        else:
            resolved.append(d)
    # Deduplicate preserving order
    seen = set()
    out = []
    for d in resolved:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return tuple(out)


def _gpu_summary() -> str:
    """Return a short GPU description for metadata."""
    parts = []
    if torch.cuda.is_available():
        parts.append(f"CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
    if torch.backends.mps.is_available():
        parts.append("MPS available")
    return "; ".join(parts) if parts else "no GPU"


def main() -> None:
    args = _parse_args()

    all_devices = _resolve_devices(args.devices)

    devices_for_short = {cfg: all_devices for cfg, _ in CONFIGS}
    # Skip CPU+img2pose on the long video (~10 min/run on M-series CPUs).
    devices_for_long = dict(devices_for_short)
    if "cpu" in all_devices:
        long_img2pose = tuple(d for d in all_devices if d != "cpu")
        devices_for_long["img2pose"] = long_img2pose if long_img2pose else ("cpu",)

    print("# py-feat detector benchmark", flush=True)
    print(f"# GPU: {_gpu_summary()}", flush=True)
    print(
        f"# OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'unset')}",
        flush=True,
    )
    print(
        f"# axes: devices={list(all_devices)}, batches={args.batches}, "
        f"workers={args.workers}",
        flush=True,
    )
    print("# Each timed call has one warmup beforehand.", flush=True)

    batch_sizes = tuple(args.batches)
    num_workers_options = tuple(args.workers)
    has_workers_axis = num_workers_options != (0,)
    rows: list[Row] = []

    if not args.skip_short_video:
        run_video_sweep(
            VIDEO_SHORT, "short (72 frames)", devices_for_short,
            batch_sizes, num_workers_options, rows,
        )
    if not args.skip_long_video:
        run_video_sweep(
            VIDEO_LONG, "long (472 frames)", devices_for_long,
            batch_sizes, num_workers_options, rows,
        )
    if not args.skip_images:
        run_image_sweep(
            [IMG_MULTI] * 16, "16 x multi_face.jpg = 80 faces",
            all_devices, batch_sizes, num_workers_options, rows,
        )

    # Optionally render a markdown report.
    if args.markdown is not None:
        if args.markdown == "__default__":
            sha = _git_commit_short()
            today = datetime.date.today().isoformat()
            out_path = Path("docs/benchmarks") / f"{today}-{sha}.md"
        else:
            out_path = Path(args.markdown)
        write_markdown(rows, out_path, args, all_devices, has_workers_axis)


if __name__ == "__main__":
    main()
