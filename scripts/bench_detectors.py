"""Apples-to-apples benchmark for Detector + MPDetector.

Sweeps face_model, device, and batch_size on a fixed real video and a
fixed image set. Prints per-frame ms and frames-per-second to stdout.
Reproducible via the test fixtures already in feat/tests/data; no
external assets required.

Usage:
    python scripts/bench_detectors.py

What it covers (all three configurations process the same input):
    1. Detector(face_model='img2pose', au_model='svm')         (default path)
    2. Detector(face_model='retinaface_r34', au_model='svm')   (new fast path)
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
  skops. svm AU is the apples-to-apples constant.
- Each timed call is preceded by one untimed warmup. Reports the wall
  time of the timed call.
"""
from __future__ import annotations

import os
import time
import warnings

warnings.simplefilter("ignore")

import torch

from feat.utils.io import get_test_data_path

VIDEO_LONG = os.path.join(get_test_data_path(), "WolfgangLanger_Pexels.mp4")
VIDEO_SHORT = os.path.join(get_test_data_path(), "single_face.mp4")
IMG_MULTI = os.path.join(get_test_data_path(), "multi_face.jpg")


def time_one(detector, inputs, data_type: str, batch_size: int):
    """Run one warmup call + one timed call. Returns (seconds, n_rows)."""
    detector.detect(inputs, data_type=data_type, batch_size=batch_size)
    t = time.perf_counter()
    fex = detector.detect(inputs, data_type=data_type, batch_size=batch_size)
    return time.perf_counter() - t, len(fex)


def banner(s: str) -> None:
    print(f"\n=== {s} ===", flush=True)


def _build_d_img2pose(device):
    from feat.detector import Detector
    return Detector(face_model="img2pose", au_model="svm", device=device)


def _build_d_retinaface(device):
    from feat.detector import Detector
    return Detector(face_model="retinaface_r34", au_model="svm", device=device)


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
    ("retinaface_r34", _build_d_retinaface),
    ("MPDetector retinaface", _build_mp_detector),
]


def run_video_sweep(video_path: str, label: str, devices_per_config) -> None:
    print(f"\n# Video: {os.path.basename(video_path)} ({label})", flush=True)
    for cfg_label, build in CONFIGS:
        banner(f"VIDEO {label}: {cfg_label}")
        print(f"{'device':>8} {'batch':>6} {'sec':>8} {'ms/frame':>10} {'fps':>8}", flush=True)
        for device in devices_per_config[cfg_label]:
            det = build(device)
            for bs in (1, 4, 16):
                sec, n = time_one(det, video_path, "video", bs)
                print(
                    f"{device:>8} {bs:>6} {sec:>8.2f} "
                    f"{sec / n * 1000:>10.1f} {n / sec:>8.1f}",
                    flush=True,
                )
            del det
            if device == "mps":
                torch.mps.empty_cache()


def run_image_sweep(image_paths: list[str], label: str) -> None:
    print(f"\n# Images: {label} ({len(image_paths)} images)", flush=True)
    for cfg_label, build in CONFIGS:
        banner(f"IMAGES {label}: {cfg_label}")
        print(f"{'device':>8} {'batch':>6} {'sec':>8} {'ms/img':>10} {'rows':>6}", flush=True)
        for device in ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",):
            det = build(device)
            for bs in (1, 4, 16):
                sec, n = time_one(det, image_paths, "image", bs)
                print(
                    f"{device:>8} {bs:>6} {sec:>8.2f} "
                    f"{sec / len(image_paths) * 1000:>10.1f} {n:>6}",
                    flush=True,
                )
            del det
            if device == "mps":
                torch.mps.empty_cache()


def main() -> None:
    mps_available = torch.backends.mps.is_available()
    devices_for_short = {
        cfg: ("cpu", "mps") if mps_available else ("cpu",) for cfg, _ in CONFIGS
    }
    # Skip CPU+img2pose on the long video (~10 min/run on M-series CPUs).
    devices_for_long = dict(devices_for_short)
    devices_for_long["img2pose"] = ("mps",) if mps_available else ("cpu",)

    print("# py-feat detector benchmark", flush=True)
    print(
        "# Hardware: see torch.backends.mps.is_available()=",
        mps_available,
        flush=True,
    )
    print("# Each timed call has one warmup beforehand.", flush=True)

    run_video_sweep(VIDEO_SHORT, "short (72 frames)", devices_for_short)
    run_video_sweep(VIDEO_LONG, "long (472 frames)", devices_for_long)
    run_image_sweep([IMG_MULTI] * 16, "16 x multi_face.jpg = 80 faces")


if __name__ == "__main__":
    main()
