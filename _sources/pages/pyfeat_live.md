# Py-feat Live

[Py-feat Live](https://github.com/cosanlab/pyfeat-live) puts Py-Feat behind a
graphical interface — run facial expression analysis on your webcam in real
time, batch-process recorded videos, and scrub the results frame by frame, with
no code. It wraps the same `Detector` models you use from Python, including the
new **Detectorv2 (v2.5)** multitask model.

> **Note:** add screenshots/GIFs for the Live, Analyze, and Viewer sections
> before publishing (the app is GUI-first and reads much better with visuals).

## Download

<div class="marimo-book-release-download" data-mb-release-download data-repo="cosanlab/pyfeat-live" data-app-name="Py-feat Live" data-platforms='[{&quot;key&quot;: &quot;mac-arm&quot;, &quot;label&quot;: &quot;macOS (Apple Silicon)&quot;, &quot;match&quot;: &quot;aarch64.dmg&quot;}, {&quot;key&quot;: &quot;mac-intel&quot;, &quot;label&quot;: &quot;macOS (Intel)&quot;, &quot;match&quot;: &quot;x64.dmg&quot;}, {&quot;key&quot;: &quot;windows&quot;, &quot;label&quot;: &quot;Windows&quot;, &quot;match&quot;: &quot;.msi&quot;}, {&quot;key&quot;: &quot;linux&quot;, &quot;label&quot;: &quot;Linux (AppImage)&quot;, &quot;match&quot;: &quot;.appimage&quot;}]'><div class="marimo-book-release-download__loading">Loading the latest release…</div><noscript><a href="https://github.com/cosanlab/pyfeat-live/releases/latest" target="_blank" rel="noopener">Download the latest Py-feat Live release on GitHub</a></noscript></div>

The button always points at the [latest release](https://github.com/cosanlab/pyfeat-live/releases/latest) (signed macOS `.dmg`; more platforms to come) — or run it from source, see the repo README. The **first launch downloads model weights** (a few minutes; cached afterward). See the [changelog](pyfeatlive_changelog.md) for what's new in each release.

## Detectors

| Detector | What it runs |
| --- | --- |
| **Detectorv2** (default) | The v2.5 multitask model — 20 AUs, 7 emotions, valence/arousal, a 478-point face mesh, 6-DoF head pose, gaze, 52 ARKit/MediaPipe blendshapes, and ArcFace/FaceNet identity, all from a single forward pass. |
| **Detectorv1** | The classic Py-Feat detector (RetinaFace / img2pose face + landmark / AU / emotion models). |
| **MPDetector** | MediaPipe FaceMesh-based detection. |

## Live

Real-time detection on your webcam. The overlay locks to the exact frame
detection ran on, so the landmarks and mesh never lag the video. Per-face panels
show emotions, valence/arousal, and head pose; you can toggle overlays (mesh,
gaze, AU heatmap) and adjust smoothing from the overlay-settings panel. Record a
session to disk — clean or overlaid video plus a per-frame `Fex` CSV — for later
review.

## Analyze (Extract)

Batch-process recorded videos. Drop files into the queue, pick a **preset**
(extraction defaults to Detectorv2), choose your compute device (CPU / MPS /
CUDA) and batch size, and run. Identity tracking groups faces across frames, and
each job writes a full native-schema `Fex` CSV.

## Viewer

Open a recorded session and scrub it frame by frame. The same overlays and
per-face emotion / valence-arousal / pose panels as Live render over the video,
and a timeseries panel plots any detected variable over time — grouped into
collapsible sets (emotions, AUs, blendshapes, pose, gaze, valence/arousal) so
the 80-plus v2.5 columns stay manageable. You can also cluster and label
identities and mark exclude-ranges as annotations.

---

*Py-feat Live is open source: [cosanlab/pyfeat-live](https://github.com/cosanlab/pyfeat-live).*
