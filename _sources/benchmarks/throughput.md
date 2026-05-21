# Throughput benchmarks

Tracks per-release wall-time of `Detector` and `MPDetector` over time. Each
entry is a single run produced by `python scripts/bench_detectors.py
--markdown`. Accuracy benchmarks live in [accuracy.md](accuracy.md).

## Methodology

`scripts/bench_detectors.py` measures the **detection + landmark + AU +
emotion + identity** path on reproducible test fixtures from
`feat/tests/data/`:

- `single_face.mp4` (72 frames, 1 face/frame)
- `WolfgangLanger_Pexels.mp4` (472 frames, 1 face/frame)
- `multi_face.jpg` × 16 = 80 faces

Three configurations are timed head-to-head:

1. `Detector(face_model='img2pose', au_model='xgb', emotion_model='resmasknet', identity_model='arcface')`
2. `Detector(face_model='retinaface', au_model='xgb', emotion_model='resmasknet', identity_model='arcface')`
3. `MPDetector(face_model='retinaface', landmark_model='mp_facemesh_v2', au_model='mp_blendshapes', emotion_model='resmasknet', identity_model='arcface')`

Swept axes: `device × batch_size × num_workers`.

## History

| date | run |
|---|---|
| 2026-05-14 | [2026-05-14-c716340.md](2026-05-14-c716340.md) |
| 2026-05-04 | [2026-05-04-f44ccb1.md](2026-05-04-f44ccb1.md) |
| 2026-05-03 | [2026-05-03-d71c0d7.md](2026-05-03-d71c0d7.md) |
| 2026-05-03 | [2026-05-03-864962c.md](2026-05-03-864962c.md) |
| 2026-05-03 | [2026-05-03-437b651.md](2026-05-03-437b651.md) |
| 2026-05-03 | [2026-05-03-09980f9.md](2026-05-03-09980f9.md) |
