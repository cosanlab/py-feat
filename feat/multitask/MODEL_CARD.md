---
license: other
license_name: research-only
license_link: LICENSE
library_name: py-feat
tags:
  - facial-expression-analysis
  - action-units
  - emotion-recognition
  - gaze-estimation
  - face-landmarks
  - head-pose
  - blendshapes
  - multitask
pipeline_tag: image-classification
---

# face_multitask_v2

A single multi-task convolutional model for facial behavior analysis, used by
[py-feat](https://github.com/cosanlab/py-feat)'s `Detectorv2`. From one face crop
it jointly predicts **action units, categorical emotion, valence/arousal,
eye gaze, a 478-point face mesh, 6-DoF head pose, and 52 MediaPipe/ARKit
blendshapes** (the v2.6 model; replaces v2.5).

- **Backbone:** ConvNeXt-V2 Tiny (FCMAE + IN-22k/IN-1k pretrained)
- **Heads:** ME-GraphAU AU graph (AFG/FGG/SC) + unified-feature emotion/V-A heads
  + landmark, pose, and **blendshape** regression heads + the v2.6 **eye-aware
  gaze head**: RoI-pooled eye features (localized by the predicted mesh),
  conditioned on predicted head pose (6D), with an L2CS-style binned prediction
  over the full ±180° range
- **Params:** ~41M · **Input:** 224×224 RGB (from a 256×256 face crop)
- **File:** `face_multitask_v26.safetensors` (safetensors; `ModelV2Config` JSON in the file metadata)

## Outputs

| Task | Output | Notes |
|---|---|---|
| Action Units | 20 probabilities [0,1] | AU01,02,04,05,06,07,09,10,11,12,14,15,17,20,23,24,25,26,28,43 |
| Emotion | 7-class softmax | Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger |
| Valence / Arousal | 2 × [−1,1] | tanh |
| Gaze | (yaw, pitch) radians | head-centric; yaw+ = right, pitch+ = up |
| Face mesh | 478 × (x,y,z) | MediaPipe topology, chip-pixel coords (z = relative depth) |
| Head pose | (yaw, pitch, roll, tx, ty, tz) | radians / pixels |
| 68 landmarks | derived | dlib-68 subset sampled from the 478 mesh |
| Blendshapes | 52 coefficients [0,1] | MediaPipe/ARKit standard names (browInnerUp, jawOpen, mouthSmileLeft, …) |

## Benchmarks (held-out, file-verified — v2.6 deployed checkpoint)

All gaze splits are identity-disjoint from training (held-out subjects), and
EYEDIAP is never trained on by any py-feat model.

| Task | Dataset | Metric | v2.6 | v2.5 |
|---|---|---|---|---|
| AU | DISFA+ (12-AU, Cheong protocol) | macro-F1 | **0.696** | 0.693 |
| AU | DISFA+ (8-AU subset) | macro-F1 | 0.738 | **0.740** |
| Emotion | AffectNet val (7-cls, drop Contempt) | acc / macro-F1 | 0.615 / 0.610 | **0.616 / 0.612** |
| Valence/Arousal | AffectNet val | CCC (V / A) | 0.775 / **0.653** | **0.780** / 0.646 |
| Gaze | ETH-XGaze (held-out subjects) | mean angular err | **5.0°** | 43.2° |
| Gaze | EYEDIAP (never-train, 15.2K frames) | mean angular err | **13.4°** | 15.3° |
| Gaze | Gaze360 (held-out split) | mean angular err | 13.0° | **12.9°** |
| Gaze | MPIIGaze (leave-subject-out) | mean angular err | 7.4° | **7.0°** |
| Gaze | Columbia (held-out subjects) | mean angular err | **5.4°** | — (trained) |

Notes: **v2.6 = v2.5 + an eye-aware gaze head** (eye RoI features, head-pose
conditioning, binned ±180° prediction) trained with ETH-XGaze added to the gaze
pool. It transforms extreme-head-pose gaze (ETH-XGaze 43°→5°) and
out-of-distribution gaze (EYEDIAP −2°), decouples eye gaze from head pose
(v2.5's gaze tracked the head; v2.6 tracks the eyes), and holds AU / emotion /
valence-arousal at v2.5 levels within noise. The small MPII / Gaze360 deltas
(+0.1–0.4°) are the cost of the pose-robust training mix. Numbers are from the
deployed checkpoint (v2.6 stage-3 `v24_best`, ep07), weight-verified against
the published `.safetensors`.

## Usage

```python
from feat import Detectorv2
detector = Detectorv2(device="cuda")
fex = detector.detect("image.jpg")   # returns a py-feat Fex
```

The model expects a face crop produced by RetinaFace + py-feat's
`extract_face_from_bbox_torch(frame, bbox, face_size=256, expand_bbox=1.2)`,
then resized to 224 and ImageNet-normalized. `Detectorv2` handles this.

## License

**Research / non-commercial use only.** Trained on datasets (AffectNet, DISFA+,
RAF-DB, Aff-Wild2, BP4D, etc.) whose licenses restrict use to academic research.
The ConvNeXt-V2 backbone is MIT-licensed. Confirm each constituent dataset's
terms before any non-research use.
