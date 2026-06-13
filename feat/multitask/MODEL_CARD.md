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
blendshapes** (the v2.5 model; replaces v2.4).

- **Backbone:** ConvNeXt-V2 Tiny (FCMAE + IN-22k/IN-1k pretrained)
- **Heads:** ME-GraphAU AU graph (AFG/FGG/SC) + unified-feature emotion/V-A and
  gaze heads + landmark, pose, and **blendshape** regression heads
- **Params:** ~30M · **Input:** 224×224 RGB (from a 256×256 face crop)
- **File:** `face_multitask_v2.safetensors` (safetensors; `ModelV2Config` JSON in the file metadata)

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

## Benchmarks (held-out, file-verified — v2.5 deployed checkpoint)

| Task | Dataset | Metric | Score |
|---|---|---|---|
| AU | DISFA+ (12-AU, Cheong protocol) | macro-F1 | **0.693** |
| AU | DISFA+ (8-AU subset) | macro-F1 | **0.740** |
| Emotion | RAF-DB official test (7-cls) | acc / macro-F1 | **0.910 / 0.885** |
| Emotion | AffectNet val (7-cls, drop Contempt) | acc / macro-F1 | **0.616 / 0.612** |
| Valence/Arousal | Aff-Wild2 official validation | CCC (V / A) | **0.852 / 0.799** |
| Gaze | MPIIGaze (leave-subject-out) | mean angular err | 7.05° |
| Gaze | Gaze360 (held-out split) | mean angular err | 12.89° |

Notes: **v2.5 = v2.4 architecture + a blendshape head**, and it beats v2.4 on every
accuracy benchmark — most dramatically AffectNet emotion (acc 0.35→0.62) and
Aff-Wild2 V/A (0.82/0.78 → 0.85/0.80). **Gaze numbers are now leave-subject-out
held-out** (honest generalization); the lower v2.4 figures (3.92°/6.81°) came from a
leaky evaluation that included training subjects, so they are not comparable — the
v2.5 numbers are the real ones. Numbers are from the deployed checkpoint
(`v25c_release_ep14`), weight-verified against the published `.safetensors`.

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
