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
blendshapes** (the v2.7 model; replaces v2.6).

- **Backbone:** ConvNeXt-V2 Tiny (FCMAE + IN-22k/IN-1k pretrained)
- **Heads:** ME-GraphAU AU graph (AFG/FGG/SC) + unified-feature emotion/V-A heads
  + landmark, pose, and **blendshape** regression heads + the v2.6 **eye-aware
  gaze head**: RoI-pooled eye features (localized by the predicted mesh),
  conditioned on predicted head pose (6D), with an L2CS-style binned prediction
  over the full ±180° range (v2.7: 2° bins)
- **Params:** ~42M · **Input:** 224×224 RGB (from a 256×256 face crop)
- **File:** `face_multitask_v27.safetensors` (safetensors; `ModelV2Config` JSON in the file metadata)
- **Weights:** soup (equal average) of five consecutive fine-tuning checkpoints

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

## Benchmarks (held-out, file-verified — v2.7 deployed checkpoint)

All gaze splits are identity-disjoint from training (held-out subjects), and
EYEDIAP is never trained on by any py-feat model.

| Task | Dataset | Metric | v2.7 | v2.6 | v2.5 |
|---|---|---|---|---|---|
| AU | DISFA+ (12-AU, Cheong protocol) | macro-F1 | 0.686 | **0.696** | 0.693 |
| AU | DISFA+ (8-AU subset) | macro-F1 | **0.740** | 0.738 | **0.740** |
| Emotion | AffectNet val (7-cls, drop Contempt) | acc / macro-F1 | 0.612 / 0.607 | 0.615 / 0.610 | **0.616 / 0.612** |
| Emotion | RAF-DB test | acc / macro-F1 | 0.876 / 0.817 | 0.873 / 0.818 | **0.910 / 0.885** |
| Valence/Arousal | AffectNet val | CCC (V / A) | 0.773 / 0.647 | 0.775 / **0.653** | **0.780** / 0.646 |
| Valence/Arousal | AFEW-VA | CCC (V / A) | 0.711 / 0.480 | 0.718 / 0.411 | **0.833 / 0.863** |
| Valence/Arousal | Aff-Wild2 val | CCC (V / A) | 0.331 / 0.418 | 0.397 / 0.458 | **0.852 / 0.799** |
| Gaze | ETH-XGaze (held-out subjects) | mean angular err | 5.1° | **5.0°** | 43.2° |
| Gaze | EYEDIAP (never-train, 15.2K frames) | mean angular err | **12.6°** | 13.4° | 15.3° |
| Gaze | Gaze360 (held-out split) | mean angular err | 13.0° | 13.0° | **12.9°** |
| Gaze | MPIIGaze (leave-subject-out) | mean angular err | 8.0° | 7.4° | **7.0°** |
| Gaze | Columbia (held-out subjects) | mean angular err | **4.1°** | 5.4° | — (trained) |
| Blendshapes | FacePlace (teacher agreement) | mean active-ch. r | **0.761** | 0.748 | 0.756 |

Notes: **v2.7 = the v2.6 architecture retrained** with a rescaled gaze loss,
a rebalanced within-gaze data mix, per-source augmentation, 2° gaze bins, and
a head-pose label fix. Vs v2.6 it improves out-of-distribution gaze (EYEDIAP
−0.7°, Columbia −1.3°), RAF-DB macro-F1 end-to-end (+2.7), AFEW-VA arousal,
blendshape fidelity (best of any release), and occlusion robustness, at the
cost of ~0.6° on frontal MPIIGaze and 0.01 on the 12-AU set.

**Known limitation (v2.6 and v2.7):** continuous valence/arousal on
*video-frame* corpora (Aff-Wild2, AFEW-VA) is substantially below v2.5
(e.g. Aff-Wild2 CCC-V 0.85 → 0.33). AffectNet (still-image) V/A is unaffected.
If frame-wise continuous V/A on video is your primary measure, prefer the v2.5
weights (`face_multitask_v2.safetensors`, still published in this repo). A fix
is under investigation. Numbers are from the deployed checkpoint (v2.7
stage-3 soup ep05-09), weight-verified against the published `.safetensors`.

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
