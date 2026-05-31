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
  - multitask
pipeline_tag: image-classification
---

# face_multitask_v1

A single multi-task convolutional model for facial behavior analysis, used by
[py-feat](https://github.com/cosanlab/py-feat)'s `Detectorv2`. From one face crop
it jointly predicts **action units, categorical emotion, valence/arousal,
eye gaze, a 478-point face mesh, and 6-DoF head pose**.

- **Backbone:** ConvNeXt-V2 Tiny (FCMAE + IN-22k/IN-1k pretrained)
- **Heads:** ME-GraphAU AU graph (AFG/FGG/MEFL/SC) + unified-feature emotion/V-A
  and gaze heads (v2.3 architecture) + landmark and pose regression heads
- **Params:** ~30M · **Input:** 224×224 RGB (from a 256×256 face crop) · **File:** `face_multitask_v1.pt` (174 MB)

## Outputs

| Task | Output | Notes |
|---|---|---|
| Action Units | 24 probabilities [0,1] | AU01,02,04,05,06,07,09,10,11,12,14,15,16,17,18,20,23,24,25,26,27,28,43,45 |
| Emotion | 8-class softmax | Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt |
| Valence / Arousal | 2 × [−1,1] | tanh |
| Gaze | (yaw, pitch) radians | head-centric; yaw+ = right, pitch+ = up |
| Face mesh | 478 × (x,y,z) | MediaPipe topology, chip-pixel coords (z = relative depth) |
| Head pose | (yaw, pitch, roll, tx, ty, tz) | radians / pixels |
| 68 landmarks | derived | dlib-68 subset sampled from the 478 mesh |

## Benchmarks (held-out, file-verified)

| Task | Dataset | Metric | Score |
|---|---|---|---|
| AU | DISFA+ (12-AU, Cheong protocol) | macro-F1 | **0.679** |
| Emotion | RAF-DB official test (7-cls) | acc / macro-F1 | **0.839 / 0.751** |
| Emotion | AffectNet val (7-cls, drop Contempt) | acc / macro-F1 | 0.304 / 0.264 |
| Valence/Arousal | Aff-Wild2 official validation | CCC (V / A) | **0.788 / 0.736** |
| Gaze | MPIIGaze | mean angular err | **3.33°** |
| Gaze | Gaze360 | mean angular err | **5.81°** |

Notes: emotion is strong on RAF-DB / Aff-Wild2 but weaker on AffectNet
(label noise + class imbalance); AffectNet-specific, not architectural.

## Usage

```python
from feat import Detectorv2
detector = Detectorv2(device="cuda")
fex = detector.detect("image.jpg")   # returns a py-feat Fex
```

The model expects a face crop produced by RetinaFace + py-feat's
`extract_face_from_bbox_torch(frame, bbox, face_size=256, expand_bbox=1.2)`,
then center-cropped to 224 and ImageNet-normalized. `Detectorv2` handles this.

## License

**Research / non-commercial use only.** Trained on datasets (AffectNet, DISFA+,
RAF-DB, Aff-Wild2, BP4D, etc.) whose licenses restrict use to academic research.
The ConvNeXt-V2 backbone is MIT-licensed. Confirm each constituent dataset's
terms before any non-research use.
