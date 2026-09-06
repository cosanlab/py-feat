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
blendshapes**.

Current default: **`face_multitask_v28.safetensors`**.

- **Backbone:** ConvNeXt-V2 Tiny (FCMAE + IN-22k/IN-1k pretrained)
- **Heads:** ME-GraphAU AU graph (AFG/FGG/SC) + unified-feature emotion/V-A and
  gaze heads + landmark, pose, and blendshape regression heads
- **Params:** 41.7M (41,694,779 across 373 tensors; the ConvNeXt-V2-Tiny backbone alone is 27.9M) · **Input:** 224×224 RGB (from a 256×256 face crop)
- **Format:** safetensors, with the `ModelV2Config` JSON in the file metadata
- **Weights:** uniform weight average ("model soup") of three training epochs

Older files in this repo (`face_multitask_v2`, `_v26`, `_v27`) are retained so
existing installs keep working. Each py-feat release pins the filename it was
built against — older code cannot construct newer architectures.

## Outputs

| Task | Output | Notes |
|---|---|---|
| Action Units | 20 probabilities [0,1] | AU01,02,04,05,06,07,09,10,11,12,14,15,17,20,23,24,25,26,28,43 |
| Emotion | 7-class softmax | Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger |
| Valence / Arousal | 2 × [−1,1] | tanh |
| Gaze | (yaw, pitch) radians | **RAW convention is y-down**: yaw+ = subject's right (image-left), pitch+ = looking DOWN. `Detectorv2` negates pitch so Fex columns are canonical +up (since py-feat 2.1.1) |
| Face mesh | 478 × (x,y,z) | MediaPipe topology, chip-pixel coords (z = relative depth) |
| Head pose | (pitch, yaw, roll, tx, ty, tz) | radians / pixels; RAW pitch+ = down (img2pose teacher frame); `Detectorv2` outputs canonical +up (since py-feat 2.1.1) |
| 68 landmarks | derived | dlib-68 subset sampled from the 478 mesh |
| Blendshapes | 52 coefficients [0,1] | MediaPipe/ARKit standard names |

## Evaluation protocol

Every benchmark below is **held out from training at the split level**, and
DISFA+ additionally at the **identity** level. The training run reserves 16
splits: `affectnet:val`, `raf_db:test`, `ferplus:{test,val}`,
`meld:{bench,val}`, `afew_va:val`, `aff_wild2:bench`, `aff_wild2_va:val`,
`gaze360:{bench,val}`, `mpii_gaze:test`, `mpii_facegaze:test`,
`columbia_gaze:test`, `ethxgaze:test`, `eyediap:test`. The `disfaplus` and
`mpii_facegaze` sources are excluded outright, and DISFA+ identities appearing
in other corpora are excluded as well.

## Benchmarks

Chip-protocol inference on held-out splits.

| Task | Dataset | Metric | Score |
|---|---|---|---|
| AU | DISFA+ (12-AU, Cheong protocol) | macro-F1 | **0.682** |
| AU | DISFA+ (common-8 subset) | macro-F1 | **0.772** |
| Emotion | AffectNet val (7-cls, drop Contempt) | acc / macro-F1 | **0.628 / 0.625** |
| Emotion | RAF-DB official test (7-cls) | acc / macro-F1 | **0.869 / 0.806** |
| Valence/Arousal | AffectNet val | CCC (V / A) | **0.773 / 0.650** |
| Valence/Arousal | Aff-Wild2 official validation | CCC (V / A) | **0.376 / 0.477** |
| Valence/Arousal | AFEW-VA validation | CCC (V / A) | **0.687 / 0.548** |
| Gaze | Gaze360 (held-out split) | mean angular err | **13.04°** |
| Gaze | MPIIGaze (leave-subject-out) | mean angular err | **8.26°** |
| Gaze | ETH-XGaze (test) | mean angular err | **4.76°** |
| Gaze | Columbia (test) | mean angular err | **3.76°** |
| Gaze | EYEDIAP (test, never trained on) | mean angular err | **10.61°** |

### Cross-tool, end-to-end

Full shipped pipeline (detect → align → predict) on raw frames, scored on the
images all tools processed. AU presence = DISFA+ intensity ≥ 2, prediction ≥ 0.5.

| Benchmark | This model | OpenFace 3.0 | LibreFace |
|---|---:|---:|---:|
| DISFA+ AU, common-8 macro-F1 | **0.774** | 0.732 | 0.492 |
| DISFA+ AU, 12-AU macro-F1 | **0.671** | — (8 AUs only) | 0.397 |
| AffectNet-7 accuracy | **0.632** | 0.587 | 0.458 |
| AffectNet-7 macro-F1 | **0.632** | 0.587 | 0.410 |
| RAF-DB-7 accuracy | **0.880** | 0.673 | 0.746 |
| RAF-DB-7 macro-F1 | **0.815** | 0.586 | 0.580 |

### AU robustness — perturbed DISFA+ (Cheong 2023 protocol)

Same 57,150 aligned DISFA+ crops as the main AU benchmark, with black-bar
occlusion and luminance shifts. The 8-AU column uses the cross-tool common set
(AU01/02/04/06/09/12/25/26).

| Perturbation | 12-AU macro-F1 | common-8 macro-F1 |
|---|---:|---:|
| none (baseline) | 0.685 | 0.774 |
| eyes occluded | 0.542 | 0.667 |
| mouth occluded | 0.538 | 0.632 |
| nose occluded | 0.678 | 0.785 |
| brightened | 0.593 | 0.684 |
| darkened | 0.675 | 0.765 |

Mouth occlusion is the worst case (−0.147 on 12-AU vs baseline).

## Known limitations

- **AU20 is effectively non-functional** (per-AU F1 **0.057** against a 0.682
  macro). AU15 (0.479) and AU06 (0.526) are also well below the macro average.
  Do not rely on lip-stretch (AU20) predictions; treat AU15/AU06 with caution.
- **Gaze pitch is uneven across domains.** Per-axis pitch MAE is 3.06° on
  ETH-XGaze and 7.05° on EYEDIAP, but 5.21° on MPIIGaze with a lower
  prediction/ground-truth correlation (r = 0.755) — frontal, screen-directed
  gaze has the narrowest pitch range and is the weakest case. Yaw is
  consistently stronger than pitch on all three. Validate before depending on
  absolute gaze pitch in a frontal-camera setting.
- **Landmarks are a secondary output.** The 68 points are sampled from the 478
  mesh rather than predicted by a dedicated landmark head, and 300W NME is
  correspondingly weaker than tools with a native 68-point head.
- Trained on posed and in-the-wild adult face imagery; performance on children,
  heavy occlusion, or extreme pose is not characterized.

## Usage

```python
from feat import Detectorv2
detector = Detectorv2(device="cuda")
fex = detector.detect("image.jpg")   # returns a py-feat Fex
```

To pin a specific checkpoint (e.g. an older published file):

```python
detector = Detectorv2(device="cuda",
                      multitask_weights="/path/to/face_multitask_v28.safetensors")
```

The model expects a face crop produced by RetinaFace + py-feat's
`extract_face_from_bbox_torch(frame, bbox, face_size=256, expand_bbox=1.2)`,
then center-cropped to 224 and ImageNet-normalized. `Detectorv2` handles this.

## License

**Research / non-commercial use only.** Trained on datasets (AffectNet, DISFA+,
RAF-DB, Aff-Wild2, AFEW-VA, BP4D, ETH-XGaze, EYEDIAP, etc.) whose licenses
restrict use to academic research. The ConvNeXt-V2 backbone is MIT-licensed.
Confirm each constituent dataset's terms before any non-research use.
