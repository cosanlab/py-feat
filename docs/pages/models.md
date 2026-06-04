# Included pre-trained detectors

Below is a list of detectors included in Py-Feat and ready to use. The model names are in the titles followed by the reference publications. Bolded models are defaults.

!!! note
    The face / landmark / AU / emotion / identity models in the sections below are the *swappable components of the modular* **`Detector` (v1)**. **`Detectorv2` (v2)** instead bundles a single multi-task network — see [Detectorv2: the multi-task model](#detectorv2-the-multi-task-model) below and the [two-detector overview](/#two-detectors-detector-and-detectorv2).

You can specify any of these models for use in the `Detector` class by passing in the name as a string, e.g.

```
from feat import Detector

detector = Detector(emotion_model='svm')
```

!!! note
    Models names are case-insensitive: `'resmasknet' == 'ResMaskNet'`

## Face and Facial Pose detection

- **`img2pose`: Face Alignment and Detection via 6DoF, Face Pose Estimation** ([Albiero et al., 2020](https://arxiv.org/pdf/2012.07791v2.pdf)). Performs simultaneous (one-shot) face detection and head pose estimation

## Facial landmark detection

- **`mobilefacenet`: Efficient CNNs for accurate real time face verification on mobile devices** ([Chen et al, 2018](https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf))
- `mobilenet`: Efficient convolutional neural networks for mobile vision applications ([Howard et al, 2017](https://arxiv.org/pdf/1704.04861v1.pdf))
- `pfld`: Practical Facial Landmark Detector by ([Guo et al, 2019](https://arxiv.org/pdf/1902.10859.pdf))

## Action Unit detection

- **`xgb`: XGBoost Classifier model trained on Histogram of Oriented Gradients\*** extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets
- `svm`: SVM model trained on Histogram of Oriented Gradients\*\* extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets

!!! note
    \*For AU07, our `xbg` detector was trained with hinge-loss instead of cross-entropy loss like other AUs as this yielded substantially better detection performance given the labeled data available for this AU. This means that while it returns continuous probability predictions,  these are more likely to appear binary in practice (i.e. be 0 or 1) and should be interpreted as *proportion of decision-trees with a detection* rather than *average decision-tree confidence* like other AU values.

!!! note
    \*\* Our `svm` detector uses the [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC) implementation from `sklearn` and thus returns **binary values** for each AU rather than probabilities. If your use-case requires continuous-valued detections, we recommend the `xgb` detector instead.

## Emotion detection

- **`resmasknet`: Facial expression recognition using residual masking network** by ([Pham et al., 2020](https://ieeexplore.ieee.org/document/9411919))
- `svm`: SVM model trained on Histogram of Oriented Gradients extracted from ExpW, CK+, and JAFFE datasets

## Identity detection

- **`facenet`: FaceNet: A unified embedding for face recognition and clustering ([Schroff et al, 2015](https://arxiv.org/abs/1503.03832))**. Inception Resnet (V1) pretrained on VGGFace2 and CASIA-Webface.

## Detectorv2: the multi-task model

- **`face_multitask_v2`** ([`py-feat/face_multitask_v2`](https://huggingface.co/py-feat/face_multitask_v2)). The single network behind **`Detectorv2`**. A ConvNeXt V2-Tiny backbone with lightweight task heads jointly predicts, from one forward pass: **20 Action Units**, **7-class emotion** (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger), continuous **valence / arousal**, **gaze**, a **478-point 3D MediaPipe FaceMesh**, and **6-DoF head pose**. Face detection is handled by RetinaFace and identity by an optional ArcFace branch (`identity_model='arcface'`, the default). Replacing v1's per-task model chain with one network makes `Detectorv2` substantially faster — especially on single frames — and adds valence/arousal + gaze that v1 does not produce. Input is a 256×256 aligned face crop (center-cropped to 224, ImageNet-normalized). Trained on a mix of public AU, emotion, gaze, and landmark datasets. On py-feat's end-to-end harness it beats the v1 (`xgb` + `resmasknet`) detector on AU (DISFA+ 8-AU F1 0.756 vs 0.588) and RAF-DB emotion (0.849 vs 0.542), and is competitive with OpenFace 3.0. The ArcFace identity weights are non-commercial-research only.

```python
from feat import Detectorv2
detector = Detectorv2(device="cuda")          # downloads face_multitask_v2 on first use
fex = detector.detect("face.jpg", data_type="image")
```

## Visualization & translation models

These linear PLS / PCA models translate between the different feature spaces py-feat works in. They are not face detectors — they take detector outputs (AUs, blendshapes, landmarks) and produce other view-friendly representations. All four are versioned at `py-feat/<repo>` on HuggingFace and downloaded on first use.

- **`bs_to_au`** ([`py-feat/bs_to_au`](https://huggingface.co/py-feat/bs_to_au)). PLS regression mapping the 52 MediaPipe / ARKit blendshapes (`MPDetector` output) to the 20 FACS AU intensities used by py-feat. Lets `MPDetector` surface AU columns alongside its blendshapes. Trained on ~350K paired frames from 10K CelebV-HQ celebrity videos. OOS variance-weighted R² = 0.236 ± 0.008 (3-fold GroupKFold by video). On a DISFA+ benchmark (n=57,150 frames, 12 labeled AUs) this beats both the standard xgb-on-dlib baseline and an alternative MP-mesh→68→xgb path on mean Pearson r (0.51 vs 0.40 vs 0.49).

- **`au_to_landmarks`** ([`py-feat/au_to_landmarks`](https://huggingface.co/py-feat/au_to_landmarks)). PLS regression for the AU → 68-pt dlib-style landmark visualization (`feat.plotting.plot_face`, `feat.plotting.predict`). The repo hosts both the v2 model (default, trained on ~10K CelebV-HQ wild-celebrity videos with pose covariates; OOS R² = 0.794) and the legacy v1 model for backwards-compatible loading via `load_viz_model("pyfeat_aus_to_landmarks")`.

- **`au_to_mesh`** ([`py-feat/au_to_mesh`](https://huggingface.co/py-feat/au_to_mesh)). PLS regression for AU → 478-vertex MediaPipe FaceMesh, consumed by `plot_face_mesh` and `predict_face_mesh`. This is the **478-mesh** visualization for the MediaPipe-mesh detectors — **`Detectorv2` and `MPDetector`**. (`Detector` (v1) emits 68-point landmarks, so its AU visualization uses the 68-point **`au_to_landmarks`** model above, *not* this mesh model.) Output lives in a pose-canonical, Procrustes-aligned, frontalized frame. The **default is `v4`** (`model_version="v4"`), fit from `Detectorv2`'s own predicted mesh on ~733K CelebV-HQ frames in the 20-AU `AU_LANDMARK_MAP['Feat']` space; the neutral mesh is aligned to the canonical MediaPipe face (frontal) and aspect-corrected to natural proportions. Earlier `v2` (original 20-AU) and `v3` (Detectorv2 24-AU) models remain available via `model_version=`.

- **`landmarks68_to_mesh478`** ([`py-feat/landmarks68_to_mesh478`](https://huggingface.co/py-feat/landmarks68_to_mesh478)). PCA-bottleneck linear regression mapping 68-pt dlib landmarks (`Detector` output) to the 478-vertex MP mesh. Used by `predict_mesh_from_dlib68` so users with a `Detector` Fex can render the 3D mesh without a separate `MPDetector` pass. Trained on ~340K paired frames; OOS R² = 0.479 — substantially higher than the AU → mesh model because dlib landmarks share rich spatial information with the MP mesh that 20 AU intensities can't encode.
