# Included pre-trained detectors

Below is a list of detectors included in Py-Feat and ready to use. The model names are in the titles followed by the reference publications. Bolded models are defaults.

!!! note
    All py-feat models are hosted on the [**`py-feat` organization on HuggingFace**](https://huggingface.co/py-feat) and downloaded automatically to a local cache the first time they are used. Each model below links to its HuggingFace model card, where you can find the upstream code, reference paper, and full license terms. The project homepage is [py-feat.org](https://py-feat.org/).

!!! note
    The face / landmark / AU / emotion / identity / gaze models in the sections below are the *swappable components of the modular* **`Detector` (v1)** (also importable as `Detectorv1`). **`Detectorv2` (v2)** instead bundles a single multi-task network — see [Detectorv2: the multi-task model](#detectorv2-the-multi-task-model) below and the [Py-Feat v2 overview](./intro.md#introducing-py-feat-v2).

You can specify any of these models for use in the `Detector` class by passing in the name as a string, e.g.

```
from feat import Detector

detector = Detector(emotion_model='svm')
```

!!! note
    Models names are case-insensitive: `'resmasknet' == 'ResMaskNet'`

## Face detection

- **`retinaface`: RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild** ([Deng et al., 2019](https://arxiv.org/abs/1905.00641)). A single-shot detector (ResNet-34 backbone) that localizes faces and five facial keypoints in one pass. This is the default face detector as of v0.7. [Model card](https://huggingface.co/py-feat/retinaface_r34) · License: MIT.
- `img2pose`: Face Alignment and Detection via 6DoF, Face Pose Estimation ([Albiero et al., 2020](https://arxiv.org/abs/2012.07791)). Performs simultaneous (one-shot) face detection **and** 6DoF head-pose estimation. [Model card](https://huggingface.co/py-feat/img2pose) · License: CC BY-NC 4.0 (non-commercial).

!!! note
    The `face_model` kwarg accepts `'retinaface'` or `'img2pose'`. (`'retinaface_r34'`, a v0.6 alias, was removed in v0.7 — use `'retinaface'`. The HuggingFace repo keeps the backbone-tagged name `py-feat/retinaface_r34`.)

!!! note
    `retinaface` detects faces only, so when it is selected py-feat estimates **6DoF head pose** from the detected facial landmarks using a small pose MLP ([`py-feat/pose_mlp_v2`](https://huggingface.co/py-feat/pose_mlp_v2)). `img2pose` instead returns head pose directly from its single forward pass.

All three landmark models use weights adapted from [`cunjian/pytorch_face_landmark`](https://github.com/cunjian/pytorch_face_landmark) and are redistributed under Py-Feat's [MIT license](https://github.com/cosanlab/py-feat/blob/master/LICENSE).

- **`mobilefacenet`: Efficient CNNs for accurate real time face verification on mobile devices** ([Chen et al, 2018](https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf)). [Model card](https://huggingface.co/py-feat/mobilefacenet) · License: MIT.
- `mobilenet`: Efficient convolutional neural networks for mobile vision applications ([Howard et al, 2017](https://arxiv.org/pdf/1704.04861v1.pdf)). [Model card](https://huggingface.co/py-feat/mobilenet) · License: MIT.
- `pfld`: Practical Facial Landmark Detector by ([Guo et al, 2019](https://arxiv.org/pdf/1902.10859.pdf)). [Model card](https://huggingface.co/py-feat/pfld) · License: MIT.

## Action Unit detection

- **`xgb`: XGBoost Classifier model trained on Histogram of Oriented Gradients\*** extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets. [Model card](https://huggingface.co/py-feat/xgb_au) · License: MIT.
- `svm`: SVM model trained on Histogram of Oriented Gradients\*\* extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets. [Model card](https://huggingface.co/py-feat/svm_au) · License: MIT.

!!! note
    \*For AU07, our `xbg` detector was trained with hinge-loss instead of cross-entropy loss like other AUs as this yielded substantially better detection performance given the labeled data available for this AU. This means that while it returns continuous probability predictions,  these are more likely to appear binary in practice (i.e. be 0 or 1) and should be interpreted as *proportion of decision-trees with a detection* rather than *average decision-tree confidence* like other AU values.

!!! note
    \*\* Our `svm` detector uses the [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC) implementation from `sklearn` and thus returns **binary values** for each AU rather than probabilities. If your use-case requires continuous-valued detections, we recommend the `xgb` detector instead.

## Emotion detection

- **`resmasknet`: Facial expression recognition using residual masking network** by ([Pham et al., 2020](https://ieeexplore.ieee.org/document/9411919)). [Model card](https://huggingface.co/py-feat/resmasknet) · License: MIT.
- `svm`: SVM model trained on Histogram of Oriented Gradients extracted from ExpW, CK+, and JAFFE datasets. [Model card](https://huggingface.co/py-feat/svm_emo) · License: MIT.

## Identity detection

- **`arcface`: ArcFace: Additive Angular Margin Loss for Deep Face Recognition** ([Deng et al., 2019](https://arxiv.org/abs/1801.07698)). InsightFace's ResNet-50 recognition model (the `buffalo_l` `w600k_r50` weights, trained on WebFace600K), producing 512-d embeddings. Its angular-margin objective disentangles identity from pose and expression far better than triplet-loss embeddings, which makes it the default for clustering identities across video frames. [Model card](https://huggingface.co/py-feat/arcface_r50) · License: code MIT; **pretrained weights are non-commercial research only** (InsightFace terms; the WebFace600K training data is also research-only). Flag this for any commercial deployment.
- `facenet`: FaceNet: A unified embedding for face recognition and clustering ([Schroff et al, 2015](https://arxiv.org/abs/1503.03832)). Inception Resnet (V1) pretrained on VGGFace2 and CASIA-Webface; the v0.6 default, still available via `identity_model='facenet'`. [Model card](https://huggingface.co/py-feat/facenet) · License: MIT.

## Gaze estimation

- **`l2cs`: L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments** ([Abdelrahman et al., 2022](https://arxiv.org/abs/2203.03339)). A ResNet-based gaze regressor (trained on Gaze360 + MPIIFaceGaze) that predicts pitch/yaw gaze angles from a face crop. Default gaze model for both `Detector` and `MPDetector`. [Model card](https://huggingface.co/py-feat/l2cs) · License: MIT.

!!! note
    `MPDetector` can alternatively compute a lightweight **geometric** gaze estimate directly from the MediaPipe iris landmarks (no extra model) via `gaze_model='geometric'`. `Detectorv2` produces gaze as one of the outputs of its single multi-task network — see below.

## Detectorv2: the multi-task model

- **`face_multitask_v2`** ([`py-feat/face_multitask_v2`](https://huggingface.co/py-feat/face_multitask_v2)). The single network behind **`Detectorv2`**. A ConvNeXt V2-Tiny backbone with lightweight task heads jointly predicts, from one forward pass: **20 Action Units**, **7-class emotion** (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger), continuous **valence / arousal**, **gaze**, a **478-point 3D MediaPipe FaceMesh**, **6-DoF head pose**, and **52 MediaPipe/ARKit blendshapes** (`fex.blendshapes`). Input is a 256×256 aligned face crop (resized to 224, ImageNet-normalized). Trained on a mix of public AU, emotion, gaze, and landmark datasets. Weights ship as a single `.safetensors` file (model config in the file metadata). [Model card](https://huggingface.co/py-feat/face_multitask_v2) · License: non-commercial research only.

  - **Face detection** is handled by **RetinaFace** (`Detectorv2` does not take a `face_model` kwarg — it hardcodes `'retinaface'`).
  - **Identity** is an optional branch: the default is `identity_model='arcface'`, and `identity_model='facenet'` is also supported (or `None` to skip identity embeddings). See the [Identity detection](#identity-detection) section above for the model cards and licenses. The ArcFace identity weights are non-commercial-research only.

  Replacing v1's per-task model chain with one network makes `Detectorv2` substantially faster — especially on single frames — and adds valence/arousal + gaze that v1 does not produce. For up-to-date accuracy and speed comparisons against `Detector` (v1) and other toolkits, see the [accuracy](../benchmarks/accuracy.md) and [speed](../benchmarks/Speed.md) benchmarks.

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

## Model licenses

Py-Feat itself is [MIT-licensed](https://github.com/cosanlab/py-feat/blob/master/LICENSE), but **each bundled model carries its own license**, and you must cite and respect the license of every model you use. The repository [`LICENSE`](https://github.com/cosanlab/py-feat/blob/master/LICENSE) file is the authoritative source and links to each upstream license; the table below summarizes it. **Several models are non-commercial only** — validate license compatibility before any commercial deployment.

| Model | License | Notes |
|---|---|---|
| `retinaface` | MIT | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface/blob/master/LICENSE.MIT) |
| `img2pose` | **CC BY-NC 4.0** | Non-commercial ([upstream license](https://github.com/vitoralbiero/img2pose/blob/main/license.md)) |
| `mobilefacenet`, `mobilenet`, `pfld` | MIT | Weights from [cunjian/pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark), redistributed under Py-Feat's MIT license |
| `xgb`, `svm` (AU & emotion) | MIT | Trained by the Py-Feat team |
| `resmasknet` | MIT | [phamquiluan/ResidualMaskingNetwork](https://github.com/phamquiluan/ResidualMaskingNetwork) |
| `arcface` | Code MIT; **weights non-commercial research** | InsightFace code is MIT; the `w600k_r50` weights and WebFace600K training data are research-only |
| `facenet` | MIT | [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch/blob/master/LICENSE.md) |
| `l2cs` | MIT | [Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net) |
| `face_multitask_v2` (`Detectorv2`) | **Non-commercial research** | Inherits the non-commercial restriction of its ArcFace identity branch and research-only training data |
