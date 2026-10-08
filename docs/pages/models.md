# Included pre-trained detectors

Py-Feat ships a set of **pre-trained models**, all hosted on the [`py-feat` organization on HuggingFace](https://huggingface.co/py-feat) and downloaded to a local cache the first time they're used. Each model below links to its HuggingFace model card for provenance and references. Model-card license labels do not establish all upstream weight or dataset permissions; see the [licensing overview](licensing.md) and the [v1](licensing_v1.md) and [v2](licensing_v2.md) notices. There are two ways to run them — the single multi-task **`Detectorv2`** network, or the modular **`Detectorv1`** pipeline; see the [Py-Feat v2 overview](../index.md#introducing-py-feat-v2) for how to choose between them. Bolded model names are the defaults, and model names are case-insensitive (`'resmasknet'` == `'ResMaskNet'`).

## Detectorv2: the multi-task model

`Detectorv2` runs a single network for the whole pipeline, so there are no components to choose.

**`face_multitask_v2`** ([`py-feat/face_multitask_v2`](https://huggingface.co/py-feat/face_multitask_v2)) is the network behind it: a ConvNeXt V2-Tiny backbone with lightweight task heads that, from one forward pass, jointly predicts **20 Action Units**, **7-class emotion** (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger), continuous **valence / arousal**, **gaze**, a **478-point 3D MediaPipe FaceMesh**, **6-DoF head pose**, and **52 MediaPipe/ARKit blendshapes** (`fex.blendshapes`). It takes a 256×256 aligned face crop (resized to 224, ImageNet-normalized) and ships as a single `.safetensors` file with the model config in the file metadata. [Model card](https://huggingface.co/py-feat/face_multitask_v2) · Weights: **non-commercial research**; see the [v2 notice](licensing_v2.md) for the CC BY-NC 4.0 pretrained backbone and checkpoint-specific distribution questions.

Face detection is handled by **RetinaFace** — `Detectorv2` hardcodes it and takes no `face_model` kwarg. Identity is an optional branch: the default is `identity_model='arcface'`, with `identity_model='facenet'` also supported, or `None` to skip identity embeddings (see [Identity detection](#identity-detection) below for those model cards). Folding v1's per-task chain into one network makes `Detectorv2` substantially faster — especially on single frames — and adds valence/arousal and gaze that v1 doesn't produce. For accuracy and speed comparisons against `Detectorv1`, see the [accuracy](../benchmarks/accuracy.md) and [speed](../benchmarks/Speed.md) benchmarks.

```python
from feat import Detectorv2
detector = Detectorv2(device="cuda")          # downloads face_multitask_v2 on first use
fex = detector.detect("face.jpg", data_type="image")
```

## Detectorv1: modular components

`Detectorv1` is the classic *modular* pipeline: it runs a separate model for each stage — face detection, facial landmarks, Action Units, emotion, identity, and gaze — and you can swap any of them or turn one off by passing `None`. Select a model by name:

```python
from feat import Detectorv1

detector = Detectorv1(emotion_model='svm')
```

The sections below are those swappable components.

### Face detection

- **`retinaface`: RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild** ([Deng et al., 2019](https://arxiv.org/abs/1905.00641)). A single-shot detector (ResNet-34 backbone) that localizes faces and five facial keypoints in one pass — the default since v0.7. [Model card](https://huggingface.co/py-feat/retinaface_r34) · Code and existing model metadata: MIT; WIDER FACE dataset terms are separate.
- `img2pose`: **Face Alignment and Detection via 6DoF Face Pose Estimation** ([Albiero et al., 2020](https://arxiv.org/abs/2012.07791)). Performs face detection **and** 6-DoF head-pose estimation in a single shot. [Model card](https://huggingface.co/py-feat/img2pose) · License: CC BY-NC 4.0 (non-commercial).

The `face_model` kwarg accepts `'retinaface'` or `'img2pose'`. Because `retinaface` only detects faces, when it's selected py-feat estimates **6-DoF head pose** from the detected landmarks with a small pose MLP ([`py-feat/pose_mlp_v2`](https://huggingface.co/py-feat/pose_mlp_v2), distilled from img2pose at ~5° average error); `img2pose` returns head pose directly from its forward pass. (`'retinaface_r34'`, a v0.6 alias, was removed in v0.7 — use `'retinaface'`; the HuggingFace repo keeps the backbone-tagged name `py-feat/retinaface_r34`.)

### Facial landmark detection

All three landmark models use weights adapted from [`cunjian/pytorch_face_landmark`](https://github.com/cunjian/pytorch_face_landmark). A sufficiently explicit upstream grant for these checkpoints has not been established; Py-Feat's MIT license for its own contributions does not establish third-party weight redistribution rights. See the [v1 notice](licensing_v1.md).

- **`mobilefacenet`: Efficient CNNs for accurate real-time face verification on mobile devices** ([Chen et al., 2018](https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf)). [Model card](https://huggingface.co/py-feat/mobilefacenet) · Upstream checkpoint license: unresolved.
- `mobilenet`: **Efficient convolutional neural networks for mobile vision applications** ([Howard et al., 2017](https://arxiv.org/pdf/1704.04861v1.pdf)). [Model card](https://huggingface.co/py-feat/mobilenet) · Upstream checkpoint license: unresolved.
- `pfld`: **Practical Facial Landmark Detector** ([Guo et al., 2019](https://arxiv.org/pdf/1902.10859.pdf)). [Model card](https://huggingface.co/py-feat/pfld) · Upstream checkpoint license: unresolved.

### Action Unit detection

- **`xgb`: XGBoost classifier** trained on Histogram of Oriented Gradients features. The paper supplement names BP4D, BP4D+, DISFA, CK+, and UNBC-McMaster shoulder pain; existing model documentation additionally names Aff-Wild2. The complete checkpoint-specific mixture requires verification; see the [v1 provenance notice](licensing_v1.md). [Model card](https://huggingface.co/py-feat/xgb_au) · Existing model metadata: MIT; training-data and trained-parameter distribution permissions require separate review. For AU07 the `xgb` detector was trained with hinge loss rather than cross-entropy (substantially better given the available labels), so its continuous probabilities tend to look binary in practice and should be read as the *proportion of decision trees that fired* rather than average confidence.
- `svm`: **SVM** trained on the same HOG features and datasets. [Model card](https://huggingface.co/py-feat/svm_au) · Existing model metadata: MIT; training-data and trained-parameter distribution permissions require separate review. It uses scikit-learn's [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), so it returns **binary** values per AU rather than probabilities — use `xgb` if you need continuous detections.

### Emotion detection

- **`resmasknet`: Facial expression recognition using a residual masking network** ([Pham et al., 2020](https://ieeexplore.ieee.org/document/9411919)). [Model card](https://huggingface.co/py-feat/resmasknet) · Upstream project and model metadata: MIT; FER2013 training-data rights are separate.
- `svm`: **SVM** trained on Histogram of Oriented Gradients from ExpW, CK+, and JAFFE. [Model card](https://huggingface.co/py-feat/svm_emo) · Existing model metadata: MIT; training-data and trained-parameter distribution permissions require separate review.

### Identity detection

- **`arcface`: ArcFace: Additive Angular Margin Loss for Deep Face Recognition** ([Deng et al., 2019](https://arxiv.org/abs/1801.07698)). InsightFace's ResNet-50 recognition model (the `buffalo_l` `w600k_r50` weights, trained on WebFace600K), producing 512-d embeddings. Its angular-margin objective disentangles identity from pose and expression far better than triplet-loss embeddings, which makes it the default for clustering identities across video frames. [Model card](https://huggingface.co/py-feat/arcface_r50) · License: code MIT; **pretrained weights are non-commercial research only** (InsightFace terms; the WebFace600K training data is also research-only) — flag this for any commercial deployment.
- `facenet`: **FaceNet: A unified embedding for face recognition and clustering** ([Schroff et al., 2015](https://arxiv.org/abs/1503.03832)). An Inception-ResNet (V1) using the VGGFace2-trained `20180402-114759` checkpoint (`facenet_20180402_114759_vggface2.pth`); CASIA-WebFace corresponds to a separate upstream checkpoint. FaceNet was the v0.6 default, still available via `identity_model='facenet'`. [Model card](https://huggingface.co/py-feat/facenet) · Implementation and model metadata: MIT; selecting this checkpoint does not establish clearance under the separate dataset terms.

### Gaze estimation

- **`l2cs`: L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments** ([Abdelrahman et al., 2022](https://arxiv.org/abs/2203.03339)). A ResNet-based gaze regressor using the **Gaze360** checkpoint (`l2cs_gaze360_resnet50.safetensors`; MPIIFaceGaze has separate upstream checkpoints) that predicts pitch/yaw gaze angles from a face crop — the default gaze model for `Detectorv1`. [Model card](https://huggingface.co/py-feat/l2cs) · Upstream code: MIT; model metadata: other (historically MIT); Gaze360's source license explicitly excludes commercial applications of models trained on its data, and public weight-sharing permission requires separate review. (`Detectorv2` produces gaze directly as one of its multi-task outputs.)

## Model licenses

Py-Feat's own software remains [MIT-licensed](https://github.com/cosanlab/py-feat/blob/main/LICENSE), including commercial use. Third-party source, pretrained weights, fitted parameters, and datasets have separate terms. The following table distinguishes explicit upstream restrictions from existing distribution labels and unresolved grants. A dataset restriction does not automatically relicense all learned parameters, and an MIT code license does not waive an applicable data agreement. Existing valid grants are not revoked by these clarifications.

See the [licensing overview](licensing.md), [v1 notice](licensing_v1.md), [v2 notice](licensing_v2.md), and [dataset register](https://github.com/cosanlab/py-feat/blob/main/LICENSES/DATASETS.md) for source terms and remaining permission questions.

| Model | Code / published label | Weight and data scope |
|---|---|---|
| `retinaface` | MIT implementation; existing model metadata MIT | ResNet34 source: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch/blob/main/LICENSE). WIDER FACE has separate CC BY-NC-ND dataset terms. |
| `img2pose` | **CC BY-NC 4.0** | Preserve the [upstream noncommercial terms](https://github.com/vitoralbiero/img2pose/blob/main/license.md) and attribution. |
| `mobilefacenet`, `mobilenet`, `pfld` | Py-Feat integration MIT; upstream checkpoint grant unresolved | [cunjian/pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark) checkpoints cannot be described as relicensed under Py-Feat's MIT license. |
| `xgb`, `svm` (AU & emotion) | Py-Feat integration MIT; existing model metadata MIT | Training agreements and trained-parameter sharing rights require checkpoint-specific review. This includes fitted PCA/scaler parameters. |
| `resmasknet` | [Upstream project MIT](https://github.com/phamquiluan/ResidualMaskingNetwork/blob/master/LICENSE) | Preserve that notice; FER2013 provenance and image rights are separate. |
| `arcface` | Code MIT; **weights non-commercial research** | [InsightFace policy](https://github.com/deepinsight/insightface#license) covers the `buffalo_l` / `w600k_r50` pretrained weights. |
| `facenet` | [Implementation MIT](https://github.com/timesler/facenet-pytorch/blob/master/LICENSE.md); existing model metadata MIT | Shipped checkpoint is VGGFace2-trained. Data permissions require separate assessment. |
| `l2cs` | [Upstream code MIT](https://github.com/Ahmednull/L2CS-Net/blob/main/LICENSE); model metadata other (historically MIT) | Shipped checkpoint is Gaze360-trained; the source license explicitly excludes commercial applications of trained models. Public weight-sharing scope remains unresolved. |
| `pose_mlp_v2` | Py-Feat implementation MIT; existing model metadata MIT | Distilled from img2pose (**CC BY-NC 4.0**, not BSD-3) on CelebV-HQ. Student-weight permissions require separate assessment. |
| `face_multitask_v2` (`Detectorv2`) | Py-Feat integration MIT; weights designated **non-commercial research** | Pretrained ConvNeXt V2 backbone weights are **CC BY-NC 4.0**. Dataset and teacher terms require additional review. ArcFace is a separate optional download; disabling it does not remove the multitask network's own restrictions. |

The applicability of training agreements, amendments, and renewals must be established for each checkpoint. The original Aff-Wild agreement and agreement chronology remain unresolved. A research-only label does not itself authorize unrestricted public model distribution.
