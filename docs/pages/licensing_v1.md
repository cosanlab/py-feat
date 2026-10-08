<!-- Mirrored from LICENSES/PYFEAT-V1.md; keep the notices synchronized. -->

# Py-Feat v1.0 licensing

This document covers the classic model family described by Cheong et al.
(2023), now exposed as `Detectorv1`. The family name is not a claim that a
PyPI package version 1.0 was released. Reviewed October 8, 2026.

## Software

Py-Feat's original source code and documentation are licensed under the
[MIT License](https://github.com/cosanlab/py-feat/blob/main/LICENSE), including commercial use. Third-party source
files retain their own copyright notices and licenses. This notice does
not add a research-only restriction to the software.

## Models

There is no single license covering every `Detectorv1` configuration.
Downloaded weights, fitted PCA/scaler/PLS parameters, datasets, and software
must be considered separately. The repository's MIT license does not grant
rights that belong to model or dataset providers. Existing valid license
grants are not revoked by this clarification.

| Component / Hugging Face repository | Software / upstream notice | Weight-specific position |
|---|---|---|
| `retinaface`, `retinaface_r34` | [RetinaFace MIT](https://github.com/biubug6/Pytorch_Retinaface/blob/master/LICENSE.MIT) | Existing model cards declare MIT; WIDER FACE data has separate noncommercial terms. A code license alone does not establish all weight rights. |
| `img2pose` | [CC BY-NC 4.0](https://github.com/vitoralbiero/img2pose/blob/main/license.md) | Preserve the upstream noncommercial license and attribution; not MIT or BSD-3. |
| `mobilefacenet`, `mobilenet`, `pfld` | [Upstream implementation and checkpoints](https://github.com/cunjian/pytorch_face_landmark) | A sufficiently explicit upstream weight redistribution grant has not been established in this review. Do not describe these as relicensed under Py-Feat's MIT license. |
| `xgb_au`, `svm_au` | Py-Feat integration: MIT | Previously labeled MIT; training agreements require separate review of trained-parameter redistribution. The MIT label is not a grant of dataset-provider rights. |
| `svm_emo` | Py-Feat integration: MIT | Previously labeled MIT; ExpW, CK+, and JAFFE training-data permissions must be considered separately. |
| `resmasknet` | [ResidualMaskingNetwork MIT](https://github.com/phamquiluan/ResidualMaskingNetwork/blob/master/LICENSE) | Preserve the upstream MIT notice; FER2013 provenance is separate from the implementation license. |
| `facenet` | [facenet-pytorch MIT](https://github.com/timesler/facenet-pytorch/blob/master/LICENSE.md) | The shipped checkpoint uses VGGFace2. Its data terms must be assessed separately; selecting FaceNet does not by itself establish commercial clearance. |
| `arcface_r50` | [InsightFace code MIT; pretrained-model terms](https://github.com/deepinsight/insightface#license) | Pretrained weights are for noncommercial research. Conversion to safetensors does not change that. |
| `l2cs` | [L2CS-Net MIT](https://github.com/Ahmednull/L2CS-Net/blob/main/LICENSE) | The shipped Gaze360 checkpoint has an additional source license expressly excluding commercial applications of models trained on Gaze360; sharing scope needs review. MPIIFaceGaze is a separate checkpoint configuration. |
| `pose_mlp_v2` | Py-Feat implementation: MIT | Distilled from img2pose using CelebV-HQ. The previous card's BSD-3 teacher claim was incorrect. Student-weight permissions require their own assessment. |

The paper-era FaceBoxes and MTCNN options have their own
[3DDFA_V2](https://github.com/cleardusk/3DDFA_V2/blob/master/LICENSE) and
[MTCNN](https://github.com/ipazc/mtcnn/blob/master/LICENSE) source notices.
They are historical components, not additional current `Detectorv1` options.

## Training, validation, and visualization

The [2023 paper and supplementary materials](https://doi.org/10.1007/s42761-023-00191-4)
describe these distinct uses:

| Artifact | Training | Validation / evaluation |
|---|---|---|
| Classic SVM and XGBoost AU detectors | The supplement explicitly names BP4D, BP4D+, DISFA, CK+, and UNBC-McMaster Shoulder Pain; current model documentation additionally names Aff-Wild2. | Three-fold cross-validation for tuning; DISFA+ for held-out AU evaluation. |
| Emotion SVM | ExpW, CK+, JAFFE | Three-fold cross-validation for tuning; separate paper benchmarks. JAFFE is training data for this artifact. |
| Original AU-to-68-landmark visualization PLS | EmotioNet, BP4D, **DISFA+** | Three-fold cross-validation. DISFA+ is not globally evaluation-only across Py-Feat artifacts. |
| Later `au_to_landmarks`, `bs_to_au`, and `landmarks68_to_mesh478` mappings | CelebV-HQ with detector/teacher-generated targets | See the exact checkpoint's model card; these are not the original paper's PLS weights. |

The supplement's AU-training sentence ends with references 99-101 without
a dataset name, and its training-dataset section also describes EmotioNet.
The exact AU classifier mixture therefore requires a checkpoint-specific
manifest; the list above must not be read as exhaustive. Separate training
runs and file-format conversions must
retain their own provenance. For dataset conditions and unresolved rights,
see [the dataset register](licensing_datasets.md).

## Distribution status

The reviewed agreements do not establish unrestricted public redistribution
or commercial use for every trained artifact above. In particular, the
Aff-Wild2 agreements and the BP4D/BP4D+ parameter-set clauses require review
against the relevant training date and checkpoint. Provider permissions,
amendments, and renewals must be checked alongside the original agreements.

This document records scope and provenance; it is not a new sublicense for
third-party datasets or weights and does not assert that dataset terms
automatically attach to every trained model or inference output.
