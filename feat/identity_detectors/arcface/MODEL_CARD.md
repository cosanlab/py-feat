---
license: other
license_name: insightface-non-commercial-research
license_link: https://github.com/deepinsight/insightface/blob/master/LICENSE
tags:
  - face-recognition
  - arcface
  - face-identification
  - face-embeddings
  - pytorch
  - safetensors
  - py-feat
library_name: pytorch
pipeline_tag: image-feature-extraction
language:
  - en
---

# ArcFace ResNet50 (WebFace600K) for py-feat

PyTorch port of the **ArcFace ResNet50** face-recognition model
distributed by InsightFace as part of the
[`buffalo_l`](https://github.com/deepinsight/insightface/tree/master/python-package#use-the-buffalo_l-pack)
pack (file `w600k_r50.onnx`). Repackaged as a `.safetensors` file for
use as a drop-in identity detector inside
[`py-feat`](https://github.com/cosanlab/py-feat) ≥ 0.7.

This repo distributes **only** the recognition (face-embedding) model
from `buffalo_l`. The face-detection / landmarks models in `buffalo_l`
are not used by py-feat (we have our own RetinaFace-R34 detector).

## Quick start

```python
from feat import Detector

# ArcFace is the default identity model in py-feat ≥ 0.7
detector = Detector()                                # implicitly uses arcface
# or explicitly:
detector = Detector(identity_model="arcface")

fex = detector.detect("video.mp4")
# fex.identity_embeddings is a [n_faces, 512] embedding table.
# fex.Identity is a connected-components cluster label per face.
```

## Model details

| Property | Value |
|---|---|
| Architecture | IResNet-50 (Improved ResNet, fused-BN form) |
| Parameters | 43.6 M |
| Input | RGB face crop, 112×112, pixel range `[0, 1]` (the wrapper rescales to `[-1, 1]`) |
| Output | 512-dim embedding vector (L2-normalized at use time, not at output) |
| File size | 166 MB (safetensors, fp32) |
| Loss | ArcFace additive angular margin softmax (Deng et al., 2019) |
| Training data | WebFace600K — Tsinghua's curated subset of WebFace260M (≈600K identities) |
| Trainer / source | InsightFace (Guo, Deng et al.) — from `buffalo_l` v0.7 release |

## Reported benchmarks

Benchmark numbers below are *as reported by InsightFace* on the original
ONNX model. The PyTorch port we ship is verified bit-equivalent (max
absolute output difference 5.9e-6, per-row cosine similarity = 1.0 on
random inputs), so these numbers carry over.

| Benchmark | Score |
|---|---|
| LFW (verification accuracy) | 99.83 % |
| CFP-FP | 98.86 % |
| AgeDB-30 | 98.13 % |
| **IJB-C** (TAR @ FAR = 1e-4) | **96.18 %** |
| IJB-B (TAR @ FAR = 1e-4) | 95.05 % |

For the FEAT use case (clustering identities across video frames with
varied pose and expression), IJB-C is the most relevant benchmark. The
prior default (FaceNet/VGGFace2) reaches roughly 80 % TAR @ FAR=1e-4 on
IJB-C — ArcFace is ~16 percentage points better, which is the gap
that drives FEAT's clustering quality lift.

## In py-feat: empirical comparison

On `multi_face.jpg` (5 different people, retinaface_r34 face crops):

| Identity model | Min off-diag pairwise cos | Max off-diag pairwise cos | Mean off-diag pairwise cos |
|---|---|---|---|
| FaceNet (prior default) | -0.07 | **0.76** | 0.23 |
| ArcFace R50 (this model) | 0.05 | **0.35** | 0.13 |

At a typical clustering threshold of 0.5, the FaceNet's max cross-similarity
of 0.76 falsely merges two different people into one identity. ArcFace's
max of 0.35 leaves comfortable margin and keeps all 5 identities distinct.

Inference cost change is small: 13.0 → 13.5 ms/frame (+4 %) on M5 MBP MPS,
batch=16, full retinaface_r34 + svm AU + identity pipeline on a 472-frame
video. ArcFace's larger 43.6M-param IR-50 backbone is offset by its
smaller 112×112 input (vs. FaceNet's 160×160).

## Intended use

- Face-identity clustering in videos / image batches for affective-science
  research, where multiple subjects need to be tracked across frames.
- Retrieval / verification (cosine similarity over the 512-dim embedding).
- Building feature tables that downstream analyses can group on.

### Out of scope

- **Surveillance / law-enforcement identification** of unconsented subjects.
  The training data (WebFace600K) was assembled by web scraping; subjects
  did not opt in to face-recognition training. Use that informs your
  consent / IRB process accordingly.
- **High-stakes identity verification** (e.g., access control, financial
  authentication) where false-positive costs are real. Demographic
  fairness was not measured by us; the InsightFace authors document
  some cross-group disparity.
- **Biometric identification of children.** WebFace600K is dominated by
  adult subjects.

## Limitations and biases

- **Demographic bias.** Web-scraped face datasets like WebFace260M /
  WebFace600K are known to over-represent young / Asian / male subjects
  relative to the global population. Verification accuracy varies
  measurably across demographic groups; downstream studies should
  validate per-group performance before publishing claims.
- **Pose extremes.** The model was trained primarily on ~frontal faces.
  Profile views (yaw > 60°) produce noticeably noisier embeddings.
- **Occlusion / image quality.** Heavily occluded faces (masks, hands,
  partial visibility) and very small / blurry crops will return
  embeddings, but those embeddings should be treated as low-confidence.
  py-feat has no built-in quality filtering yet (planned for a future
  release; see `docs/superpowers/specs/2026-05-02-identity-detection-roadmap.md`
  in the py-feat repo).
- **Children.** Performance on faces under ~12 years old is not
  characterized.

## License

This model is distributed under the **InsightFace non-commercial
research license**:
[https://github.com/deepinsight/insightface/blob/master/LICENSE](https://github.com/deepinsight/insightface/blob/master/LICENSE).

Two layers to be aware of:

1. **Model code (architecture and training scripts):** MIT-licensed by
   InsightFace.
2. **Pretrained weights (this repo's `.safetensors`):** Non-commercial
   research only. The underlying training data (WebFace600K, derived
   from WebFace260M assembled by Tsinghua University) is also research-
   only.

py-feat itself is MIT-licensed, but the MIT license does **not** override the
upstream weight license. Commercial users must independently validate
that their use case is compatible with the InsightFace and WebFace260M
terms — py-feat does not grant you commercial rights to these weights.

## Reproducing the conversion

```bash
# 1. Pull the buffalo_l pack (~275 MB) from InsightFace's GitHub release
curl -L -o /tmp/buffalo_l.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -o /tmp/buffalo_l.zip w600k_r50.onnx -d /tmp/

# 2. Convert + verify against the ONNX runtime output
python scripts/convert_arcface_onnx_to_safetensors.py \
    --onnx /tmp/w600k_r50.onnx \
    --out  /tmp/arcface_r50.safetensors \
    --backbone r50 \
    --verify
# Expected: max |diff| < 1e-5, cosine similarity = 1.0 across two random inputs.
```

The converter walks the ONNX graph in topological order to map Conv
weights, Conv biases (which are stored under numeric ONNX ids), PReLU
slopes, and BN params onto the matching PyTorch parameter names. The
fused-BN ONNX export collapses most BatchNorm layers into the adjacent
Conv via `W' = γW/σ`; the architecture in
`feat/identity_detectors/arcface/iresnet.py` mirrors that fused
structure so the load is bit-exact (modulo float32 rounding).

## Citation and attribution

If you use this model, please cite both the **method paper** and the
**InsightFace project** (which trained and distributes the weights):

```bibtex
@inproceedings{deng2019arcface,
  title     = {ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author    = {Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4690--4699},
  year      = {2019}
}
```

```bibtex
@misc{insightface2018,
  author  = {Guo, Jia and Deng, Jiankang and An, Xiang and Yu, Jack},
  title   = {InsightFace: 2D and 3D Face Analysis Project},
  year    = {2018--},
  howpublished = {\url{https://github.com/deepinsight/insightface}}
}
```

The training dataset (WebFace260M / WebFace600K) is from Tsinghua
University:

```bibtex
@inproceedings{zhu2021webface260m,
  title     = {WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition},
  author    = {Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10492--10502},
  year      = {2021}
}
```

The IResNet (Improved ResNet) backbone follows the deep residual
learning framework:

```bibtex
@inproceedings{he2016deep,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016}
}
```

If you cite py-feat itself:

```bibtex
@article{cheong2023pyfeat,
  title     = {Py-Feat: Python Facial Expression Analysis Toolbox},
  author    = {Cheong, Jin Hyun and Jolly, Eshin and Xie, Tiankang and Byrne, Sophie and Kenney, Matthew and Chang, Luke J.},
  journal   = {Affective Science},
  year      = {2023},
  publisher = {Springer},
  doi       = {10.1007/s42761-023-00191-4}
}
```

## Provenance summary

| Layer | Authors / source | License |
|---|---|---|
| ArcFace loss | Deng, Guo, Xue, Zafeiriou (2019) | Method paper, freely usable |
| IResNet architecture | He et al. (2016) + InsightFace's fused-BN export | Method paper / MIT (InsightFace code) |
| Pretrained weights | InsightFace, trained on WebFace600K | Non-commercial research |
| Training data (WebFace260M / 600K) | Zhu et al. / Tsinghua University (2021) | Non-commercial research |
| ONNX → safetensors conversion | py-feat maintainers | MIT (the script) — but the weights it produces inherit the upstream license |
| py-feat integration code | cosanlab / py-feat contributors | MIT |

## Contact

For issues with the **conversion / py-feat integration**:
[github.com/cosanlab/py-feat/issues](https://github.com/cosanlab/py-feat/issues).

For issues with the **underlying model** (training, accuracy, behavior):
[github.com/deepinsight/insightface/issues](https://github.com/deepinsight/insightface/issues).
