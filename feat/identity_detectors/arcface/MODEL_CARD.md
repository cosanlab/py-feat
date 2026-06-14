---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
library_name: py-feat
pipeline_tag: image-feature-extraction
license: other
license_name: insightface-non-commercial-research
license_link: https://github.com/deepinsight/insightface/blob/master/LICENSE
---

# ArcFace ResNet50

## Model Description
ArcFace is a face-recognition model trained with an additive angular margin softmax loss that constrains identities to disjoint angular regions of the embedding sphere. The result is much tighter intra-identity clusters under pose and expression variation than prior triplet-loss embeddings (e.g., FaceNet) — exactly the regime FEAT video data hits. This py-feat distribution is the ResNet50 backbone (`w600k_r50.onnx`) from InsightFace's `buffalo_l` pack, trained on WebFace600K, repackaged as a PyTorch `safetensors` checkpoint. Available in py-feat ≥ 0.7 as the default `identity_model='arcface'` for both `Detectorv1` and `MPDetector`. Empirically on `multi_face.jpg`, FaceNet's max off-diagonal cosine similarity between *different* people is 0.76 (false-merging at typical thresholds); ArcFace's is 0.35 (clean separation).

## Model Details
- **Model Type**: Convolutional Neural Network (CNN)
- **Architecture**: IResNet-50 (Improved ResNet, fused-BN form matching InsightFace's ONNX export)
- **Input Size**: 112 x 112 pixels, RGB, pixel range `[0, 1]` (the wrapper rescales to `[-1, 1]`)
- **Output**: 512-dim face embedding (unnormalized; cosine similarity normalizes downstream)
- **Framework**: PyTorch
- **Training data**: WebFace600K — Tsinghua-curated subset of WebFace260M (Zhu et al., 2021)
- **Reported benchmarks**: LFW 99.83 %, IJB-C 96.18 % TAR @ FAR=1e-4 (InsightFace upstream)

## Model Sources
- **Repository (training code, MIT)**: [deepinsight/insightface](https://github.com/deepinsight/insightface)
- **Distribution (`buffalo_l` pack, ONNX source)**: [InsightFace v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip)
- **Paper (ArcFace)**: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **Paper (WebFace260M)**: [WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition](https://arxiv.org/abs/2103.04098)

## Citation
If you use this model in your research or application, please cite the ArcFace paper and the InsightFace project:

J. Deng, J. Guo, N. Xue, S. Zafeiriou. ArcFace: Additive Angular Margin Loss for Deep Face Recognition, CVPR, 2019, arXiv:1801.07698.

```
@inproceedings{deng2019arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4690--4699},
  year={2019}
}
```

```
@misc{insightface2018,
  author={Guo, Jia and Deng, Jiankang and An, Xiang and Yu, Jack},
  title={InsightFace: 2D and 3D Face Analysis Project},
  year={2018--},
  howpublished={\url{https://github.com/deepinsight/insightface}}
}
```

```
@inproceedings{zhu2021webface260m,
  title={WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10492--10502},
  year={2021}
}
```

## Acknowledgements
We thank the InsightFace project (Guo, Deng et al.) for releasing the architecture and training scripts under the MIT license, and Tsinghua University (Zhu et al.) for the WebFace260M / WebFace600K dataset under non-commercial-research terms.

## License Note
The InsightFace **code** is MIT-licensed. The **pretrained weights distributed in this repo** are released under InsightFace's non-commercial-research license, and the underlying training data (WebFace600K / WebFace260M) is also non-commercial-research only. Py-Feat's conversion code and integration are MIT-licensed, but the converted weights inherit the upstream restriction. Commercial users must independently validate license compatibility and may need to substitute a different identity model.

## Example Usage

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from feat.identity_detectors.arcface.arcface_model import ArcFace
from feat.utils.io import get_resource_path

device = 'cpu'
identity_detector = ArcFace(backbone='r50')
arcface_file = hf_hub_download(
    repo_id="py-feat/arcface_r50",
    filename="arcface_r50.safetensors",
    cache_dir=get_resource_path(),
)
identity_detector.net.load_state_dict(load_file(arcface_file), strict=False)
identity_detector.eval()
identity_detector.to(device)

# Forward through a batch of [N, 3, H, W] face crops in [0, 1] range:
# embeddings = identity_detector(face_crops)   # [N, 512] unnormalized
```
