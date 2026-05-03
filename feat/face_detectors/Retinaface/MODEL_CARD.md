---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
library_name: py-feat
pipeline_tag: object-detection
license: mit
---

# RetinaFace ResNet34

## Model Description
RetinaFace is a single-shot face detector that jointly predicts bounding boxes, 5-keypoint landmarks (eyes, nose, mouth corners), and a face/no-face score per anchor. This py-feat distribution uses a ResNet34 backbone trained on WIDERFACE, reaching 88.9% AP on WIDERFACE-Hard versus img2pose's 55.5% (per Cheong et al., Affective Science 2023). Postprocessing (priors, decode, NMS) runs fully batched on-device via `torchvision.ops.batched_nms`. Available in py-feat ≥ 0.7 as `Detector(face_model='retinaface_r34')` and `MPDetector(face_model='retinaface')`.

## Model Details
- **Model Type**: Convolutional Neural Network (CNN), single-shot face detector
- **Architecture**: ResNet34 backbone + Feature Pyramid Network + SSH context modules + 3 heads (Class / Bbox / 5-keypoint Landmark)
- **Input Size**: any spatial resolution (anchors generated per (H, W) and cached)
- **Framework**: PyTorch
- **Training data**: WIDERFACE (Yang et al., 2016)

## Model Sources
- **Repository (port used by py-feat)**: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch)
- **Repository (original PyTorch reference)**: [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- **Paper**: [RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)

## Citation
If you use this model in your research or application, please cite the following paper:

J. Deng, J. Guo, E. Ververas, I. Kotsia, S. Zafeiriou. RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild, CVPR, 2020, arXiv:1905.00641.

```
@inproceedings{deng2020retinaface,
  title={RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
  author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={5203--5212},
  year={2020}
}
```

## Acknowledgements
We thank Yakhyokhuja Valikhujaev for the ResNet34-backbone PyTorch implementation, biubug6 for the original PyTorch reference, and the WIDERFACE authors (Yang, Luo, Loy, Tang) for the training data — all distributed under permissive terms.

## Example Usage

```python
import torch
from feat.face_detectors.Retinaface.Retinaface_test import Retinaface
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from feat.utils.io import get_resource_path

device = 'cpu'
detector = Retinaface(device=device)  # py-feat helper that lazy-fetches weights from py-feat/retinaface_r34

# Or, to load weights directly:
weights_file = hf_hub_download(
    repo_id="py-feat/retinaface_r34",
    filename="model.safetensors",
    cache_dir=get_resource_path(),
)
state_dict = load_file(weights_file)
# ... pass to your own RetinaFace module instance.
```
