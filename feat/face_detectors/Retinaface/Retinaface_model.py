"""RetinaFace model for py-feat.

Architecture ported from yakhyo/retinaface-pytorch (MIT, Nov 2024):
    https://github.com/yakhyo/retinaface-pytorch

The upstream code uses a clean Conv2dNormActivation primitive and ships
weights for ResNet18/34 backbones. py-feat ships only ResNet34 here:
ResNet34 is faster than ResNet50 (~3.6 GFLOPs vs ~4.1) AND more
accurate at 88.9% WIDERFACE Hard AP (vs ~84% for biubug6's ResNet50,
~55% for img2pose, ~35% for the previously-shipped MobileNet0.25
RetinaFace per Cheong et al, Affective Science 2023).

Adapted to py-feat conventions:
- Uses torchvision.models.resnet34 directly (no vendored backbone)
- Inherits PyTorchModelHubMixin for HuggingFace weight loading
- Returns module-level shapes / names compatible with the wrapper in
  Retinaface_test.py
- Module name `body` for the backbone wrapper (was `fx` upstream),
  module names ClassHead/BboxHead/LandmarkHead (each holding a
  ModuleList) for the heads (was `class_head.class_head` etc.). The
  remap from yakhyo's saved state_dict happens once in
  scripts/prepare_retinaface_r34.py before the weights are uploaded
  to HuggingFace, so the runtime load is a strict match.
"""

from typing import List, Optional, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models import resnet34
from torchvision.models._utils import IntermediateLayerGetter
from huggingface_hub import PyTorchModelHubMixin


class Conv2dNormActivation(nn.Sequential):
    """Conv2d + BatchNorm + (optional) activation. Mirrors yakhyo's primitive
    so saved state_dicts that were trained against this layout load cleanly."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.LeakyReLU,
        dilation: int = 1,
        negative_slope: Optional[float] = None,
        bias: bool = False,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {"inplace": True}
            if negative_slope is not None:
                params["negative_slope"] = negative_slope
            layers.append(activation_layer(**params))
        super().__init__(*layers)


class SSH(nn.Module):
    """Single Stage Headless feature extractor: 3×3, 5×5, 7×7 conv branches
    concatenated then ReLU'd. Identical to upstream."""

    def __init__(self, in_channel: int, out_channels: int) -> None:
        super().__init__()
        assert out_channels % 4 == 0, "Output channel must be divisible by 4."
        leaky = 0.1 if out_channels <= 64 else 0

        self.conv3X3 = Conv2dNormActivation(
            in_channel, out_channels // 2, kernel_size=3, activation_layer=None
        )
        self.conv5X5_1 = Conv2dNormActivation(
            in_channel, out_channels // 4, kernel_size=3, negative_slope=leaky
        )
        self.conv5X5_2 = Conv2dNormActivation(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            activation_layer=None,
        )
        self.conv7X7_2 = Conv2dNormActivation(
            out_channels // 4, out_channels // 4, kernel_size=3, negative_slope=leaky
        )
        self.conv7x7_3 = Conv2dNormActivation(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            activation_layer=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        conv3X3 = self.conv3X3(x)
        conv5X5 = self.conv5X5_2(self.conv5X5_1(x))
        # Note: upstream re-applies conv5X5_1 here as the input to conv7X7_2.
        # That looks like a typo (conv7X7_2 takes the same input as conv5X5_2),
        # but keeping it preserves bit-identical output to the trained weights.
        conv7X7 = self.conv7x7_3(self.conv7X7_2(self.conv5X5_1(x)))
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        return F.relu(out, inplace=True)


class FPN(nn.Module):
    """3-level Feature Pyramid Network with 1×1 lateral and 3×3 merge convs."""

    def __init__(self, in_channels_list: List[int], out_channels: int) -> None:
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0
        self.output1 = Conv2dNormActivation(
            in_channels_list[0], out_channels, kernel_size=1, negative_slope=leaky
        )
        self.output2 = Conv2dNormActivation(
            in_channels_list[1], out_channels, kernel_size=1, negative_slope=leaky
        )
        self.output3 = Conv2dNormActivation(
            in_channels_list[2], out_channels, kernel_size=1, negative_slope=leaky
        )
        self.merge1 = Conv2dNormActivation(
            out_channels, out_channels, kernel_size=3, negative_slope=leaky
        )
        self.merge2 = Conv2dNormActivation(
            out_channels, out_channels, kernel_size=3, negative_slope=leaky
        )

    def forward(self, inputs) -> List[Tensor]:
        inputs = list(inputs.values())
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        upsample3 = F.interpolate(output3, size=output2.shape[2:], mode="nearest")
        output2 = self.merge2(output2 + upsample3)

        upsample2 = F.interpolate(output2, size=output1.shape[2:], mode="nearest")
        output1 = self.merge1(output1 + upsample2)

        return [output1, output2, output3]


class _Head(nn.Module):
    """Base for ClassHead/BboxHead/LandmarkHead. Holds a per-FPN-level ModuleList
    of 1×1 convs and concatenates their outputs along the anchor dimension."""

    def __init__(self, in_channels: int, num_anchors: int, fpn_num: int, out_per_anchor: int) -> None:
        super().__init__()
        self._out_per_anchor = out_per_anchor
        self._convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_anchors * out_per_anchor,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                )
                for _ in range(fpn_num)
            ]
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        outputs = []
        for feature, layer in zip(x, self._convs):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())
        return torch.cat(
            [out.view(out.shape[0], -1, self._out_per_anchor) for out in outputs],
            dim=1,
        )


class ClassHead(_Head):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2, fpn_num: int = 3) -> None:
        # 2 outputs per anchor: face / not-face logits
        super().__init__(in_channels, num_anchors, fpn_num, out_per_anchor=2)


class BboxHead(_Head):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2, fpn_num: int = 3) -> None:
        # 4 outputs per anchor: cx, cy, w, h regression deltas
        super().__init__(in_channels, num_anchors, fpn_num, out_per_anchor=4)


class LandmarkHead(_Head):
    def __init__(self, in_channels: int = 512, num_anchors: int = 2, fpn_num: int = 3) -> None:
        # 10 outputs per anchor: 5 (x, y) keypoints
        super().__init__(in_channels, num_anchors, fpn_num, out_per_anchor=10)


# Default config for the ResNet34 backbone. Exposed as a module-level constant
# so the wrapper, the migration script, and any future code path can share one
# source of truth. Returns layer2/3/4 of resnet34 to match the FPN's expected
# input channel widths [128, 256, 512].
RETINAFACE_R34_CFG = {
    "name": "resnet34",
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 64,  # base; FPN expects [in_channel*2, *4, *8] = [128, 256, 512]
    "out_channel": 128,
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "num_anchors": 2,
}


class RetinaFace(nn.Module, PyTorchModelHubMixin):
    """RetinaFace face detector with 5-keypoint landmarks and per-anchor scores.

    The backbone is torchvision's ResNet34 (ImageNet-pretrained init when
    constructed in training mode; loaded weights overwrite both backbone and
    detection-head parameters at inference time). The trained weights are
    distributed via HuggingFace at ``py-feat/retinaface_r34``.
    """

    def __init__(self, cfg: Optional[dict] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = RETINAFACE_R34_CFG
        self.cfg = cfg

        if cfg["name"] != "resnet34":
            raise ValueError(
                f"This RetinaFace implementation only ships the resnet34 "
                f"backbone; got cfg['name']={cfg['name']!r}. Add a new "
                "branch + bundle weights via scripts/prepare_retinaface_r34.py "
                "to support additional backbones."
            )

        backbone = resnet34(weights=None)
        self.body = IntermediateLayerGetter(backbone, cfg["return_layers"])

        base_in = cfg["in_channel"]
        in_channels_list = [base_in * 2, base_in * 4, base_in * 8]
        out_channels = cfg["out_channel"]
        num_anchors = cfg["num_anchors"]

        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = ClassHead(out_channels, num_anchors=num_anchors, fpn_num=3)
        self.BboxHead = BboxHead(out_channels, num_anchors=num_anchors, fpn_num=3)
        self.LandmarkHead = LandmarkHead(out_channels, num_anchors=num_anchors, fpn_num=3)

    def forward(self, x: Tensor):
        out = self.body(x)
        fpn = self.fpn(out)

        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = self.BboxHead(features)
        classifications = self.ClassHead(features)
        ldm_regressions = self.LandmarkHead(features)

        if self.training:
            return bbox_regressions, classifications, ldm_regressions
        return bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions


def generate_priors(min_sizes, steps, image_size, clip=False, device="cpu"):
    """Generate anchor box centers + sizes (cx, cy, w, h) in normalized coords.

    Returns a tensor of shape [num_priors, 4] on the requested device. The
    wrapper caches this per (image_size, device) tuple so we don't pay the
    Python loop cost on every call."""
    feature_maps = [
        [int(torch.ceil(torch.tensor(image_size[0] / step))),
         int(torch.ceil(torch.tensor(image_size[1] / step)))]
        for step in steps
    ]
    anchors = []
    for k, f in enumerate(feature_maps):
        for i in range(f[0]):
            for j in range(f[1]):
                for min_size in min_sizes[k]:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    cx = (j + 0.5) * steps[k] / image_size[1]
                    cy = (i + 0.5) * steps[k] / image_size[0]
                    anchors.append([cx, cy, s_kx, s_ky])
    out = torch.tensor(anchors, dtype=torch.float32, device=device)
    if clip:
        out.clamp_(min=0, max=1)
    return out


def decode_boxes(loc, priors, variances):
    """Decode batched bbox regression deltas to (x1, y1, x2, y2) in normalized coords.

    loc: [B, A, 4] regression deltas (cx, cy, log(w), log(h))
    priors: [A, 4] anchor (cx, cy, w, h) in normalized coords
    variances: [v_xy, v_wh]

    Returns: [B, A, 4] boxes in (x1, y1, x2, y2) normalized coords."""
    boxes = torch.cat(
        (
            priors[:, :2].unsqueeze(0) + loc[:, :, :2] * variances[0] * priors[:, 2:].unsqueeze(0),
            priors[:, 2:].unsqueeze(0) * torch.exp(loc[:, :, 2:] * variances[1]),
        ),
        dim=2,
    )
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def decode_landmarks(pre, priors, variances):
    """Decode batched 5-point landmark regression deltas to (x, y) in normalized coords.

    pre: [B, A, 10] landmark deltas (5 keypoints × (dx, dy))
    priors: [A, 4]
    variances: [v_xy, v_wh]

    Returns: [B, A, 10] landmarks in (x1,y1, x2,y2, ..., x5,y5) normalized coords."""
    cxcy = priors[:, :2].unsqueeze(0).unsqueeze(2)  # [1, A, 1, 2]
    wh = priors[:, 2:].unsqueeze(0).unsqueeze(2)  # [1, A, 1, 2]
    deltas = pre.view(pre.size(0), -1, 5, 2)  # [B, A, 5, 2]
    landms = cxcy + deltas * variances[0] * wh
    return landms.view(pre.size(0), -1, 10)


def postprocess_retinaface(*args, **kwargs):
    """Importable stub for the v0.6 mobilenet0.25 postprocess function.

    The v0.6 wrapper called this from a per-image Python loop with
    CPU/numpy round-trips on every iteration. The v0.7 ResNet34 wrapper
    (``Retinaface_test.Retinaface``) does anchor decode, NMS, and
    threshold all on-device in a single pass, so the standalone
    postprocess function has no role in the current pipeline.

    The function is preserved as an importable name so external code
    that ``from feat.face_detectors.Retinaface.Retinaface_model import
    postprocess_retinaface`` doesn't break at import time. Restoring
    MobileNet0.25 as a selectable backbone (alongside the current
    ResNet34) is on the post-v0.7 roadmap; when that lands, this
    function will be re-implemented to call the on-device path of the
    revived backbone.
    """
    raise NotImplementedError(
        "postprocess_retinaface is a v0.6 entry point; the v0.7 RetinaFace "
        "integration moved postprocess on-device. Use "
        "feat.face_detectors.Retinaface.Retinaface_test.Retinaface, which "
        "produces per-image lists of [xmin, ymin, xmax, ymax, score] "
        "directly. Restoring a MobileNet0.25 backbone option is planned "
        "post-v0.7."
    )
