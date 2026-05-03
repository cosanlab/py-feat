"""IResNet (Improved ResNet) backbone for ArcFace face recognition.

This module implements the *fused-BN* form of the InsightFace IR-50/100
architecture, matching the structure exported in the ``buffalo_l`` /
``antelopev2`` ONNX packs from
``github.com/deepinsight/insightface/releases``.

The reference InsightFace ``arcface_torch/backbones/iresnet.py`` defines
each ``IBasicBlock`` as ``BN1 -> Conv1 -> BN2 -> PReLU -> Conv2 -> BN3 +
skip``. When exported to ONNX, the BN2/BN3 (and the per-stage downsample
BN, and the stem BN) are folded into the immediately preceding /
following Conv via the standard W' = gamma*W/sigma fold. Only ``bn1``
(which sees the block's *input* and therefore can't be merged with a
predecessor in this graph) survives as a separate node, plus the head's
``bn2`` (post-stage-4) and the final BN1d (``features``).

So the *fused* IBasicBlock is:

    bn1 -> conv1(+bias) -> prelu -> conv2(+bias, may stride) + skip

And the stem is just ``conv1(+bias) -> prelu`` (the would-be stem BN
folded into ``conv1``).

This file matches that fused structure so the ONNX initializer names
(``layer1.0.bn1.weight``, ``layer1.0.conv1.weight``, etc.) map straight
onto our PyTorch parameters with no name surgery.
"""

from typing import List

import torch
import torch.nn as nn


__all__ = ["iresnet50", "iresnet100", "IResNet"]


def _conv3x3(in_c: int, out_c: int, stride: int = 1, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(
        in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=bias
    )


def _conv1x1(in_c: int, out_c: int, stride: int = 1, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=bias)


class IBasicBlock(nn.Module):
    """Fused-BN improved residual block.

    Layout: ``bn1 -> conv1 -> prelu -> conv2 + skip``.

    The bn2/bn3 of the unfused reference architecture are absorbed into
    conv1 and conv2 respectively; the optional downsample is a single
    Conv (its BN folded in too). All Convs carry a bias to absorb the
    folded BN beta term.
    """

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c, eps=1e-5)
        self.conv1 = _conv3x3(in_c, out_c)
        self.prelu = nn.PReLU(out_c)
        self.conv2 = _conv3x3(out_c, out_c, stride=stride)
        if stride != 1 or in_c != out_c:
            self.downsample = _conv1x1(in_c, out_c, stride=stride)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        return out + identity


class IResNet(nn.Module):
    """IR backbone for ArcFace, fused-BN form.

    Args:
        layers: number of blocks per stage (4 stages). IR-50 = ``[3, 4,
            14, 3]``; IR-100 = ``[3, 13, 30, 3]``.
        embedding_size: output feature dimension (typically 512).
        input_size: square input resolution (typically 112).
    """

    def __init__(
        self,
        layers: List[int],
        embedding_size: int = 512,
        input_size: int = 112,
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size

        # Stem: would-be Conv->BN->PReLU collapses to Conv->PReLU after fusion.
        self.conv1 = _conv3x3(3, 64)
        self.prelu = nn.PReLU(64)

        # 4 stages with channel widths 64 -> 64 -> 128 -> 256 -> 512.
        # First block of each stage downsamples (stride=2), or in stage 1
        # also adopts stride=2 so the input 112 -> 7 across 4 stages.
        self.layer1 = self._make_stage(64, 64, layers[0], stride=2)
        self.layer2 = self._make_stage(64, 128, layers[1], stride=2)
        self.layer3 = self._make_stage(128, 256, layers[2], stride=2)
        self.layer4 = self._make_stage(256, 512, layers[3], stride=2)

        # Head: BN -> Dropout(p=0 at inference) -> Flatten -> Linear -> BN1d
        # Spatial size after 4 stride-2 stages on a 112-input is 7.
        feat_spatial = input_size // 16
        flat = 512 * feat_spatial * feat_spatial
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=0.0, inplace=True)
        self.fc = nn.Linear(flat, embedding_size)
        # Final BN1d is *affine* in the InsightFace ONNX (matches their
        # ``arcface_torch`` reference with ``num_features=512, affine=True``).
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-5, affine=True)

    @staticmethod
    def _make_stage(in_c: int, out_c: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [IBasicBlock(in_c, out_c, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(IBasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet50(embedding_size: int = 512, input_size: int = 112) -> IResNet:
    """IR-50: 3-4-14-3 block layout. The standard ArcFace-R50 backbone."""
    return IResNet([3, 4, 14, 3], embedding_size=embedding_size, input_size=input_size)


def iresnet100(embedding_size: int = 512, input_size: int = 112) -> IResNet:
    """IR-100: 3-13-30-3 block layout."""
    return IResNet([3, 13, 30, 3], embedding_size=embedding_size, input_size=input_size)
