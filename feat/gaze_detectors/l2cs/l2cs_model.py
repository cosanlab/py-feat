"""L2CS-Net gaze regressor — ported from Ahmednull/L2CS-Net (MIT, 2022).

Architecture
------------
- ResNet backbone (default ResNet50, optionally lighter ResNet18)
- Two parallel FC heads, each producing logits over ``num_bins`` (90)
  classification bins covering [-180°, +180°] at 4°/bin resolution
- Inference: softmax over each head, expected-value over bin centers
  (i.e. ``E[bin_idx] * 4° - 180°``), then convert to radians

Output convention matches MPDetector's existing geometric path:
``gaze_pitch``, ``gaze_yaw`` in radians, head-centric frame.

Reference
---------
Abdelrahman et al., "L2CS-Net: Fine-Grained Gaze Estimation in
Unconstrained Environments", arxiv:2203.03339 (2022).
Original code: https://github.com/Ahmednull/L2CS-Net
License: MIT.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock


_NUM_BINS = 90
_BIN_SIZE_DEG = 4.0  # 360° / 90 bins
_BIN_OFFSET_DEG = 180.0  # bin 0 = -180°, bin 89 = +176°

# ImageNet normalization (matches upstream L2CS).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class L2CS(nn.Module):
    """ResNet backbone + dual classification heads (yaw, pitch)."""

    def __init__(self, block, layers, num_bins: int = _NUM_BINS):
        super().__init__()
        self.inplanes = 64
        self.num_bins = num_bins

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 512 * block.expansion
        self.fc_yaw_gaze = nn.Linear(feat_dim, num_bins)
        self.fc_pitch_gaze = nn.Linear(feat_dim, num_bins)

        # NOTE: upstream defines an `fc_finetune` layer for a fine-tuning
        # head (512+3 -> 3) but it's unused at inference; we keep the
        # attribute for state-dict compatibility when loading upstream
        # checkpoints.
        self.fc_finetune = nn.Linear(feat_dim + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        yaw_logits = self.fc_yaw_gaze(x)
        pitch_logits = self.fc_pitch_gaze(x)
        return yaw_logits, pitch_logits


def l2cs_resnet50(num_bins: int = _NUM_BINS) -> L2CS:
    return L2CS(Bottleneck, [3, 4, 6, 3], num_bins=num_bins)


def l2cs_resnet18(num_bins: int = _NUM_BINS) -> L2CS:
    return L2CS(BasicBlock, [2, 2, 2, 2], num_bins=num_bins)


class L2CSPipeline:
    """Run face-crop -> gaze pitch/yaw (radians) for a batch.

    Inference behavior matches upstream demo:
      1. Resize face crop to 224x224 (BGR->RGB if input is BGR)
      2. ImageNet-normalize: (img/255 - mean) / std
      3. Forward through L2CS
      4. Softmax bin logits, expected-value over bin centers
      5. Convert bins -> degrees -> radians

    Input: a [N, 3, H, W] uint8 tensor (any H, W; will be resized).
    Output: pitch, yaw as float32 numpy arrays of shape [N], in radians.

    This wrapper does NOT do face detection — pass face crops only.

    Bin geometry must match the training-time gaze range:
      - Gaze360: 90 bins x 4° = [-180°, +176°] (full 360° coverage)
      - MPIIGaze: 28 bins x 3° = [-42°, +42°] (constrained screen-viewing
        range); pass ``bin_size_deg=3.0, bin_offset_deg=42.0``.
    """

    def __init__(
        self,
        model: L2CS,
        device: str = "cpu",
        bin_size_deg: float = _BIN_SIZE_DEG,
        bin_offset_deg: float = _BIN_OFFSET_DEG,
    ):
        self.model = model.eval().to(device)
        self.device = device
        # Cache the bin-center tensor. The center of bin k is
        # k * bin_size - bin_offset (in degrees).
        idx = torch.arange(model.num_bins, dtype=torch.float32, device=device)
        self._bin_centers_deg = idx * bin_size_deg - bin_offset_deg  # [num_bins]
        self._imagenet_mean = _IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
        self._imagenet_std = _IMAGENET_STD.to(device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, face_crops: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """face_crops: [N, 3, H, W] in [0, 1] float32 or [0, 255] uint8."""
        if face_crops.dtype == torch.uint8:
            x = face_crops.float() / 255.0
        else:
            x = face_crops.float()
            if x.max() > 1.5:  # likely [0, 255]
                x = x / 255.0
        x = x.to(self.device)
        # Resize to 224 if needed.
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._imagenet_mean) / self._imagenet_std

        yaw_logits, pitch_logits = self.model(x)
        yaw_probs = F.softmax(yaw_logits, dim=1)
        pitch_probs = F.softmax(pitch_logits, dim=1)
        yaw_deg = (yaw_probs * self._bin_centers_deg.unsqueeze(0)).sum(dim=1)
        pitch_deg = (pitch_probs * self._bin_centers_deg.unsqueeze(0)).sum(dim=1)
        yaw_rad = (yaw_deg * math.pi / 180.0).cpu().numpy()
        pitch_rad = (pitch_deg * math.pi / 180.0).cpu().numpy()
        return pitch_rad, yaw_rad
