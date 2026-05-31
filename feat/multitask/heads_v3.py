"""v2.3 head architectures.

Design (OpenFace 3 inspired + lessons from v2.2):
  - Unified features = (backbone GAP) ∥ (projected mesh x,y coords)
    → gives heads direct access to landmark geometry without backbone
       needing to encode it
  - GazeHeadV3: 4 FCs (left/right × yaw/pitch), L2CS-Net pattern,
       no eye cross-attention (landmarks already in input)
  - EmotionVAHeadV3: linear 8-way classifier + linear V/A regressor,
       no AU-node attention (avoids v2's AU04-degenerate routing)
  - Cooccurrence head DROPPED entirely (BP4D-skewed priors hurt DISFA+
       transfer in v2 stage 3)

Drop z from mesh: MediaPipe z is depth-rank not metric depth (noisier),
and emotion/gaze are determined by x,y geometry of mouth/brow/eye
contours which are already in 2D. Saves ~30% of proj_lmk params.
"""
from __future__ import annotations

import torch
import torch.nn as nn


N_MESH = 478


class UnifiedFeatures(nn.Module):
    """Builds the unified feature vector for all v3 heads.

    unified = cat( backbone_GAP[B, bb_ch],  proj_lmk(mesh_xy/image_size)[B, lmk_dim] )

    Mesh coords are normalized by image_size before projection — without
    this, lmk_feat magnitude (~65) dominates and downstream tanh/softmax
    saturate, killing V/A and emotion learning.

    Args:
      bb_ch: backbone output channels (768 for ConvNeXt v2 Tiny)
      lmk_dim: projected landmark feature dim (256)
      image_size: input image size in pixels (224); mesh coords assumed
        to be in [0, image_size] before normalization
    Forward:
      X_bb: backbone features [B, bb_ch, H, W]
      mesh: predicted mesh [B, N_MESH, 3]  (we ignore the z dim)
    Returns:
      unified: [B, bb_ch + lmk_dim]
    """

    def __init__(self, bb_ch: int = 768, lmk_dim: int = 256, image_size: int = 224):
        super().__init__()
        self.bb_ch = bb_ch
        self.lmk_dim = lmk_dim
        self.out_dim = bb_ch + lmk_dim
        self.image_size = float(image_size)
        # 478 landmarks × 2 (x,y) = 956 input dims
        self.proj_lmk = nn.Linear(N_MESH * 2, lmk_dim)
        # LayerNorm on each component before concatenation. Without this,
        # ConvNeXt v2 backbone GAP can have std~8 and ranges in the
        # hundreds; downstream tanh/softmax saturate, killing learning.
        self.norm_bb = nn.LayerNorm(bb_ch)
        self.norm_lmk = nn.LayerNorm(lmk_dim)

    def forward(self, X_bb: torch.Tensor, mesh: torch.Tensor) -> torch.Tensor:
        # Backbone GAP — [B, bb_ch, H, W] -> [B, bb_ch]
        bb_gap = X_bb.mean(dim=(-2, -1))
        # Mesh x,y — [B, 478, 3] -> [B, 956]. Normalize pixel coords to ~[0,1].
        B = mesh.shape[0]
        mesh_xy = mesh[:, :, :2].reshape(B, N_MESH * 2) / self.image_size
        lmk_feat = self.proj_lmk(mesh_xy)
        # LayerNorm each component so neither dominates the unified vector
        return torch.cat([self.norm_bb(bb_gap), self.norm_lmk(lmk_feat)], dim=-1)


class GazeHeadV3(nn.Module):
    """L2CS-Net style: 4 independent FCs for left/right × yaw/pitch.

    Aggregate per-axis by mean of left + right (could also concat both
    into the final gaze output for downstream eye-divergence analysis;
    keeping mean to match v2 interface that returns [B, 2] = [yaw, pitch]).
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.left_yaw   = nn.Linear(in_dim, 1)
        self.left_pitch = nn.Linear(in_dim, 1)
        self.right_yaw  = nn.Linear(in_dim, 1)
        self.right_pitch = nn.Linear(in_dim, 1)

    def forward(self, unified: torch.Tensor) -> torch.Tensor:
        yaw   = 0.5 * (self.left_yaw(unified)   + self.right_yaw(unified))
        pitch = 0.5 * (self.left_pitch(unified) + self.right_pitch(unified))
        return torch.cat([yaw, pitch], dim=-1)   # [B, 2]


class EmotionVAHeadV3(nn.Module):
    """Linear 8-way classifier + linear V/A regressor from unified features.

    Returns dict with same keys as v2 EmotionVAHead so the training loop
    doesn't need to special-case.
    """

    def __init__(self, in_dim: int, n_classes: int = 8, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(in_dim, n_classes)
        self.va_head  = nn.Linear(in_dim, 2)

    def forward(self, unified: torch.Tensor) -> dict:
        u = self.dropout(unified)
        return {
            "emotion_logits": self.cls_head(u),
            # V/A in [-1, 1] via tanh — matches v2 convention
            "va": torch.tanh(self.va_head(u)),
        }
