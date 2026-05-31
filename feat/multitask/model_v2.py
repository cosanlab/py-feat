"""ME-GraphAU v2 — multi-task face detector for py-feat.

Key differences from v1 (model.py):
  - ConvNeXt v2 Tiny backbone (FCMAE+IN22k pretrained, +1.4% IN-1k vs v1)
  - 24 AUs (added AU16, 18, 27, 45)
  - EmotionVAHead — 8-class emotion + V/A regression
  - GazeHead — eye-region landmark cross-attention + AU-eye gating + dual-eye 2-node GCN
  - MEFL identity-init (FAM/ARM output projection zero) to eliminate v1's
    stage-3 transition shock

All other modules (AFG, FGG, MEFL graph topology, LandmarkHead, PoseHead,
SCHead, CooccurrenceHead) carry over from v1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# 24 AUs for v2 (v1's 20 + AU16, AU18, AU27, AU45)
AU_NAMES = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU16", "AU17", "AU18", "AU20",
    "AU23", "AU24", "AU25", "AU26", "AU27", "AU28", "AU43", "AU45",
]
N_AU = 24
N_MESH = 478
N_EMOTION = 8     # Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt

# MediaPipe 478 mesh: eye-region vertex indices for the GazeHead.
# Conservative set: eyelid contours + iris (10 vertices total) per eye.
MP_LEFT_EYE_IDX = list(range(33, 47)) + list(range(133, 145)) + list(range(469, 473))
MP_RIGHT_EYE_IDX = list(range(263, 277)) + list(range(362, 374)) + list(range(473, 477))

# AU indices that gate gaze (eye-related AUs in our 24-AU set)
# AU05 idx=3, AU07 idx=5, AU09 idx=6, AU43 idx=22, AU45 idx=23
GAZE_GATE_AU_IDX = [3, 5, 6, 22, 23]


@dataclass
class ModelV2Config:
    backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    image_size: int = 224
    n_au: int = N_AU
    n_mesh: int = N_MESH
    n_emotion: int = N_EMOTION
    knn_k: int = 4
    afg_channels: int = 512
    mefl_channels: int = 512
    gcn_layers: int = 2
    drop_path_rate: float = 0.1
    pretrained: bool = True
    emotion_head_dropout: float = 0.25
    use_dual_eye_gcn: bool = True
    mefl_identity_init: bool = False  # legacy; kept for ckpt-cfg back-compat
    mefl_use_layer_scale: bool = True  # gate MEFL contribution via α (init 1e-2)
    use_head_v2: bool = False   # v2.1: enables EmotionVAHeadV2 + GazeHeadV2
    use_head_v3: bool = False   # v2.3: unified features + simpler heads, drops cooccur
    unified_lmk_dim: int = 256  # v2.3: projected landmark feature dim
                                #   - EmotionVAHeadV2: stronger backbone GAP path + 2-trunk fusion
                                #   - GazeHeadV2: backbone GAP + landmark cross-attn, NO AU gating,
                                #     separate yaw/pitch heads (L2CS-Net pattern)


# ============================ ANFL (unchanged from v1) ============================

class AFG(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_au: int):
        super().__init__()
        self.n_au = n_au
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_au)
        ])

    def forward(self, x: torch.Tensor):
        U_list = [head(x) for head in self.heads]
        U = torch.stack(U_list, dim=1)
        v = U.mean(dim=(-2, -1))
        return U, v


class FGG(nn.Module):
    def __init__(self, ch: int, k: int):
        super().__init__()
        self.k = k
        self.W = nn.Linear(ch, ch, bias=False)
        self.W_self = nn.Linear(ch, ch, bias=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        B, N, C = v.shape
        v_norm = F.normalize(v, dim=-1)
        sim = torch.bmm(v_norm, v_norm.transpose(1, 2))
        topk = sim.topk(self.k + 1, dim=-1).indices
        adj = torch.zeros_like(sim)
        adj.scatter_(2, topk, 1.0)
        eye = torch.eye(N, device=v.device).unsqueeze(0)
        adj = adj * (1 - eye)
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        adj = adj / deg
        msg = self.W(v)
        agg = torch.bmm(adj, msg)
        out = F.relu(self.W_self(v) + agg)
        return out


# ============================ MEFL (with identity-init for v2) ============================

class CrossAttention(nn.Module):
    """Cross-attention with Kaiming-normal init (matches original ME-GraphAU
    paper, model/graph_edge_model.py).

    NOTE: The v2-original `zero_init_output=True` path created a dead fixed
    point — proj=0 zeroes ∂L/∂(q,k,v) (chain rule), so q/k/v stayed at init
    and proj never moved off zero (weight decay dragged it back faster than
    the tiny ∂L/∂proj could grow it). Diagnosed in
    analysis/v2/diag/FINDINGS.md. Now using normal init; MEFL contribution
    is ramped in smoothly via the LayerScale α in MEFL.forward."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (ch // heads) ** -0.5
        self.q = nn.Conv2d(ch, ch, 1, bias=False)
        self.k = nn.Conv2d(ch, ch, 1, bias=False)
        self.v = nn.Conv2d(ch, ch, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)
        # PyTorch's default Kaiming-uniform on Conv2d is fine — no override.

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        B, C, H, W = q_in.shape
        Q = self.q(q_in).view(B, self.heads, C // self.heads, H * W)
        K = self.k(kv_in).view(B, self.heads, C // self.heads, H * W)
        V = self.v(kv_in).view(B, self.heads, C // self.heads, H * W)
        attn = torch.einsum("bhdn,bhdm->bhnm", Q, K) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, V)
        out = out.reshape(B, C, H, W)
        return self.proj(out)


class GatedGCNLayer(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.U = nn.Linear(ch, ch)
        self.V = nn.Linear(ch, ch)
        self.A = nn.Linear(ch, ch)
        self.B = nn.Linear(ch, ch)
        self.C = nn.Linear(ch, ch)
        self.ln_h = nn.LayerNorm(ch)
        self.ln_e = nn.LayerNorm(ch)

    def forward(self, h: torch.Tensor, e: torch.Tensor):
        B, N, C = h.shape
        Ah = self.A(h).unsqueeze(2).expand(-1, -1, N, -1)
        Bh = self.B(h).unsqueeze(1).expand(-1, N, -1, -1)
        Ce = self.C(e)
        e_msg = F.relu(self.ln_e(Ah + Bh + Ce))
        e_new = e + e_msg
        sigmoid_e = torch.sigmoid(e_new)
        Vh = self.V(h).unsqueeze(1).expand(-1, N, -1, -1)
        aggregated = (sigmoid_e * Vh).sum(dim=2)
        Uh = self.U(h)
        h_msg = F.relu(self.ln_h(Uh + aggregated))
        h_new = h + h_msg
        return h_new, e_new


class MEFL(nn.Module):
    """Multi-dim Edge Feature Learning (Luo et al. IJCAI 2022).

    FAM/ARM use Kaiming-normal init (paper convention). A LayerScale α
    (init=0.01, no weight decay) gates MEFL's contribution into the edge
    tensor so we ramp smoothly without the v2-era identity-init dead trap.
    Set use_layer_scale=False to recover the paper-exact behavior.
    """

    def __init__(self, ch: int, n_au: int, gcn_layers: int,
                 use_layer_scale: bool = True, layer_scale_init: float = 1e-2):
        super().__init__()
        self.n_au = n_au
        self.fam = CrossAttention(ch, heads=4)
        self.arm = CrossAttention(ch, heads=4)
        self.gcn = nn.ModuleList([GatedGCNLayer(ch) for _ in range(gcn_layers)])
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # Single learnable scalar α per CrossAttention output. α starts at
            # 0.01 (MEFL contributes ~1% at init), grows under gradient if
            # MEFL helps. Excluded from weight decay (see train_v2_stage3.py).
            self.alpha_fam = nn.Parameter(torch.full((), layer_scale_init))
            self.alpha_arm = nn.Parameter(torch.full((), layer_scale_init))

    def reinit_proj(self, std_fan_in: bool = True):
        """Re-initialize FAM/ARM proj weights with Kaiming-normal.
        Call after loading a stage-2 checkpoint where these were zeroed by
        the legacy identity-init path."""
        for m in (self.fam, self.arm):
            ch = m.proj.weight.shape[0]
            fan_in = ch  # 1x1 conv
            std = math.sqrt(2.0 / fan_in) if std_fan_in else 0.02
            nn.init.normal_(m.proj.weight, mean=0.0, std=std)
            # Also re-init q/k/v if they're at PyTorch's default (they will
            # have been stuck at init since proj=0 killed gradient to them).
            for sub in (m.q, m.k, m.v):
                nn.init.normal_(sub.weight, mean=0.0, std=std)

    def forward(self, U: torch.Tensor, X: torch.Tensor, v_fgg: torch.Tensor):
        B, N, C, H, W = U.shape
        U_flat = U.reshape(B * N, C, H, W)
        X_rep = X.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W)
        F_AS = self.fam(U_flat, X_rep)
        if self.use_layer_scale:
            F_AS = self.alpha_fam * F_AS
        F_AS = F_AS.reshape(B, N, C, H, W)

        e = torch.zeros(B, N, N, C, device=U.device, dtype=U.dtype)
        for i in range(N):
            Fi = F_AS[:, i].unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W)
            Fj = F_AS.reshape(B * N, C, H, W)
            F_AR = self.arm(Fj, Fi)
            if self.use_layer_scale:
                F_AR = self.alpha_arm * F_AR
            e_i = F_AR.mean(dim=(-2, -1)).view(B, N, C)
            e[:, i, :, :] = e_i

        h = v_fgg
        for layer in self.gcn:
            h, e = layer(h, e)
        return h, e


# ============================ Heads ============================

class SCHead(nn.Module):
    """Per-AU presence prob via cosine similarity to learnable prototypes."""

    def __init__(self, n_au: int, ch: int):
        super().__init__()
        self.s = nn.Parameter(torch.randn(n_au, ch) * 0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        s = F.relu(self.s)
        h = F.relu(h)
        h_norm = F.normalize(h, dim=-1)
        s_norm = F.normalize(s, dim=-1)
        cos = (h_norm * s_norm.unsqueeze(0)).sum(dim=-1)
        return cos.clamp(0.0, 1.0)


class CooccurrenceHead(nn.Module):
    def __init__(self, ch: int, n_classes: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * ch, ch),
            nn.ReLU(inplace=True),
            nn.Linear(ch, n_classes),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        e_concat = torch.cat([e, e.transpose(1, 2)], dim=-1)
        return self.fc(e_concat)


class LandmarkHead(nn.Module):
    def __init__(self, in_ch: int, out: int = N_MESH * 3, hidden: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.mean(dim=(-2, -1))
        return self.mlp(z).view(-1, N_MESH, 3)


class PoseHead(nn.Module):
    def __init__(self, in_ch: int, out: int = 6, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.mean(dim=(-2, -1)))


class EmotionVAHead(nn.Module):
    """Emotion classification + valence/arousal regression.

    Reads MEFL's refined per-AU node features (h_final) via attention pool,
    fuses with the global backbone GAP, and predicts both:
    - 8-class softmax (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
    - 2-dim continuous V/A in [-1, 1] (via tanh)
    """

    def __init__(self, mefl_ch: int = 512, bb_ch: int = 768, n_classes: int = N_EMOTION,
                 dropout: float = 0.25):
        super().__init__()
        self.attn = nn.Linear(mefl_ch, 1)
        self.bb_proj = nn.Linear(bb_ch, mefl_ch)
        self.trunk = nn.Sequential(
            nn.Linear(mefl_ch * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(512, n_classes)
        self.va_head = nn.Linear(512, 2)

    def forward(self, h_final: torch.Tensor, X_bb: torch.Tensor) -> dict:
        # h_final: [B, N_AU, mefl_ch];  X_bb: [B, bb_ch, H, W]
        w = self.attn(h_final).softmax(dim=1)
        au_pooled = (h_final * w).sum(dim=1)            # [B, mefl_ch]
        bb_pooled = X_bb.mean(dim=(-2, -1))             # [B, bb_ch]
        fused = torch.cat([au_pooled, self.bb_proj(bb_pooled)], dim=-1)
        z = self.trunk(fused)
        return {
            "emotion_logits": self.cls_head(z),
            "va": torch.tanh(self.va_head(z)),          # [B, 2]
        }


class GazeHead(nn.Module):
    """Eye-region landmark-conditional gaze prediction.

    Architecture:
      1. Query backbone tokens with eye-region landmark positions (left + right)
         via multi-head cross-attention. Two separate query sets per eye.
      2. Optional 2-node graph (left ↔ right eye) with 1 GatedGCN layer for
         binocular reasoning.
      3. Gate using AU-related features from h_final (AU05/07/09/43/45).
      4. MLP → (yaw, pitch).
    """

    def __init__(self, bb_ch: int = 768, mefl_ch: int = 512, ch: int = 256,
                 use_dual_eye_gcn: bool = True):
        super().__init__()
        self.use_dual_eye_gcn = use_dual_eye_gcn

        self.lmk_proj = nn.Linear(2, bb_ch)
        self.attn = nn.MultiheadAttention(bb_ch, num_heads=4, batch_first=True)
        self.feat_proj = nn.Linear(bb_ch, ch)
        if use_dual_eye_gcn:
            self.dual_gcn = GatedGCNLayer(ch)

        n_gate = len(GAZE_GATE_AU_IDX)
        self.gate_proj = nn.Linear(mefl_ch * n_gate, ch)

        in_dim = ch * 2 + ch    # left + right + au_gate
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(),
            nn.Linear(512, 2),
        )

    def forward(self, X_bb: torch.Tensor, mesh_pred: torch.Tensor,
                h_final: torch.Tensor) -> torch.Tensor:
        # X_bb: [B, bb_ch, H, W];  mesh_pred: [B, 478, 3];  h_final: [B, N_AU, mefl_ch]
        B, C, H, W = X_bb.shape
        kv = X_bb.flatten(2).transpose(1, 2)            # [B, H*W, bb_ch]

        # Per-eye queries from predicted landmarks
        left_q = self.lmk_proj(mesh_pred[:, MP_LEFT_EYE_IDX, :2])
        right_q = self.lmk_proj(mesh_pred[:, MP_RIGHT_EYE_IDX, :2])

        left_attn, _ = self.attn(left_q, kv, kv)
        right_attn, _ = self.attn(right_q, kv, kv)
        left_feat = self.feat_proj(left_attn.mean(dim=1))   # [B, ch]
        right_feat = self.feat_proj(right_attn.mean(dim=1))

        if self.use_dual_eye_gcn:
            nodes = torch.stack([left_feat, right_feat], dim=1)  # [B, 2, ch]
            edges = torch.zeros(B, 2, 2, left_feat.shape[-1], device=X_bb.device)
            nodes, _ = self.dual_gcn(nodes, edges)
            left_feat, right_feat = nodes[:, 0], nodes[:, 1]

        # AU gating: eye-related AUs from MEFL h_final
        au_gate_feats = h_final[:, GAZE_GATE_AU_IDX, :]      # [B, 5, mefl_ch]
        au_gate = self.gate_proj(au_gate_feats.flatten(1))   # [B, ch]

        combined = torch.cat([left_feat, right_feat, au_gate], dim=-1)
        return self.mlp(combined)                            # [B, 2]


# ============================ v2.1 heads ============================

class EmotionVAHeadV2(nn.Module):
    """v2.1: Two parallel paths into the emotion+V/A trunk.

    Path A — AU-derived (from MEFL h_final):  attention pool → mefl_ch
    Path B — Backbone GAP (from raw backbone): 2-layer MLP → mefl_ch
                                                (was 1-layer Linear in v1)
    fused = concat([A, B]) → trunk → cls_head + va_head

    Rationale:
      v1 head had backbone GAP shoved through one Linear(768→512) and
      concatenated with the AU pool. The AU-derived path dominated because
      it already carries refined per-AU semantic features.
      v2.1 strengthens path B with a real MLP (Linear→GELU→Dropout→Linear)
      so emotion/V/A can leverage spatial backbone features beyond what
      gets distilled through the AU graph.

    Identical output structure to v1 (emotion_logits + va) — drop-in.
    """

    def __init__(self, mefl_ch: int = 512, bb_ch: int = 768,
                 n_classes: int = N_EMOTION, dropout: float = 0.25):
        super().__init__()
        self.attn = nn.Linear(mefl_ch, 1)
        # Path B: deeper backbone-GAP encoder
        self.bb_trunk = nn.Sequential(
            nn.Linear(bb_ch, mefl_ch * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mefl_ch * 2, mefl_ch),
        )
        # Fusion trunk (input: AU-pool ⊕ backbone-MLP = 2 × mefl_ch)
        self.trunk = nn.Sequential(
            nn.Linear(mefl_ch * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(512, n_classes)
        self.va_head = nn.Linear(512, 2)

    def forward(self, h_final: torch.Tensor, X_bb: torch.Tensor) -> dict:
        # Path A: attention pool over the 24 AU node features
        w = self.attn(h_final).softmax(dim=1)
        au_pooled = (h_final * w).sum(dim=1)                  # [B, mefl_ch]
        # Path B: backbone GAP through 2-layer MLP
        bb_pooled = X_bb.mean(dim=(-2, -1))                   # [B, bb_ch]
        bb_feat = self.bb_trunk(bb_pooled)                    # [B, mefl_ch]
        fused = torch.cat([au_pooled, bb_feat], dim=-1)
        z = self.trunk(fused)
        return {
            "emotion_logits": self.cls_head(z),
            "va": torch.tanh(self.va_head(z)),
        }


class GazeHeadV2(nn.Module):
    """v2.1: Backbone-GAP + landmark cross-attention, NO AU gating,
    separate yaw and pitch regression heads (L2CS-Net pattern).

    Rationale:
      v1 GazeHead gated gaze prediction with eye-related AU features
      (AU05, AU07, AU09, AU43, AU45) via gate_proj on h_final. But AU05
      and AU43 are still at 0.0 F1 even after stage 3 — they're broken
      signals, so the gating injected noise.

      SOTA gaze models (L2CS-Net, GazeTR, GazeSymCAT) skip AU coupling
      entirely. L2CS-Net specifically uses separate yaw and pitch FC
      layers with their own losses — empirically more accurate than
      a single 2-output regression.

    Architecture:
      X_bb GAP                 ─┐
                                ├─ fuse via concat → trunk
      eye_cross_attn (optional) ┘                       │
                                                        ├─→ yaw_head (B, 1)
                                                        └─→ pitch_head (B, 1)

    The eye-cross-attention is kept as an optional residual signal (toggle
    via `use_eye_cross_attn`). When True, queries from predicted mesh
    eye-region landmarks attend over backbone tokens — same as v1 but
    no GCN over the two eyes (was minor) and no AU gating.
    """

    def __init__(self, bb_ch: int = 768, mefl_ch: int = 512, ch: int = 256,
                 use_eye_cross_attn: bool = True):
        super().__init__()
        self.use_eye_cross_attn = use_eye_cross_attn

        # Backbone GAP path (primary)
        self.bb_trunk = nn.Sequential(
            nn.Linear(bb_ch, ch * 2),
            nn.GELU(),
            nn.Linear(ch * 2, ch),
            nn.GELU(),
        )

        # Optional eye-region cross-attention path
        if use_eye_cross_attn:
            self.lmk_proj = nn.Linear(2, bb_ch)
            self.attn = nn.MultiheadAttention(bb_ch, num_heads=4, batch_first=True)
            self.eye_feat_proj = nn.Linear(bb_ch, ch)
            fused_dim = ch + ch * 2   # backbone + left eye + right eye
        else:
            fused_dim = ch

        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        # Separate yaw & pitch heads (L2CS-Net pattern)
        self.yaw_head = nn.Linear(256, 1)
        self.pitch_head = nn.Linear(256, 1)

    def forward(self, X_bb: torch.Tensor, mesh_pred: torch.Tensor) -> torch.Tensor:
        # X_bb: [B, bb_ch, H, W];  mesh_pred: [B, 478, 3]
        B, C, H, W = X_bb.shape

        # Backbone GAP primary feature
        bb_pooled = X_bb.mean(dim=(-2, -1))           # [B, bb_ch]
        bb_feat = self.bb_trunk(bb_pooled)            # [B, ch]
        parts = [bb_feat]

        if self.use_eye_cross_attn:
            kv = X_bb.flatten(2).transpose(1, 2)      # [B, H*W, bb_ch]
            left_q = self.lmk_proj(mesh_pred[:, MP_LEFT_EYE_IDX, :2])
            right_q = self.lmk_proj(mesh_pred[:, MP_RIGHT_EYE_IDX, :2])
            left_attn, _ = self.attn(left_q, kv, kv)
            right_attn, _ = self.attn(right_q, kv, kv)
            left_feat = self.eye_feat_proj(left_attn.mean(dim=1))
            right_feat = self.eye_feat_proj(right_attn.mean(dim=1))
            parts.extend([left_feat, right_feat])

        fused = torch.cat(parts, dim=-1)
        z = self.trunk(fused)
        yaw = self.yaw_head(z)                         # [B, 1]
        pitch = self.pitch_head(z)                     # [B, 1]
        return torch.cat([yaw, pitch], dim=-1)         # [B, 2]


# ============================ Top-level ============================

class MEGraphAUv2(nn.Module):
    def __init__(self, cfg: ModelV2Config):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=[-1],
            drop_path_rate=cfg.drop_path_rate,
        )
        in_ch = self.backbone.feature_info.channels()[-1]
        self.proj = (
            nn.Conv2d(in_ch, cfg.afg_channels, 1, bias=False)
            if in_ch != cfg.afg_channels
            else nn.Identity()
        )
        self.afg = AFG(cfg.afg_channels, cfg.afg_channels, cfg.n_au)
        self.fgg = FGG(cfg.afg_channels, k=cfg.knn_k)
        self.mefl = MEFL(cfg.mefl_channels, cfg.n_au, cfg.gcn_layers,
                         use_layer_scale=cfg.mefl_use_layer_scale)
        self.sc = SCHead(cfg.n_au, cfg.mefl_channels)
        # Cooccurrence head: only present in v2 / v2.1 / v2.2 (use_head_v3=False).
        # v2.3 drops it entirely — BP4D-skewed priors hurt DISFA+ transfer.
        if not cfg.use_head_v3:
            self.cooccur = CooccurrenceHead(cfg.mefl_channels)
        self.lmk = LandmarkHead(cfg.afg_channels)
        self.pose = PoseHead(cfg.afg_channels)

        # Head dispatch — v3 (v2.3) > v2 (v2.1) > v1 default
        if cfg.use_head_v3:
            from feat.multitask.heads_v3 import (
                UnifiedFeatures, GazeHeadV3, EmotionVAHeadV3,
            )
            self.unified = UnifiedFeatures(bb_ch=in_ch, lmk_dim=cfg.unified_lmk_dim,
                                            image_size=cfg.image_size)
            self.emotion = EmotionVAHeadV3(
                in_dim=self.unified.out_dim,
                n_classes=cfg.n_emotion,
                dropout=cfg.emotion_head_dropout,
            )
            self.gaze = GazeHeadV3(in_dim=self.unified.out_dim)
        elif cfg.use_head_v2:
            self.emotion = EmotionVAHeadV2(
                mefl_ch=cfg.mefl_channels, bb_ch=in_ch,
                n_classes=cfg.n_emotion, dropout=cfg.emotion_head_dropout,
            )
            self.gaze = GazeHeadV2(
                bb_ch=in_ch, mefl_ch=cfg.mefl_channels,
                use_eye_cross_attn=cfg.use_dual_eye_gcn,
            )
        else:
            self.emotion = EmotionVAHead(
                mefl_ch=cfg.mefl_channels, bb_ch=in_ch,
                n_classes=cfg.n_emotion, dropout=cfg.emotion_head_dropout,
            )
            self.gaze = GazeHead(
                bb_ch=in_ch, mefl_ch=cfg.mefl_channels,
                use_dual_eye_gcn=cfg.use_dual_eye_gcn,
            )

        # Homoscedastic σ (Kendall) — one per task
        self.log_var_au = nn.Parameter(torch.zeros(cfg.n_au))
        self.log_var_mesh = nn.Parameter(torch.zeros(()))
        self.log_var_pose = nn.Parameter(torch.zeros(()))
        # log_var_cooccur only when cooccur loss is active
        if not cfg.use_head_v3:
            self.log_var_cooccur = nn.Parameter(torch.zeros(()))
        self.log_var_emotion = nn.Parameter(torch.zeros(()))
        self.log_var_va = nn.Parameter(torch.zeros(()))
        self.log_var_gaze = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.backbone(x)[-1]                # [B, bb_ch, H, W]
        X = self.proj(feats)                         # [B, afg_ch, H, W]
        U, v = self.afg(X)                           # [B, N, C, H, W], [B, N, C]
        v_fgg = self.fgg(v)                          # [B, N, C]
        h, e = self.mefl(U, X, v_fgg)                # [B, N, C], [B, N, N, C]

        p_au = self.sc(h)                            # [B, N]
        mesh = self.lmk(X)                           # [B, 478, 3]
        pose = self.pose(X)                          # [B, 6]

        # Head dispatch
        if self.cfg.use_head_v3:
            # v2.3: unified features = backbone GAP ∥ proj(mesh_xy)
            unified = self.unified(feats, mesh)      # [B, bb_ch + lmk_dim]
            emotion_out = self.emotion(unified)
            gaze = self.gaze(unified)
            cooc_logits = None
        else:
            # v2 / v2.1: heads use h (AU node features) + raw backbone feats
            cooc_logits = self.cooccur(e)            # [B, N, N, 4]
            emotion_out = self.emotion(h, feats)
            if isinstance(self.gaze, GazeHeadV2):
                gaze = self.gaze(feats, mesh)
            else:
                gaze = self.gaze(feats, mesh, h)

        return {
            "p_au": p_au,
            "mesh": mesh,
            "pose": pose,
            "cooccur": cooc_logits,
            "emotion_logits": emotion_out["emotion_logits"],
            "va": emotion_out["va"],
            "gaze": gaze,
            "h": h,
            "e": e,
        }

    # ------------- per-stage freeze utilities -------------

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)
        for p in self.proj.parameters():
            p.requires_grad_(not freeze)

    def freeze_anfl(self, freeze: bool = True):
        for m in (self.afg, self.fgg, self.sc):
            for p in m.parameters():
                p.requires_grad_(not freeze)

    def freeze_mefl(self, freeze: bool = True):
        for m in (self.mefl, self.cooccur):
            for p in m.parameters():
                p.requires_grad_(not freeze)
