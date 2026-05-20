"""Landmark-only pose MLP inference.

Replaces ``feat.utils.face_pose_pnp`` for non-img2pose face_model paths.
Loaded MLP takes 68 face landmarks (normalized to the face bbox) and
emits 6DoF head pose calibrated to img2pose's coordinate frame.

Training: distillation from img2pose on CelebV-HQ (~570k frames). See
``scripts/train_pose_mlp.py``. Validation MAE on held-out CelebV-HQ:
pitch 2.78°, roll 2.05°, yaw 1.64° — comparable to img2pose's reported
~4° avg MAE on BIWI (different dataset; smaller is better).

Weights: ``models/pose_mlp_v1.safetensors`` locally, will be at
``py-feat/pose_mlp_v1`` on HuggingFace.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PoseMLP(nn.Module):
    """Mirror of the architecture in ``scripts/train_pose_mlp.py``.

    v2 architecture: Linear → LayerNorm → GELU → Dropout per hidden
    block, with wider hidden layers (default 512/256/128). v1 used a
    bare Linear→ReLU→Dropout stack (256/128/64); we keep backward
    compatibility by inferring the architecture from the checkpoint
    when loading.
    """

    def __init__(self, hidden=(512, 256, 128), dropout: float = 0.15,
                 use_layernorm: bool = True):
        super().__init__()
        in_dim = 136
        layers: list[nn.Module] = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            if use_layernorm:
                layers.extend([nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            else:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


_CACHE: dict[str, tuple[PoseMLP, np.ndarray, np.ndarray]] = {}


def _resolve_weights_path() -> Path | None:
    """Find the pose-MLP weights.

    Lookup order:
      1. ``FEAT_POSE_MLP_PATH`` env var (full path to .safetensors)
      2. ``models/pose_mlp_v1.safetensors`` relative to repo root
      3. HuggingFace ``py-feat/pose_mlp_v1`` (when uploaded)
    """
    env = os.environ.get("FEAT_POSE_MLP_PATH")
    if env and Path(env).exists():
        return Path(env)
    # Repo-relative
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    local = repo_root / "models" / "pose_mlp_v1.safetensors"
    if local.exists():
        return local
    # HuggingFace fallback (handled by caller)
    return None


def _load_pose_mlp(device: str = "cpu") -> tuple[PoseMLP, np.ndarray, np.ndarray] | None:
    """Load the pose-MLP model + output normalization stats. Cached per-device."""
    cache_key = str(device)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    weights_path = _resolve_weights_path()
    meta_path: Path | None = None
    if weights_path is not None:
        meta_path = weights_path.with_suffix(".json")

    if weights_path is None or not meta_path.exists():
        # Try HuggingFace
        try:
            from huggingface_hub import hf_hub_download

            from feat.utils.io import get_resource_path

            weights_path = Path(
                hf_hub_download(
                    repo_id="py-feat/pose_mlp_v1",
                    filename="pose_mlp_v1.safetensors",
                    cache_dir=get_resource_path(),
                )
            )
            meta_path = Path(
                hf_hub_download(
                    repo_id="py-feat/pose_mlp_v1",
                    filename="pose_mlp_v1.json",
                    cache_dir=get_resource_path(),
                )
            )
        except Exception:
            return None

    if weights_path is None or meta_path is None or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    hidden = tuple(meta["architecture"]["hidden"])
    dropout = float(meta["architecture"]["dropout"])
    norm = meta["architecture"]["output_normalization"]
    y_mean = np.asarray(norm["mean"], dtype=np.float32)
    y_std = np.asarray(norm["std"], dtype=np.float32)

    from safetensors.torch import load_file
    state = load_file(str(weights_path))
    # Detect architecture variant from state-dict keys: v2 has LayerNorm
    # weights (`net.1.weight` is shape [hidden_dim, ]); v1 doesn't.
    use_layernorm = "net.1.weight" in state and state["net.1.weight"].dim() == 1

    model = PoseMLP(hidden=hidden, dropout=dropout, use_layernorm=use_layernorm)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    _CACHE[cache_key] = (model, y_mean, y_std)
    return _CACHE[cache_key]


def pose_from_landmarks_mlp(
    landmarks_2d: torch.Tensor,
    bboxes: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Estimate 6DoF pose from 68 2D landmarks.

    Bbox-free: normalizes landmarks by their own centroid + inter-eye
    distance, so the MLP is decoupled from upstream face-detector bbox
    conventions (img2pose loose vs retinaface tight).

    Args:
        landmarks_2d: ``[B, 68, 2]`` landmark coordinates.
        bboxes: ignored (kept in signature for backward compatibility).

    Returns:
        ``[B, 6]`` pose tensor with columns
        ``(Pitch, Roll, Yaw, X, Y, Z)`` matching ``FEAT_FACEPOSE_COLUMNS_6D``.
        Returns ``None`` if the pose-MLP weights are not available.
    """
    device = landmarks_2d.device
    loaded = _load_pose_mlp(device=str(device))
    if loaded is None:
        return None
    model, y_mean, y_std = loaded
    y_mean_t = torch.as_tensor(y_mean, device=device, dtype=torch.float32)
    y_std_t = torch.as_tensor(y_std, device=device, dtype=torch.float32)

    if landmarks_2d.dim() != 3 or landmarks_2d.shape[-1] != 2:
        raise ValueError(
            f"landmarks_2d must be [B, 68, 2], got {tuple(landmarks_2d.shape)}"
        )
    if landmarks_2d.shape[1] != 68:
        raise ValueError(
            f"pose-MLP requires 68 landmarks, got {landmarks_2d.shape[1]}"
        )

    x = landmarks_2d[..., 0]  # [B, 68]
    y = landmarks_2d[..., 1]
    cx = x.mean(dim=1, keepdim=True)
    cy = y.mean(dim=1, keepdim=True)
    # Inter-eye distance (dlib-68: 36 = left-eye outer corner, 45 = right-eye outer corner).
    dx = x[:, 36] - x[:, 45]
    dy = y[:, 36] - y[:, 45]
    iod = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6).unsqueeze(1)
    x_norm = (x - cx) / iod
    y_norm = (y - cy) / iod
    feat = torch.cat([x_norm, y_norm], dim=1).to(torch.float32)

    with torch.inference_mode():
        z = model(feat)
        pose = z * y_std_t + y_mean_t

    return pose
