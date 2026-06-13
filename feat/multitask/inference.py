"""Inference wrapper for the multitask model inside py-feat.

Loads the checkpoint from the HuggingFace hub (``py-feat/face_multitask_v2``,
the v2.5 model: 20 AUs / 7 emotions + 52 blendshapes), preprocesses RetinaFace
face crops into model chips *exactly* as training did, runs a forward pass, and
decodes the raw output dict into labelled arrays.

Weights are distributed as a single ``.safetensors`` file (no pickle / arbitrary
code execution), with the ModelV2Config JSON-encoded in the file's metadata. A
legacy ``.pt`` checkpoint (config+model dict) is still accepted for local paths.

Preprocessing contract (must match training — deep/augment.py SyncedAugment,
train=False): a 256x256 uint8 RGB chip -> /255 -> center-crop to 224 ->
ImageNet normalize. The chip itself is produced upstream by
``extract_face_from_bbox_torch(frame_in_[0,1], bbox, face_size=256,
expand_bbox=1.2)`` — the same py-feat helper training used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

from feat.multitask.model_v2 import MEGraphAUv2, ModelV2Config, AU_NAMES
from feat.multitask import (
    EMOTION_NAMES, AU_NAMES_V24, EMOTION_NAMES_V24,
)
from feat.utils.blendshape_to_au import DLIB68_FROM_MP478

# Chip geometry — fixed by how training chips were produced.
CHIP_SIZE = 256          # extract_face_from_bbox_torch face_size
MODEL_INPUT = 224        # model image_size (center-crop of the chip)
EXPAND_BBOX = 1.2        # bbox margin multiplier used at extraction
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HF_REPO = "py-feat/face_multitask_v2"
HF_WEIGHTS_FILE = "face_multitask_v2.safetensors"

# dlib-68 vertex indices into the MediaPipe-478 mesh (py-feat canonical map).
_DLIB68_IDX = torch.tensor(DLIB68_FROM_MP478, dtype=torch.long)


def _load_multitask_weights(path):
    """Return (config_dict, state_dict) from a .safetensors (config in metadata)
    or a legacy .pt (config+model dict). safetensors is the distribution format;
    .pt is accepted for local/dev checkpoints."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        if "config" not in meta:
            raise RuntimeError(f"{path}: safetensors metadata missing 'config'")
        return json.loads(meta["config"]), load_file(path, device="cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["config"], ckpt["model"]


@dataclass
class MultitaskOutput:
    """Decoded, labelled model output for a batch of N faces."""
    au: np.ndarray              # [N, n_au] probabilities in [0, 1] (v2.4: 20)
    au_names: list              # n_au AU names (derived from the checkpoint)
    emotion: np.ndarray         # [N, n_emotion] softmax probabilities (v2.4: 7)
    emotion_names: list         # n_emotion emotion names
    valence: np.ndarray         # [N] in [-1, 1]
    arousal: np.ndarray         # [N] in [-1, 1]
    gaze: np.ndarray            # [N, 2] (yaw, pitch) radians
    pose: np.ndarray            # [N, 6] (yaw, pitch, roll, tx, ty, tz)
    mesh478: np.ndarray         # [N, 478, 3] chip-pixel coords (224 space)
    landmarks68: np.ndarray     # [N, 68, 2] dlib-68 (x, y) sampled from mesh478
    blendshapes: np.ndarray     # [N, 52] MediaPipe/ARKit coefficients in [0, 1]


class MultitaskModel:
    """Loads + runs the v2.3 multitask model. Detection-agnostic: it consumes
    256x256 face chips and returns decoded predictions."""

    def __init__(self, device="cpu", weights_path=None, amp=None, compile=False):
        self.device = torch.device(device)
        # Mixed precision: the model was trained with bf16 autocast, and bf16
        # inference matches fp32 to within model noise (AU <0.01, gaze <0.03 deg)
        # while running ~3x faster on GPU. Default on for CUDA, off on CPU
        # (CPU autocast is slow/uneven for these ops). Pass amp=False to force fp32.
        self.amp = (self.device.type == "cuda") if amp is None else amp
        # torch.compile fuses the MEFL edge loop for ~1.7x on the model forward
        # (90->53ms on Blackwell, batch 80). Off by default: (a) the first call
        # pays multi-second compile and each new batch size recompiles, a footgun
        # for interactive use; (b) compiled-vs-eager AU output diverges ~0.16 in
        # default mode under bf16 autocast — not yet validated as safe. Enable
        # only for throughput jobs where that drift has been checked acceptable.
        self.compile = compile
        saved_cfg, state_dict = _load_multitask_weights(
            self._resolve_weights(weights_path))
        valid = set(ModelV2Config.__dataclass_fields__.keys())
        filtered = {k: v for k, v in saved_cfg.items()
                    if k in valid and k not in ("pretrained", "use_head_v2")}
        cfg = ModelV2Config(**filtered, pretrained=False,
                            use_head_v2=saved_cfg.get("use_head_v2", False))
        self.cfg = cfg
        # AU / emotion names depend on the head dims (v2.4/v2.5 = 20 AU / 7 emotion;
        # v2.3 = 24 / 8). Derive from cfg so this loader handles both.
        self.au_names = AU_NAMES_V24 if cfg.n_au == 20 else list(AU_NAMES)
        self.emotion_names = EMOTION_NAMES_V24 if cfg.n_emotion == 7 else list(EMOTION_NAMES)
        model = MEGraphAUv2(cfg)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"multitask checkpoint mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys")
        self.model = model.to(self.device).eval()
        if self.compile:
            self.model = torch.compile(self.model)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(3, 1, 1)
        self._idx68 = _DLIB68_IDX.to(self.device)

    @staticmethod
    def _resolve_weights(weights_path):
        if weights_path is not None:
            return weights_path
        # Optional local override via env (FEAT_MULTITASK_WEIGHTS) — lets users
        # point at a local checkpoint instead of the HF default (HF_REPO below).
        import os
        env_w = os.environ.get("FEAT_MULTITASK_WEIGHTS")
        if env_w:
            return env_w
        from huggingface_hub import hf_hub_download
        from feat.utils.io import get_resource_path
        return hf_hub_download(repo_id=HF_REPO, filename=HF_WEIGHTS_FILE,
                               cache_dir=get_resource_path())

    def preprocess(self, chips):
        """chips: [N, 3, 256, 256] float in [0, 1] (RetinaFace crops).
        Returns [N, 3, 224, 224] ImageNet-normalized model input."""
        chips = chips.to(self.device)
        if chips.shape[-1] != CHIP_SIZE:
            chips = F.interpolate(chips, size=(CHIP_SIZE, CHIP_SIZE),
                                  mode="bilinear", align_corners=False)
        chips = TF.center_crop(chips, [MODEL_INPUT, MODEL_INPUT])
        return (chips - self._mean) / self._std

    @torch.inference_mode()
    def __call__(self, chips):
        """chips: [N, 3, 256, 256] float in [0, 1]. Returns MultitaskOutput."""
        x = self.preprocess(chips)
        if self.amp:
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                out = self.model(x)
        else:
            out = self.model(x)

        emotion = F.softmax(out["emotion_logits"], dim=-1)
        va = out["va"]
        mesh = out["mesh"]                      # [N, 478, 3] in 224-px coords
        lmk68 = mesh[:, self._idx68, :2]        # [N, 68, 2]
        bs = out.get("blendshapes")             # [N, 52] in [0, 1] or None (v2.4)
        n = out["p_au"].shape[0]
        blendshapes = (bs.float().cpu().numpy() if bs is not None
                       else np.full((n, 52), np.nan, dtype=np.float32))

        return MultitaskOutput(
            au=out["p_au"].float().cpu().numpy(),
            au_names=list(self.au_names),
            emotion=emotion.float().cpu().numpy(),
            emotion_names=list(self.emotion_names),
            valence=va[:, 0].float().cpu().numpy(),
            arousal=va[:, 1].float().cpu().numpy(),
            gaze=out["gaze"].float().cpu().numpy(),
            pose=out["pose"].float().cpu().numpy(),
            mesh478=mesh.float().cpu().numpy(),
            landmarks68=lmk68.float().cpu().numpy(),
            blendshapes=blendshapes,
        )
