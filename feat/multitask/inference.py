"""Inference wrapper for the v2.3 multitask model inside py-feat.

Loads the checkpoint from the HuggingFace hub (``py-feat/face_multitask_v1``),
preprocesses RetinaFace face crops into model chips *exactly* as training did,
runs a forward pass, and decodes the raw output dict into labelled arrays.

Preprocessing contract (must match training — deep/augment.py SyncedAugment,
train=False): a 256x256 uint8 RGB chip -> /255 -> center-crop to 224 ->
ImageNet normalize. The chip itself is produced upstream by
``extract_face_from_bbox_torch(frame_in_[0,1], bbox, face_size=256,
expand_bbox=1.2)`` — the same py-feat helper training used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

from feat.multitask.model_v2 import MEGraphAUv2, ModelV2Config, AU_NAMES
from feat.multitask import EMOTION_NAMES
from feat.utils.blendshape_to_au import DLIB68_FROM_MP478

# Chip geometry — fixed by how training chips were produced.
CHIP_SIZE = 256          # extract_face_from_bbox_torch face_size
MODEL_INPUT = 224        # model image_size (center-crop of the chip)
EXPAND_BBOX = 1.2        # bbox margin multiplier used at extraction
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HF_REPO = "py-feat/face_multitask_v1"
HF_WEIGHTS_FILE = "face_multitask_v1.pt"

# dlib-68 vertex indices into the MediaPipe-478 mesh (py-feat canonical map).
_DLIB68_IDX = torch.tensor(DLIB68_FROM_MP478, dtype=torch.long)


@dataclass
class MultitaskOutput:
    """Decoded, labelled model output for a batch of N faces."""
    au: np.ndarray              # [N, 24] probabilities in [0, 1]
    au_names: list              # 24 AU names
    emotion: np.ndarray         # [N, 8] softmax probabilities
    emotion_names: list         # 8 emotion names
    valence: np.ndarray         # [N] in [-1, 1]
    arousal: np.ndarray         # [N] in [-1, 1]
    gaze: np.ndarray            # [N, 2] (yaw, pitch) radians
    pose: np.ndarray            # [N, 6] (yaw, pitch, roll, tx, ty, tz)
    mesh478: np.ndarray         # [N, 478, 3] chip-pixel coords (224 space)
    landmarks68: np.ndarray     # [N, 68, 2] dlib-68 (x, y) sampled from mesh478


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
        ckpt = torch.load(self._resolve_weights(weights_path),
                          map_location=self.device, weights_only=False)
        saved_cfg = ckpt["config"]
        valid = set(ModelV2Config.__dataclass_fields__.keys())
        filtered = {k: v for k, v in saved_cfg.items()
                    if k in valid and k not in ("pretrained", "use_head_v2")}
        cfg = ModelV2Config(**filtered, pretrained=False,
                            use_head_v2=saved_cfg.get("use_head_v2", False))
        self.cfg = cfg
        model = MEGraphAUv2(cfg)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
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

        return MultitaskOutput(
            au=out["p_au"].float().cpu().numpy(),
            au_names=list(AU_NAMES),
            emotion=emotion.float().cpu().numpy(),
            emotion_names=list(EMOTION_NAMES),
            valence=va[:, 0].float().cpu().numpy(),
            arousal=va[:, 1].float().cpu().numpy(),
            gaze=out["gaze"].float().cpu().numpy(),
            pose=out["pose"].float().cpu().numpy(),
            mesh478=mesh.float().cpu().numpy(),
            landmarks68=lmk68.float().cpu().numpy(),
        )
