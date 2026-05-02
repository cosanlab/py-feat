"""RetinaFace face-detector wrapper for py-feat.

Replaces the previous mobilenet0.25-based wrapper. The old wrapper had
two problems: (a) the HuggingFace download path was broken and (b) it ran
postprocess in a per-image Python for loop with CPU/numpy round-trips
inside the inner loop, defeating any batched throughput gains. This
wrapper:

- ships only the resnet34 backbone (88.9% WIDERFACE Hard AP per yakhyo)
- accepts a batched [B, C, H, W] uint8 or float32 tensor
- does mean-subtract, model forward, anchor decode, NMS, and threshold
  all on-device with no .cpu() / .numpy() trips until the very end
- caches priors per (image_height, image_width, device) so the Python
  anchor-generation loop runs once per image-shape, not once per call
- uses torchvision.ops.batched_nms which natively groups boxes by image
  index — single batched-NMS call across the whole batch

Output contract (matches the legacy wrapper for drop-in compatibility):
    list of length B, where each entry is a list of [xmin, ymin, xmax,
    ymax, score] entries (in pixel coords) for the detected faces in
    that image. Empty list for images with no detections.
"""

from __future__ import annotations

from typing import List

import torch
from torchvision.ops import batched_nms

from feat.face_detectors.Retinaface.Retinaface_model import (
    RETINAFACE_R34_CFG,
    RetinaFace,
    decode_boxes,
    decode_landmarks,
    generate_priors,
)
from feat.utils import set_torch_device


# BGR mean from the upstream training preprocessing. Subtracted from raw
# pixel values [0, 255]. Held as a 4D tensor (1, 3, 1, 1) so it broadcasts
# against any [B, 3, H, W] input. Materialized on-device in the wrapper.
_BGR_MEAN = torch.tensor([104.0, 117.0, 123.0]).view(1, 3, 1, 1)


class Retinaface:
    """Batched RetinaFace face detector.

    Parameters
    ----------
    device : str | torch.device
        ``'cpu'``, ``'cuda'``, ``'mps'``, ``'auto'``, or a ``torch.device``.
    detection_threshold : float
        Final score threshold (after NMS) below which detections are
        dropped. Default 0.5.
    nms_threshold : float
        IoU threshold for non-max suppression. Default 0.4.
    confidence_threshold : float
        Pre-NMS score threshold. Default 0.02.
    top_k : int
        Pre-NMS top-K detections kept per image. Default 5000.
    keep_top_k : int
        Post-NMS top-K detections kept per image. Default 750.
    pretrained : str
        ``'huggingface'`` to download from ``py-feat/retinaface_r34`` (the
        default). Anything else suppresses weight loading; the caller is
        expected to load weights manually (mostly for tests).
    """

    def __init__(
        self,
        device="auto",
        detection_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        confidence_threshold: float = 0.02,
        top_k: int = 5000,
        keep_top_k: int = 750,
        pretrained: str = "huggingface",
    ) -> None:
        torch.set_grad_enabled(False)
        self.device = set_torch_device(device=device)
        self.cfg = RETINAFACE_R34_CFG

        self.detection_threshold = detection_threshold
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k

        self.net = RetinaFace(self.cfg)
        if pretrained == "huggingface":
            from feat.utils import hf_hub_download_with_fallback

            # PyTorchModelHubMixin's `from_pretrained` would ignore our
            # pre-constructed `self.net`. Use the fallback helper to grab
            # the safetensors file, then load it explicitly. v1 fallback
            # mirrors the AU classifier rollout pattern (PR #273).
            from safetensors.torch import load_file
            from feat.utils.io import get_resource_path

            weights_path = hf_hub_download_with_fallback(
                repo_id="py-feat/retinaface_r34",
                filename="model.safetensors",
                fallback_filename="retinaface_r34.safetensors",
                cache_dir=get_resource_path(),
            )
            state_dict = load_file(weights_path)
            self.net.load_state_dict(state_dict, strict=True)

        self.net = self.net.to(self.device).eval()

        # Per-(H, W, device) prior cache. The Python prior-generation
        # loop is non-trivial (~hundreds of µs at 640²) and the priors
        # don't depend on input pixel values, so we memoize them.
        self._prior_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

        # Mean tensor materialized on the right device once.
        self._mean = _BGR_MEAN.to(self.device)

    def _get_priors(self, height: int, width: int) -> torch.Tensor:
        key = (height, width, self.device)
        priors = self._prior_cache.get(key)
        if priors is None:
            priors = generate_priors(
                self.cfg["min_sizes"],
                self.cfg["steps"],
                (height, width),
                clip=self.cfg["clip"],
                device=self.device,
            )
            self._prior_cache[key] = priors
        return priors

    @torch.inference_mode()
    def __call__(self, img: torch.Tensor) -> List[List[List[float]]]:
        """Detect faces in a batch of images.

        Args:
            img: ``[B, 3, H, W]`` tensor of pixel values in [0, 255]
                (float or uint8 — coerced to float internally).

        Returns:
            list of length B; each entry is a list of detections, where
            each detection is ``[xmin, ymin, xmax, ymax, score]`` in
            pixel coordinates of the input image's resolution.
        """
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(
                f"img must be [B, 3, H, W]; got shape {tuple(img.shape)}"
            )

        img = img.to(self.device, dtype=torch.float32)
        img = img - self._mean

        B, _, H, W = img.shape

        # Single batched forward.
        loc, conf, landms = self.net(img)  # [B, A, 4], [B, A, 2], [B, A, 10]

        priors = self._get_priors(H, W)
        boxes = decode_boxes(loc, priors, self.cfg["variance"])  # [B, A, 4] normalized
        # landmarks decoded but unused in the legacy output contract; computed
        # so the cost is paid on-device in one shot if we expose them later.
        _ = decode_landmarks(landms, priors, self.cfg["variance"])
        scores = conf[..., 1]  # [B, A] face probabilities

        # Scale to pixel coords.
        scale = torch.tensor([W, H, W, H], dtype=torch.float32, device=self.device)
        boxes = boxes * scale  # [B, A, 4]

        # Flatten to [B*A, ...] with image-index, filter pre-NMS by confidence,
        # then run a single batched_nms across the whole flattened batch.
        A = boxes.shape[1]
        batch_idx = torch.arange(B, device=self.device).repeat_interleave(A)
        flat_boxes = boxes.view(B * A, 4)
        flat_scores = scores.reshape(B * A)

        keep_mask = flat_scores > self.confidence_threshold
        flat_boxes = flat_boxes[keep_mask]
        flat_scores = flat_scores[keep_mask]
        batch_idx = batch_idx[keep_mask]

        # Per-image pre-NMS top_k. We keep this simple: sort globally by score
        # and let batched_nms's per-group NMS handle the rest. For very dense
        # detections the per-image top_k truncation can be reintroduced.
        if flat_scores.numel() > self.top_k * B:
            order = torch.argsort(flat_scores, descending=True)[: self.top_k * B]
            flat_boxes = flat_boxes[order]
            flat_scores = flat_scores[order]
            batch_idx = batch_idx[order]

        keep = batched_nms(flat_boxes, flat_scores, batch_idx, self.nms_threshold)
        flat_boxes = flat_boxes[keep]
        flat_scores = flat_scores[keep]
        batch_idx = batch_idx[keep]

        # Final detection-threshold filter.
        final_mask = flat_scores >= self.detection_threshold
        flat_boxes = flat_boxes[final_mask]
        flat_scores = flat_scores[final_mask]
        batch_idx = batch_idx[final_mask]

        # Group detections back per-image. Single CPU trip at the end.
        flat_boxes_cpu = flat_boxes.cpu()
        flat_scores_cpu = flat_scores.cpu()
        batch_idx_cpu = batch_idx.cpu()

        result: List[List[List[float]]] = [[] for _ in range(B)]
        for i in range(flat_scores_cpu.numel()):
            b = int(batch_idx_cpu[i].item())
            xmin, ymin, xmax, ymax = flat_boxes_cpu[i].tolist()
            score = float(flat_scores_cpu[i].item())
            # keep_top_k per image
            if len(result[b]) < self.keep_top_k:
                result[b].append([xmin, ymin, xmax, ymax, score])
        return result
