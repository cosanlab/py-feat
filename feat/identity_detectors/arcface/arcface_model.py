"""ArcFace face-recognition wrapper.

Loads InsightFace's ArcFace ResNet50 (the ``buffalo_l`` pack's
``w600k_r50.onnx`` converted to PyTorch ``.safetensors``) and exposes a
``forward(face_crops) -> [N, 512]`` interface compatible with the
existing identity-detector contract in ``feat.detector.Detectorv1`` and
``feat.MPDetector.MPDetector``.

Why ArcFace is now the default
------------------------------
py-feat used to default to ``facenet`` (InceptionResnetV1 trained with
triplet loss on VGGFace2). Triplet-loss embeddings entangle identity
with pose and expression — a serious problem for FEAT's intended use
case of clustering identities across video frames where pose and
expression vary deliberately. On the multi_face.jpg fixture, FaceNet's
max off-diagonal cosine similarity between *different* people was 0.76
(merging 2 identities at a typical 0.5 threshold); ArcFace's was 0.35
(all 5 stayed separate).

ArcFace replaces triplet loss with an angular-margin softmax that
constrains identities to disjoint angular regions of the embedding
sphere, producing tighter intra-identity clusters. Published numbers
on IJB-C (the hardest standard face-recognition benchmark, with
diverse poses and expressions) put ArcFace-R50/WebFace600K at 96.18%
TAR @ FAR=1e-4 vs. ~80% for FaceNet/VGGFace2.

Inference cost is essentially unchanged: 13.0 -> 13.5 ms/frame on M5
MBP MPS (4% slower) for the full retinaface_r34 + svm AU + identity
pipeline on a 472-frame video. ArcFace's larger backbone (43.6M vs
~25M params) is offset by smaller input (112x112 vs 160x160).

License
-------
The InsightFace code is MIT-licensed. The pretrained weights are
distributed under InsightFace's non-commercial-research terms; the
underlying training data (WebFace600K, Tsinghua's curated subset of
WebFace260M) is also research-only. Both layers are documented in
the model card at https://github.com/deepinsight/insightface and in
``feat/identity_detectors/arcface/README.md``. Commercial users
should validate license compatibility; the default FaceNet weights
inherited a similar VGGFace2 research-only restriction, so this is
not a new license category for py-feat.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from feat.identity_detectors.arcface.iresnet import iresnet50, iresnet100


# ArcFace expects pixel values in [-1, 1]. Our pipeline carries faces in
# [0, 1] after the standard ``frame / 255.0`` rescale, so the wrapper
# applies ``(x - 0.5) / 0.5 = 2x - 1`` itself rather than asking callers
# to normalize.
_ARCFACE_INPUT_SIZE = 112


class ArcFace(nn.Module):
    """ArcFace recognition wrapper that consumes the same face-crop
    tensors as the existing facenet path.

    Args:
        backbone: ``"r50"`` (default) or ``"r100"``.

    The forward pass:

    1. Bilinearly resizes the input to 112x112 if it isn't already.
    2. Maps [0, 1] pixel values to [-1, 1] with a single multiply-add.
    3. Runs the IResNet backbone to produce a 512-d embedding per face.

    L2 normalization is intentionally NOT applied here because the
    downstream ``cluster_identities`` already runs ``cosine_similarity``,
    which normalizes internally. Returning unnormalized embeddings keeps
    the magnitudes available for any future quality-aware filtering
    (e.g. MagFace-style quality scoring).
    """

    def __init__(self, backbone: str = "r50"):
        super().__init__()
        if backbone == "r50":
            self.net = iresnet50()
        elif backbone == "r100":
            self.net = iresnet100()
        else:
            raise ValueError(
                f"Unknown ArcFace backbone {backbone!r}; expected 'r50' or 'r100'."
            )
        self.input_size = _ARCFACE_INPUT_SIZE

    def forward(self, face_crops: torch.Tensor) -> torch.Tensor:
        """Compute embeddings from face crops.

        Args:
            face_crops: ``[N, 3, H, W]`` tensor of face crops with pixel
                values in ``[0, 1]``. ``H``/``W`` need not equal
                ``self.input_size``; the wrapper resizes if needed.

        Returns:
            ``[N, 512]`` embedding tensor (NOT L2-normalized).
        """
        if face_crops.shape[-1] != self.input_size or face_crops.shape[-2] != self.input_size:
            face_crops = F.interpolate(
                face_crops,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        # [0, 1] -> [-1, 1]. Done as a fused op so it's one kernel on
        # MPS / CUDA rather than a separate sub + div.
        x = face_crops * 2.0 - 1.0
        return self.net(x)


def load_arcface_identity_detector(device, backbone: str = "r50"):
    """Build an ArcFace detector with the trained w600k_r50 weights loaded.

    ``ArcFace.__init__`` only constructs the architecture (random init), so the
    embeddings are meaningless until the checkpoint is loaded. Both Detectorv1 and
    Detectorv2 need that load; keep it here as the single source of truth.

    strict=False because BatchNorm's ``num_batches_tracked`` buffer isn't in the
    converted safetensors (it's absent from the source ONNX). The missing/
    unexpected keys are validated so a wrong- or empty-file load fails loudly
    rather than silently producing garbage embeddings.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from feat.utils.io import get_resource_path

    detector = ArcFace(backbone=backbone)
    arcface_path = os.environ.get("FEAT_ARCFACE_R50_PATH")
    if arcface_path is None:
        arcface_path = hf_hub_download(
            repo_id="py-feat/arcface_r50",
            filename="arcface_r50.safetensors",
            cache_dir=get_resource_path(),
        )
    missing, unexpected = detector.net.load_state_dict(
        load_file(arcface_path), strict=False
    )
    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    if real_missing or unexpected:
        raise RuntimeError(
            f"ArcFace weights at {arcface_path!r} are inconsistent "
            f"with the architecture. Missing: {real_missing}; "
            f"unexpected: {list(unexpected)}. Re-download from "
            f"py-feat/arcface_r50 or re-run "
            f"scripts/convert_arcface_onnx_to_safetensors.py."
        )
    detector.eval()
    detector.to(device)
    return detector
