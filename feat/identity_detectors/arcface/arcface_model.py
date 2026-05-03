"""ArcFace face-recognition wrapper.

Loads InsightFace's ArcFace ResNet50 (the ``buffalo_l`` pack's
``w600k_r50.onnx`` converted to PyTorch ``.safetensors``) and exposes a
``forward(face_crops) -> [N, 512]`` interface compatible with the
existing facenet identity detector in ``feat.detector.Detector`` and
``feat.MPDetector.MPDetector``.

Why ArcFace
-----------
The default identity detector in py-feat is ``facenet`` (InceptionResnetV1
trained with triplet loss on VGGFace2). Triplet-loss embeddings are
known to entangle identity with pose and expression — a serious problem
for FEAT's intended use case (clustering identities across video frames
where pose and expression vary deliberately).

ArcFace replaces triplet loss with an angular-margin softmax loss that
constrains identities to occupy disjoint angular regions of the
embedding sphere. The result is much tighter intra-identity clusters
under pose/expression variation. Published numbers on IJB-C (the
hardest standard face-recognition benchmark, with diverse poses and
expressions) put ArcFace-R50/WebFace600K at ~96% TAR @ FAR=1e-4 vs.
~80% for FaceNet/VGGFace2.

License
-------
The InsightFace code is MIT-licensed. The pretrained weights are
distributed under InsightFace's "non-commercial research" terms; see
https://github.com/deepinsight/insightface for the model card. py-feat
inherits the same license-risk profile that already applied to the
default facenet weights (VGGFace2-trained, also research-only).
Commercial users should validate license compatibility for their use
case.
"""

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
