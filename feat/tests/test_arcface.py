"""Tests for the ArcFace identity detector.

These tests exercise the architecture and the integration without
requiring the HuggingFace upload to exist. The full ONNX-vs-PyTorch
numerical equivalence test lives in
``scripts/convert_arcface_onnx_to_safetensors.py --verify`` and is
expected to be run once at conversion time, not on every CI run.

To run the smoke test that loads real weights, set
``FEAT_ARCFACE_R50_PATH`` to a local converted ``.safetensors`` file:

    FEAT_ARCFACE_R50_PATH=/path/to/arcface_r50.safetensors \\
        pytest feat/tests/test_arcface.py -k smoke
"""

import os
import pytest
import torch

from feat.identity_detectors.arcface.arcface_model import ArcFace
from feat.identity_detectors.arcface.iresnet import iresnet50, iresnet100


def test_iresnet50_shape():
    m = iresnet50()
    m.eval()
    x = torch.zeros(2, 3, 112, 112)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 512)


def test_iresnet100_shape():
    m = iresnet100()
    m.eval()
    x = torch.zeros(2, 3, 112, 112)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 512)


def test_iresnet50_param_count_matches_arcface_r50():
    """ArcFace-R50 has ~43.6M params. Catches unintended architecture
    drift (e.g. accidentally adding/removing a BN somewhere)."""
    m = iresnet50()
    n = sum(p.numel() for p in m.parameters())
    assert 43_500_000 < n < 43_700_000, f"Got {n} params"


def test_arcface_wrapper_accepts_arbitrary_input_size():
    """The wrapper must resize inputs to 112x112 internally so callers
    can pass through whatever face crops the rest of the pipeline
    produces (112 for retinaface_r34, 256 for MediaPipe)."""
    m = ArcFace(backbone="r50")
    m.eval()
    for in_size in [56, 112, 160, 224, 256]:
        x = torch.zeros(1, 3, in_size, in_size)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 512), f"Failed at in_size={in_size}"


def test_arcface_wrapper_normalizes_input():
    """The wrapper applies (2x - 1) to map [0,1] -> [-1,1]. Verify by
    feeding constant inputs and observing different outputs."""
    m = ArcFace(backbone="r50")
    m.eval()
    zeros = torch.zeros(1, 3, 112, 112)  # [0,1] -> -1
    ones = torch.ones(1, 3, 112, 112)    # [0,1] -> +1
    halves = torch.full((1, 3, 112, 112), 0.5)  # [0,1] -> 0
    with torch.no_grad():
        y_zeros = m(zeros)
        y_ones = m(ones)
        y_halves = m(halves)
    # Three different normalized inputs should produce three different
    # embeddings (otherwise the input pipeline isn't doing anything).
    assert not torch.allclose(y_zeros, y_ones)
    assert not torch.allclose(y_zeros, y_halves)
    assert not torch.allclose(y_halves, y_ones)


def test_arcface_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Unknown ArcFace backbone"):
        ArcFace(backbone="r37")


@pytest.mark.skipif(
    "FEAT_ARCFACE_R50_PATH" not in os.environ,
    reason="No local arcface weights available; set FEAT_ARCFACE_R50_PATH to enable.",
)
def test_arcface_smoke_load_and_forward():
    """Full smoke test with real weights — only runs when the local path
    env var is set, so CI doesn't need to download from HF."""
    from safetensors.torch import load_file

    m = ArcFace(backbone="r50")
    state = load_file(os.environ["FEAT_ARCFACE_R50_PATH"])
    missing, unexpected = m.net.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    assert not real_missing, f"Missing weights: {real_missing}"
    assert not unexpected, f"Unexpected weights: {unexpected}"

    m.eval()
    # Two random face-shaped tensors — expect non-degenerate, nonidentical embeddings.
    torch.manual_seed(0)
    x = torch.rand(2, 3, 112, 112)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 512)
    # Embeddings shouldn't collapse to zero or to identical vectors.
    assert y.norm(dim=1).min() > 1.0
    assert torch.nn.functional.cosine_similarity(y[0:1], y[1:2]).item() < 0.99
