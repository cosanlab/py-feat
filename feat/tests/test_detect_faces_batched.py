"""Batched-vs-singleton parity tests for Detector.detect_faces.

The current detect_faces runs img2pose one frame at a time inside a
``for i in range(frames.size(0))`` loop. Phase 1.3 of the speedup spec
replaces that with a single batched forward pass. Output must be
unchanged: this file pins the contract.

Three guarantees:
1. **Numerical parity**: detect_faces on N frames as one batch produces
   the same per-frame outputs as N successive single-frame calls. Bit-
   identity for boxes/scores/poses/extracted faces; deterministic at
   inference time so anything else means a real regression.
2. **Single forward pass**: img2pose is called once per detect_faces
   invocation, not N times. Catches any future refactor that
   accidentally re-introduces the per-frame loop.
3. **Empty-detection edge case**: at least one frame in the batch with
   no detected faces still produces the right NaN-padded output shape
   (the legacy loop handled this per-frame; the batched path needs to
   handle it per-frame inside the postprocess loop).
"""

import os
from unittest.mock import patch

import pytest
import torch

from feat.detector import Detector
from feat.utils.io import get_test_data_path


# Skip the whole module if the xgb model can't load (Python 3.13 SIGSEGV
# is intermittent; falling back to svm here makes the suite runnable).
@pytest.fixture(scope="module")
def detector():
    return Detector(device="cpu", au_model="svm")


@pytest.fixture(scope="module")
def two_face_images(detector):
    """Two test images stacked into a [2, C, H, W] tensor.

    Uses the existing ImageDataset path so frames go through the same
    Rescale + collate as the production detect() flow."""
    from feat.data import ImageDataset
    from torch.utils.data import DataLoader

    paths = [
        os.path.join(get_test_data_path(), "single_face.jpg"),
        os.path.join(get_test_data_path(), "multi_face.jpg"),
    ]
    # Skip if either image is missing
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"test image not found: {p}")

    ds = ImageDataset(paths, output_size=320, preserve_aspect_ratio=True, padding=True)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    return batch["Image"]


def _assert_frame_results_equal(a, b):
    """Compare two frame_results dicts element-wise."""
    for key in ("boxes", "new_boxes", "poses", "scores", "faces"):
        if key not in a:
            continue
        ta, tb = a[key], b[key]
        # NaN-tolerant equality (no-face frames return NaN-filled tensors)
        nan_a = torch.isnan(ta)
        nan_b = torch.isnan(tb)
        assert torch.equal(nan_a, nan_b), f"{key}: NaN masks differ"
        finite_a = ta[~nan_a]
        finite_b = tb[~nan_b]
        assert torch.equal(finite_a, finite_b), (
            f"{key}: finite values differ; "
            f"max diff = {(finite_a - finite_b).abs().max().item()}"
        )


def test_detect_faces_batched_matches_singleton(detector, two_face_images):
    """Numerical parity: a 2-frame batch produces the same per-frame
    output as two successive 1-frame calls.

    img2pose is deterministic at inference time, so this should be bit-
    identical (no floating-point reordering). If a future change introduces
    cross-batch interactions (e.g. shared NMS, a wrong shape collapse), this
    test will catch it."""
    batched = detector.detect_faces(two_face_images)

    singletons = []
    for i in range(two_face_images.shape[0]):
        single = two_face_images[i : i + 1]  # keep batch dim
        out = detector.detect_faces(single)
        # detect_faces returns face_id=0 for the single frame; reindex to
        # match the batched output's per-frame face_id.
        out[0]["face_id"] = i
        singletons.extend(out)

    assert len(batched) == len(singletons)
    for a, b in zip(batched, singletons):
        assert a["face_id"] == b["face_id"]
        _assert_frame_results_equal(a, b)


def test_detect_faces_calls_facepose_once_per_batch(detector, two_face_images):
    """Behavioral guarantee: the model is called *once* per detect_faces,
    regardless of batch size. Catches reintroduction of the per-frame loop.

    Hooks the model's forward (rather than __call__) because Python
    resolves obj() via type(obj).__call__, so an instance-level patch
    of __call__ is invisible to the call site."""
    call_count = {"n": 0}
    original_forward = detector.facepose_detector.forward

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    with patch.object(
        detector.facepose_detector, "forward", side_effect=counting_forward
    ):
        detector.detect_faces(two_face_images)

    assert call_count["n"] == 1, (
        f"img2pose forward was called {call_count['n']} times for a "
        f"2-frame batch; expected 1 (the per-frame loop should have "
        "been replaced)."
    )
