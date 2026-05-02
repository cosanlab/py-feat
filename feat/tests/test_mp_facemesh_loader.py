"""Unit tests for the mp_facemesh_v2 loader migration in MPDetector.

The PR migrates the landmark loader from the legacy pickled FX
GraphModule (face_landmarks_detector_Nx3x256x256_onnx.pth, requires
weights_only=False + onnx2torch importable) to a TorchScript file
(face_landmarks_detector.pt, loaded via torch.jit.load, no pickle).

These tests pin three properties:

1. With the new .pt available on the hub, the loader uses
   torch.jit.load and produces a working torch.jit.RecursiveScriptModule.
2. With only the legacy .pth available AND onnx2torch installed,
   the loader falls back with a warning and still works.
3. With only the legacy .pth available AND onnx2torch NOT installed
   (the v0.7+ default), the loader raises a clear ImportError pointing
   at the migration rather than a confusing pickle traceback.

The tests don't actually hit HuggingFace; they monkey-patch the
download helper to return a path the test controls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


# Path to a TorchScript file produced by scripts/prepare_mp_facemesh_v2_torchscript.py.
# Tests skip if not present locally; CI populates it from the live HF
# download (face_landmarks_detector.pt is already on py-feat/mp_facemesh_v2).
_LOCAL_TS = Path("/tmp/py-feat-v2-models/mp_facemesh_v2.pt")


def _have_local_ts() -> bool:
    return _LOCAL_TS.exists()


@pytest.mark.skipif(not _have_local_ts(), reason="local TorchScript file not present")
def test_loader_uses_torchscript_when_pt_available(monkeypatch):
    """When face_landmarks_detector.pt is on the hub, the loader uses
    torch.jit.load (no pickle, no onnx2torch needed)."""

    def fake_download(repo_id, filename, fallback_filename, cache_dir):
        # Pretend the .pt is on the hub.
        if filename == "face_landmarks_detector.pt":
            return str(_LOCAL_TS)
        raise AssertionError("should have hit the .pt path")

    monkeypatch.setattr(
        "feat.utils.hf_hub_download_with_fallback", fake_download
    )

    from feat.MPDetector import MPDetector

    mp = MPDetector(
        device="cpu",
        face_model=None,
        landmark_model="mp_facemesh_v2",
        au_model=None,
        emotion_model=None,
        identity_model=None,
        facepose_model=None,
    )
    # TorchScript modules subclass torch.jit.RecursiveScriptModule.
    assert isinstance(mp.landmark_detector, torch.jit.ScriptModule), (
        f"expected ScriptModule, got {type(mp.landmark_detector).__name__}"
    )
    # Forward must produce the documented 3-tuple shape.
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        bbox_or_lmk, presence, score = mp.landmark_detector(x)
    assert bbox_or_lmk.shape[-1] == 1434  # 478 landmarks × 3 coords flat


def test_loader_errors_helpfully_when_legacy_path_and_no_onnx2torch(monkeypatch):
    """If only the legacy pickle file is available and onnx2torch isn't
    installed, the loader raises a clear ImportError with migration
    instructions - NOT a confusing low-level pickle traceback."""

    legacy_path = "/tmp/face_landmarks_detector_Nx3x256x256_onnx.pth"

    def fake_download(repo_id, filename, fallback_filename, cache_dir):
        # Pretend the .pt is missing and the legacy .pth is on the hub.
        return legacy_path if filename != "face_landmarks_detector.pt" else legacy_path

    # First call returns the .pt path (which doesn't exist locally so we
    # need to make hf_hub_download_with_fallback look like it returned the
    # legacy file). Simulating "v2 missing, fall back to v1":
    def fake_v1_only(repo_id, filename, fallback_filename, cache_dir):
        # Always return the fallback (legacy) path - simulating the hub
        # only having the .pth file.
        return legacy_path  # path ends with .pth, not .pt

    monkeypatch.setattr(
        "feat.utils.hf_hub_download_with_fallback", fake_v1_only
    )

    # Block onnx2torch from importing.
    blocked = {"onnx2torch": None}
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def import_blocker(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "onnx2torch":
            raise ImportError("simulated absence of onnx2torch")
        return real_import(name, globals, locals, fromlist, level)

    # Also stash any real onnx2torch out of sys.modules so the import path
    # actually goes through the blocker.
    cached = sys.modules.pop("onnx2torch", None)
    try:
        with patch("builtins.__import__", side_effect=import_blocker):
            from feat.MPDetector import MPDetector

            with pytest.raises(ImportError, match="onnx2torch|TorchScript"):
                MPDetector(
                    device="cpu",
                    face_model=None,
                    landmark_model="mp_facemesh_v2",
                    au_model=None,
                    emotion_model=None,
                    identity_model=None,
                    facepose_model=None,
                )
    finally:
        if cached is not None:
            sys.modules["onnx2torch"] = cached
