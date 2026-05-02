"""Maintainer tool: convert the mp_facemesh_v2 FX GraphModule to TorchScript.

Why this exists
---------------
The mp_facemesh_v2 weights distributed at
    https://huggingface.co/py-feat/mp_facemesh_v2/blob/main/face_landmarks_detector_Nx3x256x256_onnx.pth
are a torch.fx.GraphModule produced by an ONNX -> PyTorch conversion via
``onnx2torch``. Loading them requires ``torch.load(weights_only=False)``,
which is a documented arbitrary-code-execution path, AND requires
``onnx2torch`` to be importable at load time. Both are issues we want to
get rid of (closes py-feat #249).

This script re-serializes the same model as a **TorchScript** module:
``torch.jit.load`` doesn't execute arbitrary Python at load time and
doesn't need ``onnx2torch`` to be present.

The conversion is **lossless**: the script verifies bit-identical outputs
between the original GraphModule and the traced TorchScript module across
multiple batch sizes (1, 2, 4, 8) on random inputs.

Usage
-----
    python scripts/prepare_mp_facemesh_v2_torchscript.py \\
        --in face_landmarks_detector_Nx3x256x256_onnx.pth \\
        --out face_landmarks_detector.pt

If ``--in`` is omitted, the script downloads the upstream HuggingFace file
into a temp path. Use ``--force`` to overwrite an existing output.

After running, upload the output file to ``py-feat/mp_facemesh_v2`` on
HuggingFace alongside the legacy file (don't replace it - older py-feat
installs still pin the legacy filename).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    parser.add_argument(
        "--in", dest="src", default=None,
        help="path to legacy face_landmarks_detector_Nx3x256x256_onnx.pth "
             "(downloaded from HF if omitted)"
    )
    parser.add_argument(
        "--out", dest="dst", required=True,
        help="path for the output TorchScript file (.pt)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite output file if it already exists"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="skip bit-identical-output parity check"
    )
    args = parser.parse_args()

    dst = Path(args.dst)
    if dst.exists() and not args.force:
        parser.error(f"output exists; pass --force to overwrite: {dst}")

    # Late imports so --help works without torch / huggingface_hub installed.
    import torch

    if args.src is None:
        from huggingface_hub import hf_hub_download

        print("downloading legacy file from py-feat/mp_facemesh_v2 ...")
        src = Path(hf_hub_download(
            repo_id="py-feat/mp_facemesh_v2",
            filename="face_landmarks_detector_Nx3x256x256_onnx.pth",
            cache_dir=tempfile.gettempdir(),
        ))
    else:
        src = Path(args.src)
        if not src.exists():
            parser.error(f"input file does not exist: {src}")

    print(f"loading {src} via torch.load(weights_only=False) ...")
    # weights_only=False is required for the legacy file - that's the
    # exact problem this script exists to fix. We pay it once here as a
    # maintainer running a known artifact.
    legacy = torch.load(src, map_location="cpu", weights_only=False)
    legacy.eval()
    print(f"  loaded {type(legacy).__name__}")

    print("tracing to TorchScript with a 1x3x256x256 example input ...")
    example = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        # strict=True enables the tracer's full correctness checks (tuple
        # output shapes, no Python control flow, no dict outputs). The
        # upstream FX GraphModule passes; confirmed at conversion time.
        traced = torch.jit.trace(legacy, example, strict=True)

    print(f"saving {dst} ...")
    torch.jit.save(traced, str(dst))

    print(f"verifying torch.jit.load works on {dst} ...")
    loaded = torch.jit.load(str(dst), map_location="cpu")
    loaded.eval()

    if not args.no_verify:
        print("verifying bit-identical outputs across batch sizes ...")
        torch.manual_seed(0)
        ok = True
        for batch in (1, 2, 4, 8):
            x = torch.randn(batch, 3, 256, 256)
            with torch.no_grad():
                a = legacy(x)
                b = loaded(x)
            for i, (ta, tb) in enumerate(zip(a, b)):
                # Tracing should produce a graph that's bit-identical to the
                # original GraphModule on the same inputs. Tightened from a
                # 1e-6 tolerance to literal 0.0 so any future regression
                # surfaces loud rather than getting swallowed by float drift.
                if not torch.equal(ta, tb):
                    ok = False
                    diff = (ta - tb).abs().max().item()
                    print(f"  batch={batch} out[{i}]: max diff {diff:.2e} (FAIL)")
        if ok:
            print("  all outputs bit-identical")
        else:
            print("ERROR: outputs diverged after TorchScript conversion", file=sys.stderr)
            return 2

    print(f"done: {dst} ({dst.stat().st_size / (1 << 20):.2f} MB)")
    print(
        "\nNext: upload to HuggingFace alongside the legacy file:\n"
        f"  hf upload py-feat/mp_facemesh_v2 {dst} face_landmarks_detector.pt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
