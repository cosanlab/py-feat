"""Convert an InsightFace ArcFace ONNX checkpoint to a PyTorch
safetensors file usable by ``feat.identity_detectors.arcface``.

Usage:

    python scripts/convert_arcface_onnx_to_safetensors.py \\
        --onnx /path/to/w600k_r50.onnx \\
        --out  feat/resources/arcface_r50.safetensors \\
        --backbone r50

Then verify embedding equality between ONNX and PyTorch:

    python scripts/convert_arcface_onnx_to_safetensors.py \\
        --onnx /path/to/w600k_r50.onnx \\
        --out  feat/resources/arcface_r50.safetensors \\
        --backbone r50 \\
        --verify

Why this script exists
----------------------
InsightFace distributes its ArcFace recognition models as ONNX in the
``buffalo_l`` / ``antelopev2`` packs at
``github.com/deepinsight/insightface/releases``. py-feat is pure PyTorch
and ships weights as ``.safetensors``. The InsightFace ONNX exporter
folds most BatchNorms into adjacent Convs (saves ~30% file size and
inference time); our ``feat.identity_detectors.arcface.iresnet`` mirrors
that fused structure so initializer names map 1:1 with no surgery.

Only PReLU slope params and Conv biases need positional/manual mapping
because they're stored under numeric ONNX initializer ids rather than
PyTorch-style names. The rest (Conv weights, BN params, FC weight/bias,
final BN1d weight/bias/running stats) carry their PyTorch names through
the ONNX export.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import torch

# Use absolute import path so the script runs from anywhere; rely on
# PYTHONPATH or a `pip install -e .` to expose `feat`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feat.identity_detectors.arcface.iresnet import iresnet50, iresnet100


def _onnx_initializer_dict(model_path: str) -> Dict[str, np.ndarray]:
    """Read an ONNX file and return ``{initializer_name: numpy_array}``."""
    m = onnx.load(model_path)
    out: Dict[str, np.ndarray] = {}
    for init in m.graph.initializer:
        out[init.name] = onnx.numpy_helper.to_array(init)
    return out


def _build_state_dict(
    onnx_inits: Dict[str, np.ndarray],
    onnx_model: onnx.ModelProto,
    torch_model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Map ONNX initializers onto the IResNet PyTorch parameters.

    The named initializers (``layer1.0.bn1.weight`` etc.) align with
    PyTorch parameter names by construction. The unnamed numeric
    initializers carry Conv bias and PReLU slopes; we identify them by
    walking the graph in topological order and matching node consumers
    to model module names.
    """
    state_dict: Dict[str, torch.Tensor] = {}

    # 1) Direct-by-name copies for every initializer whose name already
    # matches a parameter or buffer name in the PyTorch module.
    pytorch_names = set()
    for n, _ in torch_model.named_parameters():
        pytorch_names.add(n)
    for n, _ in torch_model.named_buffers():
        pytorch_names.add(n)

    for ini_name, arr in onnx_inits.items():
        if ini_name in pytorch_names:
            state_dict[ini_name] = torch.from_numpy(arr.copy())

    # 2) For Conv nodes: match the Conv's weight initializer (stored
    # directly with a numeric name in the ONNX) and bias initializer to
    # the right module by walking nodes in topological order alongside
    # the Conv layers in the model.
    pytorch_convs = []
    pytorch_prelus = []
    for name, module in torch_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pytorch_convs.append((name, module))
        elif isinstance(module, torch.nn.PReLU):
            pytorch_prelus.append((name, module))

    onnx_convs = [n for n in onnx_model.graph.node if n.op_type == "Conv"]
    onnx_prelus = [n for n in onnx_model.graph.node if n.op_type == "PRelu"]

    if len(onnx_convs) != len(pytorch_convs):
        raise RuntimeError(
            f"Conv count mismatch: ONNX has {len(onnx_convs)}, "
            f"PyTorch has {len(pytorch_convs)}"
        )
    if len(onnx_prelus) != len(pytorch_prelus):
        raise RuntimeError(
            f"PReLU count mismatch: ONNX has {len(onnx_prelus)}, "
            f"PyTorch has {len(pytorch_prelus)}"
        )

    # 3) Conv: weights are at input[1], bias at input[2] (when present).
    for (pname, pmod), onode in zip(pytorch_convs, onnx_convs):
        w_name = onode.input[1]
        w = onnx_inits[w_name]
        if w.shape != tuple(pmod.weight.shape):
            raise RuntimeError(
                f"Conv shape mismatch for {pname}: "
                f"PyTorch={tuple(pmod.weight.shape)} vs ONNX={w.shape}"
            )
        state_dict[f"{pname}.weight"] = torch.from_numpy(w.copy())
        if pmod.bias is not None and len(onode.input) > 2:
            b = onnx_inits[onode.input[2]]
            state_dict[f"{pname}.bias"] = torch.from_numpy(b.copy())

    # 4) PReLU: slope at input[1]. ONNX stores it as [C, 1, 1] for 4D
    # broadcast; PyTorch wants [C].
    for (pname, pmod), onode in zip(pytorch_prelus, onnx_prelus):
        slope_name = onode.input[1]
        slope = onnx_inits[slope_name]
        slope = np.squeeze(slope)
        if slope.shape != tuple(pmod.weight.shape):
            raise RuntimeError(
                f"PReLU slope shape mismatch for {pname}: "
                f"PyTorch={tuple(pmod.weight.shape)} vs ONNX={slope.shape}"
            )
        state_dict[f"{pname}.weight"] = torch.from_numpy(slope.copy())

    # 5) Linear (Gemm): there's exactly one in the head. The ONNX naming
    # already uses `fc.weight` and `fc.bias` so step 1 covered it.

    return state_dict


def _verify(onnx_path: str, torch_model: torch.nn.Module) -> None:
    """Run the same input through ONNX and the PyTorch model, compare."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).randn(2, 3, 112, 112).astype(np.float32)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x)).numpy()

    abs_diff = np.abs(onnx_out - torch_out)
    print(f"  ONNX out shape: {onnx_out.shape}")
    print(f"  PyTorch out shape: {torch_out.shape}")
    print(f"  max |diff|: {abs_diff.max():.3e}")
    print(f"  mean |diff|: {abs_diff.mean():.3e}")
    # Cosine similarity per row
    cos = (onnx_out * torch_out).sum(1) / (
        np.linalg.norm(onnx_out, axis=1) * np.linalg.norm(torch_out, axis=1) + 1e-12
    )
    print(f"  per-row cosine similarity: {cos}")
    if abs_diff.max() > 1e-3:
        raise SystemExit(
            f"Verification failed: max |diff| = {abs_diff.max():.3e} exceeds 1e-3."
        )
    print("  Verification PASSED.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, type=str, help="Path to source .onnx")
    p.add_argument("--out", required=True, type=str, help="Output .safetensors path")
    p.add_argument(
        "--backbone",
        default="r50",
        choices=["r50", "r100"],
        help="Backbone architecture (decides 50- vs 100-layer).",
    )
    p.add_argument("--verify", action="store_true", help="Compare ONNX and PyTorch outputs")
    args = p.parse_args()

    print(f"Reading ONNX initializers from {args.onnx}...")
    onnx_inits = _onnx_initializer_dict(args.onnx)
    onnx_model = onnx.load(args.onnx)
    print(f"  {len(onnx_inits)} initializers found.")

    print(f"Building IResNet-{args.backbone} PyTorch module...")
    torch_model = (iresnet50() if args.backbone == "r50" else iresnet100())
    torch_model.eval()

    print("Mapping ONNX initializers onto PyTorch state_dict...")
    state_dict = _build_state_dict(onnx_inits, onnx_model, torch_model)

    # Sanity: confirm we covered every PyTorch param/buffer.
    expected = set()
    for n, _ in torch_model.named_parameters():
        expected.add(n)
    for n, _ in torch_model.named_buffers():
        if "num_batches_tracked" not in n:
            expected.add(n)
    missing = expected - set(state_dict.keys())
    if missing:
        raise SystemExit(f"State dict missing {len(missing)} entries:\n  " + "\n  ".join(sorted(missing)))
    print(f"  state_dict has {len(state_dict)} tensors covering all expected params.")

    print("Loading state_dict into PyTorch model (strict)...")
    missing_keys, unexpected_keys = torch_model.load_state_dict(state_dict, strict=False)
    # `num_batches_tracked` is not in ONNX; this is fine.
    real_missing = [k for k in missing_keys if "num_batches_tracked" not in k]
    if real_missing:
        raise SystemExit(f"Strict load failed: missing {real_missing}")
    if unexpected_keys:
        raise SystemExit(f"Strict load failed: unexpected {unexpected_keys}")
    print("  Loaded cleanly.")

    if args.verify:
        print("Verifying ONNX vs PyTorch on random input...")
        _verify(args.onnx, torch_model)

    # Save as safetensors. State dict from load_state_dict isn't quite
    # the right thing — re-export the loaded module's state_dict to
    # include things like `num_batches_tracked` defaults.
    from safetensors.torch import save_file

    final_state = {k: v.contiguous() for k, v in torch_model.state_dict().items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(final_state, args.out)
    sz_mb = Path(args.out).stat().st_size / (1024 * 1024)
    print(f"Wrote {args.out} ({sz_mb:.1f} MB).")


if __name__ == "__main__":
    main()
