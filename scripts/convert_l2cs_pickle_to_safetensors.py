"""Convert upstream L2CS-Net pickle weights to safetensors.

Upstream Ahmednull/L2CS-Net distributes weights as ``L2CSNet_gaze360.pkl``
(a pickled state_dict on a Google Drive link). py-feat distributes
weights as ``.safetensors`` from the ``py-feat/l2cs`` HF repo to avoid
the pickle deserialization code path (no arbitrary-code-execution risk
on download).

This script: load .pkl (allow_pickle=True, weights_only=False, one-time
trusted-source operation), extract state_dict, strip the ``module.``
prefix if present, and save as a safetensors file.

Usage:
    # After downloading L2CSNet_gaze360.pkl from the upstream Google Drive
    python scripts/convert_l2cs_pickle_to_safetensors.py \\
        --input ~/Downloads/L2CSNet_gaze360.pkl \\
        --output models/l2cs_gaze360_resnet50.safetensors

Then ``huggingface-cli upload py-feat/l2cs models/l2cs_gaze360_resnet50.safetensors``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Upstream L2CSNet_gaze360.pkl path")
    p.add_argument("--output", type=Path, required=True,
                   help="Destination .safetensors path")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    print(f"loading {args.input} (pickle, trusted source)...")
    # weights_only=False because upstream stores a plain state_dict pickle.
    # The trade-off is acknowledged in this script's docstring: only run
    # on weights downloaded directly from the official L2CS-Net Drive.
    state = torch.load(str(args.input), map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Strip "module." prefix from DataParallel checkpoints
    state = {k.replace("module.", "", 1): v.contiguous() for k, v in state.items()}

    print(f"writing safetensors: {args.output} ({len(state)} tensors)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(args.output))
    print("done.")
    print()
    print("Next: upload to HuggingFace")
    print(f"  huggingface-cli upload py-feat/l2cs {args.output} \\")
    print(f"      l2cs_gaze360_resnet50.safetensors")
    print()
    print("Also upload feat/gaze_detectors/l2cs/MODEL_CARD.md as README.md:")
    print("  huggingface-cli upload py-feat/l2cs \\")
    print("      feat/gaze_detectors/l2cs/MODEL_CARD.md README.md")


if __name__ == "__main__":
    raise SystemExit(main())
