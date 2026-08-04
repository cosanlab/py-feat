#!/usr/bin/env python
"""Export a multitask training checkpoint (.pt) to the safetensors distribution
format py-feat downloads from the hub, and optionally upload it.

The .safetensors carries the model state_dict plus the ModelV2Config as a JSON
string under metadata key 'config' — exactly what
feat.multitask.inference._load_multitask_weights expects.

Ship NEW versions under a NEW filename (e.g. face_multitask_v26.safetensors)
and bump HF_WEIGHTS_FILE in feat/multitask/inference.py in the same release:
older py-feat code cannot build newer architectures, so replacing the file
in-place would break every existing install at download time.

    python scripts/export_multitask_safetensors.py \
        --checkpoint /path/v24_best.pt \
        --out face_multitask_v26.safetensors \
        [--upload py-feat/face_multitask_v2]
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--upload", default=None,
                    help="HF repo id to upload to (e.g. py-feat/face_multitask_v2). "
                         "Omit to only write the local file.")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if not cfg:
        raise SystemExit(f"{args.checkpoint}: no saved config — cannot export")
    sd = {k: v.contiguous() for k, v in ckpt["model"].items()}

    # Only ModelV2Config-relevant keys; training-only args stay out of the wheel.
    from feat.multitask.model_v2 import ModelV2Config
    valid = set(ModelV2Config.__dataclass_fields__.keys())
    cfg = {k: v for k, v in cfg.items() if k in valid}

    meta = {"config": json.dumps(cfg),
            "source_checkpoint": args.checkpoint.name,
            "format": "pt"}
    save_file(sd, args.out, metadata=meta)
    mb = args.out.stat().st_size / 1e6
    print(f"[export] {args.out} ({mb:.1f} MB, {len(sd)} tensors)")

    # Round-trip verification through the real loader before any upload.
    from feat.multitask.inference import MultitaskModel
    m = MultitaskModel(device="cpu", weights_path=str(args.out))
    n = sum(p.numel() for p in m.model.parameters()) / 1e6
    print(f"[verify] loads via MultitaskModel ({n:.1f}M params)")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(path_or_fileobj=str(args.out),
                        path_in_repo=args.out.name, repo_id=args.upload)
        print(f"[upload] {args.upload}/{args.out.name}")


if __name__ == "__main__":
    main()
