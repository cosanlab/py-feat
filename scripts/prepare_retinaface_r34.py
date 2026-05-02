"""Maintainer tool: convert yakhyo's RetinaFace-R34 .pth to py-feat safetensors.

The upstream weights at
    https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.pth
use yakhyo's class layout:
    fx.* (backbone), fpn.*, ssh{1,2,3}.*, {class,bbox,landmark}_head.*

py-feat's class layout (see feat/face_detectors/Retinaface/Retinaface_model.py):
    body.* (backbone), fpn.*, ssh{1,2,3}.*, {Class,Bbox,Landmark}Head.*

This script:
1. Downloads (if needed) or reads the upstream .pth
2. Remaps state_dict keys: ``fx.* -> body.*``,
   ``class_head.class_head.{i}.* -> ClassHead._convs.{i}.*``, etc.
3. Strict-loads the remapped state_dict into a freshly-constructed
   ``feat.face_detectors.Retinaface.Retinaface_model.RetinaFace`` to
   verify zero key mismatches and that the model runs end-to-end.
4. Saves the remapped weights as ``model.safetensors``, ready for upload
   to the ``py-feat/retinaface_r34`` HuggingFace Hub repo.

Usage:
    python scripts/prepare_retinaface_r34.py \\
        --in retinaface_r34.pth \\
        --out model.safetensors

If ``--in`` is omitted the script will try to download the upstream
release into a temp file. Use ``--force`` to overwrite an existing
output file.

Once you upload the safetensors file, ``feat.face_detectors.Retinaface``
will be able to fetch it via the ``py-feat/retinaface_r34`` HF repo.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict


def remap_yakhyo_to_pyfeat(yakhyo_sd: Dict) -> Dict:
    """Map yakhyo state-dict keys to the py-feat module hierarchy.

    Backbone ``fx.*`` -> ``body.*`` (IntermediateLayerGetter wrapper rename).
    Heads ``class_head.class_head.{i}.{w,b}`` -> ``ClassHead._convs.{i}.{w,b}``
    and similarly for bbox / landmark heads.
    FPN and SSH names are identical between the two layouts, pass through.
    """
    out: Dict = {}
    for key, value in yakhyo_sd.items():
        if key.startswith("fx."):
            new_key = "body." + key[len("fx."):]
        elif key.startswith("class_head.class_head."):
            # class_head.class_head.{i}.{rest} -> ClassHead._convs.{i}.{rest}
            tail = key[len("class_head.class_head."):]  # "{i}.{rest}"
            new_key = f"ClassHead._convs.{tail}"
        elif key.startswith("bbox_head.bbox_head."):
            tail = key[len("bbox_head.bbox_head."):]
            new_key = f"BboxHead._convs.{tail}"
        elif key.startswith("landmark_head.landmark_head."):
            tail = key[len("landmark_head.landmark_head."):]
            new_key = f"LandmarkHead._convs.{tail}"
        else:
            new_key = key
        out[new_key] = value
    return out


def _download_upstream_to(path: Path) -> None:
    """Fetch yakhyo's published v0.0.1 weights into ``path``."""
    import urllib.request

    url = (
        "https://github.com/yakhyo/retinaface-pytorch/releases/download/"
        "v0.0.1/retinaface_r34.pth"
    )
    print(f"downloading {url} -> {path} ...")
    with urllib.request.urlopen(url) as resp, open(path, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert yakhyo retinaface_r34.pth into a py-feat safetensors "
            "file ready to upload to the py-feat/retinaface_r34 HF repo."
        )
    )
    parser.add_argument(
        "--in", dest="src", default=None,
        help="path to upstream retinaface_r34.pth (downloaded if omitted)"
    )
    parser.add_argument(
        "--out", dest="dst", required=True,
        help="path for the output .safetensors file"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite output file if it already exists"
    )
    args = parser.parse_args()

    dst = Path(args.dst)
    if dst.exists() and not args.force:
        parser.error(f"output exists; pass --force to overwrite: {dst}")

    # Late imports so --help works without torch etc. installed.
    import torch
    from safetensors.torch import save_file
    from feat.face_detectors.Retinaface.Retinaface_model import RetinaFace

    if args.src is None:
        tmp = Path(tempfile.gettempdir()) / "retinaface_r34_upstream.pth"
        if not tmp.exists():
            _download_upstream_to(tmp)
        src = tmp
    else:
        src = Path(args.src)
        if not src.exists():
            parser.error(f"input file does not exist: {src}")

    print(f"loading {src} ...")
    yakhyo_sd = torch.load(src, map_location="cpu", weights_only=True)
    print(f"  {len(yakhyo_sd)} keys")

    print("remapping keys to py-feat layout ...")
    pyfeat_sd = remap_yakhyo_to_pyfeat(yakhyo_sd)

    print("verifying strict load into py-feat RetinaFace ...")
    model = RetinaFace()
    model.load_state_dict(pyfeat_sd, strict=True)
    model.eval()

    print("verifying forward pass on a dummy 320x320 input ...")
    x = torch.zeros(1, 3, 320, 320)
    with torch.no_grad():
        bbox, conf, ldm = model(x)
    print(
        f"  bbox: {tuple(bbox.shape)}, conf: {tuple(conf.shape)}, "
        f"ldm: {tuple(ldm.shape)}"
    )
    assert bbox.shape[0] == 1 and bbox.shape[2] == 4
    assert conf.shape[0] == 1 and conf.shape[2] == 2
    assert ldm.shape[0] == 1 and ldm.shape[2] == 10

    print(f"saving {dst} ...")
    save_file(pyfeat_sd, str(dst))

    print(f"done: {dst} ({dst.stat().st_size / (1 << 20):.1f} MB)")
    print(
        "\nNext: upload to HuggingFace via either\n"
        f"  huggingface-cli upload py-feat/retinaface_r34 {dst} model.safetensors\n"
        "or the HfApi() Python helper."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
