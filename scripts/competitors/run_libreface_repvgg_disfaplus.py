"""LibreFace (research RepVGG, DISFA-trained) evaluated on DISFA+ -> per-AU
F1 + PCC JSON.

Runs in the LibreFace research env (the pip `libreface` venv works, since it has
torch); the LibreFace repo must be on PYTHONPATH for `models.RepVGG`. **3090
only** — LibreFace's pinned torch has no Blackwell (sm_120) kernels.

    CUDA_VISIBLE_DEVICES=<3090> ~/benchmark-envs/libreface/bin/python \
        scripts/competitors/run_libreface_repvgg_disfaplus.py \
        --manifest scripts/competitors/disfaplus_aligned_manifest.csv \
        --repo ~/benchmark-envs/LibreFace-repo \
        --out bench-results/au_local/libreface_repvgg_disfaplus.json

Important: LibreFace was TRAINED on DISFA; DISFA+ is the held-out posed
benchmark, so this measures generalization. The model is the published
AU_Recognition RepVGG checkpoint; input is DISFA+'s own aligned crops fed
through LibreFace's own test transform (Resize 256 -> CenterCrop 224 ->
ImageNet norm), output clamped to [0,5] as in their solver. We report per-AU
PCC (LibreFace's native DISFA metric) and binary F1 (intensity >= 2 on both
prediction and ground truth, matching feat.evaluation's truth convention).
"""

import argparse
import csv
import datetime
import json
import socket
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]   # DISFA 12, RepVGG output order
AU_COLS = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU09",
           "AU12", "AU15", "AU17", "AU20", "AU25", "AU26"]
THRESH = 2.0  # intensity >= 2 -> positive (feat.evaluation truth convention)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo", required=True, help="LibreFace repo root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, f"{args.repo}/AU_Recognition")
    from models.RepVGG import RepVGG

    opts = argparse.Namespace(num_labels=12, dropout=0.1, hidden_dim=128,
                              half_precision=False, fm_distillation=False,
                              add_landmark=False)
    model = RepVGG(opts).cuda().eval()
    ckpt = f"{args.repo}/AU_Recognition/new_checkpoints_fm_repvgg/DISFA/all/repvgg.pt"
    model.load_state_dict(torch.load(ckpt)["model"], strict=True)
    print(f"loaded {ckpt} on {torch.cuda.get_device_name(0)}", flush=True)

    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = list(csv.DictReader(open(args.manifest)))
    if args.limit:
        rows = rows[: args.limit]

    preds, gts = [], []
    batch_imgs, batch_lbls = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).cuda()
        with torch.no_grad():
            out = torch.clamp(model(x) * 5.0, 0.0, 5.0).cpu().numpy()
        preds.append(out)
        gts.append(np.array(batch_lbls, dtype=np.float32))
        batch_imgs.clear()
        batch_lbls.clear()

    for i, r in enumerate(rows):
        try:
            img = tfm(Image.open(r["image_path"]).convert("RGB"))
        except Exception:
            continue
        batch_imgs.append(img)
        batch_lbls.append([float(r[c]) for c in AU_COLS])
        if len(batch_imgs) >= args.batch:
            flush()
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    flush()

    pred = np.concatenate(preds)   # [N, 12] intensity
    gt = np.concatenate(gts)       # [N, 12] intensity
    n = len(pred)

    per_au_f1, per_au_pcc = {}, {}
    for j, au in enumerate(AU_COLS):
        yt = (gt[:, j] >= THRESH).astype(int)
        yp = (pred[:, j] >= THRESH).astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        denom = 2 * tp + fp + fn
        per_au_f1[au] = round(2 * tp / denom, 4) if denom else 0.0
        pcc = np.corrcoef(pred[:, j], gt[:, j])[0, 1]
        per_au_pcc[au] = round(float(pcc), 4) if np.isfinite(pcc) else None

    payload = {
        "model": "libreface_repvgg",
        "date": datetime.datetime.now().isoformat(),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "dataset": "disfaplus",
        "n_predicted": n,
        "predicted_aus": AU_COLS,
        "per_au_f1": per_au_f1,
        "per_au_pcc": per_au_pcc,
        "mean_f1_12au": round(sum(per_au_f1.values()) / 12, 4),
        "mean_pcc_12au": round(
            np.mean([v for v in per_au_pcc.values() if v is not None]), 4),
        "f1_threshold": THRESH,
        "license": "USC research-only",
        "note": "LibreFace AU_Recognition RepVGG (DISFA-trained) on DISFA+ "
                "(held-out posed; measures generalization). DISFA+ aligned crops, "
                "LibreFace test transform, output x5 clamp [0,5]. F1 at intensity "
                ">= 2 both sides. Trained on DISFA, so DISFA+ is out-of-distribution.",
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"mean F1={payload['mean_f1_12au']}  mean PCC={payload['mean_pcc_12au']}  "
          f"(n={n}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
