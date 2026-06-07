"""Run LibreFace AU detection over a DISFA+ manifest -> per-AU F1 JSON.

Runs in the ISOLATED LibreFace env (NOT py-feat):
    ~/benchmark-envs/libreface/bin/python scripts/competitors/run_libreface_disfaplus.py \
        --manifest scripts/competitors/disfaplus_manifest.csv \
        --out bench-results/au_local/libreface_disfaplus.json

LibreFace `au_intensities` are on the DISFA 0..5 intensity scale (it was
trained on DISFA), so we binarize BOTH the prediction and the ground truth at
>= 2 — matching feat.evaluation.metrics (truth >= 2). Output schema mirrors the
existing OpenFace3 result so all tools drop into the same comparison.
"""

import argparse
import csv
import datetime
import json
import socket

# DISFA+ AU column -> LibreFace au_intensities key.
AU_TO_LF = {
    "AU01": "au_1_intensity", "AU02": "au_2_intensity", "AU04": "au_4_intensity",
    "AU05": "au_5_intensity", "AU06": "au_6_intensity", "AU09": "au_9_intensity",
    "AU12": "au_12_intensity", "AU15": "au_15_intensity", "AU17": "au_17_intensity",
    "AU20": "au_20_intensity", "AU25": "au_25_intensity", "AU26": "au_26_intensity",
}
AUS = list(AU_TO_LF)
TRUTH_THRESH = 2.0   # DISFA intensity >= 2 -> positive (feat convention)
PRED_THRESH = 2.0    # LibreFace intensity is on the same 0..5 scale


def f1(tp, fp, fn):
    if tp == 0 and (fp == 0 or fn == 0):
        return 0.0 if (fp or fn) else 0.0
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import libreface  # noqa: E402  (isolated env)

    rows = list(csv.DictReader(open(args.manifest)))
    if args.limit:
        rows = rows[: args.limit]

    # counts per AU
    tp = dict.fromkeys(AUS, 0)
    fp = dict.fromkeys(AUS, 0)
    fn = dict.fromkeys(AUS, 0)
    n_ok = 0
    for i, row in enumerate(rows):
        try:
            attrs = libreface.get_facial_attributes(row["image_path"], device=args.device)
            inten = attrs["au_intensities"]
        except Exception:
            continue
        n_ok += 1
        for au in AUS:
            yt = float(row[au]) >= TRUTH_THRESH
            yp = float(inten.get(AU_TO_LF[au], 0.0)) >= PRED_THRESH
            if yt and yp:
                tp[au] += 1
            elif yp and not yt:
                fp[au] += 1
            elif yt and not yp:
                fn[au] += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)} frames", flush=True)

    per_au_f1 = {au: round(f1(tp[au], fp[au], fn[au]), 4) for au in AUS}
    mean_f1 = round(sum(per_au_f1.values()) / len(AUS), 4)

    payload = {
        "model": "libreface",
        "date": datetime.datetime.now().isoformat(),
        "host": socket.gethostname(),
        "dataset": "disfaplus",
        "n_predicted": n_ok,
        "truth_threshold": TRUTH_THRESH,
        "pred_threshold": PRED_THRESH,
        "predicted_aus": AUS,
        "missing_aus": [],
        "per_au_f1": per_au_f1,
        "mean_f1_12au": mean_f1,
        "license": "USC research-only",
        "note": "LibreFace au_intensities (DISFA 0..5 scale) binarized at >=2; "
                "ground truth DISFA+ intensity binarized at >=2.",
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"mean F1 (12 AU) = {mean_f1}  over {n_ok} frames -> {args.out}")


if __name__ == "__main__":
    main()
