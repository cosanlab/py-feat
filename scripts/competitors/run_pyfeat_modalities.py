"""py-feat emotion / valence-arousal / gaze eval -> per-tool JSON.

Companion to the AU cross-tool benchmark. Drives the SAME `feat.evaluation`
detect path the AU harness uses (so face detection / cropping is identical),
then scores:

  * AffectNet-val : 7-class emotion (top-1 accuracy + macro F1) and, for v2,
                    valence/arousal CCC (Detectorv2 emits `valence`/`arousal`;
                    the stock runner stubs these out, so we score them here).
  * Columbia Gaze : mean angular error (deg), head-frontal subset.

    PYFEAT_DATA_ROOT=/Storage/Data .venv/bin/python \
        scripts/competitors/run_pyfeat_modalities.py --detector v2 \
        --dataset affectnet columbia \
        --out docs/benchmarks/competitors/pyfeat_modalities_v2.json
"""

import argparse
import datetime
import json
import socket
import warnings

warnings.simplefilter("ignore")

import numpy as np
import torch

from feat.evaluation import datasets as ds
from feat.evaluation.metrics import (
    concordance_correlation_coefficient as ccc,
    emotion_accuracy,
    emotion_f1_macro,
)
from feat.evaluation.runner import _detect, _score_gaze, _top_face_per_frame

# Emotion column name -> AffectNet expression_int (0..6). Covers both the v1
# lowercase set (`FEAT_EMOTION_COLUMNS`) and the v2 capitalized set
# (`EMOTION_COLUMNS_V2`); AffectNet has no Contempt so v2's 7 align 1:1.
EMO_NAME_TO_AFFECTNET = {
    "neutral": 0, "Neutral": 0,
    "happiness": 1, "Happy": 1,
    "sadness": 2, "Sad": 2,
    "surprise": 3, "Surprise": 3,
    "fear": 4, "Fear": 4,
    "disgust": 5, "Disgust": 5,
    "anger": 6, "Anger": 6,
}


def _build(which, device):
    if which == "v2":
        from feat.detector_v2 import Detectorv2

        return Detectorv2(device=device, identity_model=None)
    from feat.detector import Detectorv1

    return Detectorv1(device=device)


def _predict(detector, paths, batch_size):
    fex = _detect(detector, paths, batch_size, 0, output_size=512)
    fex = _top_face_per_frame(fex)
    return fex.set_index("frame").reindex(range(len(paths))).reset_index(drop=True)


def _score_emotion_va(fex, labels):
    # emotion columns in fex order (handles both v1 and v2 naming)
    emo_cols = [c for c in fex.columns if c in EMO_NAME_TO_AFFECTNET]
    valid = fex[emo_cols].notna().all(axis=1).values
    y_pred_int = np.full(len(fex), -1, dtype=int)
    if valid.any():
        amax = fex.loc[valid, emo_cols].values.argmax(axis=1)
        y_pred_int[valid] = [EMO_NAME_TO_AFFECTNET[emo_cols[i]] for i in amax]
    y_true = labels["expression_int"].to_numpy()
    m = y_pred_int >= 0
    out = {
        "n_scored": int(m.sum()),
        "emotion_accuracy": round(emotion_accuracy(y_true[m], y_pred_int[m]), 4),
        "emotion_f1_macro": round(emotion_f1_macro(y_true[m], y_pred_int[m]), 4),
    }
    if {"valence", "arousal"}.issubset(fex.columns) and "valence" in labels and "arousal" in labels:
        vt, at = labels["valence"].to_numpy(), labels["arousal"].to_numpy()
        vp, ap = fex["valence"].to_numpy(), fex["arousal"].to_numpy()
        vm = m & np.isfinite(vp) & np.isfinite(ap)
        out["valence_ccc"] = round(ccc(vt[vm], vp[vm]), 4)
        out["arousal_ccc"] = round(ccc(at[vm], ap[vm]), 4)
        out["valence_arousal_supported"] = True
    else:
        out["valence_arousal_supported"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["v1", "v2"], default="v2")
    ap.add_argument("--dataset", nargs="+", default=["affectnet", "columbia"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--emotion-manifest", default=None,
                    help="CSV (image_path,expression_int) for emotion-only "
                         "scoring of an arbitrary dataset, e.g. RAF-DB.")
    ap.add_argument("--manifest-name", default="emotion_manifest")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    detector = _build(args.detector, args.device)
    results = {}

    if args.emotion_manifest:
        import pandas as pd
        mdf = pd.read_csv(args.emotion_manifest)
        fex = _predict(detector, mdf["image_path"].tolist(), args.batch_size)
        labels = pd.DataFrame({"expression_int": mdf["expression_int"].astype(int).values})
        results[args.manifest_name] = {
            "n_samples": len(mdf),
            **_score_emotion_va(fex, labels),
        }
        print(args.manifest_name, ":", results[args.manifest_name], flush=True)

    if "affectnet" in args.dataset:
        s = ds.load_affectnet_val()
        fex = _predict(detector, s.image_paths, args.batch_size)
        results["affectnet_val"] = {
            "n_samples": len(s.image_paths),
            **_score_emotion_va(fex, s.labels),
        }
        print("affectnet:", results["affectnet_val"], flush=True)

    if "columbia" in args.dataset:
        s = ds.load_columbia_gaze()
        if s is None:
            print("columbia: not found", flush=True)
        else:
            fex = _predict(detector, s.image_paths, args.batch_size)
            results["columbia_gaze"] = {
                "n_samples": len(s.image_paths),
                **_score_gaze(fex, s.labels),
            }
            print("columbia:", results["columbia_gaze"], flush=True)

    payload = {
        "model": f"pyfeat_{args.detector}",
        "date": datetime.datetime.now().isoformat(),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
