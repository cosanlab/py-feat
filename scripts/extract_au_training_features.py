"""Extract HOG features for AU classifier retraining.

Runs the v0.7 detection pipeline (face_model='img2pose' + mobilefacenet
landmarks + corrected HOGLayer) over the training-data sources, captures
HOG features at the point that feeds the AU classifier, and caches the
(hog_features, landmarks, au_labels, subject_id, source) tuple to disk.

This is the input the new xgb_au_v3 model will be trained on — features
extracted from the SAME pipeline users will run at inference time, so
training and inference see the same feature distribution.

Training data sources (subject-disjoint from DISFA+ benchmark):
- DISFA-original, 18 subjects (excludes SN001, SN003, SN004, SN007, SN009,
  SN010, SN013, SN025, SN027 which appear in DISFA+)
- CK+ peak frames (FACS-coded)
- EmotioNet (semi-automatic labels, downweighted)

Excluded from training (held out as benchmarks):
- DISFA+ (AU F1 benchmark)
- AffectNet val (emotion benchmark)

Output: ``<output-dir>/au_features.parquet`` plus a `.json` metadata file.

Usage:
    python scripts/extract_au_training_features.py \\
        --output-dir bench-cache/au_train \\
        --sources disfa18 ckplus emotionet \\
        --device cuda --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import torch

# DISFA+ subjects that must be excluded from DISFA training data.
DISFAPLUS_SUBJECTS = {
    "SN001", "SN003", "SN004", "SN007", "SN009",
    "SN010", "SN013", "SN025", "SN027",
}

# Canonical 20 py-feat AU columns (same order as XGBClassifier.au_keys
# in feat/au_detectors/StatLearning/SL_test.py, but using zero-padded
# names for consistency with the Detector output schema).
PYFEAT_AU_COLS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09",
    "AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20",
    "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
]

# Data root resolves the same way as feat/evaluation/datasets.py.
DATA_ROOT = Path(os.environ.get("PYFEAT_DATA_ROOT", "/Storage/Data"))


# ---------------------------------------------------------------------------
# Per-source label loaders. Each returns a DataFrame with columns:
#   image_path, subject, source, AU01..AU43 (NaN where unscored)
# ---------------------------------------------------------------------------


def load_disfa18_labels() -> pd.DataFrame:
    """DISFA original. By default excludes the 9 subjects that overlap with
    DISFA+ to avoid subject-level leakage into the DISFA+ benchmark.

    Set PYFEAT_INCLUDE_DISFA_OVERLAP=1 in the environment to include those
    9 subjects (SN001/003/004/007/009/010/013/025/027) for full-corpus
    training experiments. WARNING: doing so creates subject leakage with
    DISFA+ bench; the resulting model's DISFA+ F1 is partly memorization.
    """
    csv = DATA_ROOT / "DISFA_" / "DISFA_main_New.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv, index_col=0, low_memory=False)
    include_overlap = os.environ.get("PYFEAT_INCLUDE_DISFA_OVERLAP", "0") == "1"
    if not include_overlap:
        df = df[~df["subject"].isin(DISFAPLUS_SUBJECTS)].copy()
    else:
        print("  WARNING: DISFA loader including DISFA+ overlap subjects "
              f"({sorted(DISFAPLUS_SUBJECTS)}) — leakage risk if benching on DISFA+")
        df = df.copy()
    if df.empty:
        return df

    rename = {
        "AU1": "AU01", "AU2": "AU02", "AU4": "AU04", "AU5": "AU05",
        "AU6": "AU06", "AU9": "AU09", "AU12": "AU12", "AU15": "AU15",
        "AU17": "AU17", "AU20": "AU20", "AU25": "AU25", "AU26": "AU26",
    }
    df = df.rename(columns=rename)

    out = pd.DataFrame({
        "image_path": df["aligned_path"].astype(str).values,
        "subject": df["subject"].astype(str).values,
        "source": "disfa18",
    })
    # DISFA labels are intensities 0..5; copy with NaN for unscored AUs.
    for au in PYFEAT_AU_COLS:
        if au in df.columns:
            out[au] = df[au].astype(float).values
        else:
            out[au] = np.nan
    # Keep rows whose aligned image actually exists on disk.
    out = out[out["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return out


def load_ckplus_labels() -> pd.DataFrame:
    """CK+ peak-emotion frames with FACS coding.

    CK+ FACS labels live under ``FACS/<subject>/<seq>/<frame>_facs.txt``
    or similar. The labels file lists AU numbers + intensities per frame.
    """
    facs_root = DATA_ROOT / "CK+" / "FACS"
    img_root = DATA_ROOT / "CK+" / "cohn-kanade-images"
    if not facs_root.exists() or not img_root.exists():
        return pd.DataFrame()

    rows = []
    for subj_dir in sorted(facs_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        subject = subj_dir.name
        for seq_dir in sorted(subj_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            for facs_file in sorted(seq_dir.glob("*_facs.txt")):
                # Filename format: S010_001_00000014_facs.txt
                stem = facs_file.stem.replace("_facs", "")
                # The peak image is the frame referenced in the filename
                img_path = img_root / subject / seq_dir.name / f"{stem}.png"
                if not img_path.exists():
                    continue
                # Parse FACS file: lines like "  4 0   3 0   12 0" or
                # "<au_num> <intensity>".
                au_dict: dict[str, float] = {}
                try:
                    with facs_file.open() as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    au_num = int(float(parts[0]))
                                    intensity = float(parts[1])
                                    au_key = f"AU{au_num:02d}"
                                    if au_key in PYFEAT_AU_COLS:
                                        au_dict[au_key] = intensity if intensity > 0 else 1.0
                                except (ValueError, IndexError):
                                    continue
                except OSError:
                    continue
                row = {
                    "image_path": str(img_path),
                    "subject": subject,
                    "source": "ckplus",
                }
                # CK+ FACS files list ONLY active AUs; absent AUs are 0.
                for au in PYFEAT_AU_COLS:
                    row[au] = au_dict.get(au, 0.0)
                rows.append(row)
    return pd.DataFrame(rows)


def load_emotionet_labels() -> pd.DataFrame:
    """EmotioNet semi-automatic AU labels.

    The CSV has 60 AU columns; we filter to the 12 that overlap py-feat's
    output schema (AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20,
    AU25, AU26). 999 codes mean "not scored" -> NaN.

    Local images live under ``/Storage/Data/EmotioNet/imgs/`` or
    ``EmotioNet/Aligned/`` depending on which extraction is on disk.
    """
    csv = DATA_ROOT / "EmotioNet" / "labels" / "EmotioNet_FACS_aws_2020_24600.csv"
    if not csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv, low_memory=False)
    df.columns = [c.strip().strip("'").strip() for c in df.columns]

    # Locate images. EmotioNet local paths typically look like
    # ``imgs/N_0000000001/N_0000000001_00001.jpg`` mirroring the S3 URL.
    aligned_root = DATA_ROOT / "EmotioNet" / "Aligned"
    imgs_root = DATA_ROOT / "EmotioNet" / "imgs"
    if not aligned_root.exists() and not imgs_root.exists():
        return pd.DataFrame()

    def resolve_path(url: str) -> str | None:
        # Local EmotioNet layouts seen on /Storage:
        #   Aligned/N_0000000001_00001.jpg  (flattened, one dir)
        #   imgs/N_0000000001_00001.jpg     (same)
        # vs the S3 URL form:
        #   Images/N_0000000001/N_0000000001_00001.jpg
        # We try the flat basename first, then the nested form, then any
        # subdirectory under the relevant root.
        url = url.strip().strip("'").strip('"')
        basename = url.split("/")[-1]
        for root in (aligned_root, imgs_root):
            p = root / basename
            if p.exists():
                return str(p)
            # also try nested form for completeness
            if "amazonaws.com/emotionet/" in url:
                rel = url.split("amazonaws.com/emotionet/")[-1]
                p2 = root / rel
                if p2.exists():
                    return str(p2)
        return None

    url_col = "URL"
    rows = []
    for _, row in df.iterrows():
        url = row.get(url_col, "")
        img_path = resolve_path(str(url))
        if img_path is None:
            continue
        # Map EmotioNet AU columns to py-feat names.
        labels = {}
        for au_n in (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43):
            src_col = f"AU {au_n}"
            if src_col not in df.columns:
                continue
            v = row[src_col]
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            au_key = f"AU{au_n:02d}"
            labels[au_key] = np.nan if v == 999 else float(v)
        out = {
            "image_path": img_path,
            "subject": Path(img_path).parent.name,  # use identity-folder as subject
            "source": "emotionet",
        }
        for au in PYFEAT_AU_COLS:
            out[au] = labels.get(au, np.nan)
        rows.append(out)
    return pd.DataFrame(rows)


def _parse_bp4d_au_occ_csv(csv_path: Path, columns_are_au_index: bool) -> pd.DataFrame:
    """Parse a single BP4D / BP4D+ AU_OCC CSV.

    Two header conventions are seen:
    - BP4D (legacy): header is the integer range 0..99 where N>=1 means
      "AU N"; col 0 is the frame_id. Pass columns_are_au_index=True.
    - BP4D+ / BP4D-2016-rerelease: header explicitly names the scored AUs,
      e.g. `1,1,2,4,5,...` (35-ish cols). Col 0 is the frame_id (header
      value ignored). Pass columns_are_au_index=False.

    The `columns_are_au_index` hint is overridden by an auto-detect that
    counts header columns — files with <50 columns are treated as named.
    (Legacy BP4D files have 100 cols; named-AU files have ~35.) This lets
    us mix the original BP4D AU_OCC with re-released F013-style files
    transparently.

    Returns a DataFrame with columns [frame, AU01, AU02, ..., AU43] where
    AUs are 0/1 or NaN (9 codes are converted to NaN). Only AUs in
    PYFEAT_AU_COLS are kept.
    """
    with csv_path.open() as f:
        header_line = f.readline().strip().split(",")
    # Auto-detect format: legacy BP4D files have 100 columns (header
    # 0..99). Named-AU files (BP4D+ / F013 re-release) have ~35. Override
    # caller's hint if header column count makes it obvious.
    if len(header_line) < 50:
        columns_are_au_index = False
    # Read with positional column indices so duplicate names in the header
    # (BP4D+ has `1,1,2,4,...` where col 0 frame-id and col 1 AU1 both
    # display as "1") don't break the parse.
    data = pd.read_csv(csv_path, header=0, names=range(len(header_line)))
    frame_ids = data.iloc[:, 0].astype(int).values

    out = {"frame": frame_ids}
    if columns_are_au_index:
        # BP4D: header is 0..99; column N (N>=1) = AU N.
        for au_target in PYFEAT_AU_COLS:
            au_n = int(au_target[2:])
            if au_n < len(header_line):
                v = data.iloc[:, au_n].astype(float).values
                out[au_target] = np.where(v == 9, np.nan, v)
            else:
                out[au_target] = np.full(len(frame_ids), np.nan)
    else:
        # BP4D+: header[i] names the AU at column i (skip col 0 = frame).
        au_to_col = {}
        for i, h in enumerate(header_line):
            if i == 0:
                continue
            try:
                au_to_col[int(h)] = i
            except ValueError:
                continue
        for au_target in PYFEAT_AU_COLS:
            au_n = int(au_target[2:])
            col_idx = au_to_col.get(au_n)
            if col_idx is not None:
                v = data.iloc[:, col_idx].astype(float).values
                out[au_target] = np.where(v == 9, np.nan, v)
            else:
                out[au_target] = np.full(len(frame_ids), np.nan)
    return pd.DataFrame(out)


def load_bp4d_labels() -> pd.DataFrame:
    """BP4D — 41 subjects, AU_OCC binary labels, originals in Sequences_2D+3D/<subj>/<task>/<frame>.jpg.

    Frame filenames are inconsistent across the dataset distribution:
    F001-F023 and M001-M008 use 4-digit zero-padded names (0000.jpg);
    M009-M018 tarballs use 3-digit names (000.jpg). Probe both per task
    directory so the loader works regardless of which extraction pass
    was used.
    """
    occ_dir = DATA_ROOT / "BP4D" / "AUCoding" / "AU_OCC"
    img_root = DATA_ROOT / "BP4D" / "Sequences_2D+3D"
    if not occ_dir.exists() or not img_root.exists():
        return pd.DataFrame()
    parts = []
    for csv_path in sorted(occ_dir.glob("*.csv")):
        # Filename: F001_T1.csv etc.
        stem = csv_path.stem
        subject, task = stem.split("_", 1)
        task_dir = img_root / subject / task
        # Detect naming convention from one extant file (cheap O(1) on the
        # directory listing). Default to 4-digit if the dir is empty.
        try:
            sample = next((p for p in task_dir.iterdir() if p.suffix == ".jpg"), None)
        except FileNotFoundError:
            sample = None
        pad_width = 4
        if sample is not None:
            stem_digits = "".join(c for c in sample.stem if c.isdigit())
            if stem_digits:
                pad_width = len(stem_digits)
        labels = _parse_bp4d_au_occ_csv(csv_path, columns_are_au_index=True)
        labels["subject"] = subject
        labels["source"] = "bp4d"
        labels["image_path"] = labels["frame"].apply(
            lambda f, td=task_dir, w=pad_width: str(td / f"{int(f):0{w}d}.jpg")
        )
        labels = labels.drop(columns=["frame"])
        parts.append(labels)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # Existence filter — many BP4D frames may not actually have JPGs on disk
    # (the dataset ships per-frame .mat features in 2D+3D/, not jpgs).
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return df


def load_bp4dplus_labels() -> pd.DataFrame:
    """BP4D+ — 140 subjects, AU_OCC binary labels.

    Frame filenames use mixed padding across subjects: 99 of 140 use 3-digit
    (``000.jpg``), 41 use 4-digit (``0000.jpg``). Auto-detect per task dir
    (same as BP4D loader). Without this, labels referencing low frame numbers
    miss their JPGs and ~48K rows get filtered out by the existence check.
    """
    occ_dir = DATA_ROOT / "BP4D+_v0.2" / "AUCoding" / "AU_OCC"
    img_root = DATA_ROOT / "BP4D+_v0.2" / "2D+3D"
    if not occ_dir.exists() or not img_root.exists():
        return pd.DataFrame()
    parts = []
    for csv_path in sorted(occ_dir.glob("*.csv")):
        stem = csv_path.stem
        subject, task = stem.split("_", 1)
        task_dir = img_root / subject / task
        try:
            sample = next((p for p in task_dir.iterdir() if p.suffix == ".jpg"), None)
        except FileNotFoundError:
            sample = None
        pad_width = 4
        if sample is not None:
            stem_digits = "".join(c for c in sample.stem if c.isdigit())
            if stem_digits:
                pad_width = len(stem_digits)
        labels = _parse_bp4d_au_occ_csv(csv_path, columns_are_au_index=False)
        labels["subject"] = subject
        labels["source"] = "bp4dplus"
        labels["image_path"] = labels["frame"].apply(
            lambda f, td=task_dir, w=pad_width: str(td / f"{int(f):0{w}d}.jpg")
        )
        labels = labels.drop(columns=["frame"])
        parts.append(labels)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return df


def load_pain_labels() -> pd.DataFrame:
    """UNBC-McMaster Shoulder Pain Expression Archive (cosanlab pre-shutdown copy).

    Layout:
        aligned/<subj_id>/<sequence>/<frame>.jpg                — face crop
        Frame_Labels/FACS/<subj_id>/<sequence>/<frame>_facs.txt — FACS rows

    Each FACS .txt is either empty (frame scored, no AUs active) or
    has rows of ``AU_NUM   intensity   side1   side2`` (whitespace-separated,
    scientific notation). Subject from path component
    (e.g. ``109-ib109`` → subject ``ib109``).
    """
    root = DATA_ROOT.parent / "ShoulderPain"  # /Storage/ShoulderPain - try data_root sibling
    if not root.exists():
        # Default to /Storage/Data/ShoulderPain (what's on this machine).
        root = Path("/Storage/Data/ShoulderPain")
    if not root.exists():
        return pd.DataFrame()
    facs_root = root / "Frame_Labels" / "FACS"
    img_root = root / "aligned"
    if not facs_root.exists() or not img_root.exists():
        return pd.DataFrame()

    rows = []
    for subj_dir in sorted(facs_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        # Subject ID: "109-ib109" → take part after the dash for the cleaner ID
        subject = subj_dir.name.split("-", 1)[-1] if "-" in subj_dir.name else subj_dir.name
        for seq_dir in sorted(subj_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            for facs_file in sorted(seq_dir.glob("*_facs.txt")):
                stem = facs_file.stem.replace("_facs", "")
                img_path = img_root / subj_dir.name / seq_dir.name / f"{stem}.jpg"
                if not img_path.exists():
                    continue
                au_intensity = {}
                try:
                    with facs_file.open() as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    au_n = int(round(float(parts[0])))
                                    intensity = float(parts[1])
                                except ValueError:
                                    continue
                                au_key = f"AU{au_n:02d}"
                                if au_key in PYFEAT_AU_COLS:
                                    au_intensity[au_key] = intensity
                except OSError:
                    continue

                # Frame is scored — every AU not listed defaults to 0 (PAIN
                # FACS convention: rows present only for active AUs in this
                # frame; empty file = pure neutral).
                row = {
                    "image_path": str(img_path),
                    "subject": subject,
                    "source": "pain",
                }
                for au in PYFEAT_AU_COLS:
                    row[au] = au_intensity.get(au, 0.0)
                rows.append(row)
    return pd.DataFrame(rows)


def load_disfaplus_labels() -> pd.DataFrame:
    """DISFA+ posed-peak — 9 subjects, 12 AUs, intensity 0..5.

    Used for v3.11+ Cheong-replication training. Including DISFA+ in the
    training corpus is a data leak (it's the bench dataset), but Cheong's
    published pipeline fit PCA on all 7 datasets including DISFA+, so
    we replicate that to test whether the leak explains his F1 gap.
    """
    root = DATA_ROOT / "DISFAPlusDataset"
    labels_root = root / "Labels"
    img_root = root / "Aligned"
    if not labels_root.exists() or not img_root.exists():
        return pd.DataFrame()

    au_names = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12",
                "AU15", "AU17", "AU20", "AU25", "AU26"]
    rows = []
    for subj_dir in sorted(labels_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        subject = subj_dir.name
        for trial_dir in sorted(subj_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            trial = trial_dir.name
            au_dict: dict[str, dict[str, int]] = {}
            valid = True
            for au in au_names:
                au_path = trial_dir / f"{au}.txt"
                if not au_path.exists():
                    valid = False
                    break
                with au_path.open() as f:
                    fr_to_v = {}
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            fr_to_v[parts[0]] = int(parts[1])
                au_dict[au] = fr_to_v
            if not valid:
                continue
            common_frames = set.intersection(*(set(d) for d in au_dict.values()))
            for fid in sorted(common_frames):
                img_path = img_root / subject / trial / fid
                if not img_path.exists():
                    continue
                row = {
                    "image_path": str(img_path),
                    "subject": subject,
                    "source": "disfaplus",
                }
                for au in PYFEAT_AU_COLS:
                    # Map "AU1" → "AU01" naming convention
                    src_name = f"AU{int(au[2:])}"
                    if src_name in au_dict:
                        row[au] = float(au_dict[src_name].get(fid, 0))
                    else:
                        row[au] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


SOURCE_LOADERS = {
    "disfa18": load_disfa18_labels,
    "ckplus": load_ckplus_labels,
    "emotionet": load_emotionet_labels,
    "bp4d": load_bp4d_labels,
    "bp4dplus": load_bp4dplus_labels,
    "pain": load_pain_labels,
    "disfaplus": load_disfaplus_labels,
}


# ---------------------------------------------------------------------------
# Feature extraction — uses the same corrected HOG path the inference
# pipeline calls (feat.utils.face_mask.extract_hog_features_batched).
# ---------------------------------------------------------------------------


def extract_features(
    labels_df: pd.DataFrame,
    output_path: Path,
    device: str = "cuda",
    batch_size: int = 32,
    chunk_size: int = 2000,
    align_faces: bool = False,
    face_model: str = "retinaface",
    num_workers: int = 0,
    detect_call_size: int = 5000,
) -> None:
    """Run the v0.7 pipeline + corrected HOG over labels_df and cache to disk.

    Mechanism: load a Detector with face+landmark heads, monkey-patch
    ``extract_hog_features_batched`` to capture HOG features as they're
    computed during detect(), and accumulate matched (HOG, landmarks,
    AU labels, metadata) tuples in memory before flushing per chunk.

    Outputs:
      - ``<output_path>/chunk_NNNN.npz`` per chunk_size rows
            hog: (N, D) float32
            landmarks: (N, 136) float32 -- 68 (x, y) in aligned-crop coords
            labels: (N, 20) float32 -- AU01..AU43 (NaN where unscored)
            subjects: (N,) S20 / object
            sources: (N,) S20 / object
            image_paths: (N,) object
      - ``<output_path>/manifest.json``
    """
    import feat.utils.face_mask as face_mask
    from feat.detector import Detector

    output_path.mkdir(parents=True, exist_ok=True)
    label_cols = [c for c in labels_df.columns if c.startswith("AU")]
    label_array = labels_df[label_cols].to_numpy(dtype=np.float32)
    subjects = labels_df["subject"].astype(str).values
    sources = labels_df["source"].astype(str).values
    paths_arr = labels_df["image_path"].astype(str).values
    n = len(labels_df)

    # Build detector with au_model='xgb' so the AU path (and thus HOG
    # extraction) actually runs. AU outputs themselves are discarded; we
    # capture HOG features via the monkey-patch below.
    print(f"loading Detector on {device} (face_model={face_model})...", flush=True)
    detector = Detector(
        face_model=face_model,
        au_model="xgb",
        emotion_model=None,
        identity_model=None,
        device=device,
    )

    # Optional: load Procrustes-warp neutral template for aligned-face
    # extraction. Keeps the whole flow on GPU via warp_affine + grid_sample.
    aligned_template = None
    if align_faces:
        import torch as _torch
        import pandas as _pd
        from feat.utils.io import get_resource_path
        tpl_path = Path(get_resource_path()).parent.parent / "feat" / "resources" / "neutral_face_coordinates.csv"
        if not tpl_path.exists():
            tpl_path = Path("/home/ljchang/Github/py-feat/feat/resources/neutral_face_coordinates.csv")
        tpl_df = _pd.read_csv(tpl_path)
        aligned_template = _torch.tensor(
            tpl_df[["x", "y"]].to_numpy(), dtype=_torch.float32, device=device,
        )
        print(f"  Procrustes alignment ON, template from {tpl_path.name}: "
              f"x=[{tpl_df.x.min():.0f}, {tpl_df.x.max():.0f}], "
              f"y=[{tpl_df.y.min():.0f}, {tpl_df.y.max():.0f}]")

    orig_extract = face_mask.extract_hog_features_batched
    capture_buf: list[tuple[np.ndarray, np.ndarray]] = []

    def capturing_extract(extracted_faces, landmarks, hog_layer=None):
        # If --align-faces, warp the face crop so landmarks land on the
        # neutral template. landmarks here is [N, 136] in normalized [0,1]
        # face-crop coords (mobilefacenet output convention).
        ef = extracted_faces
        lm = landmarks
        if aligned_template is not None and ef.shape[0] > 0:
            from feat.utils.image_operations import (
                procrustes_similarity_torch,
            )
            from feat.utils.geometry import warp_affine
            face_size = ef.shape[-1]
            N = ef.shape[0]
            lm_pix = lm.view(N, 68, 2).to(ef.device, dtype=ef.dtype) * face_size
            # The template CSV is in a 256-pixel reference frame; face crops
            # are 112x112 (mobilefacenet default). Scale template to the
            # current face_size so Procrustes maps src→dst at the same scale.
            # Without this, the warp scales the face up ~2x and clips most
            # of it outside the output frame.
            target_template = aligned_template * (face_size / 256.0)
            M = procrustes_similarity_torch(lm_pix, target_template)
            ef = warp_affine(ef, M, dsize=(face_size, face_size), mode="bilinear")
            # Apply M to landmarks too so downstream HOG masking uses the
            # post-warp positions. M maps src_pix → dst_pix via M @ [x, y, 1].
            ones = torch.ones(N, 68, 1, device=lm_pix.device, dtype=lm_pix.dtype)
            lm_hom = torch.cat([lm_pix, ones], dim=-1)  # [N, 68, 3]
            lm_warped = torch.einsum("bij,bkj->bki", M, lm_hom)  # [N, 68, 2]
            lm = (lm_warped / face_size).reshape(N, 136)

        hog, lm_list = orig_extract(ef, lm, hog_layer=hog_layer)
        # hog is numpy [N, D]; lm_list is list of per-face landmark arrays
        # (each [68, 2] in aligned-crop coords). Copy out so subsequent
        # in-place ops in Detector don't affect what we save.
        hog_copy = np.array(hog, dtype=np.float32)
        lm_flat = np.array(
            [np.asarray(lm).reshape(-1) for lm in lm_list], dtype=np.float32
        ) if lm_list else np.zeros((0, 136), dtype=np.float32)
        capture_buf.append((hog_copy, lm_flat))
        return hog, lm_list

    face_mask.extract_hog_features_batched = capturing_extract

    # The Detector hot-path imports extract_hog_features_batched at module
    # load time, so patching the module attribute alone isn't enough — we
    # also patch the binding inside feat.detector.
    import feat.detector as det_mod
    det_mod.extract_hog_features_batched = capturing_extract

    chunk_idx = 0
    pending_hog: list[np.ndarray] = []
    pending_lm: list[np.ndarray] = []
    pending_labels: list[np.ndarray] = []
    pending_subjects: list[str] = []
    pending_sources: list[str] = []
    pending_paths: list[str] = []

    t0 = time.perf_counter()
    n_processed = 0
    n_faces_total = 0

    def flush_chunk():
        nonlocal chunk_idx
        if not pending_hog:
            return
        hog_arr = np.concatenate(pending_hog, axis=0)
        lm_arr = np.concatenate(pending_lm, axis=0)
        lbl_arr = np.stack(pending_labels, axis=0)
        subj_arr = np.asarray(pending_subjects, dtype=object)
        src_arr = np.asarray(pending_sources, dtype=object)
        path_arr = np.asarray(pending_paths, dtype=object)
        out_file = output_path / f"chunk_{chunk_idx:04d}.npz"
        np.savez_compressed(
            out_file,
            hog=hog_arr, landmarks=lm_arr, labels=lbl_arr,
            subjects=subj_arr, sources=src_arr, image_paths=path_arr,
        )
        print(f"  wrote {out_file.name}: {len(hog_arr)} rows, HOG dim {hog_arr.shape[1]}", flush=True)
        chunk_idx += 1
        pending_hog.clear()
        pending_lm.clear()
        pending_labels.clear()
        pending_subjects.clear()
        pending_sources.clear()
        pending_paths.clear()

    # Process in two-level batching:
    # - Outer "detect_call_size" group passes many paths to a single
    #   detector.detect() call so its DataLoader workers stay alive across
    #   ~5000 images (instead of being respawned every batch_size images).
    # - detector.detect() internally batches at `batch_size` for the GPU.
    # This avoids DataLoader fork overhead when num_workers > 0.
    for start in range(0, n, detect_call_size):
        end = min(start + detect_call_size, n)
        batch_paths = paths_arr[start:end].tolist()
        capture_buf.clear()
        try:
            fex = detector.detect(
                batch_paths, data_type="image",
                output_size=512, batch_size=batch_size,
                num_workers=num_workers, progress_bar=False,
            )
        except Exception as e:
            print(f"  detect failed on batch [{start}:{end}]: {e}", flush=True)
            continue
        if len(fex) == 0 or not capture_buf:
            continue

        # Concatenate the per-mini-batch HOG captures
        hog_concat = np.concatenate([c[0] for c in capture_buf], axis=0)
        lm_concat = np.concatenate([c[1] for c in capture_buf], axis=0)

        # fex order matches the order of detected faces, which matches
        # the order HOG features were emitted. So row i of fex corresponds
        # to row i of hog_concat / lm_concat.
        # Use `frame` (0-indexed within the batch) to map back to inputs.
        frame_ids = fex["frame"].to_numpy().astype(int)
        face_scores = fex["FaceScore"].to_numpy()

        # Keep only the highest-confidence face per input frame.
        # Build best-face-per-frame mask aligned to fex rows.
        seen: dict[int, int] = {}  # frame_id -> row idx in fex
        for i, (f, s) in enumerate(zip(frame_ids, face_scores)):
            if f not in seen or s > face_scores[seen[f]]:
                seen[f] = i
        keep_idx = sorted(seen.values())

        for i in keep_idx:
            input_pos = int(frame_ids[i])
            global_idx = start + input_pos
            if global_idx >= n:
                continue
            pending_hog.append(hog_concat[i : i + 1])
            pending_lm.append(lm_concat[i : i + 1])
            pending_labels.append(label_array[global_idx])
            pending_subjects.append(subjects[global_idx])
            pending_sources.append(sources[global_idx])
            pending_paths.append(paths_arr[global_idx])
            n_faces_total += 1

        n_processed += len(batch_paths)
        while len(pending_hog) >= chunk_size:
            flush_chunk()

        elapsed = time.perf_counter() - t0
        rate = n_processed / max(elapsed, 1e-3)
        eta = (n - n_processed) / max(rate, 1e-3)
        print(
            f"  {n_processed}/{n} images, {n_faces_total} faces captured, "
            f"{rate:.1f} img/s, elapsed {elapsed:.0f}s, ETA {eta:.0f}s",
            flush=True,
        )

    # Final flush + manifest
    flush_chunk()
    elapsed = time.perf_counter() - t0
    manifest = {
        "n_images_processed": n_processed,
        "n_faces_captured": n_faces_total,
        "elapsed_seconds": round(elapsed, 1),
        "chunks": sorted(p.name for p in output_path.glob("chunk_*.npz")),
        "label_columns": label_cols,
        "device": device,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done. processed {n_processed} images, captured {n_faces_total} faces "
          f"in {elapsed:.0f}s. manifest written.")

    # Restore the patched function
    face_mask.extract_hog_features_batched = orig_extract
    det_mod.extract_hog_features_batched = orig_extract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("bench-cache/au_train"))
    p.add_argument("--sources", nargs="+",
                   default=["disfa18", "ckplus", "emotionet", "bp4d", "bp4dplus"],
                   choices=list(SOURCE_LOADERS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers for image loading. Set >0 to parallelize "
                        "disk I/O (especially helpful on NAS). 4-8 typically saturates.")
    p.add_argument("--detect-call-size", type=int, default=5000,
                   help="Number of paths to pass to each detector.detect() call. "
                        "DataLoader workers stay alive across this many images, "
                        "amortizing fork overhead. Detector still batches internally "
                        "at --batch-size for GPU work.")
    p.add_argument("--face-model", default="retinaface",
                   help="Face detector: 'retinaface' (v0.7 default, batched) or "
                        "'img2pose' (v0.6 default, matches existing chunks).")
    p.add_argument("--limit", type=int, default=None,
                   help="cap on total training rows (for quick smoke tests)")
    p.add_argument("--inventory-only", action="store_true",
                   help="just enumerate the label files; skip feature extraction")
    p.add_argument("--align-faces", action="store_true",
                   help="Procrustes-warp face crops to a canonical 68-pt template "
                   "(feat/resources/neutral_face_coordinates.csv) before HOG. "
                   "Keeps the whole pipeline on GPU via warp_affine + grid_sample.")
    args = p.parse_args()

    # Load + concatenate label tables
    parts = []
    for src in args.sources:
        print(f"loading {src} labels...", flush=True)
        t0 = time.perf_counter()
        df = SOURCE_LOADERS[src]()
        elapsed = time.perf_counter() - t0
        print(f"  {src}: {len(df)} examples in {elapsed:.1f}s", flush=True)
        if not df.empty:
            parts.append(df)
    if not parts:
        print("no labels loaded; check PYFEAT_DATA_ROOT and source availability", flush=True)
        return 1
    labels_df = pd.concat(parts, ignore_index=True)

    print(f"\ncombined: {len(labels_df)} examples across {labels_df['source'].nunique()} sources")
    print(f"per-source counts: {labels_df['source'].value_counts().to_dict()}")
    print(f"AU coverage (non-NaN counts per AU):")
    au_cols = [c for c in labels_df.columns if c.startswith("AU")]
    cov = labels_df[au_cols].notna().sum().sort_values(ascending=False)
    for au, n_labeled in cov.items():
        print(f"  {au}: {n_labeled:>7} / {len(labels_df):>7}  ({n_labeled/len(labels_df)*100:.1f}%)")

    if args.limit:
        labels_df = labels_df.sample(n=min(args.limit, len(labels_df)), random_state=42)
        print(f"\n--limit {args.limit} -> sampling {len(labels_df)} rows")

    if args.inventory_only:
        print("\n--inventory-only: skipping feature extraction")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(args.output_dir / "training_labels.parquet")
    print(f"wrote labels: {args.output_dir / 'training_labels.parquet'}")

    extract_features(
        labels_df,
        output_path=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        align_faces=args.align_faces,
        face_model=args.face_model,
        num_workers=args.num_workers,
        detect_call_size=args.detect_call_size,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
