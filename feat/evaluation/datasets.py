"""Dataset loaders for accuracy benchmarks.

Each loader returns a ``DatasetSplit`` describing image paths + ground-truth
labels for a held-out subset. Loaders are tolerant — if the dataset is not
present on the local filesystem they return ``None`` and the runner skips
the dataset. This lets the same bench script run on any machine.

Data root lookup order:
1. ``$PYFEAT_DATA_ROOT`` env var
2. ``/Storage/Data`` (the cosanlab workstation default)

Image paths returned are the **aligned crops** (``aligned_path`` on DISFA;
``aligned_path`` on AffectNet ``validation_aligned.csv``). This skips
end-to-end face-detection accuracy but matches how Cheong et al. and the
upstream benchmark protocols evaluate AU/emotion/landmark heads. A
face-detection-IoU benchmark on WIDER FACE / 300-W is out of scope for
Phase 0.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Map AffectNet expression integer codes to py-feat emotion column names.
# AffectNet codes 7 (contempt), 8 (none), 9 (uncertain), 10 (non-face) have
# no py-feat counterpart and are filtered out before scoring.
AFFECTNET_EMOTION_MAP: dict[int, str] = {
    0: "neutral",
    1: "happiness",
    2: "sadness",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "anger",
}

# DISFA AU columns in the source CSV (no zero-padding) → py-feat output
# (zero-padded "AU01" etc.). The 12 DISFA AUs are a strict subset of the
# 20 py-feat AUs, so every DISFA AU has a counterpart.
DISFA_AU_RENAME: dict[str, str] = {
    "AU1": "AU01",
    "AU2": "AU02",
    "AU4": "AU04",
    "AU5": "AU05",
    "AU6": "AU06",
    "AU9": "AU09",
    "AU12": "AU12",
    "AU15": "AU15",
    "AU17": "AU17",
    "AU20": "AU20",
    "AU25": "AU25",
    "AU26": "AU26",
}


@dataclass
class DatasetSplit:
    """Image paths + ground-truth labels for one evaluation subset.

    Fields are intentionally minimal so loaders for different label types
    (AU intensity, emotion class, valence/arousal, identity pairs,
    identity-search) share one shape.

    ``metric_kind`` selects the runner path:
      - ``"au_intensity"`` — per-image AU labels in ``labels``
      - ``"emotion_class"`` — per-image emotion + V/A labels in ``labels``
      - ``"identity_pairs"`` — ``labels`` has columns
        ``["path_a", "path_b", "same_id", "fold"]`` (one row per pair).
        ``image_paths`` is empty; the runner reads from ``labels``.
      - ``"identity_search"`` — TinyFace-style 1:N. ``image_paths`` is
        all probe paths; ``extras`` holds gallery paths and id arrays.
    """

    name: str
    image_paths: list[str]
    labels: pd.DataFrame  # one row per sample; columns are dataset-specific
    metric_kind: str
    notes: str = ""
    extras: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.image_paths) if self.image_paths else len(self.labels)


def data_root() -> Path:
    return Path(os.environ.get("PYFEAT_DATA_ROOT", "/Storage/Data"))


def identity_data_root() -> Path:
    """Separate root for identity benchmarks since they live outside ``Data/``."""
    return Path(os.environ.get("PYFEAT_IDENTITY_ROOT", "/Storage/IdentityDatasets"))


def _subset_json_path(dataset_name: str) -> Path:
    return Path(__file__).parent / "subsets" / f"{dataset_name}.json"


def _load_subset_index(dataset_name: str) -> list[str] | None:
    """Return the frozen list of sample IDs for a dataset, or None if not pinned."""
    p = _subset_json_path(dataset_name)
    if not p.exists():
        return None
    return json.loads(p.read_text())["ids"]


# ---------------------------------------------------------------------------
# DISFA
# ---------------------------------------------------------------------------


def load_disfaplus(
    subset_size: int | None = None,
    seed: int = 42,
    use_aligned: bool = False,
) -> DatasetSplit | None:
    """Load DISFA+ — 9 posed-peak subjects, 12 AUs.

    DISFA+ ("Extended DISFA Plus") differs from raw DISFA: subjects pose
    specific AUs on cue, peak intensity is reached, then they relax. The
    Cheong et al. 2023 paper uses this dataset for the AU F1 benchmark
    (Table 6) — reported Feat-XGB average F1 = 0.52.

    Structure on disk:
        Labels/<SN###>/<trial>/AU<N>.txt   — frame_id intensity (0..5)
        Images/<SN###>/<trial>/<frame>.jpg — originals
        Aligned/<SN###>/<trial>/<frame>.jpg — face-aligned crops

    AU labels are stored per-trial; each AU has its own .txt under the
    trial directory. We merge per-frame across the 12 AU files.

    Subject IDs in DISFA+ (SN001, SN003, SN004, SN007, SN009, SN010,
    SN013, SN025, SN027) are all also present in raw DISFA. Any training
    pipeline that uses DISFA must exclude these subjects to avoid leakage
    into the DISFA+ benchmark.
    """
    root = data_root() / "DISFAPlusDataset"
    labels_root = root / "Labels"
    img_root = root / ("Aligned" if use_aligned else "Images")
    if not labels_root.exists() or not img_root.exists():
        return None

    rows = []
    au_names = [f"AU{n}" for n in (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)]
    for subj_dir in sorted(labels_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        subject = subj_dir.name
        for trial_dir in sorted(subj_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            trial = trial_dir.name
            # Read each AU file → {frame_id_str: intensity}
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
            # Intersect frame IDs across the 12 AUs
            common_frames = set.intersection(*(set(d) for d in au_dict.values()))
            for fid in sorted(common_frames):
                img_path = img_root / subject / trial / fid
                if not img_path.exists():
                    continue
                row = {
                    "image_path": str(img_path),
                    "subject": subject,
                    "trial": trial,
                    "frame": fid,
                }
                for au in au_names:
                    # Rename to zero-padded AU01..AU26 for consistency with
                    # py-feat's detector output column names.
                    row[f"AU{int(au[2:]):02d}"] = au_dict[au][fid]
                rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)

    # Optional deterministic downsampling (stratified by subject so each
    # subject contributes proportionally).
    if subset_size is not None and subset_size < len(df):
        rng = np.random.default_rng(seed)
        subjects = sorted(df["subject"].unique())
        per_subj = max(1, subset_size // len(subjects))
        parts = []
        for s in subjects:
            sdf = df[df["subject"] == s]
            if len(sdf) <= per_subj:
                parts.append(sdf)
            else:
                idx = rng.choice(sdf.index.values, size=per_subj, replace=False)
                parts.append(sdf.loc[sorted(idx)])
        df = pd.concat(parts).reset_index(drop=True)

    labels = df.drop(columns=["image_path"]).copy()
    return DatasetSplit(
        name="disfaplus",
        image_paths=df["image_path"].tolist(),
        labels=labels,
        metric_kind="au_intensity",
        notes=(
            f"DISFA+ posed-peak; {df['subject'].nunique()} subjects, "
            f"{df['trial'].nunique()} trials; AU intensity 0..5; "
            f"{'aligned' if use_aligned else 'original'} face crops. "
            f"Subjects: {sorted(df['subject'].unique())}"
        ),
    )


def load_disfa(
    split: str = "P3",
    subset_size: int | None = None,
    seed: int = 42,
) -> DatasetSplit | None:
    """Load DISFA aligned face crops + per-frame AU intensities.

    The DISFA CSV ships with a ``data_split`` column ("P1", "P2", "P3")
    encoding the canonical 3-fold subject-disjoint split. We default to
    P3 as the held-out fold for regression evaluation.

    ``subset_size`` downsamples *within* the fold, distributing roughly
    evenly across subjects so each subject contributes the same number of
    frames. Selection is deterministic given ``seed``.
    """
    csv_path = data_root() / "DISFA_" / "DISFA_main_New.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    df = df[df["data_split"] == split].copy()
    if df.empty:
        return None

    # Prefer the frozen subset list if one is committed in subsets/.
    frozen = _load_subset_index(f"disfa_{split.lower()}")
    if frozen is not None:
        # Each id is "<subject>/<frame>" — match against aligned_path basename.
        df["__id"] = df.apply(
            lambda r: f"{r['subject']}/{int(r['frame'])}", axis=1
        )
        df = df[df["__id"].isin(set(frozen))].copy()
    elif subset_size is not None:
        subjects = sorted(df["subject"].unique())
        rng = np.random.default_rng(seed)
        per_subject = max(1, subset_size // len(subjects))
        parts = []
        for subj in subjects:
            sdf = df[df["subject"] == subj]
            if len(sdf) <= per_subject:
                parts.append(sdf)
            else:
                idx = rng.choice(sdf.index.values, size=per_subject, replace=False)
                parts.append(sdf.loc[sorted(idx)])
        df = pd.concat(parts).sort_index().reset_index(drop=True)

    # Verify the aligned crops actually exist on disk; drop missing ones.
    df = df[df["aligned_path"].apply(os.path.exists)].reset_index(drop=True)
    if df.empty:
        return None

    labels = df[list(DISFA_AU_RENAME)].rename(columns=DISFA_AU_RENAME).copy()
    labels["subject"] = df["subject"].values
    labels["frame"] = df["frame"].values

    return DatasetSplit(
        name=f"disfa_{split.lower()}",
        image_paths=df["aligned_path"].tolist(),
        labels=labels,
        metric_kind="au_intensity",
        notes=(
            f"DISFA fold={split}; {df['subject'].nunique()} subjects; "
            f"AU intensity 0..5; aligned face crops."
        ),
    )


# ---------------------------------------------------------------------------
# AffectNet
# ---------------------------------------------------------------------------


def load_affectnet_val(
    subset_size: int | None = 1000,
    seed: int = 42,
) -> DatasetSplit | None:
    """Load AffectNet manually-annotated validation set (aligned crops).

    Uses ``Manual_Annot_files/validation_aligned.csv`` which has the
    pre-aligned face crop paths. Filters to classes 0..6 (the 7 emotions
    py-feat predicts) and optionally takes a stratified subset.
    """
    csv_path = data_root() / "AffectNet" / "Manual_Annot_files" / "validation_aligned.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    df = df[df["expression"].isin(AFFECTNET_EMOTION_MAP)].copy()
    if df.empty:
        return None

    frozen = _load_subset_index("affectnet_val")
    if frozen is not None:
        df["__id"] = df["subDirectory_filePath"].astype(str)
        df = df[df["__id"].isin(set(frozen))].copy()
    elif subset_size is not None and subset_size < len(df):
        # Stratified subsample by expression class.
        rng = np.random.default_rng(seed)
        per_class = max(1, subset_size // len(AFFECTNET_EMOTION_MAP))
        parts = []
        for cls in sorted(AFFECTNET_EMOTION_MAP):
            cdf = df[df["expression"] == cls]
            if len(cdf) <= per_class:
                parts.append(cdf)
            else:
                idx = rng.choice(cdf.index.values, size=per_class, replace=False)
                parts.append(cdf.loc[sorted(idx)])
        df = pd.concat(parts).reset_index(drop=True)

    # The CSV's ``aligned_path`` points at ``Manual_Annot/Aligned/`` which
    # wasn't extracted from the shipped rar archives on this machine.
    # Fall back to the original images under
    # ``Manually_Annotated_Images/<subDirectory_filePath>``; the full
    # detector pipeline will run face detection on them.
    orig_root = data_root() / "AffectNet" / "Manual_Annot" / "Manually_Annotated_Images"
    df["image_path"] = df["subDirectory_filePath"].apply(
        lambda p: str(orig_root / p)
    )
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    if df.empty:
        return None

    labels = pd.DataFrame(
        {
            "expression_int": df["expression"].astype(int).values,
            "expression": df["expression"]
            .astype(int)
            .map(AFFECTNET_EMOTION_MAP)
            .values,
            "valence": df["valence"].astype(float).values,
            "arousal": df["arousal"].astype(float).values,
            "subDirectory_filePath": df["subDirectory_filePath"].astype(str).values,
            # Carry the GT face bbox so the runner can IoU-match the
            # detector's output to the labeled face when an image
            # contains multiple detections.
            "face_x": df["face_x"].astype(float).values,
            "face_y": df["face_y"].astype(float).values,
            "face_width": df["face_width"].astype(float).values,
            "face_height": df["face_height"].astype(float).values,
        }
    )

    return DatasetSplit(
        name="affectnet_val",
        image_paths=df["image_path"].tolist(),
        labels=labels,
        metric_kind="emotion_class",
        notes=(
            "AffectNet manual val; classes 0..6 (7 emotions); "
            "full detector pipeline on original images."
        ),
        extras={"has_valence_arousal": True},
    )


# ---------------------------------------------------------------------------
# Identity verification: CALFW / CPLFW
# ---------------------------------------------------------------------------


def _load_lfw_style_pairs(
    pairs_txt: Path,
    image_dir: Path,
    landmark_dir: Path | None = None,
    landmark_suffix: str = "_5loc_attri.txt",
) -> pd.DataFrame | None:
    """Parse CALFW/CPLFW-style ``pairs_*.txt``.

    Format: 12000 lines = 6000 pairs (consecutive line pairs). On positive
    pairs both lines carry the same fold tag 1..10; on negative pairs both
    carry label 0. First half is positive, second half negative; folds are
    300 same + 300 different per fold (LFW protocol).

    When ``landmark_dir`` is provided, returns an extra ``landmark_a`` /
    ``landmark_b`` column pointing at the matching ``*_5loc_attri.txt``.
    The runner uses these for InsightFace-template similarity alignment
    so that ArcFace embeddings match the alignment its training used.
    """
    if not pairs_txt.exists() or not image_dir.exists():
        return None
    with pairs_txt.open() as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) % 2:
        raise ValueError(f"{pairs_txt} has odd line count {len(lines)}")
    rows = []
    for i in range(0, len(lines), 2):
        name_a, tag_a = lines[i].rsplit(" ", 1)
        name_b, tag_b = lines[i + 1].rsplit(" ", 1)
        if tag_a != tag_b:
            raise ValueError(f"Pair {i//2} has mismatched tags {tag_a} vs {tag_b}")
        same = int(tag_a) > 0
        fold = (i // 2 // 300) % 10
        row = {
            "path_a": str(image_dir / name_a),
            "path_b": str(image_dir / name_b),
            "same_id": int(same),
            "fold": int(fold),
        }
        if landmark_dir is not None:
            stem_a = Path(name_a).stem
            stem_b = Path(name_b).stem
            row["landmark_a"] = str(landmark_dir / f"{stem_a}{landmark_suffix}")
            row["landmark_b"] = str(landmark_dir / f"{stem_b}{landmark_suffix}")
        rows.append(row)
    return pd.DataFrame(rows)


def load_calfw() -> DatasetSplit | None:
    """CALFW with InsightFace-template alignment.

    Uses the *original* 250x250 images + the 5-landmark files shipped
    under ``images&landmarks/`` so the runner can produce ArcFace-aligned
    112x112 crops at evaluation time. The shipped ``aligned images/``
    folder is DeepFunneled-style — not what ArcFace was trained on —
    which gives ~73% accuracy vs. the ~96% published using
    ArcFace-template alignment.
    """
    root = identity_data_root() / "calfw"
    lm_root = root / "images&landmarks" / "images&landmarks"
    df = _load_lfw_style_pairs(
        pairs_txt=root / "pairs_CALFW.txt",
        image_dir=lm_root / "images",
        landmark_dir=lm_root / "CA_landmarks",
    )
    if df is None:
        return None
    return DatasetSplit(
        name="calfw",
        image_paths=[],
        labels=df,
        metric_kind="identity_pairs",
        notes="CALFW 6000 pairs (3000 same + 3000 diff), 10-fold CV, ArcFace-template aligned from originals",
        extras={"alignment": "arcface_5pt"},
    )


def load_cplfw() -> DatasetSplit | None:
    root = identity_data_root() / "cplfw"
    df = _load_lfw_style_pairs(
        pairs_txt=root / "pairs_CPLFW.txt",
        image_dir=root / "images",
        landmark_dir=root / "CP_landmarks",
    )
    if df is None:
        return None
    return DatasetSplit(
        name="cplfw",
        image_paths=[],
        labels=df,
        metric_kind="identity_pairs",
        notes="CPLFW 6000 pairs (3000 same + 3000 diff), 10-fold CV, ArcFace-template aligned from originals",
        extras={"alignment": "arcface_5pt"},
    )


# ---------------------------------------------------------------------------
# Identity 1:N search: TinyFace
# ---------------------------------------------------------------------------


def load_tinyface(use_distractors: bool = True) -> DatasetSplit | None:
    """TinyFace closed-set face identification (1:N).

    Reads the .mat protocol files for probe + gallery_match ordering.
    Distractor gallery is huge (153k images) — toggle via
    ``use_distractors`` to skip it for fast smoke tests.
    """
    try:
        import scipy.io as sio
    except ImportError:
        return None

    root = identity_data_root() / "tinyface"
    probe_mat = root / "Testing_Set" / "probe_img_ID_pairs.mat"
    gal_mat = root / "Testing_Set" / "gallery_match_img_ID_pairs.mat"
    if not probe_mat.exists() or not gal_mat.exists():
        return None

    pm = sio.loadmat(str(probe_mat))
    gm = sio.loadmat(str(gal_mat))
    probe_names = [str(x[0][0]) for x in pm["probe_set"]]
    probe_ids = pm["probe_ids"].ravel().astype(int).tolist()
    gallery_names = [str(x[0][0]) for x in gm["gallery_set"]]
    gallery_ids = gm["gallery_ids"].ravel().astype(int).tolist()

    probe_paths = [str(root / "Testing_Set" / "Probe" / n) for n in probe_names]
    gallery_paths = [str(root / "Testing_Set" / "Gallery_Match" / n) for n in gallery_names]

    distractor_paths: list[str] = []
    if use_distractors:
        ddir = root / "Testing_Set" / "Gallery_Distractor"
        if ddir.exists():
            distractor_paths = sorted(str(p) for p in ddir.glob("*.jpg"))

    labels = pd.DataFrame({"probe_id": probe_ids, "probe_path": probe_paths})

    return DatasetSplit(
        name="tinyface",
        image_paths=probe_paths,
        labels=labels,
        metric_kind="identity_search",
        notes=(
            f"TinyFace 1:N identification: {len(probe_paths)} probes, "
            f"{len(gallery_paths)} mated gallery, "
            f"{len(distractor_paths)} distractors"
        ),
        extras={
            "gallery_paths": gallery_paths,
            "gallery_ids": gallery_ids,
            "distractor_paths": distractor_paths,
        },
    )


# ---------------------------------------------------------------------------
# Columbia Gaze
# ---------------------------------------------------------------------------


def _columbia_gaze_root() -> Path | None:
    """Find Columbia Gaze on disk. Looks for the unzipped dataset dir."""
    # Common layouts after unzip: "Columbia Gaze Data Set/" or
    # "columbia_gaze_data_set/" depending on host conventions.
    candidates = [
        data_root() / "ColumbiaGaze" / "Columbia Gaze Data Set",
        data_root() / "ColumbiaGaze" / "columbia_gaze_data_set",
        data_root() / "ColumbiaGaze",
    ]
    for p in candidates:
        if p.exists() and any(p.glob("[0-9][0-9][0-9][0-9]")):
            return p
    return None


def load_columbia_gaze(
    head_pose_filter: str | None = "0P",
    subset_size: int | None = None,
    seed: int = 42,
) -> DatasetSplit | None:
    """Columbia Gaze — 5,880 images, 56 subjects, controlled lab capture.

    Smith, Yin, Feiner, Nayar 2013. http://www.cs.columbia.edu/CAVE/databases/columbia_gaze_data_set/

    Filename schema: ``<subj>_<dist>_<head>P_<vert>V_<horiz>H.jpg``
        subj  — 4-digit subject ID, 0001..0056
        dist  — camera distance, '2m' (the only published value)
        head  — head yaw, one of {-30, -15, 0, +15, +30} in degrees (P suffix)
        vert  — gaze pitch in degrees, one of {-10, 0, +10} (V suffix)
        horiz — gaze yaw in degrees, one of {-15, -10, -5, 0, +5, +10, +15} (H suffix)

    Default filters to ``head_pose='0P'`` (frontal head) because Columbia's
    gaze annotations are in the **camera frame** while L2CS outputs gaze in
    the **head-centric frame**; on the head-frontal subset the two frames
    coincide and we get a clean apples-to-apples comparison. To eval the
    head-pose-rotated subset, the bench runner would need to compose head
    rotation into the predicted gaze vector — pass ``head_pose_filter=None``
    to load everything and let the runner handle that.

    Returns DatasetSplit with labels = DataFrame[gaze_pitch_rad, gaze_yaw_rad,
    head_pose_deg, subject]. metric_kind='gaze_angular'.
    """
    root = _columbia_gaze_root()
    if root is None:
        return None

    import re
    # Match "0001_2m_0P_-15V_+10H.jpg" or with explicit + signs.
    # Allow optional sign before the digits.
    pat = re.compile(
        r"(?P<subj>\d{4})_(?P<dist>\dm)_(?P<head>[+-]?\d+)P_"
        r"(?P<vert>[+-]?\d+)V_(?P<horiz>[+-]?\d+)H\.jpg",
        re.IGNORECASE,
    )

    rows = []
    paths = []
    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.isdigit():
            continue
        subject = subj_dir.name
        for img in sorted(subj_dir.iterdir()):
            m = pat.match(img.name)
            if not m:
                continue
            head_pose_deg = int(m.group("head"))
            if head_pose_filter is not None:
                # Allow either "0P" or "0" form
                want = int(head_pose_filter.rstrip("P").rstrip("p"))
                if head_pose_deg != want:
                    continue
            vert_deg = int(m.group("vert"))
            horiz_deg = int(m.group("horiz"))
            paths.append(str(img))
            rows.append({
                "subject": subject,
                "head_pose_deg": head_pose_deg,
                # Columbia's +V = subject looks up; L2CS-Gaze360 convention
                # matches (+pitch = up). No flip needed.
                "gaze_pitch_rad": np.deg2rad(vert_deg),
                # Columbia's +H = subject looks to their LEFT (camera's right).
                # The full-set convention sweep (all axis/sign/unit combos vs
                # L2CS output, June 2026) shows +H maps to L2CS-Gaze360's +yaw
                # directly: mean angular error drops to ~2.7° with this sign and
                # ~17.5° with the opposite. (The earlier 100-sample sweep that
                # preferred the flip was underpowered.) See
                # feat/evaluation/metrics.py for the angular error definition.
                "gaze_yaw_rad": np.deg2rad(horiz_deg),
            })

    if not rows:
        return None
    df = pd.DataFrame(rows)

    if subset_size is not None and subset_size < len(df):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=subset_size, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        paths = [paths[i] for i in idx]

    return DatasetSplit(
        name="columbia_gaze",
        image_paths=paths,
        labels=df,
        metric_kind="gaze_angular",
        notes=(
            f"Columbia Gaze, head_pose_filter={head_pose_filter!r}, "
            f"n={len(df)} from {df['subject'].nunique()} subjects"
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def available() -> dict[str, str]:
    """Discover which datasets are present on this machine."""
    found = {}
    if (data_root() / "DISFA_" / "DISFA_main_New.csv").exists():
        found["disfa"] = str(data_root() / "DISFA_")
    if (data_root() / "DISFAPlusDataset" / "Labels").exists():
        found["disfaplus"] = str(data_root() / "DISFAPlusDataset")
    if (data_root() / "AffectNet" / "Manual_Annot_files" / "validation_aligned.csv").exists():
        found["affectnet_val"] = str(data_root() / "AffectNet")
    if (identity_data_root() / "calfw" / "pairs_CALFW.txt").exists():
        found["calfw"] = str(identity_data_root() / "calfw")
    if (identity_data_root() / "cplfw" / "pairs_CPLFW.txt").exists():
        found["cplfw"] = str(identity_data_root() / "cplfw")
    if (identity_data_root() / "tinyface" / "Testing_Set" / "probe_img_ID_pairs.mat").exists():
        found["tinyface"] = str(identity_data_root() / "tinyface")
    if _columbia_gaze_root() is not None:
        found["columbia_gaze"] = str(_columbia_gaze_root())
    return found


def load(name: str, **kwargs) -> DatasetSplit | None:
    """Single dispatch by dataset name. Returns None if dataset absent."""
    if name == "disfaplus":
        return load_disfaplus(**kwargs)
    if name == "disfa" or name.startswith("disfa_"):
        split = "P3"
        if "_" in name and name.split("_")[-1].upper().startswith("P"):
            split = name.split("_")[-1].upper()
        return load_disfa(split=split, **kwargs)
    if name == "affectnet_val":
        return load_affectnet_val(**kwargs)
    if name == "calfw":
        return load_calfw()
    if name == "cplfw":
        return load_cplfw()
    if name == "tinyface":
        return load_tinyface(**kwargs)
    if name == "columbia_gaze":
        return load_columbia_gaze(**kwargs)
    raise ValueError(f"Unknown dataset: {name!r}")
