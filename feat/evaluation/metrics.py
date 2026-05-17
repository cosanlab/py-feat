"""Metric primitives for accuracy benchmarks.

All functions take 1-D numpy arrays (or pandas Series) and return a Python
float. No detector dependencies — these are pure NumPy / scikit-learn so
they're trivially unit-testable with synthetic data.

Conventions:
- AU ground truth (DISFA): integer intensity 0..5; we binarize at >= 2 per
  the standard DISFA convention.
- AU prediction (py-feat): probability in [0, 1]; we binarize at >= 0.5.
- Emotion ground truth (AffectNet): integer class 0..6 mapped to
  ``["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]``.
- Valence / arousal: continuous, [-1, 1].
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def binarize_au_truth(y: np.ndarray, threshold: int = 2) -> np.ndarray:
    """Convert DISFA-style 0..5 intensity to binary {0, 1}."""
    return (np.asarray(y) >= threshold).astype(np.int8)


def binarize_au_pred(y: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert py-feat probability output to binary {0, 1}."""
    return (np.asarray(y) >= threshold).astype(np.int8)


def au_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 for a single AU. Returns 0.0 if no positives in truth and pred."""
    y_true = np.asarray(y_true).astype(np.int8)
    y_pred = np.asarray(y_pred).astype(np.int8)
    if y_true.sum() == 0 and y_pred.sum() == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, zero_division=0))


def au_icc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ICC(3,1) — two-way mixed, single-rater, absolute agreement.

    Implemented directly to avoid a pingouin dependency. Formula:

        ICC(3,1) = (MSR - MSE) / (MSR + (k-1) * MSE)

    where MSR is mean square between subjects (rows) and MSE is residual
    mean square error from a two-way ANOVA. For k=2 raters (truth, pred):

        MSR = SS_subjects / (n - 1)
        MSE = SS_error    / ((n - 1) * (k - 1))

    Returns 0.0 when variance is degenerate (all-equal columns).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size != y_pred.size or y_true.size < 2:
        return 0.0

    ratings = np.stack([y_true, y_pred], axis=1)  # shape (n, k=2)
    n, k = ratings.shape
    grand_mean = ratings.mean()
    subj_means = ratings.mean(axis=1)
    rater_means = ratings.mean(axis=0)

    ss_subj = k * np.sum((subj_means - grand_mean) ** 2)
    ss_rater = n * np.sum((rater_means - grand_mean) ** 2)
    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_error = ss_total - ss_subj - ss_rater

    msr = ss_subj / (n - 1) if n > 1 else 0.0
    mse = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0
    denom = msr + (k - 1) * mse
    if denom <= 1e-12:
        return 0.0
    return float((msr - mse) / denom)


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's CCC — used for valence/arousal evaluation in AffectNet.

        CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))**2)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2:
        return 0.0
    mt, mp = y_true.mean(), y_pred.mean()
    vt = np.mean((y_true - mt) ** 2)
    vp = np.mean((y_pred - mp) ** 2)
    cov = np.mean((y_true - mt) * (y_pred - mp))
    denom = vt + vp + (mt - mp) ** 2
    if denom <= 1e-12:
        return 0.0
    return float(2.0 * cov / denom)


def emotion_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 accuracy over integer-coded emotion labels."""
    return float(accuracy_score(y_true, y_pred))


def emotion_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 over integer-coded emotion labels."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def cosine_similarity_pairs(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two ``[N, D]`` embedding arrays."""
    a = np.asarray(emb_a, dtype=np.float64)
    b = np.asarray(emb_b, dtype=np.float64)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a_norm * b_norm, axis=1)


def verification_accuracy_lfw_10fold(
    similarity: np.ndarray,
    same_id: np.ndarray,
    fold: np.ndarray,
) -> dict:
    """LFW-protocol 10-fold cross-validated verification accuracy.

    For each test fold:
      1. Find the threshold on the 9 *training* folds that maximizes
         binary accuracy.
      2. Apply that threshold to the held-out fold.

    Returns mean accuracy across folds, std, best-mean-threshold, and
    AUC. Matches the standard CALFW/CPLFW/LFW reporting convention used
    in the InsightFace paper tables.
    """
    sim = np.asarray(similarity, dtype=np.float64)
    y = np.asarray(same_id, dtype=np.int8)
    f = np.asarray(fold, dtype=np.int32)
    folds = sorted(np.unique(f).tolist())
    accs, thresholds = [], []

    candidate_thresh = np.linspace(sim.min(), sim.max(), 400)

    for held in folds:
        train_mask = f != held
        test_mask = f == held
        # find best train threshold by binary accuracy
        accs_train = [
            float(((sim[train_mask] >= t) == y[train_mask].astype(bool)).mean())
            for t in candidate_thresh
        ]
        best_t = candidate_thresh[int(np.argmax(accs_train))]
        thresholds.append(float(best_t))
        acc_test = float(((sim[test_mask] >= best_t) == y[test_mask].astype(bool)).mean())
        accs.append(acc_test)

    # AUC via Mann-Whitney (no sklearn dep for the metric itself; we use
    # sklearn elsewhere but keep this self-contained for readability)
    pos = sim[y == 1]
    neg = sim[y == 0]
    if pos.size and neg.size:
        n_pos, n_neg = pos.size, neg.size
        all_vals = np.concatenate([pos, neg])
        ranks = all_vals.argsort().argsort().astype(np.float64) + 1.0
        sum_ranks_pos = ranks[: n_pos].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    else:
        auc = float("nan")

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auc": float(auc),
        "threshold_mean": float(np.mean(thresholds)),
        "n_pairs": int(sim.size),
        "n_folds": len(folds),
    }


def rank_k_identification(
    probe_emb: np.ndarray,
    probe_ids: np.ndarray,
    gallery_emb: np.ndarray,
    gallery_ids: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict:
    """Closed-set top-K identification accuracy.

    For each probe, rank gallery rows by cosine similarity (descending)
    and report the fraction of probes whose true identity appears in the
    top-K gallery rows. Distractors (gallery rows whose id never matches
    a probe id) should be appended to ``gallery_emb`` / ``gallery_ids``
    upstream — they hurt rank-K naturally.
    """
    pe = np.asarray(probe_emb, dtype=np.float32)
    ge = np.asarray(gallery_emb, dtype=np.float32)
    pid = np.asarray(probe_ids)
    gid = np.asarray(gallery_ids)

    # L2-normalize so cosine == dot product.
    pe = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-12)
    ge = ge / (np.linalg.norm(ge, axis=1, keepdims=True) + 1e-12)

    # Chunk probe rows to keep peak memory bounded.
    chunk = 256
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    n_scored = 0
    for start in range(0, pe.shape[0], chunk):
        end = min(start + chunk, pe.shape[0])
        sims = pe[start:end] @ ge.T  # [chunk, gallery]
        topk_idx = np.argpartition(-sims, kth=max_k - 1, axis=1)[:, :max_k]
        # Sort each row's top-k for correct rank ordering.
        for row, ti in enumerate(topk_idx):
            order = ti[np.argsort(-sims[row, ti])]
            true_id = pid[start + row]
            for k in ks:
                if true_id in gid[order[:k]]:
                    hits[k] += 1
            n_scored += 1
    return {f"rank_{k}": hits[k] / n_scored for k in ks} | {"n_probes": n_scored}


def gaze_unit_vector(pitch_rad: np.ndarray, yaw_rad: np.ndarray) -> np.ndarray:
    """Convert head-centric (pitch, yaw) in radians to a 3D unit gaze vector.

    Standard L2CS / Gaze360 convention:
      x = -cos(pitch) * sin(yaw)   # +x = subject's left (camera's right)
      y = -sin(pitch)              # +y = up
      z = -cos(pitch) * cos(yaw)   # +z = away from camera (gaze points TO target)
    """
    pitch = np.asarray(pitch_rad, dtype=np.float64)
    yaw = np.asarray(yaw_rad, dtype=np.float64)
    cos_p = np.cos(pitch)
    vec = np.stack([
        -cos_p * np.sin(yaw),
        -np.sin(pitch),
        -cos_p * np.cos(yaw),
    ], axis=-1)
    return vec


def angular_error_degrees(
    pitch_true_rad: np.ndarray, yaw_true_rad: np.ndarray,
    pitch_pred_rad: np.ndarray, yaw_pred_rad: np.ndarray,
) -> np.ndarray:
    """Per-sample angular gaze error in degrees.

    Computed as arccos(dot(unit_true, unit_pred)). Robust to small numerical
    overflow above 1.0 via clipping. The standard metric used in MPIIFaceGaze,
    Gaze360, and ETH-XGaze benchmarks.
    """
    v_true = gaze_unit_vector(pitch_true_rad, yaw_true_rad)
    v_pred = gaze_unit_vector(pitch_pred_rad, yaw_pred_rad)
    dot = np.clip((v_true * v_pred).sum(axis=-1), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def iou_xywh(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two boxes in (x, y, w, h) format."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def summarize_au_metrics(
    y_true_intensity: dict[str, np.ndarray],
    y_pred_prob: dict[str, np.ndarray],
    truth_threshold: int = 2,
    pred_threshold: float = 0.5,
) -> dict:
    """Compute per-AU F1 + ICC and aggregate means.

    Inputs are dicts keyed by canonical AU name (``"AU01"``, ``"AU02"``, ...).
    Returns:
        ``{"per_au": {"AU01": {"f1": ..., "icc": ...}, ...},
           "mean_f1": ..., "mean_icc": ..., "n_aus": ...}``
    Only AUs present in *both* inputs are evaluated.
    """
    common = sorted(set(y_true_intensity) & set(y_pred_prob))
    per_au: dict[str, dict] = {}
    f1s, iccs = [], []
    for au in common:
        t_int = np.asarray(y_true_intensity[au])
        p_prob = np.asarray(y_pred_prob[au])
        mask = ~(np.isnan(t_int.astype(np.float64)) | np.isnan(p_prob.astype(np.float64)))
        t_int, p_prob = t_int[mask], p_prob[mask]
        if t_int.size == 0:
            continue
        t_bin = binarize_au_truth(t_int, threshold=truth_threshold)
        p_bin = binarize_au_pred(p_prob, threshold=pred_threshold)
        f1 = au_f1_binary(t_bin, p_bin)
        icc = au_icc(t_int.astype(np.float64), p_prob.astype(np.float64))
        per_au[au] = {"f1": f1, "icc": icc, "n": int(t_int.size)}
        f1s.append(f1)
        iccs.append(icc)
    return {
        "per_au": per_au,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_icc": float(np.mean(iccs)) if iccs else 0.0,
        "n_aus": len(per_au),
    }
