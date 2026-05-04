"""Parity tests for the batched face mask preparation path.

The legacy per-face Python loop in feat/utils/image_operations.py does:
   align_face -> ConvexHull -> grid_points_in_poly -> mask_image

The batched replacement in feat/utils/face_mask.py runs the same
sequence with the alignment and final mask application batched on the
input device. The convex-hull mask construction itself stays on CPU
(via the same scipy + skimage calls) for bit-for-bit parity with the
legacy AU classifier feature space.
"""

from __future__ import annotations

import numpy as np
import torch

from feat.utils.face_mask import (
    align_faces_batched,
    extract_faces_from_landmarks_batched,
)
from feat.utils.image_operations import (
    extract_face_from_landmarks,
    align_face,
)


def _synthetic_face_batch(n_faces=4, face_size=160, seed=0):
    """Synthetic [N, C, H, W] face batch with deterministic landmarks
    that approximate a realistic 68-landmark face layout.

    The landmarks are placed on an ellipse-ish contour for the jaw,
    a horizontal line for the brows, and small clusters for eyes,
    nose, and mouth. Real values don't matter for parity — only that
    the geometry exercises align_face's anchor math non-trivially.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    faces = torch.rand(n_faces, 3, face_size, face_size)

    # Build a canonical 68-landmark layout in face_size space.
    cx, cy = face_size / 2, face_size / 2
    rx, ry = face_size * 0.35, face_size * 0.45

    # Jaw 0..16 on a downward arc.
    jaw_t = np.linspace(np.pi * 0.85, np.pi * 0.15, 17)
    jaw = np.stack([cx - rx * np.cos(jaw_t), cy + ry * np.sin(jaw_t)], axis=1)

    # Brows 17..26 (left 17-21, right 22-26).
    brow_l = np.stack([np.linspace(cx - rx * 0.7, cx - rx * 0.1, 5), np.full(5, cy - ry * 0.45)], axis=1)
    brow_r = np.stack([np.linspace(cx + rx * 0.1, cx + rx * 0.7, 5), np.full(5, cy - ry * 0.45)], axis=1)

    # Nose 27..35 (vertical bridge 27-30, base 31-35).
    nose_bridge = np.stack([np.full(4, cx), np.linspace(cy - ry * 0.3, cy, 4)], axis=1)
    nose_base = np.stack([np.linspace(cx - rx * 0.15, cx + rx * 0.15, 5), np.full(5, cy + ry * 0.05)], axis=1)

    # Eyes 36..47 (left 36-41, right 42-47).
    eye_l = np.stack(
        [np.linspace(cx - rx * 0.55, cx - rx * 0.2, 6), np.full(6, cy - ry * 0.25)], axis=1
    )
    eye_r = np.stack(
        [np.linspace(cx + rx * 0.2, cx + rx * 0.55, 6), np.full(6, cy - ry * 0.25)], axis=1
    )

    # Mouth 48..67. Outer 48-59, inner 60-67. Just place along a horizontal line.
    mouth_outer = np.stack(
        [np.linspace(cx - rx * 0.35, cx + rx * 0.35, 12), np.full(12, cy + ry * 0.4)], axis=1
    )
    mouth_inner = np.stack(
        [np.linspace(cx - rx * 0.25, cx + rx * 0.25, 8), np.full(8, cy + ry * 0.4)], axis=1
    )

    pts = np.concatenate([jaw, brow_l, brow_r, nose_bridge, nose_base, eye_l, eye_r, mouth_outer, mouth_inner], axis=0)
    assert pts.shape == (68, 2), pts.shape

    # Per-face jitter so all N faces aren't identical.
    landmarks = np.empty((n_faces, 68, 2), dtype=np.float32)
    for i in range(n_faces):
        offset = rng.uniform(-3, 3, size=(1, 2))
        rot_t = rng.uniform(-0.05, 0.05)
        c, s = np.cos(rot_t), np.sin(rot_t)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        rotated = (pts - [cx, cy]) @ R.T + [cx, cy]
        landmarks[i] = (rotated + offset).astype(np.float32)

    return faces, torch.from_numpy(landmarks)


# -------------------- align_faces_batched --------------------


def test_align_faces_batched_shape_and_dtype():
    faces, landmarks = _synthetic_face_batch(n_faces=3, face_size=160)
    aligned, aligned_lm_int, M = align_faces_batched(faces, landmarks, face_size=112)
    assert aligned.shape == (3, 3, 112, 112)
    assert aligned.dtype == torch.float32
    assert aligned_lm_int.shape == (3, 68, 2)
    assert aligned_lm_int.dtype == torch.int64
    assert M.shape == (3, 2, 3)


def test_align_faces_batched_matches_legacy_per_face():
    """For each face, the batched path must reproduce the legacy
    align_face output. Tolerance is bilinear-warp epsilon: aligned
    landmarks are integer-truncated so they should match exactly;
    aligned images may differ by sub-pixel sampling at the boundary
    but the bulk of the image must match closely.
    """
    faces, landmarks = _synthetic_face_batch(n_faces=4, face_size=160)
    face_size = 112

    batched_aligned, batched_lm, _ = align_faces_batched(
        faces, landmarks, face_size=face_size
    )

    for i in range(faces.shape[0]):
        legacy_aligned, legacy_lm = align_face(
            faces[i],
            landmarks[i].flatten().tolist(),
            landmark_type=68,
            box_enlarge=2.5,
            img_size=face_size,
        )
        # Legacy aligned is [1, C, H, W]; squeeze.
        legacy_aligned = legacy_aligned[0]

        # Landmark exact match (both use int truncation on the same float coords).
        np.testing.assert_array_equal(
            batched_lm[i].cpu().numpy(),
            legacy_lm,
            err_msg=f"face {i}: aligned landmark mismatch",
        )

        # Image: 99% of pixels should be within bilinear epsilon. Some
        # boundary pixels can flip due to fill_value vs zero-padding
        # interaction at the edge; allow up to 1% disagreement at >1e-2.
        diff = (batched_aligned[i] - legacy_aligned).abs().max(dim=0).values
        n_disagree = (diff > 1e-2).sum().item()
        n_total = diff.numel()
        assert n_disagree / n_total < 0.01, (
            f"face {i}: {n_disagree}/{n_total} pixels differ by >1e-2 "
            f"(max diff: {diff.max().item():.4f})"
        )


# -------------------- extract_faces_from_landmarks_batched (full pipeline) --------------------


def test_extract_faces_from_landmarks_batched_shape():
    faces, landmarks = _synthetic_face_batch(n_faces=4, face_size=160)
    masked, lm_list = extract_faces_from_landmarks_batched(
        faces, landmarks, face_size=112
    )
    assert masked.shape == (4, 3, 112, 112)
    assert masked.dtype == torch.float32
    assert len(lm_list) == 4
    for arr in lm_list:
        assert arr.shape == (68, 2)


def test_extract_faces_from_landmarks_batched_parity_with_legacy():
    """The masked aligned image and aligned landmarks must match the
    legacy per-face loop output. Tolerances:
      - landmarks: exact (both use int truncation)
      - masked image: per-pixel |new - old| < 5e-2 for >=98% of pixels
        (bilinear epsilon + ray-cast vs grid_points_in_poly edge ambiguity)
    """
    faces, landmarks = _synthetic_face_batch(n_faces=4, face_size=160)

    # Batched.
    batched_masked, batched_lm_list = extract_faces_from_landmarks_batched(
        faces, landmarks, face_size=112
    )

    for i in range(faces.shape[0]):
        legacy_masked, legacy_lm = extract_face_from_landmarks(
            faces[i],
            landmarks[i].flatten(),
            face_size=112,
        )
        # legacy_masked is [1, C, H, W].
        legacy_masked = legacy_masked[0]

        # Landmarks: exact.
        np.testing.assert_array_equal(
            batched_lm_list[i],
            legacy_lm,
            err_msg=f"face {i}: aligned landmark mismatch",
        )

        # Masked image: tolerance for warp + mask edge ambiguity.
        diff = (batched_masked[i] - legacy_masked).abs()
        max_diff = diff.max().item()
        per_pixel_diff = diff.max(dim=0).values  # [H, W]
        n_disagree = (per_pixel_diff > 5e-2).sum().item()
        n_total = per_pixel_diff.numel()
        frac = n_disagree / n_total
        assert frac < 0.02, (
            f"face {i}: {n_disagree}/{n_total} ({frac:.4f}) pixels differ "
            f"by >5e-2 (max diff: {max_diff:.4f})"
        )


# -------------------- end-to-end HOG output parity --------------------


def test_extract_hog_features_batched_matches_legacy():
    """Drop-in replacement test: HOG output matches legacy to float32 ulp.

    extract_hog_features_batched is a drop-in replacement for the legacy
    extract_hog_features (normalized [0, 1] landmark input contract). The
    batched path computes the convex-hull mask via the same skimage call
    as the legacy loop, so the masked-pixel set is identical. HOG output
    matches the legacy path to within float32 epsilon (~5e-6 absolute,
    ~5e-5 relative) — the residual is from numpy-vs-torch last-bit
    differences in float64 division/matmul during affine matrix
    construction, well below any xgboost leaf-split threshold and
    confirmed not to flip end-to-end AU predictions on test images.
    """
    from feat.utils.face_mask import extract_hog_features_batched
    from feat.utils.image_operations import extract_hog_features

    # Generate synthetic faces with NORMALIZED landmarks in [0, 1]
    # (matches the legacy extract_hog_features contract: landmarks come
    # from mobilefacenet output in [0, 1] face-crop coords).
    faces, _ = _synthetic_face_batch(n_faces=4, face_size=112)
    # Re-derive landmarks in normalized space directly.
    torch.manual_seed(7)
    landmarks_norm = torch.rand(faces.shape[0], 68, 2) * 0.6 + 0.2  # in [0.2, 0.8]
    # Build a more face-shaped layout: jaw on bottom arc, eyes upper-mid, etc.
    # Use the same canonical layout as _synthetic_face_batch but in [0, 1].
    cx, cy = 0.5, 0.5
    rx, ry = 0.35, 0.45
    jaw_t = np.linspace(np.pi * 0.85, np.pi * 0.15, 17)
    jaw = np.stack([cx - rx * np.cos(jaw_t), cy + ry * np.sin(jaw_t)], axis=1)
    brow_l = np.stack([np.linspace(cx - rx * 0.7, cx - rx * 0.1, 5), np.full(5, cy - ry * 0.45)], axis=1)
    brow_r = np.stack([np.linspace(cx + rx * 0.1, cx + rx * 0.7, 5), np.full(5, cy - ry * 0.45)], axis=1)
    nose_b = np.stack([np.full(4, cx), np.linspace(cy - ry * 0.3, cy, 4)], axis=1)
    nose_e = np.stack([np.linspace(cx - rx * 0.15, cx + rx * 0.15, 5), np.full(5, cy + ry * 0.05)], axis=1)
    eye_l = np.stack([np.linspace(cx - rx * 0.55, cx - rx * 0.2, 6), np.full(6, cy - ry * 0.25)], axis=1)
    eye_r = np.stack([np.linspace(cx + rx * 0.2, cx + rx * 0.55, 6), np.full(6, cy - ry * 0.25)], axis=1)
    mouth_o = np.stack([np.linspace(cx - rx * 0.35, cx + rx * 0.35, 12), np.full(12, cy + ry * 0.4)], axis=1)
    mouth_i = np.stack([np.linspace(cx - rx * 0.25, cx + rx * 0.25, 8), np.full(8, cy + ry * 0.4)], axis=1)
    pts = np.concatenate([jaw, brow_l, brow_r, nose_b, nose_e, eye_l, eye_r, mouth_o, mouth_i], axis=0)
    landmarks_norm = torch.from_numpy(pts).unsqueeze(0).repeat(faces.shape[0], 1, 1).float()
    landmarks_flat = landmarks_norm.reshape(faces.shape[0], -1)

    legacy_features, _ = extract_hog_features(faces, landmarks_flat)
    new_features, _ = extract_hog_features_batched(faces, landmarks_flat)

    assert new_features.shape == legacy_features.shape
    diff = np.abs(new_features - legacy_features)
    max_abs = diff.max()
    mean_abs = diff.mean()
    # float32 epsilon is ~1.2e-7. Numpy-vs-torch last-bit differences
    # in the float64 affine matrix construction propagate through the
    # warp_affine cast to float32 and produce ~5e-6 max drift. Tight
    # enough that no xgboost leaf split can flip on it; loose enough
    # to allow numpy/torch float64 last-bit differences.
    assert max_abs < 1e-5, (
        f"max |new - legacy| = {max_abs:.6e}; expected float32-ulp drift "
        f"(~5e-6). Drift this large suggests the batched path diverged "
        f"from the legacy mask or affine. mean |diff| = {mean_abs:.6e}."
    )
