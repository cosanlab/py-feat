"""Tests for ``feat.utils.image_operations.procrustes_align_2d_batched``.

The 2D Umeyama similarity helper is consumed by the dlib-68 → MP-478 bridge
in ``feat.plotting`` (PR #305) and may be reused by other landmark-frame
canonicalization paths. These tests pin its algebraic contract independent
of any model.
"""
from __future__ import annotations

import numpy as np
import pytest

from feat.utils.image_operations import procrustes_align_2d_batched


# A small reference anchor configuration that's not pathological — the
# 8 stable upper-face dlib points used by the v2 bridge.
DLIB_ANCHOR_INDICES = np.array(
    [27, 28, 29, 30, 36, 39, 42, 45], dtype=np.int64,
)
REF_ANCHORS = np.array([
    [320.0, 250.0],
    [322.0, 266.0],
    [323.0, 281.0],
    [324.0, 295.0],
    [270.0, 260.0],
    [299.0, 258.0],
    [343.0, 254.0],
    [372.0, 250.0],
], dtype=np.float32)


def _seed_landmarks_with_anchors_at_ref(K=68):
    """Build a (K, 2) landmark set where the anchor subset equals REF_ANCHORS.
    Other landmarks fill in zeros — irrelevant for alignment, which depends
    only on the anchor subset."""
    landmarks = np.zeros((K, 2), dtype=np.float32)
    landmarks[DLIB_ANCHOR_INDICES] = REF_ANCHORS
    return landmarks


# ---------------------------------------------------------------------
# Algebraic correctness
# ---------------------------------------------------------------------

class TestAlgebra:
    def test_identity_when_anchors_already_match(self):
        """If anchors == reference, alignment is a no-op (transform ≈ identity)."""
        lm = _seed_landmarks_with_anchors_at_ref()
        aligned = procrustes_align_2d_batched(
            lm[None], DLIB_ANCHOR_INDICES, REF_ANCHORS,
        )
        np.testing.assert_allclose(
            aligned[0, DLIB_ANCHOR_INDICES], REF_ANCHORS, atol=1e-3,
        )

    def test_recovers_reference_after_known_similarity(self):
        """Apply a known rotate+scale+translate, then align — anchors should
        come back within numerical tolerance of REF_ANCHORS."""
        lm = _seed_landmarks_with_anchors_at_ref()
        theta = np.deg2rad(30.0)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
        s = 1.5
        t = np.array([50.0, -20.0], dtype=np.float32)
        transformed = (s * lm @ R.T + t).astype(np.float32)
        aligned = procrustes_align_2d_batched(
            transformed[None], DLIB_ANCHOR_INDICES, REF_ANCHORS,
        )
        np.testing.assert_allclose(
            aligned[0, DLIB_ANCHOR_INDICES], REF_ANCHORS, atol=1e-2,
        )

    def test_batched_per_face_independence(self):
        """Two faces in a batch must align independently — different
        per-face transforms must not cross-contaminate."""
        lm0 = _seed_landmarks_with_anchors_at_ref()
        # Face 1: rotated 45° + translated
        theta = np.deg2rad(45.0)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
        lm1 = (lm0 @ R.T + np.array([10.0, -5.0], dtype=np.float32)).astype(np.float32)
        batch = np.stack([lm0, lm1])
        aligned = procrustes_align_2d_batched(
            batch, DLIB_ANCHOR_INDICES, REF_ANCHORS,
        )
        # Both faces should land their anchors on REF_ANCHORS independently.
        np.testing.assert_allclose(
            aligned[0, DLIB_ANCHOR_INDICES], REF_ANCHORS, atol=1e-2,
        )
        np.testing.assert_allclose(
            aligned[1, DLIB_ANCHOR_INDICES], REF_ANCHORS, atol=1e-2,
        )

    def test_works_with_arbitrary_K(self):
        """The function isn't 68-specific — should accept any K >= len(anchors)."""
        # K=200 (e.g., a future denser face model)
        K = 200
        lm = np.zeros((K, 2), dtype=np.float32)
        # Place the anchors at indices 0..7 for variety
        anchor_idx = np.arange(8, dtype=np.int64)
        lm[anchor_idx] = REF_ANCHORS
        aligned = procrustes_align_2d_batched(lm[None], anchor_idx, REF_ANCHORS)
        assert aligned.shape == (1, K, 2)
        np.testing.assert_allclose(
            aligned[0, anchor_idx], REF_ANCHORS, atol=1e-3,
        )


# ---------------------------------------------------------------------
# Edge cases / errors
# ---------------------------------------------------------------------

class TestErrorPaths:
    def test_rejects_2d_input(self):
        """Caller must pass a leading batch dim; (K, 2) alone must fail."""
        with pytest.raises(ValueError, match=r"\(n, K, 2\)"):
            procrustes_align_2d_batched(
                np.zeros((68, 2), dtype=np.float32),
                DLIB_ANCHOR_INDICES, REF_ANCHORS,
            )

    def test_rejects_wrong_last_dim(self):
        """3-D shape (n, K, 3) — last dim must be 2."""
        with pytest.raises(ValueError, match=r"\(n, K, 2\)"):
            procrustes_align_2d_batched(
                np.zeros((1, 68, 3), dtype=np.float32),
                DLIB_ANCHOR_INDICES, REF_ANCHORS,
            )

    def test_rejects_anchor_ref_length_mismatch(self):
        """anchor_idx and ref_anchors must have matching lengths."""
        with pytest.raises(ValueError, match=r"ref_anchors shape"):
            procrustes_align_2d_batched(
                np.zeros((1, 68, 2), dtype=np.float32),
                DLIB_ANCHOR_INDICES,           # length 8
                REF_ANCHORS[:5],               # length 5 — mismatch
            )

    def test_empty_batch(self):
        """Zero-row batch should pass through cleanly (parity for the
        no-faces-detected upstream case)."""
        out = procrustes_align_2d_batched(
            np.zeros((0, 68, 2), dtype=np.float32),
            DLIB_ANCHOR_INDICES, REF_ANCHORS,
        )
        assert out.shape == (0, 68, 2)
