"""Smoke + parity tests for MPDetector.plot_face_landmarks.

Pre-PR: the plotting function ran 9 separate per-face Python loops, one
per face part (tessellation, lips, irises, eyes, eyebrows, oval). Each
loop did the same iteration with a different connection set + style.
This test pins down the line-count contract so a refactor that
collapses the loops can't silently lose a part or change the
draw-order semantics.

Each face draws 2688 line segments (sum of connections across the 9
parts). For N faces, the axis ends up with N * 2688 Line2D children.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from feat.MPDetector import plot_face_landmarks
from feat.data import Fex
from feat.utils import MP_LANDMARK_COLUMNS
from feat.utils.io import get_test_data_path


# Sum of FACE_LANDMARKS_* connection counts across the 9 parts:
# tessellation 2556 + lips 40 + L_iris 4 + L_eye 16 + L_brow 8
# + R_iris 4 + R_eye 16 + R_brow 8 + face_oval 36 = 2688
EXPECTED_LINES_PER_FACE = 2688


def _synthetic_mp_fex(n_faces: int) -> Fex:
    """Build a minimal Fex with mediapipe-style landmarks for testing.

    n_faces faces all in frame=0, integer-indexed 0..n-1. Landmark
    coords are random within [0, 200] so plotted lines fall inside the
    test image. Only the columns plot_face_landmarks reads are
    populated; everything else is filler.
    """
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 200, size=(n_faces, len(MP_LANDMARK_COLUMNS)))
    df = pd.DataFrame(coords, columns=MP_LANDMARK_COLUMNS)
    df["frame"] = 0
    df["input"] = os.path.join(get_test_data_path(), "single_face.jpg")
    fex = Fex(df, landmark_columns=MP_LANDMARK_COLUMNS)
    return fex


@pytest.mark.parametrize("n_faces", [1, 3])
def test_plot_face_landmarks_line_count(n_faces):
    """The plotted axis must contain exactly n_faces * 2688 Line2D
    segments — one per (face, connection) across the 9 parts."""
    fex = _synthetic_mp_fex(n_faces=n_faces)
    fig, ax = plt.subplots()
    try:
        plot_face_landmarks(fex, frame_idx=0, ax=ax)
        assert len(ax.lines) == n_faces * EXPECTED_LINES_PER_FACE
    finally:
        plt.close(fig)


def test_plot_face_landmarks_returns_axis():
    """Function must return the axis it drew on (not None)."""
    fex = _synthetic_mp_fex(n_faces=1)
    fig, ax = plt.subplots()
    try:
        result = plot_face_landmarks(fex, frame_idx=0, ax=ax)
        assert result is ax
    finally:
        plt.close(fig)


def test_plot_face_landmarks_draw_order_part_major():
    """Draw order must be part-major: all faces' tessellation drawn
    first (backdrop), all faces' face ovals drawn last (foreground).

    matplotlib stacks Line2D children in insertion order. With
    n_faces=2 and the loop refactor, the first 2*N_TESSELATION lines
    must be the tessellation lines and the LAST 2*N_OVAL lines must
    be the face oval lines. If the refactor accidentally swapped to
    face-major order, face 1's oval would land mid-list and the test
    would catch it.
    """
    from feat.utils.mp_plotting import FaceLandmarksConnections

    n_tess = len(FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)
    n_oval = len(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)

    fex = _synthetic_mp_fex(n_faces=2)
    fig, ax = plt.subplots()
    try:
        plot_face_landmarks(
            fex,
            frame_idx=0,
            ax=ax,
            tesselation_color="red",
            oval_color="blue",
        )
        # First 2 * n_tess lines are tessellation (red).
        for line in ax.lines[: 2 * n_tess]:
            assert line.get_color() == "red"
        # Last 2 * n_oval lines are face oval (blue).
        for line in ax.lines[-2 * n_oval :]:
            assert line.get_color() == "blue"
    finally:
        plt.close(fig)
