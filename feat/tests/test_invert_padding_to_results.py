"""Parity test for the vectorized invert_padding_to_results helper.

Replaces a per-frame loop in `Detector.detect()` and `MPDetector.detect()` that
recomputed `compute_original_image_size`, extracted Padding/Scale numpy arrays,
and rewrote ~140 columns *per frame* via `df.loc[mask, col]`. The helper does
the same work in O(rows + landmarks) by hoisting the once-per-batch lookups
and running every column update with vectorized numpy ops.

This file's parity test reimplements the prior loop verbatim and asserts
element-wise equivalence with the new helper on a synthetic batch.
"""

import numpy as np
import pandas as pd
import torch

from feat.utils.image_operations import (
    compute_original_image_size,
    invert_padding_to_results,
)


def _legacy_invert_padding(batch_results, batch_data, n_landmarks):
    """Faithful reproduction of the prior per-frame inversion loop."""
    for j, frame_idx in enumerate(batch_results["frame"].unique()):
        batch_results.loc[
            batch_results["frame"] == frame_idx, ["FrameHeight", "FrameWidth"]
        ] = (
            compute_original_image_size(batch_data)[j, :]
            .repeat(
                len(batch_results.loc[batch_results["frame"] == frame_idx, "frame"]),
                1,
            )
            .numpy()
        )
        batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectX"] = (
            batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectX"]
            - batch_data["Padding"]["Left"].detach().numpy()[j]
        ) / batch_data["Scale"].detach().numpy()[j]
        batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectY"] = (
            batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectY"]
            - batch_data["Padding"]["Top"].detach().numpy()[j]
        ) / batch_data["Scale"].detach().numpy()[j]
        batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectWidth"] = (
            batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectWidth"]
            / batch_data["Scale"].detach().numpy()[j]
        )
        batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectHeight"] = (
            batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectHeight"]
            / batch_data["Scale"].detach().numpy()[j]
        )
        for i in range(n_landmarks):
            batch_results.loc[batch_results["frame"] == frame_idx, f"x_{i}"] = (
                batch_results.loc[batch_results["frame"] == frame_idx, f"x_{i}"]
                - batch_data["Padding"]["Left"].detach().numpy()[j]
            ) / batch_data["Scale"].detach().numpy()[j]
            batch_results.loc[batch_results["frame"] == frame_idx, f"y_{i}"] = (
                batch_results.loc[batch_results["frame"] == frame_idx, f"y_{i}"]
                - batch_data["Padding"]["Top"].detach().numpy()[j]
            ) / batch_data["Scale"].detach().numpy()[j]
    return batch_results


def _make_synthetic_batch(n_frames=4, n_faces_per_frame=2, n_landmarks=68, seed=0):
    """Build matching `batch_results` (DataFrame) and `batch_data` (dict)."""
    rng = np.random.default_rng(seed)

    rows = []
    for f in range(n_frames):
        for _ in range(n_faces_per_frame):
            row = {
                "frame": 100 + f,  # not zero-indexed, to catch index assumptions
                "FaceRectX": float(rng.uniform(0, 200)),
                "FaceRectY": float(rng.uniform(0, 200)),
                "FaceRectWidth": float(rng.uniform(50, 100)),
                "FaceRectHeight": float(rng.uniform(50, 100)),
                "FrameHeight": 0.0,  # Will be overwritten
                "FrameWidth": 0.0,
            }
            for i in range(n_landmarks):
                row[f"x_{i}"] = float(rng.uniform(0, 400))
                row[f"y_{i}"] = float(rng.uniform(0, 400))
            rows.append(row)
    batch_results = pd.DataFrame(rows)

    pad_left = torch.tensor(
        [int(rng.integers(0, 50)) for _ in range(n_frames)], dtype=torch.float32
    )
    pad_top = torch.tensor(
        [int(rng.integers(0, 50)) for _ in range(n_frames)], dtype=torch.float32
    )
    pad_right = torch.tensor(
        [int(rng.integers(0, 50)) for _ in range(n_frames)], dtype=torch.float32
    )
    pad_bottom = torch.tensor(
        [int(rng.integers(0, 50)) for _ in range(n_frames)], dtype=torch.float32
    )
    scale = torch.tensor(
        [float(rng.uniform(0.5, 2.0)) for _ in range(n_frames)], dtype=torch.float32
    )
    image = torch.zeros(n_frames, 3, 480, 640)

    batch_data = {
        "Image": image,
        "Padding": {
            "Left": pad_left,
            "Top": pad_top,
            "Right": pad_right,
            "Bottom": pad_bottom,
        },
        "Scale": scale,
    }
    return batch_results, batch_data


def test_invert_padding_matches_legacy_68():
    legacy_input, batch_data = _make_synthetic_batch(n_landmarks=68, seed=1)
    new_input = legacy_input.copy()

    legacy_result = _legacy_invert_padding(legacy_input, batch_data, n_landmarks=68)
    new_result = invert_padding_to_results(new_input, batch_data, n_landmarks=68)

    pd.testing.assert_frame_equal(
        legacy_result.reset_index(drop=True),
        new_result.reset_index(drop=True),
        check_exact=False,
        atol=1e-6,
    )


def test_invert_padding_matches_legacy_478():
    """Same parity, with the MediaPipe-sized 478-landmark schema."""
    legacy_input, batch_data = _make_synthetic_batch(n_landmarks=478, seed=2)
    new_input = legacy_input.copy()

    legacy_result = _legacy_invert_padding(legacy_input, batch_data, n_landmarks=478)
    new_result = invert_padding_to_results(new_input, batch_data, n_landmarks=478)

    pd.testing.assert_frame_equal(
        legacy_result.reset_index(drop=True),
        new_result.reset_index(drop=True),
        check_exact=False,
        atol=1e-6,
    )


def test_invert_padding_handles_unsorted_frames():
    """The helper must respect first-appearance order, not numerical order,
    matching the prior `frame.unique()` semantics. This test scrambles the
    rows so frame numbers don't appear in numerical order."""
    legacy_input, batch_data = _make_synthetic_batch(n_landmarks=68, seed=3)
    # Reverse the rows
    legacy_input = legacy_input.iloc[::-1].reset_index(drop=True)
    new_input = legacy_input.copy()

    legacy_result = _legacy_invert_padding(legacy_input, batch_data, n_landmarks=68)
    new_result = invert_padding_to_results(new_input, batch_data, n_landmarks=68)

    pd.testing.assert_frame_equal(
        legacy_result.reset_index(drop=True),
        new_result.reset_index(drop=True),
        check_exact=False,
        atol=1e-6,
    )


def test_invert_padding_empty_input():
    _, batch_data = _make_synthetic_batch(n_landmarks=68)
    empty = pd.DataFrame(
        columns=[
            "frame",
            "FaceRectX",
            "FaceRectY",
            "FaceRectWidth",
            "FaceRectHeight",
            "FrameHeight",
            "FrameWidth",
            *[f"x_{i}" for i in range(68)],
            *[f"y_{i}" for i in range(68)],
        ]
    )
    out = invert_padding_to_results(empty, batch_data, n_landmarks=68)
    assert len(out) == 0
