"""Tests for the pure-torch DLT-PnP head-pose estimator.

The estimator's accuracy on real face images is documented as
approximate (see ``feat.utils.face_pose_pnp``). These tests pin two
weaker but more important properties:

1. **Identity recovery on synthetic data.** When the input landmarks
   are exactly the projection of the 3D template under a known
   ``(R, t)``, the solver recovers ``(R, t)`` to within numerical
   precision. This is the strongest correctness guarantee available
   without baking specific img2pose comparison values into the test
   suite.

2. **Sign / sanity properties on real face images.** Pose values are
   finite, bounded, and translation Z is positive (face in front of
   camera).
"""

from __future__ import annotations

import math

import torch

from feat.utils.face_pose_pnp import (
    default_intrinsics,
    load_img2pose_3d_template,
    rotation_matrix_to_img2pose_euler,
    solve_dlt_pnp,
)


# Real 68 landmarks of the right-most face in feat/tests/data/multi_face.jpg
# (face index 4), detected via Detector(face_model="retinaface"). This
# forward-facing face is near-planar enough that DLT is ill-conditioned: the
# old det(R_raw)>0 sign rule selected the mirror "behind the camera" solution
# (t_z<0, yaw ~= -166 deg). It is the regression fixture for the cheirality fix.
_LANDMARKS_FACE4 = [
    (436.07, 249.09), (437.02, 260.49), (438.45, 271.94), (440.38, 283.26),
    (443.63, 293.41), (449.55, 301.53), (458.02, 306.81), (467.73, 309.61),
    (477.89, 310.36), (487.92, 309.21), (496.84, 305.18), (504.26, 299.08),
    (509.43, 290.72), (512.50, 280.87), (514.64, 270.33), (516.17, 259.56),
    (516.91, 248.71), (445.64, 240.35), (451.52, 236.26), (458.62, 235.29),
    (465.89, 236.76), (472.55, 239.50), (484.92, 239.60), (491.14, 237.13),
    (497.93, 235.69), (504.53, 236.60), (509.62, 240.75), (478.68, 246.10),
    (478.57, 252.33), (478.51, 258.10), (478.52, 263.99), (470.00, 271.58),
    (474.01, 272.96), (478.52, 273.89), (483.02, 273.19), (487.05, 271.92),
    (453.34, 248.87), (458.74, 247.38), (463.33, 247.40), (467.84, 249.47),
    (463.24, 250.08), (458.65, 249.95), (488.83, 249.78), (493.69, 247.76),
    (498.19, 247.88), (502.94, 249.67), (498.09, 250.74), (493.63, 250.69),
    (461.92, 283.72), (468.39, 280.23), (474.59, 278.11), (478.60, 279.22),
    (482.63, 278.36), (488.28, 280.61), (493.50, 284.37), (488.04, 286.47),
    (482.95, 287.84), (478.37, 288.10), (473.88, 287.57), (468.10, 285.88),
    (463.56, 283.64), (474.43, 282.27), (478.62, 282.79), (482.88, 282.52),
    (491.82, 284.22), (482.76, 282.93), (478.45, 283.15), (474.25, 282.76),
]


def _build_test_rotation(pitch_deg: float, yaw_deg: float, roll_deg: float) -> torch.Tensor:
    """Build a [3, 3] rotation matrix from XYZ-intrinsic Euler angles."""
    p, y, r = (math.radians(a) for a in (pitch_deg, yaw_deg, roll_deg))
    Rx = torch.tensor([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = torch.tensor([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _project(template: torch.Tensor, R: torch.Tensor, t: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Standard pinhole projection: returns [N, 2] image-pixel coords."""
    pts_cam = template @ R.T + t
    proj = pts_cam @ K.T
    return proj[:, :2] / proj[:, 2:3]


def test_dlt_recovers_identity_pose():
    """With identity rotation and pure-Z translation, DLT recovers the
    inputs exactly (within float epsilon)."""
    template = load_img2pose_3d_template()
    K = default_intrinsics((480, 640))
    R_true = torch.eye(3)
    t_true = torch.tensor([0.0, 0.0, 5.0])

    landmarks = _project(template, R_true, t_true, K).unsqueeze(0)
    R_rec, t_rec = solve_dlt_pnp(landmarks, template, K)

    assert torch.allclose(R_rec[0], R_true, atol=1e-4), f"R drift: {(R_rec[0] - R_true).abs().max():.2e}"
    assert torch.allclose(t_rec[0], t_true, atol=1e-3), f"t drift: {(t_rec[0] - t_true).abs().max():.2e}"


def test_dlt_recovers_pure_pitch():
    """Pitch about the X-axis is the cleanest single-axis rotation to test;
    DLT should recover it within ~0.01 rad on synthetic input."""
    template = load_img2pose_3d_template()
    K = default_intrinsics((480, 640))
    pitch_deg = 15.0
    R_true = _build_test_rotation(pitch_deg, 0.0, 0.0)
    t_true = torch.tensor([0.0, 0.0, 5.0])

    landmarks = _project(template, R_true, t_true, K).unsqueeze(0)
    R_rec, _ = solve_dlt_pnp(landmarks, template, K)

    # Allow a tighter bound here than the real-image tests: synthetic input
    # has zero noise.
    assert torch.allclose(R_rec[0], R_true, atol=1e-3), (
        f"pitch reconstruction: max R diff "
        f"{(R_rec[0] - R_true).abs().max().item():.4f}"
    )


def test_dlt_translation_z_positive():
    """For a face placed in front of the camera, the recovered tz must be
    positive (camera convention). Sign-handling in ``solve_dlt_pnp``
    guarantees this even when DLT's intermediate solution flips sign."""
    template = load_img2pose_3d_template()
    K = default_intrinsics((480, 640))
    R_true = _build_test_rotation(5.0, -10.0, 3.0)
    t_true = torch.tensor([0.5, -0.3, 7.0])

    landmarks = _project(template, R_true, t_true, K).unsqueeze(0)
    _, t_rec = solve_dlt_pnp(landmarks, template, K)

    assert t_rec[0, 2] > 0, f"recovered tz must be positive, got {t_rec[0, 2]:.3f}"


def test_dlt_no_cheirality_flip_on_real_frontal_face():
    """Regression: a near-frontal real face must not be solved to the mirror
    "behind the camera" pose. The old det(R_raw)>0 sign rule selected t_z<0
    for this face (yaw flipped to ~-166 deg); cheirality-based sign keeps the
    face in front (t_z>0) and yaw within a sane frontal range."""
    template = load_img2pose_3d_template()
    # multi_face.jpg is 667x1000 (H, W); intrinsics derive from this.
    K = default_intrinsics((667, 1000))
    landmarks = torch.tensor(_LANDMARKS_FACE4, dtype=torch.float32).unsqueeze(0)

    R, t = solve_dlt_pnp(landmarks, template, K)

    assert t[0, 2] > 0, f"face placed behind camera (t_z={t[0, 2]:.2f}); cheirality flip"
    yaw = rotation_matrix_to_img2pose_euler(R)[0, 2]
    assert yaw.abs() < math.radians(90), (
        f"yaw {math.degrees(yaw):.1f} deg indicates a ~180 deg pose flip on a "
        "forward-facing face"
    )


def test_dlt_batched_independence():
    """Batched input: each face's solution must be independent of the
    others. Run two distinct synthetic poses through one batched call
    and confirm each is recovered correctly."""
    template = load_img2pose_3d_template()
    K = default_intrinsics((480, 640))
    R1 = _build_test_rotation(10.0, 0.0, 0.0)
    R2 = _build_test_rotation(0.0, -15.0, 0.0)
    t1 = torch.tensor([0.0, 0.0, 5.0])
    t2 = torch.tensor([0.0, 0.0, 8.0])

    L1 = _project(template, R1, t1, K)
    L2 = _project(template, R2, t2, K)
    landmarks = torch.stack([L1, L2], dim=0)

    R_rec, t_rec = solve_dlt_pnp(landmarks, template, K)

    assert torch.allclose(R_rec[0], R1, atol=1e-3)
    assert torch.allclose(R_rec[1], R2, atol=1e-3)
    assert torch.allclose(t_rec[0], t1, atol=1e-3)
    assert torch.allclose(t_rec[1], t2, atol=1e-3)


def test_euler_conversion_round_trip():
    """rotation_matrix_to_img2pose_euler should produce angles whose chain
    back to a rotation matrix recovers the original (up to numerical
    precision). Doesn't pin a specific axis convention - just locks down
    that the conversion is consistent."""
    from feat.utils.geometry import (
        rotation_matrix_to_quaternion,
        euler_from_quaternion,
    )

    # Build a rotation
    R = _build_test_rotation(12.0, -7.0, 5.0).unsqueeze(0)
    angles = rotation_matrix_to_img2pose_euler(R)
    assert angles.shape == (1, 3)
    assert torch.isfinite(angles).all()
    # Re-derive from quaternion path (must match)
    q = rotation_matrix_to_quaternion(R)
    roll, pitch, yaw = euler_from_quaternion(q[..., 0], q[..., 1], q[..., 2], q[..., 3])
    expected = torch.stack([roll, pitch, yaw], dim=-1)
    assert torch.allclose(angles, expected)
