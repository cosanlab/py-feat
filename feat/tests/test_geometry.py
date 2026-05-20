"""Numerical parity tests for the in-house geometry helpers vs kornia.

These tests run only if `kornia` is importable (it's installed in the
current py-feat environment for the v0.7 transition; once kornia is
removed from requirements, this test file will skip gracefully).

The helpers in `feat.utils.geometry` replace specific kornia call sites
that were used in `feat/utils/image_operations.py` and
`feat/facepose_detectors/img2pose/deps/pose_operations.py`. Parity is
held to ~1e-5 absolute tolerance.
"""

import math

import pytest
import torch

from feat.utils.geometry import (
    axis_angle_to_rotation_matrix,
    euler_from_quaternion,
    rotation_matrix_to_axis_angle,
    rotation_matrix_to_quaternion,
    warp_affine,
)

try:
    import kornia  # noqa: F401
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

requires_kornia = pytest.mark.skipif(
    not HAS_KORNIA, reason="kornia not installed; parity tests skipped"
)


@requires_kornia
def test_axis_angle_to_rotation_matrix_matches_kornia():
    torch.manual_seed(0)
    rvecs = torch.randn(8, 3)
    R_ours = axis_angle_to_rotation_matrix(rvecs)
    R_kornia = kornia.geometry.conversions.axis_angle_to_rotation_matrix(rvecs)
    torch.testing.assert_close(R_ours, R_kornia, atol=1e-5, rtol=1e-5)


def test_axis_angle_to_rotation_matrix_zero_rotation():
    """Zero rotation vector must give identity (no NaN from div by 0)."""
    rvec = torch.zeros(1, 3)
    R = axis_angle_to_rotation_matrix(rvec)
    torch.testing.assert_close(R, torch.eye(3).unsqueeze(0))


def test_rotation_matrix_to_axis_angle_round_trip():
    torch.manual_seed(1)
    rvecs = torch.randn(8, 3) * 0.5  # avoid extreme angles where wrapping kicks in
    R = axis_angle_to_rotation_matrix(rvecs)
    rvecs_recovered = rotation_matrix_to_axis_angle(R)
    torch.testing.assert_close(rvecs_recovered, rvecs, atol=1e-5, rtol=1e-5)


@requires_kornia
def test_rotation_matrix_to_axis_angle_matches_kornia():
    torch.manual_seed(2)
    rvecs = torch.randn(8, 3) * 0.5
    R = axis_angle_to_rotation_matrix(rvecs)
    rvec_ours = rotation_matrix_to_axis_angle(R)
    rvec_kornia = kornia.geometry.conversions.rotation_matrix_to_axis_angle(R)
    torch.testing.assert_close(rvec_ours, rvec_kornia, atol=1e-5, rtol=1e-5)


def test_rotation_matrix_to_quaternion_round_trip():
    """Quaternion built from R, used to rebuild R, should round-trip."""
    torch.manual_seed(3)
    rvecs = torch.randn(8, 3)
    R = axis_angle_to_rotation_matrix(rvecs)
    q = rotation_matrix_to_quaternion(R)
    # Build R back from q
    w, x, y, z = q.unbind(-1)
    R_back = torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (x * y + w * z),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - w * x),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x * x + y * y),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )
    torch.testing.assert_close(R_back, R, atol=1e-5, rtol=1e-5)


@requires_kornia
def test_rotation_matrix_to_quaternion_matches_kornia():
    torch.manual_seed(4)
    rvecs = torch.randn(8, 3)
    R = axis_angle_to_rotation_matrix(rvecs)
    q_ours = rotation_matrix_to_quaternion(R)
    q_kornia = kornia.geometry.conversions.rotation_matrix_to_quaternion(R)
    # Quaternions are equivalent up to sign; align signs before comparing.
    sign = torch.sign((q_ours * q_kornia).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q_ours_aligned = q_ours * sign
    torch.testing.assert_close(q_ours_aligned, q_kornia, atol=1e-5, rtol=1e-5)


@requires_kornia
def test_euler_from_quaternion_matches_kornia():
    torch.manual_seed(5)
    rvecs = torch.randn(8, 3)
    R = axis_angle_to_rotation_matrix(rvecs)
    q = rotation_matrix_to_quaternion(R)
    w, x, y, z = q.unbind(-1)
    e_ours = torch.stack(euler_from_quaternion(w, x, y, z), dim=-1)
    e_kornia = torch.stack(
        kornia.geometry.conversions.euler_from_quaternion(w, x, y, z), dim=-1
    )
    torch.testing.assert_close(e_ours, e_kornia, atol=1e-5, rtol=1e-5)


def test_warp_affine_identity():
    torch.manual_seed(6)
    src = torch.rand(2, 3, 64, 64)
    eye = torch.eye(2, 3).unsqueeze(0).expand(2, -1, -1)
    out = warp_affine(src, eye, (64, 64), align_corners=False)
    # Identity warp should give back the source pretty closely. With
    # align_corners=False there's an inherent half-pixel offset in
    # grid_sample; we accept a loose tolerance and check shapes/finite.
    assert out.shape == src.shape
    assert torch.isfinite(out).all()
    # Image content correlation should be very high.
    src_flat = src.flatten()
    out_flat = out.flatten()
    correlation = torch.corrcoef(torch.stack([src_flat, out_flat]))[0, 1]
    assert correlation > 0.99


@requires_kornia
def test_warp_affine_matches_kornia_translation():
    """Compare a non-trivial translation against kornia."""
    torch.manual_seed(7)
    src = torch.rand(2, 3, 64, 64)
    M = torch.tensor(
        [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]]
    ).unsqueeze(0).expand(2, -1, -1).contiguous()
    out_ours = warp_affine(src, M, (64, 64))
    out_kornia = kornia.geometry.transform.warp_affine(src, M, (64, 64))
    torch.testing.assert_close(out_ours, out_kornia, atol=1e-4, rtol=1e-4)


def test_warp_affine_accepts_fill_value():
    """`align_face` calls warp_affine with fill_value=(128, 128, 128). The
    helper must accept the kwarg and apply the constant outside the warped
    region."""
    src = torch.zeros(1, 3, 32, 32)
    src[..., 8:24, 8:24] = 1.0
    # Translate by (40, 40) px so the entire output is out-of-bounds.
    M = torch.tensor([[1.0, 0.0, 40.0], [0.0, 1.0, 40.0]]).unsqueeze(0)
    out = warp_affine(src, M, (32, 32), fill_value=(128.0, 128.0, 128.0))
    expected = torch.full_like(src, 128.0)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


def test_rotation_matrix_to_axis_angle_stable_near_pi():
    """The theta-near-pi fallback must produce sensible axis-angle output
    instead of catastrophic numerical drift. Round-trip the rotation matrix
    across angles approaching pi and assert it matches the input."""
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.7, 0.5, 0.5],
        ]
    )
    axes = axes / axes.norm(dim=-1, keepdim=True)
    thetas = [0.1, 1.0, 2.0, math.pi - 0.1, math.pi - 1e-3, math.pi - 1e-5]
    for axis in axes:
        for theta in thetas:
            rvec = (axis * theta).unsqueeze(0)
            R = axis_angle_to_rotation_matrix(rvec)
            rvec_back = rotation_matrix_to_axis_angle(R)
            R_back = axis_angle_to_rotation_matrix(rvec_back)
            # Tolerance is set to float32 precision near the theta=pi
            # singular boundary; the in-house helper round-trips to within
            # ~1e-3 of input R, which is the inherent precision of
            # `acos((trace-1)/2)` for cos_theta near -1.
            torch.testing.assert_close(
                R_back, R, atol=2e-3, rtol=2e-3
            )


@requires_kornia
def test_warp_affine_matches_kornia_rotation_scale():
    """Compare a rotation+scale warp against kornia on a different output size."""
    torch.manual_seed(8)
    src = torch.rand(1, 3, 96, 96)
    angle = math.radians(15)
    s = 0.8
    M = torch.tensor(
        [
            [s * math.cos(angle), -s * math.sin(angle), 10.0],
            [s * math.sin(angle), s * math.cos(angle), 5.0],
        ]
    ).unsqueeze(0)
    out_ours = warp_affine(src, M, (48, 48))
    out_kornia = kornia.geometry.transform.warp_affine(src, M, (48, 48))
    torch.testing.assert_close(out_ours, out_kornia, atol=1e-4, rtol=1e-4)
