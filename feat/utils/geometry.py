"""Pure-PyTorch geometric helpers used by the detector pipeline.

These replace the small handful of `kornia` calls py-feat used to make so we
don't need to take on the kornia dependency. All ops are batched, dtype/device
preserving, and work inside ``torch.inference_mode``.

Replaces:
- ``kornia.geometry.conversions.axis_angle_to_rotation_matrix``
- ``kornia.geometry.conversions.rotation_matrix_to_axis_angle``
- ``kornia.geometry.conversions.rotation_matrix_to_quaternion``
- ``kornia.geometry.conversions.euler_from_quaternion``
- ``kornia.geometry.transform.warp_affine``

Numerical parity with kornia is tested in
``feat/tests/test_geometry.py`` and held to ~1e-5 absolute tolerance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (rotation vector) to rotation matrix via Rodrigues.

    Args:
        axis_angle: ``[..., 3]`` rotation vectors (axis * angle in radians).

    Returns:
        ``[..., 3, 3]`` rotation matrices.
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"expected last dim 3, got shape {tuple(axis_angle.shape)}")

    theta = axis_angle.norm(dim=-1, keepdim=True)  # [..., 1]
    # Avoid div-by-zero when theta == 0; the eventual rotation is identity in that case.
    safe_theta = torch.where(theta > 0, theta, torch.ones_like(theta))
    axis = axis_angle / safe_theta  # [..., 3]
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    K = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*axis_angle.shape[:-1], 3, 3)

    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(K)
    sin_t = torch.sin(theta).unsqueeze(-1)  # [..., 1, 1]
    cos_t = torch.cos(theta).unsqueeze(-1)
    R = eye + sin_t * K + (1 - cos_t) * (K @ K)
    return R


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle (rotation vector).

    Three numerical regimes need separate handling:

    1. theta near 0: skew-symmetric part is ~0; the full Rodrigues
       formula is unstable. Use ``axis * 0.5`` (the leading-order term).
    2. theta in the bulk: standard recipe ``axis = skew / (2 sin(theta))``.
    3. theta near pi: ``sin(theta) ~ 0`` again, but ``axis`` from the skew
       has vanishingly small magnitude in any direction not orthogonal to
       the axis. Fall back to extracting the axis from the diagonal of
       ``(R + I) / 2``, which has rank 1 and whose dominant column gives
       the axis. Standard fallback (Hartley & Zisserman, Eigen's
       ``AngleAxis::fromRotationMatrix``).

    Args:
        R: ``[..., 3, 3]`` rotation matrices.

    Returns:
        ``[..., 3]`` axis-angle vectors (axis * angle in radians).
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected trailing shape (3, 3), got {tuple(R.shape)}")

    trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [...]
    cos_theta = ((trace - 1) / 2).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # [..., values in [0, pi]]

    skew_axis = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )  # [..., 3]; magnitude = 2 sin(theta)

    sin_theta = torch.sin(theta).unsqueeze(-1)  # [..., 1]
    safe_sin = torch.where(sin_theta.abs() > 1e-8, sin_theta, torch.ones_like(sin_theta))
    bulk_axis = skew_axis / (2 * safe_sin)  # standard branch

    # theta near pi fallback. (R + I) / 2 = axis ⊗ axis (outer product)
    # for theta = pi exactly. Pick the column with largest diagonal entry,
    # then normalize to unit length and fix sign so it agrees with skew_axis.
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    M_pi = (R + eye) / 2.0  # [..., 3, 3]
    diag = torch.stack([M_pi[..., 0, 0], M_pi[..., 1, 1], M_pi[..., 2, 2]], dim=-1)
    col_idx = diag.argmax(dim=-1)  # [...]
    # Gather the column of M_pi indexed by col_idx.
    idx = col_idx.unsqueeze(-1).unsqueeze(-1).expand(*col_idx.shape, 3, 1)  # [..., 3, 1]
    pi_axis = M_pi.gather(-1, idx).squeeze(-1)  # [..., 3]
    pi_axis = pi_axis / pi_axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    # Sign: pi_axis . skew_axis should match (positive) since both encode
    # the same physical rotation direction.
    sign = torch.sign((pi_axis * skew_axis).sum(dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    pi_axis = pi_axis * sign

    # Use the M_pi fallback whenever sin(theta) is small enough that the
    # bulk formula `axis = skew / (2 sin(theta))` loses precision. With
    # float32 the bulk path has ~1% relative error at theta = pi - 0.1
    # and degrades quickly past that, so we switch over at pi - 0.05.
    near_pi = (theta > (torch.pi - 0.05)).unsqueeze(-1)
    near_zero = (theta < 1e-6).unsqueeze(-1)

    # Compose the three branches.
    # near_zero: skew_axis * 0.5 (leading-order term).
    # near_pi: pi_axis * theta.
    # bulk: bulk_axis * theta.
    bulk = bulk_axis * theta.unsqueeze(-1)
    near_pi_out = pi_axis * theta.unsqueeze(-1)
    out = torch.where(near_pi, near_pi_out, bulk)
    out = torch.where(near_zero, skew_axis * 0.5, out)
    return out


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to unit quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability across the rotation sphere.
    Returns a ``[..., 4]`` tensor with the scalar component first, matching
    kornia's output ordering.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected trailing shape (3, 3), got {tuple(R.shape)}")

    m = R
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    # Branch by which trace term is largest for stability.
    # We compute all four candidates and pick per-element.
    t0 = 1 + trace
    t1 = 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]
    t2 = 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]
    t3 = 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]

    # Branch 0: w largest.
    s0 = 2 * torch.sqrt(t0.clamp(min=1e-12))
    w0 = 0.25 * s0
    x0 = (m[..., 2, 1] - m[..., 1, 2]) / s0
    y0 = (m[..., 0, 2] - m[..., 2, 0]) / s0
    z0 = (m[..., 1, 0] - m[..., 0, 1]) / s0

    # Branch 1: x largest.
    s1 = 2 * torch.sqrt(t1.clamp(min=1e-12))
    w1 = (m[..., 2, 1] - m[..., 1, 2]) / s1
    x1 = 0.25 * s1
    y1 = (m[..., 0, 1] + m[..., 1, 0]) / s1
    z1 = (m[..., 0, 2] + m[..., 2, 0]) / s1

    # Branch 2: y largest.
    s2 = 2 * torch.sqrt(t2.clamp(min=1e-12))
    w2 = (m[..., 0, 2] - m[..., 2, 0]) / s2
    x2 = (m[..., 0, 1] + m[..., 1, 0]) / s2
    y2 = 0.25 * s2
    z2 = (m[..., 1, 2] + m[..., 2, 1]) / s2

    # Branch 3: z largest.
    s3 = 2 * torch.sqrt(t3.clamp(min=1e-12))
    w3 = (m[..., 1, 0] - m[..., 0, 1]) / s3
    x3 = (m[..., 0, 2] + m[..., 2, 0]) / s3
    y3 = (m[..., 1, 2] + m[..., 2, 1]) / s3
    z3 = 0.25 * s3

    # Pick the branch corresponding to the largest of (trace, m00, m11, m22).
    diag = torch.stack(
        [
            trace,
            m[..., 0, 0],
            m[..., 1, 1],
            m[..., 2, 2],
        ],
        dim=-1,
    )
    branch = diag.argmax(dim=-1)  # [...] in {0, 1, 2, 3}

    w_all = torch.stack([w0, w1, w2, w3], dim=-1)
    x_all = torch.stack([x0, x1, x2, x3], dim=-1)
    y_all = torch.stack([y0, y1, y2, y3], dim=-1)
    z_all = torch.stack([z0, z1, z2, z3], dim=-1)

    idx = branch.unsqueeze(-1)
    w = w_all.gather(-1, idx).squeeze(-1)
    x = x_all.gather(-1, idx).squeeze(-1)
    y = y_all.gather(-1, idx).squeeze(-1)
    z = z_all.gather(-1, idx).squeeze(-1)

    quat = torch.stack([w, x, y, z], dim=-1)
    return F.normalize(quat, dim=-1)


def euler_from_quaternion(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    """Convert (w, x, y, z) quaternion to (roll, pitch, yaw) Euler angles.

    Matches ``kornia.geometry.conversions.euler_from_quaternion`` (XYZ
    intrinsic, returned as a 3-tuple of tensors).
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation), with safe-clamp for asin domain.
    sinp = (2 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    fill_value: tuple = None,
) -> torch.Tensor:
    """Apply a 2D affine transform to a batch of images.

    Drop-in replacement for ``kornia.geometry.transform.warp_affine``.
    Convention matches kornia: ``M`` is a ``[B, 2, 3]`` matrix mapping
    *source* pixel coordinates to *destination* pixel coordinates (the
    forward, visible transform). Internally this is inverted to feed
    ``F.affine_grid``, which wants the destination-to-source mapping.

    The pixel-coordinate normalization always uses kornia's convention
    (``2 / (dim - 1)`` scale; the same form for both ``align_corners``
    settings) — kornia's ``normal_transform_pixel`` does not branch on
    ``align_corners``, so neither do we. ``align_corners`` is still
    passed to ``F.affine_grid`` and ``F.grid_sample`` to match kornia
    end-to-end.

    Args:
        src: ``[B, C, H, W]`` source images.
        M: ``[B, 2, 3]`` affine matrices in pixel coordinates (src -> dst).
        dsize: ``(out_h, out_w)`` output spatial size.
        mode: ``"bilinear"`` or ``"nearest"`` for ``grid_sample``.
        padding_mode: passed through to ``grid_sample``.
        align_corners: passed through to ``grid_sample``. Default ``False``
            matches kornia's default.
        fill_value: optional ``(r, g, b)`` (or per-channel) constant fill
            applied where the warp would sample outside the source image.
            When provided, ``padding_mode`` is forced to ``"zeros"`` and
            an out-of-bounds mask is composited with the constant
            afterwards. Matches kornia's behavior when ``padding_mode``
            in {"fill", "zeros"} with a fill_value.

    Returns:
        ``[B, C, out_h, out_w]`` warped images.
    """
    if src.dim() != 4:
        raise ValueError(f"src must be [B, C, H, W], got {tuple(src.shape)}")
    if M.shape[-2:] != (2, 3):
        raise ValueError(f"M must have trailing shape (2, 3), got {tuple(M.shape)}")
    if M.dim() == 2:
        M = M.unsqueeze(0).expand(src.shape[0], -1, -1)

    B, C, H, W = src.shape
    out_h, out_w = dsize

    # Build [B, 3, 3] homogeneous form of M (src_pix -> dst_pix).
    M_h = torch.cat(
        [M, torch.tensor([[[0.0, 0.0, 1.0]]], dtype=M.dtype, device=M.device).expand(B, 1, 3)],
        dim=1,
    )

    # Convert to normalized coordinates (kornia's normalize_homography):
    #   src_norm -> dst_norm  =  dst_norm_from_dst_pix @ M_h @ src_pix_from_src_norm
    src_pix_from_src_norm = _norm_to_pixel_matrix(H, W, align_corners, src.dtype, src.device)
    dst_norm_from_dst_pix = _pixel_to_norm_matrix(out_h, out_w, align_corners, src.dtype, src.device)
    dst_norm_from_src_norm = dst_norm_from_dst_pix @ M_h @ src_pix_from_src_norm

    # affine_grid wants dst_norm -> src_norm, i.e. the inverse.
    src_norm_from_dst_norm = torch.linalg.inv(dst_norm_from_src_norm)
    theta = src_norm_from_dst_norm[:, :2, :]

    grid = F.affine_grid(theta, [B, C, out_h, out_w], align_corners=align_corners)

    if fill_value is None:
        return F.grid_sample(src, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    # fill_value path: sample with zero padding, then composite the constant
    # over the out-of-bounds region using a per-pixel "in-bounds" mask.
    warped = F.grid_sample(src, grid, mode=mode, padding_mode="zeros", align_corners=align_corners)
    in_bounds = ((grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1)).unsqueeze(1)  # [B, 1, H, W]
    fill = torch.tensor(fill_value, dtype=src.dtype, device=src.device).reshape(1, -1, 1, 1)
    if fill.shape[1] == 1:
        fill = fill.expand(1, C, 1, 1)
    return torch.where(in_bounds, warped, fill)


def _norm_to_pixel_matrix(h: int, w: int, align_corners: bool, dtype, device) -> torch.Tensor:
    """3x3 matrix mapping normalized coords in [-1, 1] to pixel coords [0, dim-1].

    Empirically matches kornia: with ``align_corners=False`` the scale is
    ``dim / 2`` (since the [-1, 1] range maps over ``dim`` pixels); with
    ``align_corners=True`` it's ``(dim - 1) / 2``. The translation column
    centers the origin at the image center either way.
    """
    if align_corners:
        sx = (w - 1) / 2.0
        sy = (h - 1) / 2.0
    else:
        sx = w / 2.0
        sy = h / 2.0
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    return torch.tensor(
        [[sx, 0.0, cx], [0.0, sy, cy], [0.0, 0.0, 1.0]], dtype=dtype, device=device
    )


def _pixel_to_norm_matrix(h: int, w: int, align_corners: bool, dtype, device) -> torch.Tensor:
    """3x3 matrix mapping pixel coords [0, dim-1] to normalized coords [-1, 1]."""
    return torch.linalg.inv(_norm_to_pixel_matrix(h, w, align_corners, dtype, device))
