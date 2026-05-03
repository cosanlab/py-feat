"""Pure-PyTorch PnP head-pose estimation from 2D landmarks.

When ``Detector(face_model='retinaface')`` (or any other detector that
doesn't natively regress 6DoF pose) is used, the 6DoF pose columns are
populated by solving a Perspective-n-Point (PnP) problem against the 68
2D landmarks from the landmark stage.

This module ships a closed-form **Direct Linear Transform (DLT)** PnP
solver in pure PyTorch:

  * Closed-form (single SVD, no iteration, vectorized across the batch)
  * No OpenCV / cv2.solvePnP dependency
  * Works inside ``torch.inference_mode()``
  * Reuses img2pose's published 68-point 3D template (HF config
    ``threed_points``), so the recovered pose lives in the same
    head-centric coordinate system as img2pose's regressed pose.

Accuracy caveats:
  * DLT minimizes algebraic error rather than pixel-space reprojection
    error, so it's looser than what cv2.solvePnP's iterative refinement
    or EPnP would give. Empirically on real face images it can disagree
    with img2pose's regressed pose by 5-10 degrees on Roll/Yaw and up to
    20-30 degrees on Pitch (the axis with shallowest landmark
    differentiation).
  * Default camera intrinsics use focal length = max(H, W) when the true
    intrinsics aren't known. This is approximate.
  * The pose estimate uses ONLY landmarks; img2pose's regressed pose
    uses image features directly. The two methods are not expected to
    agree exactly even at perfect convergence.

If you need img2pose-grade pose accuracy, stay on
``Detector(face_model='img2pose')``. The PnP path is documented as
approximate and intended for users who prefer faster batched detection
via ``face_model='retinaface'`` and can tolerate looser pose values.
"""

from __future__ import annotations

import json
from typing import Tuple

import torch

from feat.utils.io import get_resource_path
from huggingface_hub import hf_hub_download


_TEMPLATE_CACHE: dict = {}


def load_img2pose_3d_template(device=None) -> torch.Tensor:
    """Load img2pose's 68-point 3D face template from its HF config.

    The template is in head-centric coordinates with the same axis
    convention img2pose uses internally. Returns ``[68, 3]`` float32.

    Parses + materializes the template once per process (and once per
    requested device); subsequent calls hit a module-level cache. Without
    this cache, calling ``pose_from_landmarks_2d`` per detect() would
    re-open the HF cache file and re-parse the same JSON every frame -
    a real perf hit at video frame rates.
    """
    cpu_key = "cpu"
    if cpu_key not in _TEMPLATE_CACHE:
        config_path = hf_hub_download(
            repo_id="py-feat/img2pose",
            filename="config.json",
            cache_dir=get_resource_path(),
        )
        with open(config_path, "r") as f:
            cfg = json.load(f)
        pts = torch.tensor(cfg["threed_points"], dtype=torch.float32)
        if pts.shape != (68, 3):
            raise ValueError(
                f"expected (68, 3) template, got {tuple(pts.shape)} in img2pose config"
            )
        _TEMPLATE_CACHE[cpu_key] = pts

    if device is None:
        return _TEMPLATE_CACHE[cpu_key]

    # Cache one materialized tensor per device so repeated GPU/MPS calls
    # don't re-do the .to() copy each time.
    device_key = str(torch.device(device))
    if device_key not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[device_key] = _TEMPLATE_CACHE[cpu_key].to(device)
    return _TEMPLATE_CACHE[device_key]


def default_intrinsics(image_size: Tuple[int, int], device=None) -> torch.Tensor:
    """Build a reasonable default camera intrinsic matrix.

    Pinhole camera with focal length = max(W, H) (a common choice when the
    true camera intrinsics are unknown - the diagonal-of-image heuristic),
    principal point at image center, square pixels. Approximate but
    typically within a factor of 2 of true intrinsics for consumer cameras
    and standard webcam captures.

    The recovered (R, t) is sensitive to focal length differently for
    rotation vs. translation: a 40% focal-length error shifts ``t`` by
    ~40% but rotates by less than ~2° in practice. Treat the X/Y/Z columns
    as the unreliable ones when intrinsics are guessed; angles are robust.

    Args:
        image_size: ``(H, W)`` of the input image in pixels.

    Returns:
        ``[3, 3]`` float32 intrinsic matrix.
    """
    H, W = image_size
    f = float(max(H, W))
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor(
        [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    if device is not None:
        K = K.to(device)
    return K


def solve_dlt_pnp(
    points_2d: torch.Tensor, points_3d: torch.Tensor, K: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve PnP via the Direct Linear Transform.

    For each face in the batch, finds the rotation R and translation t
    such that, in homogeneous coords, ``K @ [R | t] @ X = lambda * x`` is
    satisfied in the least-squares sense across the 68 correspondences.

    Args:
        points_2d: ``[B, N, 2]`` image-plane landmarks in pixel coords.
        points_3d: ``[N, 3]`` head-centric 3D template (shared across the
            batch), or ``[B, N, 3]`` for per-face templates.
        K: ``[3, 3]`` camera intrinsics (shared across the batch).

    Returns:
        ``(R, t)`` where ``R`` is ``[B, 3, 3]`` and ``t`` is ``[B, 3]``.
    """
    if points_2d.dim() != 3 or points_2d.shape[-1] != 2:
        raise ValueError(f"points_2d must be [B, N, 2], got {tuple(points_2d.shape)}")
    if points_3d.dim() == 2:
        points_3d = points_3d.unsqueeze(0).expand(points_2d.shape[0], -1, -1)
    if points_3d.shape[-1] != 3:
        raise ValueError(f"points_3d must end in dim 3, got {tuple(points_3d.shape)}")

    B, N, _ = points_2d.shape
    device = points_2d.device

    # Move 2D points to normalized camera coords by left-multiplying by K^-1.
    # This decouples DLT from the intrinsics so we recover R and t in metric
    # coordinates directly.
    K_inv = torch.linalg.inv(K)  # [3, 3]
    p2_h = torch.cat([points_2d, torch.ones_like(points_2d[..., :1])], dim=-1)  # [B,N,3]
    p2_norm = p2_h @ K_inv.T  # [B, N, 3] -> normalized homogeneous
    u = p2_norm[..., 0]  # [B, N]
    v = p2_norm[..., 1]

    # Build the [B, 2N, 12] DLT matrix. For each correspondence (X, Y, Z) <-> (u, v):
    #   Row 1:  [X Y Z 1   0 0 0 0   -u*X -u*Y -u*Z -u]
    #   Row 2:  [0 0 0 0   X Y Z 1   -v*X -v*Y -v*Z -v]
    X = points_3d[..., 0]
    Y = points_3d[..., 1]
    Z = points_3d[..., 2]
    ones = torch.ones_like(X)
    zeros = torch.zeros_like(X)

    row1 = torch.stack(
        [X, Y, Z, ones, zeros, zeros, zeros, zeros, -u * X, -u * Y, -u * Z, -u],
        dim=-1,
    )  # [B, N, 12]
    row2 = torch.stack(
        [zeros, zeros, zeros, zeros, X, Y, Z, ones, -v * X, -v * Y, -v * Z, -v],
        dim=-1,
    )
    A = torch.stack([row1, row2], dim=2).reshape(B, 2 * N, 12)  # [B, 2N, 12]

    # Solve A @ p = 0 in the least-squares sense via SVD. The solution is the
    # right-singular vector corresponding to the smallest singular value.
    _, _, Vh = torch.linalg.svd(A, full_matrices=False)
    p = Vh[:, -1, :]  # [B, 12]
    P = p.view(B, 3, 4)  # [B, 3, 4]

    # The first 3x3 block of P is s*R (rotation up to scale). Extract scale
    # from the determinant of the rotation block, then renormalize.
    R_raw = P[:, :, :3]  # [B, 3, 3]
    t_raw = P[:, :, 3]  # [B, 3]

    # DLT solves A @ p = 0 up to a sign — both p and -p are solutions, but
    # only one corresponds to "face in front of camera". The wrong sign
    # gives a mirror configuration with det(R_raw) < 0, which Procrustes
    # would project to a rotation about the wrong axis. Resolve the sign
    # before Procrustes by checking det(R_raw): for a face in front of
    # camera, det(R_raw) = scale^3 * det(R) = scale^3 (since det(R) = 1
    # for a proper rotation). We need scale > 0 — equivalently det(R_raw)
    # > 0 — so flip the sign of P when it's negative.
    det_raw = torch.linalg.det(R_raw)
    sign_p = torch.sign(det_raw)
    sign_p = torch.where(sign_p == 0, torch.ones_like(sign_p), sign_p)
    R_raw = R_raw * sign_p.view(B, 1, 1)
    t_raw = t_raw * sign_p.unsqueeze(-1)

    # Project R_raw onto the closest rotation matrix via SVD (Procrustes).
    # det(U V^T) is +1 here (since we already flipped sign so det(R_raw) > 0),
    # but we keep the determinant-fix for numerical robustness.
    U, _, Vt = torch.linalg.svd(R_raw)
    det = torch.linalg.det(U @ Vt)
    flip = torch.diag_embed(
        torch.cat(
            [torch.ones(B, 2, device=device), det.unsqueeze(-1).sign()], dim=-1
        )
    )
    R = U @ flip @ Vt

    # Recover scale and unscaled translation. With sign already resolved,
    # s is positive and t[:, 2] should be > 0 for a face in front.
    s = torch.einsum("bij,bij->b", R_raw, R) / 3.0
    t = t_raw / s.unsqueeze(-1)

    return R, t


def rotation_matrix_to_img2pose_euler(R: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, 3, 3]`` rotation matrices to img2pose's 3-angle output.

    img2pose internally chains ``rotvec -> rotation_matrix -> quaternion ->
    euler_from_quaternion`` via ``feat.utils.image_operations.rotvec_to_euler_angles``.
    The final ``euler_from_quaternion`` returns ``(roll, pitch, yaw)`` —
    angles about the X, Y, Z axes respectively. The Fex DataFrame columns
    are then *labeled* (Pitch, Roll, Yaw, X, Y, Z) and assigned in that
    storage order, so:

      column "Pitch" = X-axis rotation (true roll, by standard convention)
      column "Roll"  = Y-axis rotation (true pitch)
      column "Yaw"   = Z-axis rotation (true yaw)

    This is a documented pre-existing labeling quirk in img2pose's
    integration. To stay numerically consistent with what users have
    been getting from ``Detector(face_model='img2pose').detect()``, this
    function returns the same 3 angles in the same storage order.
    """
    from feat.utils.geometry import (
        rotation_matrix_to_quaternion,
        euler_from_quaternion,
    )

    quaternion = rotation_matrix_to_quaternion(R)
    roll, pitch, yaw = euler_from_quaternion(
        quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    )
    # Same stack order as ``rotvec_to_euler_angles`` -> matches img2pose's
    # `dofs[:, :3]` storage order, which the FEAT_FACEPOSE_COLUMNS_6D
    # assignment then labels (Pitch, Roll, Yaw).
    return torch.stack([roll, pitch, yaw], dim=-1)


def pose_from_landmarks_2d(
    landmarks_2d: torch.Tensor,
    image_size: Tuple[int, int],
    points_3d: torch.Tensor | None = None,
    K: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate 6DoF head pose from 68 2D landmarks via DLT-PnP.

    Args:
        landmarks_2d: ``[B, 68, 2]`` landmark coordinates in image-pixel space.
        image_size: ``(H, W)`` of the source image (for default intrinsics).
        points_3d: optional ``[68, 3]`` template; defaults to img2pose's
            shipped 3D points so the result is in the same coordinate frame
            as img2pose's regressed pose.
        K: optional ``[3, 3]`` intrinsic matrix; defaults to ``default_intrinsics``.

    Returns:
        ``[B, 6]`` tensor of (pitch, roll, yaw, x, y, z) in (radians, radians,
        radians, normalized, normalized, normalized) where the translation
        units depend on the units of ``points_3d`` (img2pose's template uses
        unit-normalized coordinates).
    """
    device = landmarks_2d.device
    if points_3d is None:
        points_3d = load_img2pose_3d_template(device=device)
    elif points_3d.device != device:
        points_3d = points_3d.to(device)
    if K is None:
        K = default_intrinsics(image_size, device=device)
    elif K.device != device:
        K = K.to(device)

    R, t = solve_dlt_pnp(landmarks_2d, points_3d, K)
    angles = rotation_matrix_to_img2pose_euler(R)  # [B, 3]
    return torch.cat([angles, t], dim=-1)
