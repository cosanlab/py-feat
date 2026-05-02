"""
Pure-PyTorch head-pose estimation from MediaPipe Face Mesh predictions.

The MediaPipe Face Mesh outputs ~468 3D landmarks per face, in screen-relative
coordinates (x/y in image space, z is relative depth). MediaPipe also publishes
a canonical 3D face model in head-centric coordinates with the same vertex
ordering. The head pose for a detected face is therefore the rigid similarity
transform that aligns the observed mesh to the canonical mesh.

This is solved in closed form via the Umeyama (1991) algorithm using a single
SVD - no iteration, no Adam loop, no `requires_grad` workaround, and no
camera intrinsics. Works equally well in `torch.inference_mode()`.
"""

import os
import torch

from feat.utils.io import get_resource_path


_CANONICAL_FACE_MODEL_FILENAME = "canonical_face_model.pt"


def load_canonical_face_model(device=None):
    """Load MediaPipe's 468-vertex canonical face model.

    Vertex coordinates are in head-centric space, with Y up, X to the subject's
    right, and Z out of the face. Same vertex ordering as MediaPipe Face Mesh's
    first 468 output landmarks (the 10 iris landmarks at indices 468-477 are
    not part of the canonical model and must be excluded before alignment).

    Args:
        device: Optional torch.device or string. Defaults to CPU.

    Returns:
        torch.Tensor of shape [468, 3].
    """
    path = os.path.join(get_resource_path(), _CANONICAL_FACE_MODEL_FILENAME)
    canonical = torch.load(path, map_location="cpu", weights_only=True)
    if device is not None:
        canonical = canonical.to(device)
    return canonical


def umeyama_alignment(src, dst, with_scale=True):
    """Closed-form similarity transform from `src` points to `dst` points.

    Solves the Umeyama (1991) least-squares problem: find rotation R,
    translation t, and (optional) scale s such that

        dst ≈ s * R @ src + t

    in the least-squares sense. This is equivalent to OpenCV's
    `estimateAffine3D` for rigid+scale transforms, but pure-torch and batched.

    Works inside `torch.inference_mode()` (no autograd needed) and supports a
    leading batch dimension so a whole batch of faces aligns in one call.

    Args:
        src: [..., N, 3] source points (e.g., canonical face model).
        dst: [..., N, 3] target points (e.g., observed landmarks).
        with_scale: If True, recover an isotropic scale factor; otherwise s=1.

    Returns:
        R: [..., 3, 3] rotation matrix (det = +1, no reflection).
        t: [..., 3] translation vector.
        scale: [...] scale factor (always returned; is 1.0 when with_scale=False).
    """
    if src.shape != dst.shape:
        raise ValueError(
            f"src and dst must have the same shape, got {src.shape} vs {dst.shape}"
        )
    if src.shape[-1] != 3:
        raise ValueError(f"points must have last-dim 3, got {src.shape[-1]}")

    # Centroids
    src_mean = src.mean(dim=-2, keepdim=True)  # [..., 1, 3]
    dst_mean = dst.mean(dim=-2, keepdim=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Cross-covariance: H = src_c^T @ dst_c, shape [..., 3, 3]
    H = src_c.transpose(-2, -1) @ dst_c

    # SVD of H. torch.linalg.svd returns (U, S, Vh) with H = U diag(S) Vh.
    U, S, Vh = torch.linalg.svd(H)

    # Rotation. Standard recipe: R = V * diag(1, ..., 1, sign(det(V U^T))) * U^T.
    V = Vh.transpose(-2, -1)
    det = torch.det(V @ U.transpose(-2, -1))
    sign = torch.where(
        det < 0, torch.tensor(-1.0, dtype=det.dtype, device=det.device), torch.tensor(1.0, dtype=det.dtype, device=det.device)
    )
    D = torch.eye(3, dtype=src.dtype, device=src.device).expand(*sign.shape, 3, 3).clone()
    D[..., 2, 2] = sign
    R = V @ D @ U.transpose(-2, -1)

    if with_scale:
        # Variance of src (centered)
        src_var = (src_c.pow(2).sum(dim=(-2, -1)))  # [...]
        # Trace term: sum of singular values weighted by D's sign correction
        # trace(diag(S) * D) = sum(S_i * D_ii) = S_0 + S_1 + sign * S_2
        S_signed = S.clone()
        S_signed[..., 2] = S_signed[..., 2] * sign
        trace_term = S_signed.sum(dim=-1)  # [...]
        scale = trace_term / src_var.clamp(min=1e-12)
    else:
        scale = torch.ones(R.shape[:-2], dtype=src.dtype, device=src.device)

    # Translation: dst_mean = s * R @ src_mean + t
    src_mean_v = src_mean.squeeze(-2)  # [..., 3]
    dst_mean_v = dst_mean.squeeze(-2)
    t = dst_mean_v - (scale.unsqueeze(-1) * (R @ src_mean_v.unsqueeze(-1)).squeeze(-1))

    return R, t, scale


def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to (pitch, roll, yaw) Euler angles in radians.

    Uses the convention where pitch rotates around X, roll around Z, yaw
    around Y, applied in that intrinsic order. Handles the gimbal-lock
    singularity (when the X-Z column of R is near zero).

    Args:
        R: Tensor of shape [..., 3, 3].

    Returns:
        Tensor of shape [..., 3] with columns (pitch, roll, yaw) in radians.
    """
    sy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
    singular = sy < 1e-6

    pitch = torch.where(
        singular,
        torch.atan2(-R[..., 2, 1], R[..., 1, 1]),
        torch.atan2(R[..., 2, 1], R[..., 2, 2]),
    )
    roll = torch.atan2(-R[..., 2, 0], sy)
    yaw = torch.where(
        singular,
        torch.zeros_like(pitch),
        torch.atan2(R[..., 1, 0], R[..., 0, 0]),
    )
    return torch.stack([pitch, roll, yaw], dim=-1)


def estimate_face_pose_from_mesh(observed_landmarks_3d, canonical=None, return_euler_angles=True):
    """Estimate head pose from MediaPipe Face Mesh landmarks via mesh alignment.

    Args:
        observed_landmarks_3d: Tensor of shape [B, 468, 3] (or [B, 478, 3], in
            which case the last 10 iris landmarks are dropped). Coordinates in
            image / screen-relative space as produced by MediaPipe Face Mesh.
        canonical: Optional [468, 3] canonical face model. If None, loaded
            from py-feat resources.
        return_euler_angles: If True (default) return Euler angles
            (pitch, roll, yaw) instead of the full rotation matrix.

    Returns:
        If return_euler_angles:
            (euler, t): euler shape [B, 3]; t shape [B, 3].
        Else:
            (R, t): R shape [B, 3, 3]; t shape [B, 3].
    """
    if observed_landmarks_3d.dim() != 3 or observed_landmarks_3d.shape[-1] != 3:
        raise ValueError(
            f"observed_landmarks_3d must have shape [B, N, 3], got {tuple(observed_landmarks_3d.shape)}"
        )

    n_landmarks = observed_landmarks_3d.shape[1]
    if n_landmarks == 478:
        observed = observed_landmarks_3d[:, :468, :]
    elif n_landmarks == 468:
        observed = observed_landmarks_3d
    else:
        raise ValueError(
            f"Expected 468 or 478 landmarks (face mesh + optional iris), got {n_landmarks}"
        )

    if canonical is None:
        canonical = load_canonical_face_model(device=observed.device)
    else:
        canonical = canonical.to(observed.device)

    # Broadcast canonical across the batch dimension.
    canonical_b = canonical.unsqueeze(0).expand(observed.shape[0], -1, -1)

    R, t, _scale = umeyama_alignment(canonical_b, observed, with_scale=True)

    if return_euler_angles:
        return rotation_matrix_to_euler_angles(R), t
    return R, t
