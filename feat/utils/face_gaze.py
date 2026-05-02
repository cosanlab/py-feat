"""
Pure-PyTorch gaze estimation from MediaPipe Face Mesh landmarks.

The MediaPipe mesh emits 478 3D landmarks per face: 468 face vertices plus 5
iris landmarks per eye (indices 468-472 left, 473-477 right). The gaze
direction for each eye is approximated by

    gaze_eye = normalize(iris_center - eye_center)

where the centers are computed from the iris-ring landmarks and a small ring
of eye-perimeter landmarks respectively. The original implementation in
`MPDetector.estimate_gaze_direction` does this computation in *camera frame*,
which makes a turned head look like averted gaze.

This module computes gaze in the *head frame* by rotating the relevant
landmarks using the head pose recovered by `feat.utils.face_pose`. The
returned gaze vector and (pitch, yaw) angles describe gaze relative to a
neutral, camera-facing head, which is what most downstream analyses want.
"""

import torch
import torch.nn.functional as F


# Landmark indices on the MediaPipe Face Mesh.
LEFT_EYE_LANDMARKS = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
)
RIGHT_EYE_LANDMARKS = (
    263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
)
LEFT_IRIS_LANDMARKS = (468, 469, 470, 471, 472)
RIGHT_IRIS_LANDMARKS = (473, 474, 475, 476, 477)


def estimate_gaze(landmarks_3d, R=None):
    """Estimate per-eye and combined gaze from a batch of face-mesh landmarks.

    Args:
        landmarks_3d: Tensor of shape [B, 478, 3], the full MediaPipe output
            (face + iris). 468-only input is rejected because gaze needs iris.
        R: Optional [B, 3, 3] head rotation from
            `face_pose.estimate_face_pose_from_mesh(..., return_euler_angles=False)`.
            When provided, gaze is computed in head-centric frame; otherwise,
            in camera frame (a turned head will look like averted gaze).

    Returns:
        dict with keys:
            "left_vector":  [B, 3] unit gaze vector for the left eye.
            "right_vector": [B, 3] unit gaze vector for the right eye.
            "combined_vector": [B, 3] unit gaze vector (mean of L and R).
            "left_pitch_yaw":  [B, 2] (pitch, yaw) in radians for the left eye.
            "right_pitch_yaw": [B, 2] (pitch, yaw) in radians for the right eye.
            "combined_pitch_yaw": [B, 2] (pitch, yaw) for the combined gaze.
        Pitch is positive when looking up; yaw is positive when looking to the
        subject's left (i.e., the camera's right). Angles are 0 when the gaze
        is along the head's forward axis.
    """
    if landmarks_3d.dim() != 3 or landmarks_3d.shape[-1] != 3:
        raise ValueError(
            f"landmarks_3d must have shape [B, N, 3], got {tuple(landmarks_3d.shape)}"
        )
    if landmarks_3d.shape[1] != 478:
        raise ValueError(
            f"gaze requires the full 478-landmark mesh including iris (5 per eye), "
            f"got {landmarks_3d.shape[1]} landmarks"
        )

    device = landmarks_3d.device
    left_eye_idx = torch.tensor(LEFT_EYE_LANDMARKS, dtype=torch.long, device=device)
    right_eye_idx = torch.tensor(RIGHT_EYE_LANDMARKS, dtype=torch.long, device=device)
    left_iris_idx = torch.tensor(LEFT_IRIS_LANDMARKS, dtype=torch.long, device=device)
    right_iris_idx = torch.tensor(RIGHT_IRIS_LANDMARKS, dtype=torch.long, device=device)

    left_eye_center = landmarks_3d[:, left_eye_idx, :].mean(dim=1)
    right_eye_center = landmarks_3d[:, right_eye_idx, :].mean(dim=1)
    left_iris_center = landmarks_3d[:, left_iris_idx, :].mean(dim=1)
    right_iris_center = landmarks_3d[:, right_iris_idx, :].mean(dim=1)

    left_vec = F.normalize(left_iris_center - left_eye_center, dim=-1)
    right_vec = F.normalize(right_iris_center - right_eye_center, dim=-1)

    if R is not None:
        # Rotate gaze vectors from camera frame into head frame: head_v = R^T @ cam_v.
        # R is [B, 3, 3]; vectors are [B, 3].
        left_vec = torch.einsum("bij,bj->bi", R.transpose(-2, -1), left_vec)
        right_vec = torch.einsum("bij,bj->bi", R.transpose(-2, -1), right_vec)
        left_vec = F.normalize(left_vec, dim=-1)
        right_vec = F.normalize(right_vec, dim=-1)

    combined_vec = F.normalize((left_vec + right_vec) * 0.5, dim=-1)

    return {
        "left_vector": left_vec,
        "right_vector": right_vec,
        "combined_vector": combined_vec,
        "left_pitch_yaw": _vector_to_pitch_yaw(left_vec),
        "right_pitch_yaw": _vector_to_pitch_yaw(right_vec),
        "combined_pitch_yaw": _vector_to_pitch_yaw(combined_vec),
    }


def _vector_to_pitch_yaw(v):
    """Convert a unit gaze vector to (pitch, yaw) in radians.

    Convention: head-centric frame with X to subject's right, Y up, Z forward
    (out of the face). Pitch is rotation about X (positive = looking up). Yaw
    is rotation about Y (positive = looking to subject's left).

    Args:
        v: Tensor of shape [..., 3], unit-normalized along the last axis.

    Returns:
        Tensor of shape [..., 2] with columns (pitch, yaw).
    """
    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]
    pitch = torch.atan2(y, torch.sqrt(x * x + z * z))
    yaw = torch.atan2(-x, z)
    return torch.stack([pitch, yaw], dim=-1)
