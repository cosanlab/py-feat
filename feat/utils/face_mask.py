"""Experimental torch-native face mask preparation for HOG feature extraction.

The legacy path in feat/utils/image_operations.py runs a per-face Python
loop that calls scipy.spatial.ConvexHull and skimage.draw.grid_points_in_poly.
Profile evidence (8 faces on M5 MBP):

    legacy on CPU :  10.7 ms  (HOG itself = 7.7 ms, loop = 3.0 ms)
    legacy on MPS :  23.8 ms  (HOG itself = 0.5 ms, loop+sync = 23 ms)

On MPS the loop is 98% of the time because each iteration forces a
GPU<->CPU sync. This module batches the whole alignment + masking
pipeline so the GPU pipe is filled once.

This is **gated by parity**: the masked-pixel set must match the legacy
path within ~1 px tolerance per face. The convex-hull mask is
load-bearing for AU classifier occlusion robustness (Cheong et al,
Affective Science 2023) — preserving the same masked region preserves
the property by construction.

See issue #293 for design discussion.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import ConvexHull

from feat.utils.geometry import warp_affine


# OpenFace 68-landmark indices used by align_face for the 5 anchor points.
# These are loadbearing — moving an anchor changes the alignment.
_LEFT_EYE_IDX = torch.tensor([36, 37, 38, 39, 40, 41])
_RIGHT_EYE_IDX = torch.tensor([42, 43, 44, 45, 46, 47])
_NOSE_TIP_IDX = 30
_MOUTH_LEFT_IDX = 48
_MOUTH_RIGHT_IDX = 54
_JAW_LEFT_IDX = 0
_JAW_RIGHT_IDX = 16


def align_faces_batched(
    frames: torch.Tensor,
    landmarks: torch.Tensor,
    face_size: int = 112,
    box_enlarge: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched torch port of align_face for the 68-landmark layout.

    Args:
        frames: ``[N, C, H, W]`` source face crops on any device.
        landmarks: ``[N, 68, 2]`` landmark coords in source-pixel space.
        face_size: output crop size (square).
        box_enlarge: scale factor on the anchor bounding box.

    Returns:
        aligned_imgs: ``[N, C, face_size, face_size]`` warped crops.
        aligned_landmarks_int: ``[N, 68, 2]`` integer landmark coords in
            the aligned crop's coordinate frame. Matches the legacy
            ``new_landmarks.astype(int)`` truncation.
        affine_mats: ``[N, 2, 3]`` matrices used by warp_affine. Useful
            for downstream consumers that need to project additional
            points into the aligned frame.
    """
    if frames.dim() != 4:
        raise ValueError(f"frames must be [N, C, H, W], got {tuple(frames.shape)}")
    if landmarks.dim() != 3 or landmarks.shape[-2:] != (68, 2):
        raise ValueError(
            f"landmarks must be [N, 68, 2], got {tuple(landmarks.shape)}"
        )
    n = frames.shape[0]
    device = frames.device

    # Legacy align_face does the matrix algebra in numpy float64 and only
    # casts to float32 at warp_affine time (image_operations.py:408). We
    # match that to keep the affine matrix and projected landmarks within
    # float32 ulp of the legacy path: float32 throughout would introduce
    # ~1e-4 drift per HOG feature, enough to flip xgboost leaf splits on
    # OOD inputs.
    #
    # MPS doesn't support float64 ops, so we run the matrix algebra on
    # CPU regardless of the input device. The transferred tensors are
    # tiny ([N, 5, 2] anchors and [N, 2, 3] result), and the result moves
    # back to the GPU once before warp_affine consumes it.
    dtype64 = torch.float64
    dtype32 = torch.float32

    # .cpu() first, then .to(float64). MPS errors if you ask for a
    # float64 dtype change while a tensor is still on the device.
    lm_cpu = landmarks.cpu().to(dtype=dtype64)

    left_eye = lm_cpu[:, _LEFT_EYE_IDX].mean(dim=1)  # [N, 2]
    right_eye = lm_cpu[:, _RIGHT_EYE_IDX].mean(dim=1)
    nose = lm_cpu[:, _NOSE_TIP_IDX]
    mouth_l = lm_cpu[:, _MOUTH_LEFT_IDX]
    mouth_r = lm_cpu[:, _MOUTH_RIGHT_IDX]

    # mat2 in legacy: [5 anchors, (x, y, 1)]
    anchors = torch.stack([left_eye, right_eye, nose, mouth_l, mouth_r], dim=1)  # [N, 5, 2]

    # Rotation matrix mat1 from eye angle.
    delta = right_eye - left_eye  # [N, 2]
    eye_dist = torch.linalg.norm(delta, dim=1).clamp(min=1e-12)  # [N]
    cos_v = delta[:, 0] / eye_dist  # [N]
    sin_v = delta[:, 1] / eye_dist  # [N]

    # mat1: rotation that brings the eye axis horizontal.
    # Legacy: [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
    zero = torch.zeros(n, dtype=dtype64)
    mat1 = torch.stack(
        [
            torch.stack([cos_v, sin_v, zero], dim=1),
            torch.stack([-sin_v, cos_v, zero], dim=1),
        ],
        dim=1,
    )  # [N, 2, 3]

    # Apply mat1 to anchors.
    rot_anchors = torch.einsum("nij,naj->nai", mat1[:, :, :2], anchors)  # [N, 5, 2]

    # Bounding box of rotated anchors.
    rx = rot_anchors[..., 0]  # [N, 5]
    ry = rot_anchors[..., 1]
    rx_min, rx_max = rx.min(dim=1).values, rx.max(dim=1).values
    ry_min, ry_max = ry.min(dim=1).values, ry.max(dim=1).values

    center_x = (rx_max + rx_min) * 0.5
    center_y = (ry_max + ry_min) * 0.5
    width = rx_max - rx_min
    height = ry_max - ry_min
    half_size = 0.5 * box_enlarge * torch.maximum(width, height)
    scale = (face_size - 1) * 0.5 / half_size.clamp(min=1e-12)

    # mat3 = [[scale, 0, scale*(half - cx)], [0, scale, scale*(half - cy)], [0, 0, 1]]
    tx = scale * (half_size - center_x)
    ty = scale * (half_size - center_y)
    mat3 = torch.stack(
        [
            torch.stack([scale, zero, tx], dim=1),
            torch.stack([zero, scale, ty], dim=1),
        ],
        dim=1,
    )  # [N, 2, 3]

    # Compose: full = mat3 @ mat1 (in 3x3 form, but mat1 has [0,0,1] last row implicit).
    # full[:2, :2] = mat3[:2, :2] @ mat1[:2, :2]
    # full[:2, 2]  = mat3[:2, 2]   (since mat1[:2, 2] = 0)
    M64 = torch.zeros(n, 2, 3, dtype=dtype64)
    M64[:, :2, :2] = torch.einsum("nij,njk->nik", mat3[:, :, :2], mat1[:, :, :2])
    M64[:, :2, 2] = mat3[:, :, 2]

    # Move the small [N, 2, 3] matrix to the input device as float32 for
    # warp_affine. (kornia and our local warp_affine both want float32;
    # MPS doesn't support float64 grid_sample.)
    M = M64.to(device=device, dtype=dtype32)

    aligned = warp_affine(
        frames.to(dtype=dtype32),
        M,
        (face_size, face_size),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
        fill_value=(128, 128, 128),
    )

    # Project landmarks via the float64 matrix on CPU to match legacy
    # precision, then cast to int. Legacy: `(mat @ land_3d.T).T` in
    # numpy float64, then `.astype(int)` (image_operations.py:431-434).
    lm_h = torch.cat([lm_cpu, torch.ones(n, 68, 1, dtype=dtype64)], dim=2)  # [N, 68, 3]
    aligned_lm = torch.einsum("nij,nkj->nki", M64, lm_h)  # [N, 68, 2]
    aligned_lm_int = aligned_lm.to(torch.int64).to(device)

    return aligned, aligned_lm_int, M


def _build_full_masks_cpu_skimage(
    aligned_landmarks_int: torch.Tensor,
    face_size: int,
) -> torch.Tensor:
    """Build the full convex-hull-plus-forehead mask using skimage CPU calls.

    Bit-for-bit parity with the legacy per-face loop: the same
    ``scipy.spatial.ConvexHull`` and
    ``skimage.morphology.convex_hull.grid_points_in_poly`` are called with
    the same arguments, then the same forehead box is stamped. Difference
    vs the legacy loop is only that the *batch* of CPU work is performed
    in one tight numpy loop, with no per-face GPU<->CPU sync.

    Returns: ``[N, face_size, face_size]`` boolean mask on CPU.
    """
    from skimage.morphology.convex_hull import grid_points_in_poly

    lm_np = aligned_landmarks_int.detach().cpu().numpy()
    n = lm_np.shape[0]
    mask_np = np.zeros((n, face_size, face_size), dtype=bool)
    for i in range(n):
        hull = ConvexHull(lm_np[i])
        m = grid_points_in_poly(
            shape=(face_size, face_size),
            verts=list(
                zip(
                    lm_np[i][hull.vertices][:, 1],
                    lm_np[i][hull.vertices][:, 0],
                )
            ),
        )
        m[
            0 : np.min([lm_np[i][_JAW_LEFT_IDX][1], lm_np[i][_JAW_RIGHT_IDX][1]]),
            lm_np[i][_JAW_LEFT_IDX][0] : lm_np[i][_JAW_RIGHT_IDX][0],
        ] = True
        mask_np[i] = m
    return torch.from_numpy(mask_np)


def extract_faces_from_landmarks_batched(
    frames: torch.Tensor,
    landmarks: torch.Tensor,
    face_size: int = 112,
    box_enlarge: float = 2.5,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """Batched replacement for the per-face Python loop.

    Replaces the per-face loop in ``feat/utils/image_operations.py
    ::extract_face_from_landmarks``. The alignment, mask application,
    and downstream HOG run as one batched torch pipeline; the only
    work that stays per-face is the scipy/skimage convex-hull mask
    construction, which is preserved exactly to keep the AU classifier
    feature space bit-for-bit identical.

    Args:
        frames: ``[N, C, H, W]`` face crops (already in face-detector
            output coordinates; this function aligns them).
        landmarks: ``[N, 68, 2]`` landmark coords in frames' pixel space.
        face_size: aligned crop output size (square).
        box_enlarge: alignment box scale factor.

    Returns:
        masked_aligned_faces: ``[N, C, face_size, face_size]`` aligned
            crops with the convex-hull-plus-forehead mask applied.
        aligned_landmarks_list: list of N numpy arrays of shape ``[68, 2]``
            in the aligned crop's pixel coordinates. Matches the
            ``au_new_landmarks`` output of the legacy loop.
    """
    aligned, aligned_lm_int, _ = align_faces_batched(
        frames, landmarks, face_size=face_size, box_enlarge=box_enlarge
    )
    n = aligned.shape[0]

    full_mask = _build_full_masks_cpu_skimage(aligned_lm_int, face_size).to(
        aligned.device
    )

    # Apply mask: legacy uses torch.sgn(mask) * img which is just multiplication
    # by 1.0 or 0.0. For boolean masks the same effect is mask.float().
    masked = aligned * full_mask.unsqueeze(1).to(aligned.dtype)

    # Per-face landmark list to match the legacy output contract.
    aligned_lm_np = aligned_lm_int.detach().cpu().numpy()
    aligned_landmarks_list = [aligned_lm_np[i] for i in range(n)]

    return masked, aligned_landmarks_list


def extract_hog_features_batched(
    extracted_faces: torch.Tensor,
    landmarks: torch.Tensor,
    hog_layer=None,
):
    """Drop-in replacement for ``extract_hog_features`` with batched mask prep.

    Same input contract as the legacy ``extract_hog_features``:
    ``landmarks`` are in normalized ``[0, 1]`` face-crop coords (e.g.,
    output of ``mobilefacenet.forward``) flattened to ``[N, 136]``.
    Internally we inverse-transform them to pixel coords and route
    through the batched alignment + masking pipeline.

    Args:
        extracted_faces: ``[N, C, face_size, face_size]`` face crops.
        landmarks: ``[N, 136]`` flattened normalized 68-landmark coords.
        hog_layer: optional pre-built ``HOGLayer`` (cached one from
            ``Detector._hog_layer``). If None, builds one fresh.

    Returns:
        hog_features: numpy array of shape ``[N, n_features]``.
        au_new_landmarks: list of per-face landmark arrays in the
            face-aligned crop's pixel coordinates (matches the legacy
            ``au_new_landmarks`` contract).

    Raises:
        ValueError: if ``landmarks`` does not have the 68-landmark
            shape ``[N, 136]``. The HOG path's ``align_face`` is
            hardcoded to the OpenFace 68-point layout; callers with
            MediaPipe 478-landmark output must translate first (see
            issue #294).
    """
    from feat.utils.image_operations import (
        HOGLayer,
        inverse_transform_landmarks_torch,
    )

    n_faces = landmarks.shape[0]
    if n_faces == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    if landmarks.shape[-1] != 136:
        raise ValueError(
            f"extract_hog_features_batched expects [N, 136] landmarks "
            f"(68 OpenFace-layout points x (x, y)); got {tuple(landmarks.shape)}. "
            "Callers with MediaPipe 478-landmark output need a 478->68 "
            "translator first — see issue #294."
        )

    face_size = extracted_faces.shape[-1]
    bboxes = torch.tensor(
        [[0, 0, face_size, face_size]],
        device=landmarks.device,
        dtype=landmarks.dtype,
    ).expand(n_faces, 4)
    landmarks_pix_flat = inverse_transform_landmarks_torch(landmarks, bboxes)
    landmarks_pix = landmarks_pix_flat.reshape(n_faces, 68, 2)

    masked, lm_list = extract_faces_from_landmarks_batched(
        extracted_faces,
        landmarks_pix,
        face_size=face_size,
    )

    if hog_layer is None:
        hog_layer = HOGLayer(
            orientations=8,
            pixels_per_cell=8,
            cells_per_block=2,
            block_normalization="L2-Hys",
            feature_vector=True,
            device=masked.device,
        ).to(masked.device)

    with torch.inference_mode():
        features = hog_layer(masked).cpu().numpy()
    return features, lm_list
