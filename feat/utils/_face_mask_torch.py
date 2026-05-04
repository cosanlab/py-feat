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
    dtype = torch.float32

    lm = landmarks.to(device=device, dtype=dtype)

    left_eye = lm[:, _LEFT_EYE_IDX.to(device)].mean(dim=1)  # [N, 2]
    right_eye = lm[:, _RIGHT_EYE_IDX.to(device)].mean(dim=1)
    nose = lm[:, _NOSE_TIP_IDX]
    mouth_l = lm[:, _MOUTH_LEFT_IDX]
    mouth_r = lm[:, _MOUTH_RIGHT_IDX]

    # mat2 in legacy: [5 anchors, (x, y, 1)]
    anchors = torch.stack([left_eye, right_eye, nose, mouth_l, mouth_r], dim=1)  # [N, 5, 2]

    # Rotation matrix mat1 from eye angle.
    delta = right_eye - left_eye  # [N, 2]
    eye_dist = torch.linalg.norm(delta, dim=1).clamp(min=1e-6)  # [N]
    cos_v = delta[:, 0] / eye_dist  # [N]
    sin_v = delta[:, 1] / eye_dist  # [N]

    # mat1: rotation that brings the eye axis horizontal.
    # Legacy: [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
    zero = torch.zeros(n, device=device, dtype=dtype)
    mat1 = torch.stack(
        [
            torch.stack([cos_v, sin_v, zero], dim=1),
            torch.stack([-sin_v, cos_v, zero], dim=1),
        ],
        dim=1,
    )  # [N, 2, 3]

    # Apply mat1 to anchors.
    # anchors_h = [x, y, 1]; rotated.x = cos*x + sin*y; rotated.y = -sin*x + cos*y
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
    scale = (face_size - 1) * 0.5 / half_size.clamp(min=1e-6)

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

    # Compose: full = mat3 @ mat1 (in 3x3 form, but mat1 has [0,0,1] last row implicit)
    # mat1 acts as [[cos, sin, 0], [-sin, cos, 0]]; mat3 applies after.
    # full[:2, :2] = mat3[:2, :2] @ mat1[:2, :2]
    # full[:2, 2]  = mat3[:2, :2] @ mat1[:2, 2] + mat3[:2, 2]   (= mat3[:2, 2] since mat1 t = 0)
    M = torch.zeros(n, 2, 3, device=device, dtype=dtype)
    M[:, :2, :2] = torch.einsum("nij,njk->nik", mat3[:, :, :2], mat1[:, :, :2])
    M[:, :2, 2] = mat3[:, :, 2]

    aligned = warp_affine(
        frames.to(dtype=dtype),
        M,
        (face_size, face_size),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
        fill_value=(128, 128, 128),
    )

    # Project landmarks: [x, y, 1] -> M @ [x, y, 1]
    lm_h = torch.cat([lm, torch.ones(n, 68, 1, device=device, dtype=dtype)], dim=2)  # [N, 68, 3]
    aligned_lm = torch.einsum("nij,nkj->nki", M, lm_h)  # [N, 68, 2]
    # Legacy uses .astype(int) which is C-style truncation toward zero on
    # positive coords (which all aligned landmarks are by construction).
    aligned_lm_int = aligned_lm.to(torch.int64)

    return aligned, aligned_lm_int, M


def polygon_masks_batched(
    polygons: torch.Tensor,
    polygon_lens: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Batched point-in-polygon mask via vectorized ray casting.

    Args:
        polygons: ``[N, V_max, 2]`` polygon vertices in (x, y) pixel
            coordinates. Polygons with fewer than V_max vertices must
            be padded; ``polygon_lens`` indicates how many leading
            entries are valid.
        polygon_lens: ``[N]`` int tensor giving the number of valid
            vertices per polygon.
        height, width: output mask spatial size.

    Returns:
        ``[N, height, width]`` boolean mask. ``True`` for pixels inside
        the polygon (using even-odd fill rule).

    Implementation: for each pixel and each polygon edge, count
    horizontal-ray crossings; odd parity = inside.
    """
    if polygons.dim() != 3 or polygons.shape[-1] != 2:
        raise ValueError(f"polygons must be [N, V, 2], got {tuple(polygons.shape)}")
    n, v_max, _ = polygons.shape
    device = polygons.device
    dtype = polygons.dtype

    if not torch.is_floating_point(polygons):
        polygons = polygons.to(torch.float32)
        dtype = polygons.dtype

    # Build edges by pairing each vertex with its successor (with wrap).
    # For polygons shorter than v_max, the trailing edges connect the
    # last valid vertex to the first vertex via the padded region; we
    # zero those edges out by checking polygon_lens.
    next_idx = torch.arange(v_max, device=device)
    # For polygon i with len L_i, edge j is (j -> j+1) for j in [0, L_i - 1],
    # where j+1 wraps to 0 at j = L_i - 1.
    # Construct edges as polygons[:, j] -> polygons[:, (j+1) % L_i].
    # Vectorize via gather: idx2[i, j] = (j+1) % L_i.
    j = torch.arange(v_max, device=device).unsqueeze(0).expand(n, v_max)  # [N, V]
    L = polygon_lens.to(device).unsqueeze(1)  # [N, 1]
    next_j = torch.where(j + 1 < L, j + 1, torch.zeros_like(j))  # wrap to 0
    # Gather polygons[i, next_j[i, k]] for each (i, k)
    next_idx_expanded = next_j.unsqueeze(-1).expand(-1, -1, 2)  # [N, V, 2]
    edge_starts = polygons  # [N, V, 2]
    edge_ends = torch.gather(polygons, 1, next_idx_expanded)  # [N, V, 2]

    # Edge validity: edge j is valid iff j < polygon_lens[i].
    edge_valid = j < L  # [N, V]

    # Build the pixel grid in (x, y) pixel coords.
    ys = torch.arange(height, device=device, dtype=dtype)  # [H]
    xs = torch.arange(width, device=device, dtype=dtype)  # [W]
    py = ys.view(1, 1, height, 1)  # [1, 1, H, 1]
    px = xs.view(1, 1, 1, width)  # [1, 1, 1, W]

    y0 = edge_starts[..., 1].view(n, v_max, 1, 1)
    y1 = edge_ends[..., 1].view(n, v_max, 1, 1)
    x0 = edge_starts[..., 0].view(n, v_max, 1, 1)
    x1 = edge_ends[..., 0].view(n, v_max, 1, 1)

    # Ray-casting cross condition: edge crosses the horizontal line
    # y = py iff y0 <= py < y1 OR y1 <= py < y0. (Half-open interval
    # avoids double-counting at vertex y-values.)
    crosses_y = ((y0 <= py) & (py < y1)) | ((y1 <= py) & (py < y0))

    # x-coordinate where edge intersects the horizontal line at y=py.
    # safe-divide the y-difference; where crosses_y is False the result
    # is masked out anyway.
    dy = y1 - y0
    safe_dy = torch.where(dy == 0, torch.ones_like(dy), dy)
    x_at = x0 + (py - y0) / safe_dy * (x1 - x0)
    crosses = crosses_y & (px < x_at)

    # Mask invalid (padded) edges.
    valid = edge_valid.view(n, v_max, 1, 1)
    crossings = (crosses & valid).sum(dim=1)  # [N, H, W]
    return (crossings % 2 == 1)


def _compute_convex_hulls_cpu(
    aligned_landmarks_int: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-face scipy ConvexHull on aligned landmarks (CPU only).

    Returns padded (polygons, polygon_lens) ready for polygon_masks_batched.
    Per profile, ConvexHull on 68 points is ~0.02 ms — keeping it on CPU
    in a tight numpy loop avoids re-implementing convex hull on GPU.
    """
    lm_np = aligned_landmarks_int.detach().cpu().numpy()
    n = lm_np.shape[0]
    hulls = []
    for i in range(n):
        hull = ConvexHull(lm_np[i])
        # hull.vertices is a 1D array of indices into lm_np[i] in CCW order.
        # Legacy code uses (y, x) order in grid_points_in_poly; for ray-cast
        # in (x, y) we keep (x, y) — we pass polygons in [..., 2] = (x, y).
        hulls.append(lm_np[i][hull.vertices])  # [V_i, 2] in (x, y)

    v_max = max(h.shape[0] for h in hulls)
    polygons = np.zeros((n, v_max, 2), dtype=np.int64)
    lens = np.zeros((n,), dtype=np.int64)
    for i, h in enumerate(hulls):
        polygons[i, : h.shape[0]] = h
        lens[i] = h.shape[0]
    return torch.from_numpy(polygons), torch.from_numpy(lens)


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
    mask_strategy: str = "skimage",
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """Torch-native batched replacement for the per-face Python loop.

    Replaces the per-face loop in ``feat/utils/image_operations.py
    ::extract_face_from_landmarks``. The alignment, mask application,
    and downstream HOG run as one batched torch pipeline; the only
    work that stays per-face is the convex-hull mask construction
    itself (under ``mask_strategy='skimage'``, which preserves bit-for-bit
    parity).

    Args:
        frames: ``[N, C, H, W]`` face crops (already in face-detector
            output coordinates; this function aligns them).
        landmarks: ``[N, 68, 2]`` landmark coords in frames' pixel space.
        face_size: aligned crop output size (square).
        box_enlarge: alignment box scale factor.
        mask_strategy: how to build the convex-hull mask.
            ``"skimage"`` (default) calls the legacy
            ``skimage.morphology.convex_hull.grid_points_in_poly`` per
            face on CPU — bit-for-bit parity with the legacy loop, but
            keeps a small CPU step in the pipeline.
            ``"torch"`` uses an experimental fully on-device ray-cast
            (see ``polygon_masks_batched``); fastest but currently
            disagrees with skimage on ~7 boundary pixels per face,
            which propagates to a measurable HOG drift. Not safe for
            classifiers trained on the skimage path; gated for
            ablation use only.

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

    if mask_strategy == "skimage":
        full_mask = _build_full_masks_cpu_skimage(aligned_lm_int, face_size).to(
            aligned.device
        )
    elif mask_strategy == "torch":
        polygons, polygon_lens = _compute_convex_hulls_cpu(aligned_lm_int)
        polygons = polygons.to(aligned.device)
        polygon_lens = polygon_lens.to(aligned.device)
        mask = polygon_masks_batched(
            polygons, polygon_lens, face_size, face_size
        )  # [N, H, W]
        # Forehead extension: rectangle above min(y0, y16), cols x0..x16.
        jaw_l = aligned_lm_int[:, _JAW_LEFT_IDX]
        jaw_r = aligned_lm_int[:, _JAW_RIGHT_IDX]
        top_y = torch.minimum(jaw_l[:, 1], jaw_r[:, 1]).clamp(min=0, max=face_size)
        left_x = jaw_l[:, 0].clamp(min=0, max=face_size)
        right_x = jaw_r[:, 0].clamp(min=0, max=face_size)
        ys = torch.arange(face_size, device=aligned.device).view(1, face_size, 1)
        xs = torch.arange(face_size, device=aligned.device).view(1, 1, face_size)
        above_jaw = ys < top_y.view(n, 1, 1)
        in_cols = (xs >= left_x.view(n, 1, 1)) & (xs < right_x.view(n, 1, 1))
        forehead = above_jaw & in_cols
        full_mask = mask | forehead
    else:
        raise ValueError(
            f"mask_strategy must be 'skimage' or 'torch', got {mask_strategy!r}"
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
    mask_strategy: str = "skimage",
):
    """Drop-in replacement for ``extract_hog_features`` with batched mask prep.

    Same input contract as the legacy ``extract_hog_features``:
    ``landmarks`` are in normalized ``[0, 1]`` face-crop coords (e.g.,
    output of ``mobilefacenet.forward``). Internally we inverse-transform
    them to pixel coords and route through the batched alignment +
    masking pipeline.

    Args:
        extracted_faces: ``[N, C, face_size, face_size]`` face crops.
        landmarks: ``[N, n_landmarks*2]`` flattened normalized
            landmarks (same shape contract as legacy).
        hog_layer: optional pre-built ``HOGLayer`` (cached one from
            ``Detector._hog_layer``). If None, builds one fresh.
        mask_strategy: see ``extract_faces_from_landmarks_batched``.

    Returns:
        hog_features: numpy array of shape ``[N, n_features]``.
        au_new_landmarks: list of per-face landmark arrays in the
            face-aligned crop's pixel coordinates (matches the legacy
            ``au_new_landmarks`` contract).
    """
    from feat.utils.image_operations import (
        HOGLayer,
        inverse_transform_landmarks_torch,
    )

    n_faces = landmarks.shape[0]
    if n_faces == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    face_size = extracted_faces.shape[-1]
    bboxes = (
        torch.tensor([0, 0, face_size, face_size])
        .unsqueeze(0)
        .repeat(n_faces, 1)
    )
    landmarks_pix_flat = inverse_transform_landmarks_torch(landmarks, bboxes)
    landmarks_pix = landmarks_pix_flat.reshape(n_faces, 68, 2)

    masked, lm_list = extract_faces_from_landmarks_batched(
        extracted_faces,
        landmarks_pix,
        face_size=face_size,
        mask_strategy=mask_strategy,
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
