"""Facial landmark asymmetry scoring for the MediaPipe mesh.

Clinical facial motor assessment often relies on a left/right comparison.
This module computes per-frame L/R landmark asymmetry utilising the 
mirror-vertex map already computed internally for AU region maps in
feat.utils.region_maps.
"""
from __future__ import annotations

import numpy as np

from feat.utils.face_pose import load_canonical_face_model
from feat.utils.region_maps import _mirror_map

_LEFT_EYE_OUTER = 263
_RIGHT_EYE_OUTER = 33
_MIRROR_INDEX = None


def _get_mirror_index():
    global _MIRROR_INDEX
    if _MIRROR_INDEX is None:
        V = load_canonical_face_model().detach().cpu().numpy().astype(np.float64)
        mirror_map, _, _ = _mirror_map(V)
        _MIRROR_INDEX = np.array(
            [mirror_map[i] for i in range(len(mirror_map))], dtype=np.int64
        )
    return _MIRROR_INDEX


def landmark_asymmetry(x, y, normalize=True):
    """Per-frame facial landmark asymmetry score.

    Args:
        x, y: array-like. One frame of Fex.landmarks_x / Fex.landmarks_y
            (MediaPipe model only). Accepts 468 or 478 points; only the
            base 468-point mesh has a canonical mirror correspondence, so
            the 10 iris points (indices 468-477), if present, are ignored.
        normalize: if True, divide by interocular distance so scores are
            comparable across faces/distances.

    Returns:
        float. 0 = perfectly symmetric.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] not in (468, 478) or y.shape[0] not in (468, 478):
        raise ValueError(
            "landmark_asymmetry expects the MediaPipe face mesh (468 or "
            f"478 points); got {x.shape[0]} points. Not defined for the "
            "dlib-68 model."
        )
    x = x[:468]
    y = y[:468]

    mirror_index = _get_mirror_index()
    pts = np.stack([x, y], axis=1)
    mirrored_partner = pts[mirror_index]

    midline_x = (x[_LEFT_EYE_OUTER] + x[_RIGHT_EYE_OUTER]) / 2.0
    reflected = mirrored_partner.copy()
    reflected[:, 0] = 2 * midline_x - reflected[:, 0]

    disp = np.linalg.norm(pts - reflected, axis=1)
    score = float(disp.mean())

    if normalize:
        interocular = np.linalg.norm(pts[_LEFT_EYE_OUTER] - pts[_RIGHT_EYE_OUTER])
        if interocular > 0:
            score /= interocular

    return score