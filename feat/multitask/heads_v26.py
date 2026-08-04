"""v2.6 gaze head — eye-ROI features, explicit head pose, head-frame composition.

Why this exists (see notes/gaze_v26_plan.md for the measurements):

v2.5c's GazeHeadV3 sees `LN(GAP(C5)) ∥ LN(Linear(mesh_xy))` and applies four bare
linear maps. Three problems compound:

  1. C5 is stride 32 — a 7x7 map for a 224 chip — and it is GLOBAL-AVERAGE-POOLED.
     Each eye occupies well under one cell, so after pooling there is no spatial
     eye evidence left at all. GazeTech (arXiv:2308.09593) finds stride, input
     resolution and multi-region input are the determining factors for gaze.
  2. The geometry path is one linear layer over all 956 mesh coords. The 10 iris
     landmarks are 2% of that input, and the mesh is the model's OWN prediction
     from a landmark head frozen since stage 1 — whose iris accuracy degrades
     under exactly the head rotations we care about.
  3. Nothing tells the head where the head is pointing, so it must memorize the
     eye-in-head vs head-direction decomposition separately per training domain.
     MPII supervises almost pure eye-in-head (head fixed); Gaze360 supervises
     almost pure head direction. They fight over one output, which is why the
     model works within each domain and collapses on their combination
     (ETH-XGaze: corr(pred_pitch, GT pitch) = -0.096).

Each fix is independently switchable, and with all of them off this head is
numerically identical to GazeHeadV3 — see test_gaze_head_v26.py.

Note on the mesh: it is used for LOCALIZATION only (where to crop eye features),
never as a feature vector. That is a far more robust use of it — a few-pixel
landmark error barely shifts a 40 px ROI window, whereas it materially changes a
956-dim linear projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

# MediaPipe 478 eye landmarks. Corners are the stable extremes of each socket;
# the iris centre moves with gaze, so a box centred on the corners' midpoint
# keeps the whole eye in frame at any gaze angle.
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)


def rodrigues(rvec: torch.Tensor) -> torch.Tensor:
    """Batched rotation vector [B,3] -> rotation matrix [B,3,3].

    img2pose (and therefore our pose head) emits a Rodrigues vector, so this is
    what turns the predicted pose into the R_head used for gaze composition.
    """
    theta = rvec.norm(dim=-1, keepdim=True).clamp_min(1e-8)      # [B,1]
    k = rvec / theta
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zero, -kz, ky], dim=-1),
        torch.stack([kz, zero, -kx], dim=-1),
        torch.stack([-ky, kx, zero], dim=-1),
    ], dim=-2)                                                   # [B,3,3]
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).expand_as(K)
    t = theta.unsqueeze(-1)
    return I + torch.sin(t) * K + (1.0 - torch.cos(t)) * (K @ K)


def yawpitch_to_vec(gaze: torch.Tensor) -> torch.Tensor:
    """[B,2] (yaw,pitch) -> [B,3] unit vector. Matches losses_v2.gaze_angular:
    x = sin(yaw)cos(pitch), y = sin(pitch), z = -cos(yaw)cos(pitch)."""
    yaw, pitch = gaze[:, 0], gaze[:, 1]
    cp = torch.cos(pitch)
    return torch.stack([torch.sin(yaw) * cp, torch.sin(pitch),
                        -torch.cos(yaw) * cp], dim=-1)


def vec_to_yawpitch(v: torch.Tensor) -> torch.Tensor:
    """[B,3] -> [B,2] (yaw,pitch); exact inverse of yawpitch_to_vec."""
    v = F.normalize(v, dim=-1, eps=1e-8)
    pitch = torch.asin(v[:, 1].clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    yaw = torch.atan2(v[:, 0], -v[:, 2])
    return torch.stack([yaw, pitch], dim=-1)


class EyeROIEncoder(nn.Module):
    """Pool a small window around each eye from a stride-16 feature map.

    Costs no extra backbone compute: C4 is already produced in the forward pass,
    we just stop discarding it.
    """

    def __init__(self, in_ch: int, out_dim: int = 256, roi_size: int = 4,
                 box_frac: float = 0.18, image_size: int = 224,
                 feat_stride: int = 16):
        super().__init__()
        self.roi_size = roi_size
        self.box_frac = box_frac
        self.image_size = float(image_size)
        self.spatial_scale = 1.0 / feat_stride
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256), nn.GELU(),
        )
        self.proj = nn.Linear(256 * roi_size * roi_size, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def _boxes(self, mesh_px: torch.Tensor, corners) -> torch.Tensor:
        """mesh_px: [B,478,2] in chip pixels -> [B,4] xyxy box around one eye."""
        c = mesh_px[:, list(corners), :].mean(dim=1)              # [B,2]
        half = 0.5 * self.box_frac * self.image_size
        return torch.stack([c[:, 0] - half, c[:, 1] - half,
                            c[:, 0] + half, c[:, 1] + half], dim=-1)

    def forward(self, feat: torch.Tensor, mesh_px: torch.Tensor):
        B = feat.shape[0]
        idx = torch.arange(B, device=feat.device, dtype=feat.dtype).unsqueeze(-1)
        out = []
        for corners in (LEFT_EYE_CORNERS, RIGHT_EYE_CORNERS):
            boxes = torch.cat([idx, self._boxes(mesh_px, corners)], dim=-1)
            r = roi_align(feat, boxes, output_size=self.roi_size,
                          spatial_scale=self.spatial_scale, aligned=True)
            r = self.enc(r).flatten(1)
            out.append(self.norm(self.proj(r)))
        return out[0], out[1]


def make_bin_centers(n_bins: int, lo_deg: float, hi_deg: float,
                     device=None, dtype=torch.float32) -> torch.Tensor:
    """[n_bins] bin centres in RADIANS, evenly spaced over [lo_deg, hi_deg]."""
    edges = torch.linspace(lo_deg, hi_deg, n_bins + 1, device=device, dtype=dtype)
    return torch.deg2rad(0.5 * (edges[:-1] + edges[1:]))


class GazeHeadV26(nn.Module):
    """Gaze head with optional eye-ROI features, 6D head pose, head-frame output,
    and L2CS-style binned prediction.

    Keeps L2CS-Net's per-eye structure: each eye predicts the gaze and the two are
    averaged. With `use_eye_roi` each side sees its OWN eye's features, which is
    what makes the per-eye split meaningful rather than decorative (in v2.5c both
    sides read an identical input, so the four FCs were redundant by construction).

    Returns [B,2] (yaw, pitch) so every downstream consumer — losses, benches,
    py-feat's Detectorv2 — is unchanged. With `n_bins > 0` it ALSO returns the
    per-axis bin logits, and the [B,2] output is their softmax expectation, so the
    inference contract is still a plain (yaw, pitch) pair.

    n_bins is mutually exclusive with head_frame. Binning needs a per-axis angular
    output to discretize; head_frame's output is a 3-vector in a DIFFERENT (head)
    frame, so the only way to bin it is to derive eye-in-head targets by inverting
    the model's own predicted R_head — a target that moves with the pose head and
    is not something we want to debug inside a 1.5-day run. They are separately
    switchable and can be compared later.
    """

    def __init__(self, in_dim: int, eye_dim: int = 0, pose_dim: int = 0,
                 head_frame: bool = False, hidden: int = 0,
                 n_bins: int = 0, bin_lo_deg: float = -180.0,
                 bin_hi_deg: float = 180.0):
        super().__init__()
        self.eye_dim = eye_dim
        self.pose_dim = pose_dim
        self.head_frame = head_frame
        self.n_bins = int(n_bins or 0)
        if self.n_bins and head_frame:
            raise ValueError(
                "GazeHeadV26: n_bins>0 and head_frame are mutually exclusive "
                "(see the class docstring). Pick one.")
        if self.n_bins:
            self.register_buffer(
                "bin_centers", make_bin_centers(self.n_bins, bin_lo_deg, bin_hi_deg),
                persistent=False)
        # 3 for a head-frame direction vector, 2*n_bins of logits when binning,
        # else 2 for (yaw, pitch).
        self.out_dim = 3 if head_frame else (2 * self.n_bins if self.n_bins else 2)
        per_side_in = in_dim + eye_dim + pose_dim

        if hidden:
            def branch():
                return nn.Sequential(nn.Linear(per_side_in, hidden), nn.GELU(),
                                     nn.Linear(hidden, self.out_dim))
            self.left, self.right = branch(), branch()
            self.legacy = False
        elif head_frame or self.n_bins:
            self.left = nn.Linear(per_side_in, self.out_dim)
            self.right = nn.Linear(per_side_in, self.out_dim)
            self.legacy = False
        else:
            # Exactly GazeHeadV3: four independent scalar FCs, averaged per axis.
            self.left_yaw = nn.Linear(per_side_in, 1)
            self.left_pitch = nn.Linear(per_side_in, 1)
            self.right_yaw = nn.Linear(per_side_in, 1)
            self.right_pitch = nn.Linear(per_side_in, 1)
            self.legacy = True

    def forward(self, unified: torch.Tensor,
                eye_l: torch.Tensor | None = None,
                eye_r: torch.Tensor | None = None,
                pose_6d: torch.Tensor | None = None,
                R_head: torch.Tensor | None = None) -> torch.Tensor:
        def side(eye):
            parts = [unified]
            if self.eye_dim:
                if eye is None:
                    raise ValueError("GazeHeadV26 built with eye_dim>0 but no eye features")
                parts.append(eye)
            if self.pose_dim:
                if pose_6d is None:
                    raise ValueError("GazeHeadV26 built with pose_dim>0 but no pose_6d")
                parts.append(pose_6d)
            return torch.cat(parts, dim=-1) if len(parts) > 1 else unified

        xl, xr = side(eye_l), side(eye_r)

        if self.legacy:
            yaw = 0.5 * (self.left_yaw(xl) + self.right_yaw(xr))
            pitch = 0.5 * (self.left_pitch(xl) + self.right_pitch(xr))
            return torch.cat([yaw, pitch], dim=-1)

        out = 0.5 * (self.left(xl) + self.right(xr))

        if self.n_bins:
            # Average the two eyes' LOGITS (not their expectations) — the eyes are
            # two views of one gaze direction, so pooling the evidence before the
            # softmax is what makes a bimodal per-eye disagreement resolvable
            # rather than silently averaged into the gap between two modes.
            logits = out.view(-1, 2, self.n_bins)
            centers = self.bin_centers.to(logits.dtype)
            gaze = (logits.softmax(dim=-1) * centers).sum(dim=-1)   # [B,2] rad
            return gaze, logits

        if not self.head_frame:
            return out

        # Composition: the head predicts eye-in-head, and the (detached) predicted
        # head rotation carries it into camera frame. This is the piece that lets
        # eye-dominated (MPII) and head-dominated (Gaze360) supervision train the
        # SAME latent instead of pulling against each other.
        g_head = F.normalize(out, dim=-1, eps=1e-8)
        if R_head is None:
            return vec_to_yawpitch(g_head)
        return vec_to_yawpitch(torch.bmm(R_head, g_head.unsqueeze(-1)).squeeze(-1))


def pose_to_6d(pose: torch.Tensor) -> torch.Tensor:
    """6-DoF pose [B,6] (rvec|tvec) -> 6D rotation representation [B,6].

    The first two columns of R, per Zhou et al. Euler angles are deliberately not
    used: they are discontinuous at wrap, so the head would have to learn the
    discontinuity, and gaze error near +-180 yaw is exactly where we are weakest.
    """
    R = rodrigues(pose[:, :3])
    return R[:, :, :2].reshape(pose.shape[0], 6)
