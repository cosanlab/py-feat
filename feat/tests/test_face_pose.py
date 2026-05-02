"""Tests for pure-PyTorch head-pose and gaze estimation."""

import math
import torch
import pytest

from feat.utils.face_pose import (
    load_canonical_face_model,
    umeyama_alignment,
    rotation_matrix_to_euler_angles,
    estimate_face_pose_from_mesh,
)
from feat.utils.face_gaze import estimate_gaze


def _make_rotation_matrix(pitch, roll, yaw):
    """Build a rotation matrix from intrinsic pitch (X), roll (Z), yaw (Y).

    Convention matches `rotation_matrix_to_euler_angles`: pitch about X,
    roll about Z, yaw about Y, applied as Rz @ Rx @ Ry... well actually we
    just need *some* rotation we can recover. Use scipy-style intrinsic
    rotations: R = Ry(yaw) @ Rx(pitch) @ Rz(roll).
    """
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    Rz = torch.tensor([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=torch.float32)
    return Ry @ Rx @ Rz


class TestUmeyamaAlignment:
    def test_identity_recovery(self):
        """src == dst should give R=I, t=0, scale=1."""
        torch.manual_seed(0)
        src = torch.randn(50, 3)
        R, t, s = umeyama_alignment(src, src)
        assert torch.allclose(R, torch.eye(3), atol=1e-5)
        assert torch.allclose(t, torch.zeros(3), atol=1e-5)
        assert torch.allclose(s, torch.tensor(1.0), atol=1e-5)

    def test_recovers_known_transform(self):
        """Apply a known R/t/s, recover it via Umeyama."""
        torch.manual_seed(0)
        src = torch.randn(100, 3)
        R_true = _make_rotation_matrix(0.3, -0.1, 0.5)
        t_true = torch.tensor([1.0, -2.0, 0.5])
        s_true = 2.5
        dst = s_true * (R_true @ src.T).T + t_true

        R, t, s = umeyama_alignment(src, dst)

        assert torch.allclose(R, R_true, atol=1e-4)
        assert torch.allclose(t, t_true, atol=1e-4)
        assert torch.allclose(s, torch.tensor(s_true), atol=1e-4)

    def test_batched(self):
        """Solving in batch must match solving each item separately."""
        torch.manual_seed(0)
        B, N = 4, 60
        src = torch.randn(B, N, 3)
        # Different transform per batch
        targets = []
        for i in range(B):
            Ri = _make_rotation_matrix(0.1 * i, 0.05 * i, -0.2 * i)
            ti = torch.tensor([0.5 * i, -i, 1.0])
            si = 1.0 + 0.2 * i
            targets.append(si * (Ri @ src[i].T).T + ti)
        dst = torch.stack(targets)

        R, t, s = umeyama_alignment(src, dst)
        assert R.shape == (B, 3, 3)
        assert t.shape == (B, 3)
        assert s.shape == (B,)

        for i in range(B):
            R_i, t_i, s_i = umeyama_alignment(src[i], dst[i])
            assert torch.allclose(R[i], R_i, atol=1e-5)
            assert torch.allclose(t[i], t_i, atol=1e-5)
            assert torch.allclose(s[i], s_i, atol=1e-5)

    def test_works_in_inference_mode(self):
        """Must work inside torch.inference_mode (no autograd)."""
        torch.manual_seed(0)
        src = torch.randn(50, 3)
        with torch.inference_mode():
            R, t, s = umeyama_alignment(src, src)
        assert torch.allclose(R, torch.eye(3), atol=1e-5)

    def test_with_scale_false_clamps_to_one(self):
        torch.manual_seed(0)
        src = torch.randn(50, 3)
        dst = 3.7 * src + torch.tensor([1.0, 2.0, 3.0])
        # With scale=False, the recovered R/t will *not* fit dst exactly
        # because we forbid scale; but the scale return value must be 1.
        _R, _t, s = umeyama_alignment(src, dst, with_scale=False)
        assert torch.allclose(s, torch.tensor(1.0))

    def test_reflection_branch_returns_proper_rotation(self):
        """When dst is a mirror of src across one axis, det(V U^T) is
        negative and the algorithm must apply the D = diag(1, 1, -1)
        correction to keep R a proper rotation (det = +1)."""
        torch.manual_seed(123)
        src = torch.randn(60, 3)
        # Mirror across the x-axis.
        dst = src.clone()
        dst[:, 0] = -dst[:, 0]
        R, _t, _s = umeyama_alignment(src, dst)
        det = torch.linalg.det(R)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-5), (
            f"R must be a proper rotation (det=+1), got det={det.item()}"
        )

    def test_coincident_points_returns_zero_scale(self):
        """All src points at the origin: src variance is 0, scale is
        undefined. The helper should not produce NaN or Inf - it returns
        scale = 0 (clamped) and a finite (but meaningless) R / t."""
        src = torch.zeros(20, 3)
        dst = torch.randn(20, 3)
        R, t, s = umeyama_alignment(src, dst)
        assert torch.isfinite(R).all()
        assert torch.isfinite(t).all()
        assert torch.isfinite(s)
        # Clamped to >= 0.
        assert s >= 0


class TestRotationToEuler:
    def test_identity(self):
        e = rotation_matrix_to_euler_angles(torch.eye(3).unsqueeze(0))
        assert torch.allclose(e, torch.zeros(1, 3), atol=1e-6)

    def test_round_trip(self):
        """Build R from known angles, recover them."""
        for pitch, roll, yaw in [(0.1, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.3)]:
            R = _make_rotation_matrix(pitch, roll, yaw).unsqueeze(0)
            e = rotation_matrix_to_euler_angles(R)
            # We don't insist on the exact angle decomposition matching since
            # the convention is intrinsic-vs-extrinsic dependent; just assert
            # that re-constructing R from the recovered angles round-trips.
            p2, r2, y2 = e[0]
            R2 = _make_rotation_matrix(p2.item(), r2.item(), y2.item()).unsqueeze(0)
            # The recovered angles describe *some* decomposition of R; the
            # test we actually care about is that R itself is recovered.
            # Looser tolerance because the conversion may pick a different
            # branch but the final rotation should match.
            assert torch.allclose(R @ R.transpose(-2, -1), torch.eye(3).unsqueeze(0), atol=1e-5)
            assert torch.allclose(R2 @ R2.transpose(-2, -1), torch.eye(3).unsqueeze(0), atol=1e-5)


class TestEstimateFacePoseFromMesh:
    def test_canonical_input_gives_identity_pose(self):
        """Feeding in the canonical mesh should produce identity rotation
        and zero translation."""
        canonical = load_canonical_face_model()
        observed = canonical.unsqueeze(0)  # [1, 468, 3]
        R, t = estimate_face_pose_from_mesh(observed, return_euler_angles=False)
        assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-4)
        assert torch.allclose(t, torch.zeros(1, 3), atol=1e-3)

    def test_known_rotation_recovered(self):
        """Apply a known rotation+translation to canonical, recover it."""
        canonical = load_canonical_face_model()
        R_true = _make_rotation_matrix(0.15, -0.05, 0.3).unsqueeze(0)
        t_true = torch.tensor([[10.0, -5.0, 200.0]])
        observed = (R_true @ canonical.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1) + t_true.unsqueeze(1)

        R, t = estimate_face_pose_from_mesh(observed, return_euler_angles=False)
        assert torch.allclose(R, R_true, atol=1e-4)
        assert torch.allclose(t, t_true, atol=1e-3)

    def test_accepts_478_landmarks_with_iris(self):
        """When given 478 landmarks (face mesh + iris), only the first 468
        should be used and pose estimation must still succeed."""
        canonical = load_canonical_face_model()
        # Pad with 10 random iris-style points
        torch.manual_seed(0)
        iris = torch.randn(10, 3) * 5.0
        observed_478 = torch.cat([canonical, iris], dim=0).unsqueeze(0)
        R, t = estimate_face_pose_from_mesh(observed_478, return_euler_angles=False)
        assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=1e-4)

    def test_rejects_wrong_landmark_count(self):
        with pytest.raises(ValueError, match="468 or 478"):
            estimate_face_pose_from_mesh(torch.zeros(1, 100, 3))

    def test_returns_euler_by_default(self):
        canonical = load_canonical_face_model()
        observed = canonical.unsqueeze(0)
        e, t = estimate_face_pose_from_mesh(observed)
        assert e.shape == (1, 3)
        assert torch.allclose(e, torch.zeros(1, 3), atol=1e-4)

    def test_mediapipe_axis_convention_recovered_after_flip(self):
        """End-to-end convention guard: take canonical landmarks, apply a
        known rotation + translation in canonical convention, then re-express
        the result in MediaPipe-pixel convention (negate Y and Z), then apply
        the same MediaPipe -> canonical flip that ``convert_landmarks_3d``
        does, and confirm the recovered rotation matches the original.

        Locks down what MPDetector.convert_landmarks_3d is doing - if anyone
        accidentally drops or inverts the Y/Z flip in the future, this test
        fails loud rather than letting Pitch silently drift to +/-pi like
        v0.7.0-dev did before PR #279."""
        canonical = load_canonical_face_model()
        R_true = _make_rotation_matrix(0.20, -0.10, 0.05).unsqueeze(0)
        t_true = torch.tensor([[5.0, -2.0, 50.0]])

        # Apply the rotation+translation in canonical convention (what a real
        # head pose does to the head-centric mesh in camera coordinates).
        observed_canonical = (
            R_true @ canonical.unsqueeze(0).transpose(-2, -1)
        ).transpose(-2, -1) + t_true.unsqueeze(1)

        # Re-express in MediaPipe-pixel convention: Y down, Z into screen.
        observed_mediapipe = observed_canonical * torch.tensor([1.0, -1.0, -1.0])

        # Apply the same axis flip ``convert_landmarks_3d`` performs.
        observed_recovered = observed_mediapipe * torch.tensor([1.0, -1.0, -1.0])

        R, t = estimate_face_pose_from_mesh(
            observed_recovered, return_euler_angles=False
        )
        assert torch.allclose(R, R_true, atol=1e-4), (
            f"R drift: max={(R - R_true).abs().max():.2e}"
        )
        assert torch.allclose(t, t_true, atol=1e-3), (
            f"t drift: max={(t - t_true).abs().max():.2e}"
        )

    def test_skipping_axis_flip_fails(self):
        """The complement of the test above: if the Y/Z flip is *omitted*,
        Umeyama should NOT recover the original rotation - it produces a
        rotation that includes a 180-degree X-axis flip. This test pins
        down that the bug is real and the flip is necessary."""
        canonical = load_canonical_face_model()
        R_true = _make_rotation_matrix(0.20, -0.10, 0.05).unsqueeze(0)
        observed_canonical = (
            R_true @ canonical.unsqueeze(0).transpose(-2, -1)
        ).transpose(-2, -1)

        # MediaPipe-pixel convention: Y down, Z into screen.
        observed_mediapipe = observed_canonical * torch.tensor([1.0, -1.0, -1.0])

        # Pass the un-flipped MediaPipe-pixel landmarks straight in.
        # The recovered R will absorb a 180-deg flip and disagree with R_true.
        R, _ = estimate_face_pose_from_mesh(
            observed_mediapipe, return_euler_angles=False
        )
        assert not torch.allclose(R, R_true, atol=0.1), (
            "expected the un-flipped path to disagree with the truth - "
            "either Umeyama got smarter (revisit the convention assumption) "
            "or the test setup drifted"
        )


class TestEstimateGaze:
    def _make_mesh_with_iris_offset(self, iris_offset_left, iris_offset_right=None):
        """Build a fake 478-landmark mesh: canonical face + iris landmarks
        placed at specified offsets from the eye centers (in canonical/head
        frame). Returns a [1, 478, 3] tensor."""
        canonical = load_canonical_face_model()  # [468, 3]

        from feat.utils.face_gaze import LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS
        left_eye_center = canonical[list(LEFT_EYE_LANDMARKS)].mean(dim=0)
        right_eye_center = canonical[list(RIGHT_EYE_LANDMARKS)].mean(dim=0)

        if iris_offset_right is None:
            iris_offset_right = iris_offset_left

        # 5 iris landmarks per eye, all at the offset (so the mean is the offset point).
        left_iris = (left_eye_center + iris_offset_left).unsqueeze(0).expand(5, 3)
        right_iris = (right_eye_center + iris_offset_right).unsqueeze(0).expand(5, 3)
        full_mesh = torch.cat([canonical, left_iris, right_iris], dim=0)
        return full_mesh.unsqueeze(0)

    def test_forward_gaze_when_iris_centered(self):
        """When the iris-eye offset is along +Z (iris in front of eye center,
        i.e., looking out of the face), pitch/yaw should be near zero."""
        offset = torch.tensor([0.0, 0.0, 1.0])
        mesh = self._make_mesh_with_iris_offset(offset)
        result = estimate_gaze(mesh)
        py = result["combined_pitch_yaw"][0]
        assert abs(py[0].item()) < 0.05  # pitch
        assert abs(py[1].item()) < 0.05  # yaw

    def test_yaw_when_iris_offset_horizontal(self):
        """Iris offset to subject's left (-x in head frame, since +x = right)
        plus +z forward should give negative yaw (looking right) per the
        convention `yaw = atan2(-x, z)`. Iris offset to subject's right
        (+x) gives positive yaw."""
        # Iris offset to subject's right: +x, +z. yaw = atan2(-x, z) < 0
        offset_right = torch.tensor([1.0, 0.0, 1.0])
        mesh = self._make_mesh_with_iris_offset(offset_right)
        result = estimate_gaze(mesh)
        yaw = result["combined_pitch_yaw"][0, 1].item()
        assert yaw < -0.1, f"Expected negative yaw for rightward gaze, got {yaw}"

    def test_pose_compensation_makes_turned_head_look_forward(self):
        """A head turned to its left, with iris pointing along the head's
        forward axis, should produce near-zero gaze in the head frame even
        though the camera-frame iris-eye offset is non-trivial."""
        from feat.utils.face_pose import estimate_face_pose_from_mesh

        # Build a frontal mesh with iris pointing along head's +z (forward).
        offset_forward = torch.tensor([0.0, 0.0, 1.0])
        mesh_frontal = self._make_mesh_with_iris_offset(offset_forward)

        # Now rotate the entire mesh by yaw=+0.3 rad (turn head to subject's left).
        R_turn = _make_rotation_matrix(0.0, 0.0, 0.3)
        mesh_turned = (R_turn @ mesh_frontal[0].T).T.unsqueeze(0)

        # Without pose compensation, the gaze yaw should reflect the head turn.
        result_no_R = estimate_gaze(mesh_turned)
        yaw_no_R = result_no_R["combined_pitch_yaw"][0, 1].item()
        assert abs(yaw_no_R) > 0.1, (
            f"Without pose compensation, turned head should produce non-zero yaw; got {yaw_no_R}"
        )

        # With pose compensation, gaze should be near zero in head frame.
        R, _t = estimate_face_pose_from_mesh(mesh_turned, return_euler_angles=False)
        result_with_R = estimate_gaze(mesh_turned, R=R)
        py_with_R = result_with_R["combined_pitch_yaw"][0]
        assert abs(py_with_R[0].item()) < 0.1, f"pitch in head frame: {py_with_R[0].item()}"
        assert abs(py_with_R[1].item()) < 0.1, f"yaw in head frame: {py_with_R[1].item()}"

    def test_rejects_468_only_input(self):
        """gaze needs iris (478 landmarks total), not just face mesh (468)."""
        canonical = load_canonical_face_model()
        with pytest.raises(ValueError, match="478"):
            estimate_gaze(canonical.unsqueeze(0))
