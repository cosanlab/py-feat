"""Detectorv2 — RetinaFace + the v2.3 multitask model + ArcFace/FaceNet identity.

A single forward of the multitask model yields AU (24), emotion (8), valence/
arousal, gaze, head pose, and a 478-point face mesh; an identity branch
(``identity_model='arcface'`` by default, or ``'facenet'``) adds embeddings.
Outputs a native-v2 :class:`~feat.data.Fex` whose landmark block is
the dlib-68 subset derived from the 478 mesh (so Fex helpers expecting 68
points keep working), with the full 478 mesh available in ``mesh_*`` columns.

The model consumes a 256x256 RetinaFace crop (expand_bbox=1.2), exactly as its
training chips were produced; preprocessing to the 224 model input is handled by
:class:`~feat.multitask.inference.MultitaskModel`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from feat.data import Fex, ImageDataset, TensorDataset, VideoDataset
from feat.utils import set_torch_device
from feat.utils.image_operations import (
    extract_face_from_bbox_torch,
    convert_image_to_tensor,
    convert_bbox_output,
    per_face_padding_inversion_terms,
)
from feat.utils import (
    openface_2d_landmark_columns,
    FEAT_FACEBOX_COLUMNS,
    FEAT_IDENTITY_COLUMNS,
    FEAT_FACEPOSE_COLUMNS_6D,
    FEAT_GAZE_COLUMNS,
    MP_BLENDSHAPE_NAMES,
)
from feat.face_detectors.Retinaface.Retinaface_test import Retinaface
from feat.identity_detectors.arcface.arcface_model import (
    load_arcface_identity_detector,
)
from feat.identity_detectors.facenet.facenet_model import (
    load_facenet_identity_detector,
)
from feat.multitask import (
    VA_COLUMNS_V2,
    MESH_COLUMNS_V2,
)
from feat.multitask.inference import (
    MultitaskModel,
    CHIP_SIZE,
    MODEL_INPUT,
    EXPAND_BBOX,
    _DLIB68_IDX,
)

# Crop offset from the 256 chip to the 224 model input (centre crop).
_CROP_OFFSET = (CHIP_SIZE - MODEL_INPUT) // 2  # 16


class Detectorv2(nn.Module):
    """Multitask face-behavior detector (v2.3 model).

    Pipeline: RetinaFace -> 256 crop -> multitask model (AU/emotion/V-A/gaze/
    mesh/pose) + ArcFace/FaceNet identity -> Fex.
    """

    SUPPORTED_MODELS = {
        "face_model":     {"options": ["retinaface"],               "default": "retinaface"},
        "identity_model": {"options": ["arcface", "facenet", None], "default": "arcface"},
    }

    def __init__(self, device="cpu", face_detection_threshold=0.5,
                 identity_model="arcface", multitask_weights=None, amp=None,
                 compile=False):
        super().__init__()
        self.device = set_torch_device(device)
        self.face_size = CHIP_SIZE
        self.face_detection_threshold = face_detection_threshold

        self.face_detector = Retinaface(device=self.device)
        self.multitask = MultitaskModel(device=self.device,
                                        weights_path=multitask_weights,
                                        amp=amp, compile=compile)
        if identity_model in ("arcface", "arcface_r50"):
            self.identity_detector = load_arcface_identity_detector(self.device)
        elif identity_model == "facenet":
            self.identity_detector = load_facenet_identity_detector(self.device)
        elif identity_model is None:
            self.identity_detector = None
        else:
            raise ValueError(
                f"{identity_model!r} is not a supported identity_model for "
                "Detectorv2; expected 'arcface' (default), 'facenet', or None."
            )

        self._idx68 = _DLIB68_IDX.to(self.device)
        self.info = dict(
            face_model="retinaface",
            multitask_model="face_multitask_v2",
            identity_model=identity_model,
            facepose_model="multitask",
            gaze_model="multitask",
        )

    def __repr__(self):
        return (f"Detectorv2(face=retinaface, multitask=face_multitask_v2, "
                f"identity={self.info['identity_model']}, device={self.device})")

    # ------------------------------------------------------------------ #
    def detect_faces(self, images, face_detection_threshold=0.5):
        """RetinaFace detection + 256 crops. Returns per-frame dicts.

        Mirrors Detector.detect_faces' batched-crop strategy: one grid_sample
        call across all faces in the batch, with no-detection frames carrying a
        single NaN-bbox placeholder so forward() sees >= 1 row per frame.
        """
        # The DataLoader hands us a uint8 [B,3,H,W] tensor. Cast to float on the
        # GPU, not the CPU: transferring uint8 moves 1/4 the bytes and the float
        # cast is ~free on-device. (convert_image_to_tensor with img_type set
        # casts to float32 on CPU first — ~19ms + a 4x-larger H2D copy on the
        # multi-face bench.) Keep the helper only for the dim/format handling.
        frames = convert_image_to_tensor(images)              # uint8, no cast
        frames_px = frames.to(self.device, non_blocking=True).float()
        frames_unit = frames_px / 255.0

        rf_outputs = self.face_detector(frames_px)

        # Assemble all detections on CPU with numpy, then make exactly ONE
        # host->device transfer. The prior per-image torch.tensor(..., device=cuda)
        # + per-frame torch.full/torch.tensor allocations cost ~26ms/batch in
        # tiny-op + sync overhead; numpy assembly drops that to ~1ms.
        B = len(rf_outputs)
        bbox_np, score_np, n_per_frame = [], [], []
        for image_dets in rf_outputs:
            if image_dets:
                arr = np.asarray(image_dets, dtype=np.float32)
                arr = arr[arr[:, 4] >= face_detection_threshold]
            else:
                arr = np.empty((0, 5), dtype=np.float32)
            if arr.shape[0] == 0:
                # No-detection placeholder: one NaN bbox so forward() sees a row.
                bbox_np.append(np.full((1, 4), np.nan, dtype=np.float32))
                score_np.append(np.zeros((1,), dtype=np.float32))
                n_per_frame.append(1)
            else:
                bbox_np.append(arr[:, :4])
                score_np.append(arr[:, 4])
                n_per_frame.append(arr.shape[0])

        all_bboxes = torch.from_numpy(np.concatenate(bbox_np, axis=0)).to(self.device)
        all_scores = torch.from_numpy(np.concatenate(score_np, axis=0)).to(self.device)
        n_per_frame_t = torch.tensor(n_per_frame, device=self.device)
        all_frame_idx = torch.repeat_interleave(
            torch.arange(B, device=self.device), n_per_frame_t
        )

        # Temporal bbox stabilization (EMA) for streaming: smooth the box
        # BEFORE the crop so the chip — and therefore the mesh, AUs, gaze
        # and pose predicted from it — stop jittering on a still face.
        # Off unless `bbox_smoothing_alpha` is set by the caller (e.g. a
        # live stream); batch/offline runs leave it disabled.
        all_bboxes = self._smooth_bboxes(all_bboxes)

        bboxes_for_extract = torch.nan_to_num(all_bboxes, nan=0.0)
        all_faces, all_new_bboxes = extract_face_from_bbox_torch(
            frames_unit, bboxes_for_extract,
            face_size=self.face_size, expand_bbox=EXPAND_BBOX,
            frame_idx=all_frame_idx,
        )
        all_new_bboxes = all_new_bboxes.to(torch.float32)

        no_det = torch.isnan(all_bboxes).any(dim=1)
        if no_det.any():
            all_faces = all_faces.clone()
            all_faces[no_det] = 0
            all_new_bboxes = all_new_bboxes.clone()
            all_new_bboxes[no_det] = float("nan")

        image_size = tuple(frames_unit.shape[-2:])
        results, cursor = [], 0
        for i in range(B):
            n = n_per_frame[i]
            sl = slice(cursor, cursor + n)
            results.append({
                "face_id": i,
                "faces": all_faces[sl],
                "boxes": all_bboxes[sl],
                "new_boxes": all_new_bboxes[sl],
                "scores": all_scores[sl],
                "image_size": image_size,
            })
            cursor += n
        return results

    def crop_faces_from_boxes(self, images, boxes):
        """Crop faces at caller-supplied boxes WITHOUT running RetinaFace.

        A streaming counterpart to :meth:`detect_faces`: when the caller
        already knows where each face is (e.g. a tracker deriving a ROI
        from the previous frame's mesh), this skips the expensive
        RetinaFace pass and only does the 256-chip crop, returning the
        SAME per-frame ``faces_data`` structure ``forward`` consumes.
        ``scores`` are 1.0 placeholders (no detection confidence exists).

        Args:
            images: ``[B,C,H,W]`` tensor (or anything
                ``convert_image_to_tensor`` accepts) of source frames,
                pixel range 0-255 — same as :meth:`detect_faces` expects.
            boxes: a single ``[N,4]`` tensor (one-frame batch, ``B==1``)
                or a length-``B`` list of ``[Ni,4]`` tensors, each in
                ``[x1,y1,x2,y2]`` source-frame pixel coords.
                Must be torch tensors (numpy arrays are not accepted).

        Returns:
            list of ``B`` per-frame dicts keyed
            ``face_id/faces/boxes/new_boxes/scores/image_size`` —
            identical in shape to :meth:`detect_faces` output.
        """
        frames = convert_image_to_tensor(images)
        frames_px = frames.to(self.device, non_blocking=True).float()
        frames_unit = frames_px / 255.0
        B = frames_unit.shape[0]

        if torch.is_tensor(boxes):
            boxes = [boxes]
        if len(boxes) != B:
            raise ValueError(
                f"crop_faces_from_boxes: {len(boxes)} box-lists for {B} frames"
            )

        per_frame = []
        for b in boxes:
            b = b.to(self.device, torch.float32)
            if b.ndim == 1:
                b = b.reshape(1, 4)
            if b.shape[-1] != 4:
                raise ValueError(
                    f"crop_faces_from_boxes: each box tensor must be [N,4], got {tuple(b.shape)}"
                )
            per_frame.append(b)
        n_per_frame = [int(b.shape[0]) for b in per_frame]
        all_boxes = torch.cat(per_frame, dim=0)
        image_size = tuple(frames_unit.shape[-2:])

        if all_boxes.shape[0] == 0:
            # Defensive: no faces to crop on any frame.
            empty = torch.empty((0,), device=self.device)
            return [{
                "face_id": i, "faces": torch.empty((0, 3, self.face_size,
                                                    self.face_size), device=self.device),
                "boxes": torch.empty((0, 4), device=self.device),
                "new_boxes": torch.empty((0, 4), device=self.device),
                "scores": empty, "image_size": image_size,
            } for i in range(B)]

        n_per_frame_t = torch.tensor(n_per_frame, device=self.device)
        all_frame_idx = torch.repeat_interleave(
            torch.arange(B, device=self.device), n_per_frame_t
        )

        all_faces, all_new_bboxes = extract_face_from_bbox_torch(
            frames_unit, all_boxes,
            face_size=self.face_size, expand_bbox=EXPAND_BBOX,
            frame_idx=all_frame_idx,
        )
        all_new_bboxes = all_new_bboxes.to(torch.float32)
        all_scores = torch.ones(all_boxes.shape[0], device=self.device)

        results, cursor = [], 0
        for i in range(B):
            n = n_per_frame[i]
            sl = slice(cursor, cursor + n)
            results.append({
                "face_id": i,
                "faces": all_faces[sl],
                "boxes": all_boxes[sl],
                "new_boxes": all_new_bboxes[sl],
                "scores": all_scores[sl],
                "image_size": image_size,
            })
            cursor += n
        return results

    def _smooth_bboxes(self, all_bboxes):
        """Exponential-moving-average smoothing of the RetinaFace boxes
        across calls, to stabilize a live stream's crop (and thus the
        mesh/AUs/gaze/pose) on a still face.

        Disabled unless ``self.bbox_smoothing_alpha`` is set > 0 by the
        caller. Faces are matched frame-to-frame by box-center proximity;
        a new/unmatched face (or a no-detection NaN placeholder) passes
        through unsmoothed. Assumes a streaming batch of one image.
        ``alpha`` is the weight on the *current* frame (lower = smoother
        but laggier).
        """
        import numpy as np

        alpha = float(getattr(self, "bbox_smoothing_alpha", 0.0) or 0.0)
        if alpha <= 0.0:
            self._prev_boxes = None
            return all_bboxes

        cur = all_bboxes.detach().to("cpu", torch.float32).numpy()
        out = cur.copy()
        valid = np.isfinite(cur).all(axis=1)
        prev = getattr(self, "_prev_boxes", None)
        if prev is not None and len(prev):
            prev_c = (prev[:, :2] + prev[:, 2:]) / 2.0
            cur_c = (cur[:, :2] + cur[:, 2:]) / 2.0
            used: set = set()
            for i in range(cur.shape[0]):
                if not valid[i]:
                    continue
                dist = np.linalg.norm(prev_c - cur_c[i], axis=1)
                for j in used:
                    dist[j] = np.inf
                if dist.size == 0:
                    continue
                j = int(np.argmin(dist))
                width = float(cur[i, 2] - cur[i, 0])
                # Only fuse when the nearest previous box is plausibly the
                # same face (within half its width) — else it's a new face.
                if np.isfinite(dist[j]) and dist[j] < 0.5 * max(width, 1.0):
                    used.add(j)
                    out[i] = alpha * cur[i] + (1.0 - alpha) * prev[j]
        self._prev_boxes = out[valid].copy() if valid.any() else None
        return torch.from_numpy(out).to(all_bboxes.device)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def forward(self, faces_data, batch_data):
        """Run the multitask model + identity on detected faces; return a
        per-face DataFrame in original-frame coordinates."""
        faces = torch.cat([f["faces"] for f in faces_data], dim=0).to(self.device)
        new_bboxes = torch.cat([f["new_boxes"] for f in faces_data], dim=0).to(self.device)
        n_faces = faces.shape[0]

        n_per_frame = [f["faces"].shape[0] for f in faces_data]
        frame_idx = torch.repeat_interleave(
            torch.arange(len(faces_data), device=self.device),
            torch.tensor(n_per_frame, device=self.device),
        )
        pad_left, pad_top, scale, frame_h, frame_w = per_face_padding_inversion_terms(
            batch_data, frame_idx, self.device
        )

        out = self.multitask(faces)  # MultitaskOutput; faces already [0,1] 256 crops

        # ---- AUs, emotion (softmax), valence/arousal ----
        # Column names come from the loaded checkpoint (out.au_names /
        # out.emotion_names), not hardcoded constants, so this stays correct if
        # a different-width checkpoint is loaded via FEAT_MULTITASK_WEIGHTS.
        feat_aus = pd.DataFrame(out.au, columns=out.au_names)
        feat_emotions = pd.DataFrame(out.emotion, columns=out.emotion_names)
        feat_va = pd.DataFrame(
            np.column_stack([out.valence, out.arousal]), columns=VA_COLUMNS_V2
        )

        # ---- Identity (ArcFace/FaceNet on the same crops) ----
        if self.identity_detector is not None and n_faces > 0:
            emb = self.identity_detector.forward(faces)
            emb = emb.cpu().detach().numpy() if torch.is_tensor(emb) else np.asarray(emb)
        else:
            emb = np.full((n_faces, 512), np.nan)
        feat_identities = pd.DataFrame(emb, columns=FEAT_IDENTITY_COLUMNS[1:])

        # ---- Faceboxes -> original frame ----
        bboxes = torch.cat(
            [convert_bbox_output(f["new_boxes"].to(self.device),
                                 f["scores"].to(self.device)) for f in faces_data],
            dim=0,
        )
        bboxes[:, 0] = (bboxes[:, 0] - pad_left) / scale
        bboxes[:, 1] = (bboxes[:, 1] - pad_top) / scale
        bboxes[:, 2] = bboxes[:, 2] / scale
        bboxes[:, 3] = bboxes[:, 3] / scale
        feat_faceboxes = pd.DataFrame(
            bboxes.cpu().detach().numpy(), columns=FEAT_FACEBOX_COLUMNS
        )

        # ---- Mesh / landmarks: chip(224) -> [0,1] in 256 crop -> padded
        #      frame -> original frame. Applied to all 478 vertices; the
        #      dlib-68 block is a subset for Fex helpers. ----
        mesh = torch.as_tensor(out.mesh478, device=self.device)   # [N,478,3], 224-px
        mesh_orig = self._mesh_to_original_frame(
            mesh, new_bboxes, pad_left, pad_top, scale
        )                                                          # [N,478,3]
        lmk68 = mesh_orig[:, self._idx68, :2]                      # [N,68,2]

        feat_landmarks = pd.DataFrame(
            torch.cat([lmk68[:, :, 0], lmk68[:, :, 1]], dim=1).cpu().numpy(),
            columns=openface_2d_landmark_columns,
        )
        mesh_np = mesh_orig.cpu().numpy()                          # [N,478,3]
        feat_mesh = pd.DataFrame(
            np.concatenate([mesh_np[:, :, 0], mesh_np[:, :, 1], mesh_np[:, :, 2]], axis=1),
            columns=MESH_COLUMNS_V2,
        )

        # ---- Pose: multitask head -> canonical Fex [Pitch,Roll,Yaw,X,Y,Z].
        # Empirically the head's index 0 responds to PITCH and index 1 to YAW
        # (the inference docstring's [yaw,pitch,...] order is mislabeled),
        # verified on-camera against the classic Detector (img2pose / Pose-MLP).
        # Map to the canonical convention (+pitch=up, +yaw=turn to subject's
        # right, +roll=tilt to subject's right) with sign flips on pitch/roll:
        #   Pitch = -head[0]   Roll = -head[2]   Yaw = +head[1]
        # NOTE: the head under-predicts pitch magnitude (a fit limitation, not
        # a labeling bug) — pitch reads smaller than img2pose for the same nod.
        p = out.pose
        feat_poses = pd.DataFrame(
            np.column_stack([-p[:, 0], -p[:, 2], p[:, 1], p[:, 3], p[:, 4], p[:, 5]]),
            columns=FEAT_FACEPOSE_COLUMNS_6D,
        )

        # ---- Gaze: model [yaw,pitch] rad -> [gaze_pitch,gaze_yaw,gaze_angle] ----
        gaze_yaw, gaze_pitch = out.gaze[:, 0], out.gaze[:, 1]
        cos_angle = np.clip(np.cos(gaze_pitch) * np.cos(gaze_yaw), -1.0, 1.0)
        gaze_angle = np.arccos(cos_angle)
        feat_gaze = pd.DataFrame(
            np.column_stack([gaze_pitch, gaze_yaw, gaze_angle]), columns=FEAT_GAZE_COLUMNS
        )

        # ---- Blendshapes: 52 MediaPipe/ARKit coefficients in [0, 1] (v2.5) ----
        feat_blendshapes = pd.DataFrame(out.blendshapes, columns=MP_BLENDSHAPE_NAMES)

        feat_frame_meta = pd.DataFrame({
            "FrameHeight": frame_h.cpu().detach().numpy().astype(np.float64),
            "FrameWidth": frame_w.cpu().detach().numpy().astype(np.float64),
        })

        # No-detection rows carry a NaN placeholder bbox (see detect_faces).
        # The model + ArcFace still ran on a zeroed crop, so blank out every
        # prediction for those rows — only the (already-NaN) facebox and the
        # frame metadata stay meaningful, matching Detectorv1's behavior.
        no_det = np.isnan(new_bboxes.cpu().numpy()).any(axis=1)
        if no_det.any():
            for df in (feat_landmarks, feat_poses, feat_aus, feat_emotions,
                       feat_va, feat_gaze, feat_identities, feat_mesh,
                       feat_blendshapes):
                df.loc[no_det, :] = np.nan

        return pd.concat(
            [feat_faceboxes, feat_landmarks, feat_poses, feat_aus, feat_emotions,
             feat_va, feat_gaze, feat_identities, feat_mesh, feat_blendshapes,
             feat_frame_meta],
            axis=1,
        )

    def _mesh_to_original_frame(self, mesh, new_bboxes, pad_left, pad_top, scale):
        """[N,478,3] mesh in MODEL_INPUT(224)-pixel coords -> original-frame coords.

        The mesh head emits coords in MODEL_INPUT(224) pixel units (the model
        sees the 224 center-crop of the 256 chip). Mapping back to box fractions
        therefore needs TWO terms:
          * scale: divide by MODEL_INPUT (224) — the units the mesh is in,
          * offset: add ``_CROP_OFFSET / CHIP_SIZE`` (16/256) — the center-crop
            boundary expressed as a fraction of the full 256 chip / box.
        i.e. ``xy01 = mesh / MODEL_INPUT + _CROP_OFFSET / CHIP_SIZE``, then map
        into new_bboxes.

        Previously this was ``(mesh + _CROP_OFFSET) / CHIP_SIZE``, which
        normalised the mesh by CHIP_SIZE(256) instead of MODEL_INPUT(224) — a
        0.875x shrink (plus the two errors partly masked each other) that left
        the mesh too small and shifted, worst on small/angled faces. Validated
        against MPDetector's MediaPipe mesh: same shape (Procrustes scale 1.000,
        <1px residual) and now <0.5px centroid offset.

        NB: do the affine directly rather than via
        ``inverse_transform_landmarks_torch`` — that helper reshapes its input as
        interleaved (x0,y0,x1,y1,...) pairs, but our coords are axis-major, so
        feeding it here would scramble x/y scaling on non-square boxes.
        """
        xy01 = mesh[:, :, :2] / float(MODEL_INPUT) + _CROP_OFFSET / float(CHIP_SIZE)
        left = new_bboxes[:, 0]                                     # [N]
        top = new_bboxes[:, 1]
        w = new_bboxes[:, 2] - left
        h = new_bboxes[:, 3] - top
        # padded-frame coords, then invert Rescale to original frame.
        x = (xy01[:, :, 0] * w[:, None] + left[:, None] - pad_left[:, None]) / scale[:, None]
        y = (xy01[:, :, 1] * h[:, None] + top[:, None] - pad_top[:, None]) / scale[:, None]
        return torch.stack([x, y, mesh[:, :, 2]], dim=-1)          # [N,478,3]

    # ------------------------------------------------------------------ #
    def detect(self, inputs, data_type="image", output_size=None, batch_size=1,
               num_workers=0, pin_memory=False, face_identity_threshold=0.8,
               face_detection_threshold=None, skip_frames=None, progress_bar=True,
               **kwargs):
        """Detect faces + multitask features. Returns a native-v2 Fex."""
        thr = (face_detection_threshold if face_detection_threshold is not None
               else self.face_detection_threshold)

        if data_type.lower() == "image":
            loader = DataLoader(
                ImageDataset(inputs, output_size=output_size,
                             preserve_aspect_ratio=True, padding=True),
                num_workers=num_workers, batch_size=batch_size,
                pin_memory=pin_memory, shuffle=False)
        elif data_type.lower() == "tensor":
            loader = DataLoader(TensorDataset(inputs), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)
        elif data_type.lower() == "video":
            dataset = VideoDataset(inputs, skip_frames=skip_frames,
                                   output_size=output_size)
            loader = DataLoader(dataset, num_workers=num_workers,
                                batch_size=batch_size, pin_memory=pin_memory,
                                shuffle=False)
        else:
            raise ValueError(f"unknown data_type {data_type!r}")

        iterator = tqdm(loader) if progress_bar else loader
        batch_output, frame_counter = [], 0
        for batch_id, batch_data in enumerate(iterator):
            faces_data = self.detect_faces(batch_data["Image"],
                                           face_detection_threshold=thr)
            batch_results = self.forward(faces_data, batch_data)

            file_names, frame_ids = [], []
            for i, face in enumerate(faces_data):
                n = len(face["scores"])
                fid = (batch_data["Frame"].detach().numpy()[i]
                       if data_type.lower() == "video" else frame_counter + i)
                frame_ids.append(np.repeat(fid, n))
                file_names.append(np.repeat(batch_data["FileName"][i], n))
            batch_results["input"] = np.concatenate(file_names)
            batch_results["frame"] = np.concatenate(frame_ids)
            batch_output.append(batch_results)
            frame_counter += batch_data["Image"].shape[0]

        concat_df = pd.concat(batch_output).reset_index(drop=True)
        fex = Fex(
            concat_df,
            au_columns=self.multitask.au_names,
            emotion_columns=self.multitask.emotion_names,
            facebox_columns=FEAT_FACEBOX_COLUMNS,
            landmark_columns=openface_2d_landmark_columns,
            facepose_columns=FEAT_FACEPOSE_COLUMNS_6D,
            gaze_columns=FEAT_GAZE_COLUMNS,
            identity_columns=FEAT_IDENTITY_COLUMNS[1:],
            blendshape_columns=list(MP_BLENDSHAPE_NAMES),
            detector="Detectorv2",
            face_model=self.info["face_model"],
            identity_model=self.info["identity_model"],
            facepose_model=self.info["facepose_model"],
            gaze_model=self.info["gaze_model"],
        )
        if data_type.lower() == "video":
            fex["approx_time"] = [dataset.calc_approx_frame_time(x)
                                  for x in fex["frame"].to_numpy()]
        fex.compute_identities(threshold=face_identity_threshold, inplace=True)
        return fex
