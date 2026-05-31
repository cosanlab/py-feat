"""Detectorv2 — RetinaFace + the v2.3 multitask model + ArcFace identity.

A single forward of the multitask model yields AU (24), emotion (8), valence/
arousal, gaze, head pose, and a 478-point face mesh; ArcFace adds identity
embeddings. Outputs a native-v2 :class:`~feat.data.Fex` whose landmark block is
the dlib-68 subset derived from the 478 mesh (so Fex helpers expecting 68
points keep working), with the full 478 mesh available in ``mesh_*`` columns.

The model consumes a 256x256 RetinaFace crop (expand_bbox=1.2), exactly as its
training chips were produced; preprocessing to the 224 model input is handled by
:class:`~feat.multitask.inference.MultitaskModel`.
"""
from __future__ import annotations

from pathlib import Path

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
    inverse_transform_landmarks_torch,
    per_face_padding_inversion_terms,
)
from feat.utils import (
    openface_2d_landmark_columns,
    FEAT_FACEBOX_COLUMNS,
    FEAT_IDENTITY_COLUMNS,
    FEAT_FACEPOSE_COLUMNS_6D,
    FEAT_GAZE_COLUMNS,
    N_OPENFACE_LANDMARKS,
)
from feat.face_detectors.Retinaface.Retinaface_test import Retinaface
from feat.identity_detectors.arcface.arcface_model import ArcFace
from feat.multitask import (
    N_MESH,
    AU_COLUMNS_V2,
    EMOTION_COLUMNS_V2,
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
    mesh/pose) + ArcFace identity -> Fex.
    """

    def __init__(self, device="cpu", face_detection_threshold=0.5,
                 identity_model="arcface", multitask_weights=None):
        super().__init__()
        self.device = set_torch_device(device)
        self.face_size = CHIP_SIZE
        self.face_detection_threshold = face_detection_threshold

        self.face_detector = Retinaface(device=self.device)
        self.multitask = MultitaskModel(device=self.device,
                                        weights_path=multitask_weights)
        self.identity_detector = (
            ArcFace(backbone="r50") if identity_model == "arcface" else None
        )
        if self.identity_detector is not None:
            self.identity_detector.to(self.device)
            self.identity_detector.eval()

        self._idx68 = _DLIB68_IDX.to(self.device)
        self.info = dict(
            face_model="retinaface",
            multitask_model="face_multitask_v1",
            identity_model=identity_model,
            facepose_model="multitask",
            gaze_model="multitask",
        )

    def __repr__(self):
        return (f"Detectorv2(face=retinaface, multitask=face_multitask_v1, "
                f"identity={self.info['identity_model']}, device={self.device})")

    # ------------------------------------------------------------------ #
    def detect_faces(self, images, face_detection_threshold=0.5):
        """RetinaFace detection + 256 crops. Returns per-frame dicts.

        Mirrors Detector.detect_faces' batched-crop strategy: one grid_sample
        call across all faces in the batch, with no-detection frames carrying a
        single NaN-bbox placeholder so forward() sees >= 1 row per frame.
        """
        frames_unit = convert_image_to_tensor(images, img_type="float32") / 255.0
        frames_unit = frames_unit.to(self.device)
        frames_px = convert_image_to_tensor(images, img_type="float32").to(self.device)

        rf_outputs = self.face_detector(frames_px)
        per_image_dets = []
        for image_dets in rf_outputs:
            if image_dets:
                arr = torch.tensor(image_dets, dtype=torch.float32, device=self.device)
                keep = arr[:, 4] >= face_detection_threshold
                per_image_dets.append({"boxes": arr[keep, :4], "scores": arr[keep, 4]})
            else:
                per_image_dets.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "scores": torch.empty((0,), device=self.device),
                })

        B = len(per_image_dets)
        bbox_chunks, score_chunks, n_per_frame = [], [], []
        for det in per_image_dets:
            if det["boxes"].numel() != 0:
                bbox_chunks.append(det["boxes"])
                score_chunks.append(det["scores"])
                n_per_frame.append(det["boxes"].shape[0])
            else:
                bbox_chunks.append(torch.full((1, 4), float("nan"), device=self.device))
                score_chunks.append(torch.zeros((1,), device=self.device))
                n_per_frame.append(1)

        all_bboxes = torch.cat(bbox_chunks, dim=0)
        all_scores = torch.cat(score_chunks, dim=0)
        n_per_frame_t = torch.tensor(n_per_frame, device=self.device)
        all_frame_idx = torch.repeat_interleave(
            torch.arange(B, device=self.device), n_per_frame_t
        )

        bboxes_for_extract = torch.where(
            torch.isnan(all_bboxes), torch.zeros_like(all_bboxes), all_bboxes
        )
        all_faces, all_new_bboxes = extract_face_from_bbox_torch(
            frames_unit, bboxes_for_extract,
            face_size=self.face_size, expand_bbox=EXPAND_BBOX,
            frame_idx=all_frame_idx,
        )

        no_det = torch.isnan(all_bboxes).any(dim=1)
        if no_det.any():
            all_faces = all_faces.clone()
            all_faces[no_det] = 0
            all_new_bboxes = all_new_bboxes.clone().to(torch.float32)
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

        # ---- AUs (24), emotion (8 softmax), valence/arousal ----
        feat_aus = pd.DataFrame(out.au, columns=AU_COLUMNS_V2)
        feat_emotions = pd.DataFrame(out.emotion, columns=EMOTION_COLUMNS_V2)
        feat_va = pd.DataFrame(
            np.column_stack([out.valence, out.arousal]), columns=VA_COLUMNS_V2
        )

        # ---- Identity (ArcFace on the same crops) ----
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

        # ---- Pose: model [yaw,pitch,roll,tx,ty,tz] -> Fex [Pitch,Roll,Yaw,X,Y,Z] ----
        p = out.pose
        feat_poses = pd.DataFrame(
            np.column_stack([p[:, 1], p[:, 2], p[:, 0], p[:, 3], p[:, 4], p[:, 5]]),
            columns=FEAT_FACEPOSE_COLUMNS_6D,
        )

        # ---- Gaze: model [yaw,pitch] rad -> [gaze_pitch,gaze_yaw,gaze_angle] ----
        gaze_yaw, gaze_pitch = out.gaze[:, 0], out.gaze[:, 1]
        cos_angle = np.clip(np.cos(gaze_pitch) * np.cos(gaze_yaw), -1.0, 1.0)
        gaze_angle = np.arccos(cos_angle)
        feat_gaze = pd.DataFrame(
            np.column_stack([gaze_pitch, gaze_yaw, gaze_angle]), columns=FEAT_GAZE_COLUMNS
        )

        feat_frame_meta = pd.DataFrame({
            "FrameHeight": frame_h.cpu().detach().numpy().astype(np.float64),
            "FrameWidth": frame_w.cpu().detach().numpy().astype(np.float64),
        })

        return pd.concat(
            [feat_faceboxes, feat_landmarks, feat_poses, feat_aus, feat_emotions,
             feat_va, feat_gaze, feat_identities, feat_mesh, feat_frame_meta],
            axis=1,
        )

    def _mesh_to_original_frame(self, mesh, new_bboxes, pad_left, pad_top, scale):
        """[N,478,3] mesh in 224-chip coords -> original-frame coords.

        x,y go chip(224) -> +offset -> normalize by 256 -> inverse_transform via
        new_bboxes (padded frame) -> invert DataLoader Rescale. z (relative
        depth) is passed through unchanged.
        """
        N = mesh.shape[0]
        xy01 = (mesh[:, :, :2] + _CROP_OFFSET) / float(CHIP_SIZE)   # [N,478,2] in [0,1]
        flat = torch.cat([xy01[:, :, 0], xy01[:, :, 1]], dim=1)     # [N, 2*478] axis-major
        padded = inverse_transform_landmarks_torch(flat, new_bboxes)  # padded frame
        padded = padded.reshape(N, 2, N_MESH).permute(0, 2, 1)     # [N,478,2] (x,y)
        x = (padded[:, :, 0] - pad_left[:, None]) / scale[:, None]
        y = (padded[:, :, 1] - pad_top[:, None]) / scale[:, None]
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
            au_columns=AU_COLUMNS_V2,
            emotion_columns=EMOTION_COLUMNS_V2,
            facebox_columns=FEAT_FACEBOX_COLUMNS,
            landmark_columns=openface_2d_landmark_columns,
            facepose_columns=FEAT_FACEPOSE_COLUMNS_6D,
            gaze_columns=FEAT_GAZE_COLUMNS,
            identity_columns=FEAT_IDENTITY_COLUMNS[1:],
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
