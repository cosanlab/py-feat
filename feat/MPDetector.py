"""
This is an experimental module written primarily by @ljchang porting over
Tensorflow's MediaPipe face mesh model to PyTorch for better real-time
performance. It is not currently recommended for use. See this (closed) PR
for more discussion: https://github.com/cosanlab/py-feat/pull/228
"""

import json
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feat.data import Fex, ImageDataset, TensorDataset, VideoDataset
from skops.io import load, get_untrusted_types
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from feat.pretrained import AU_LANDMARK_MAP
from torch.utils.data import DataLoader
from PIL import Image
from feat.face_detectors.Retinaface.Retinaface_test import Retinaface

# Aliases that resolve to the new ResNet34-backbone Retinaface wrapper.
# 'retinaface' is the legacy MPDetector default; 'retinaface_r34' is the
# canonical name in feat.detector.Detector. Accepting both keeps existing
# MPDetector callers working while the canonical name is the documented one.
_RETINAFACE_ALIASES = ("retinaface", "retinaface_r34")


# Per-axis sign flip applied to MediaPipe Face Mesh landmarks to translate
# them into the canonical face-model coordinate convention used downstream.
# See `convert_landmarks_3d` for the full rationale.
_MEDIAPIPE_TO_CANONICAL_AXIS_FLIP = torch.tensor(
    [1.0, -1.0, -1.0], dtype=torch.float32
)
from feat.au_detectors.MP_Blendshapes.MP_Blendshapes_test import (
    MediaPipeBlendshapesMLPMixer,
)
from feat.identity_detectors.facenet.facenet_model import InceptionResnetV1
from feat.emo_detectors.ResMaskNet.resmasknet_test import (
    ResMasking,
)
from feat.emo_detectors.StatLearning.EmoSL_test import EmoSVMClassifier
from feat.utils import (
    set_torch_device,
    FEAT_EMOTION_COLUMNS,
    FEAT_FACEBOX_COLUMNS,
    FEAT_FACEPOSE_COLUMNS_6D,
    FEAT_GAZE_COLUMNS,
    FEAT_IDENTITY_COLUMNS,
    MP_LANDMARK_COLUMNS,
    MP_BLENDSHAPE_NAMES,
    MP_BLENDSHAPE_MODEL_LANDMARKS_SUBSET,
)
from feat.utils.image_operations import (
    convert_image_to_tensor,
    extract_face_from_bbox_torch,
    inverse_transform_landmarks_torch,
    extract_hog_features,
    convert_bbox_output,
    compute_original_image_size,
    invert_padding_to_results,
)
from feat.utils.io import get_resource_path
from feat.utils.mp_plotting import FaceLandmarksConnections
from feat.utils.face_pose import (
    estimate_face_pose_from_mesh,
    rotation_matrix_to_euler_angles,
)
from feat.utils.face_gaze import estimate_gaze


def convert_landmarks_3d(fex):
    """
    Converts facial landmarks from a feature extraction object into a 3D tensor.

    MediaPipe Face Mesh outputs landmarks in image-pixel space: ``x``
    increases to the right, ``y`` increases DOWN, ``z`` is relative depth
    where positive ``z`` is INTO the screen. The canonical face model used
    downstream (``feat.utils.face_pose.load_canonical_face_model``) lives
    in head-centric coordinates with ``y`` UP and ``z`` OUT of the face
    (toward the camera). Without normalizing those, the rigid-alignment
    step would absorb the convention difference into a 180-degree rotation
    about the x-axis - which surfaced as Pitch values clustered near +/- pi
    for every forward-facing portrait in v0.7.

    Flipping ``y`` and ``z`` once at this conversion boundary keeps the
    ``estimate_face_pose_from_mesh`` API generic (it accepts any landmarks
    that share the canonical's convention) while putting the MediaPipe-
    specific axis knowledge in the MediaPipe-specific code path.

    Args:
        fex (Fex): Fex DataFrame containing 478 3D landmark coordinates

    Returns:
        landmarks (torch.Tensor): A tensor of shape [batch_size, 478, 3]
        containing the 3D coordinates (x, y, z) of 478 facial landmarks for
        each instance in the batch, expressed in the canonical face model's
        coordinate convention.
    """
    # Force float32 — the canonical face model (loaded via torch.load with
    # weights_only=True) is float32, and umeyama_alignment downstream
    # requires both operands to share dtype. Without this cast, pandas's
    # default Python-float (float64) propagates through and breaks the
    # subsequent rigid-alignment matmul.
    landmarks = torch.tensor(
        fex.landmarks.astype("float32").values
    ).reshape(fex.shape[0], 478, 3)
    # Flip Y and Z to translate from MediaPipe image-pixel convention into
    # the canonical face model's head-centric convention.
    return landmarks * _MEDIAPIPE_TO_CANONICAL_AXIS_FLIP.to(landmarks.device)


def plot_face_landmarks(
    fex,
    frame_idx,
    ax=None,
    oval_color="white",
    oval_linestyle="-",
    oval_linewidth=3,
    tesselation_color="gray",
    tesselation_linestyle="-",
    tesselation_linewidth=1,
    mouth_color="white",
    mouth_linestyle="-",
    mouth_linewidth=3,
    eye_color="navy",
    eye_linestyle="-",
    eye_linewidth=2,
    iris_color="skyblue",
    iris_linestyle="-",
    iris_linewidth=2,
):
    """Plots face landmarks on the given frame using specified styles for each part.

    Args:
        fex: DataFrame containing face landmarks (x, y coordinates).
        frame_idx: Index of the frame to plot.
        ax: Matplotlib axis to draw on. If None, a new axis is created.
        oval_color, tesselation_color, mouth_color, eye_color, iris_color: Colors for each face part.
        oval_linestyle, tesselation_linestyle, mouth_linestyle, eye_linestyle, iris_linestyle: Linestyle for each face part.
        oval_linewidth, tesselation_linewidth, mouth_linewidth, eye_linewidth, iris_linewidth: Linewidth for each face part.
        n_faces: Number of faces in the frame. If None, will be determined from fex.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Get frame data
    fex_frame = fex.query("frame == @frame_idx")
    n_faces_frame = fex_frame.shape[0]

    # Add the frame image
    ax.imshow(Image.open(fex_frame["input"].unique()[0]))

    # Helper function to draw lines for a set of connections
    def draw_connections(face_idx, connections, color, linestyle, linewidth):
        for connection in connections:
            start = connection.start
            end = connection.end
            line = plt.Line2D(
                [fex.loc[face_idx, f"x_{start}"], fex.loc[face_idx, f"x_{end}"]],
                [fex.loc[face_idx, f"y_{start}"], fex.loc[face_idx, f"y_{end}"]],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            ax.add_line(line)

    # Face tessellation
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            tesselation_color,
            tesselation_linestyle,
            tesselation_linewidth,
        )

    # Mouth
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
            mouth_color,
            mouth_linestyle,
            mouth_linewidth,
        )

    # Left iris
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            iris_color,
            iris_linestyle,
            iris_linewidth,
        )

    # Left eye
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
            eye_color,
            eye_linestyle,
            eye_linewidth,
        )

    # Left eyebrow
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
            eye_color,
            eye_linestyle,
            eye_linewidth,
        )

    # Right iris
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            iris_color,
            iris_linestyle,
            iris_linewidth,
        )

    # Right eye
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
            eye_color,
            eye_linestyle,
            eye_linewidth,
        )

    # Right eyebrow
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
            eye_color,
            eye_linestyle,
            eye_linewidth,
        )

    # Face oval
    for face in range(n_faces_frame):
        draw_connections(
            face,
            FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
            oval_color,
            oval_linestyle,
            oval_linewidth,
        )

    # Optionally turn off axis for a clean plot
    ax.axis("off")

    return ax


class MPDetector(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        face_model="retinaface",
        landmark_model="mp_facemesh_v2",
        au_model="mp_blendshapes",
        facepose_model=None,
        emotion_model=None,
        identity_model=None,
        device="cpu",
    ):
        super(MPDetector, self).__init__()

        self.info = dict(
            face_model=face_model,
            landmark_model=landmark_model,
            emotion_model=emotion_model,
            facepose_model=facepose_model,
            au_model=au_model,
            identity_model=identity_model,
        )
        self.device = set_torch_device(device)

        # Initialize Face Detector. The v0.7 RetinaFace rebuild dropped the
        # MobileNet0.25 path in favor of the ResNet34 wrapper at
        # feat.face_detectors.Retinaface.Retinaface_test.Retinaface, which
        # batches its postprocess on-device and downloads weights from the
        # py-feat/retinaface_r34 HuggingFace repo. Both 'retinaface' and
        # 'retinaface_r34' resolve to the same wrapper.
        self.info["face_model"] = face_model
        if face_model is not None:
            if face_model in _RETINAFACE_ALIASES:
                self.face_detector = Retinaface(device=self.device)
            else:
                raise ValueError(f"{face_model} is not currently supported.")
        else:
            self.face_detector = None

        # Initialize Landmark Detector
        self.info["landmark_model"] = landmark_model
        if landmark_model is not None:
            if landmark_model == "mp_facemesh_v2":
                self.face_size = 256
                # Prefer the TorchScript model: torch.jit.load doesn't run
                # arbitrary Python at load time the way `torch.load(weights_only=
                # False)` does, and it doesn't require onnx2torch importable
                # at runtime (the legacy file was a torch.fx.GraphModule produced
                # by onnx2torch, which can only be unpickled if that package is
                # importable).
                #
                # If only the legacy pickle file is available on the hub
                # (e.g., a pinned older revision), fall back to the
                # weights_only=False path - but only if onnx2torch is
                # actually installed. v0.7 dropped it from requirements,
                # so this fallback only triggers for users who explicitly
                # reinstall it for backwards-compat work.
                from feat.utils import hf_hub_download_with_fallback

                landmark_model_file = hf_hub_download_with_fallback(
                    repo_id="py-feat/mp_facemesh_v2",
                    filename="face_landmarks_detector.pt",
                    fallback_filename="face_landmarks_detector_Nx3x256x256_onnx.pth",
                    cache_dir=get_resource_path(),
                )
                # Distinguish TorchScript from legacy pickle by file extension.
                # Suffix `.pt` is TorchScript (torch.jit.save); `.pth` is the
                # legacy pickle (torch.save). Use os.path.splitext rather than
                # endswith() so a file at any path basename is handled correctly.
                import os as _os
                _ext = _os.path.splitext(landmark_model_file)[1].lower()
                _is_torchscript = _ext == ".pt"
                if _is_torchscript:
                    # torch.jit.load(map_location=...) places parameters on
                    # the requested device. The .to(self.device) below is a
                    # no-op for the TorchScript path but kept for symmetry
                    # with the legacy path.
                    self.landmark_detector = torch.jit.load(
                        landmark_model_file, map_location=self.device
                    )
                else:
                    try:
                        import onnx2torch  # noqa: F401
                    except ImportError as e:
                        raise ImportError(
                            "MPDetector got the legacy mp_facemesh_v2 pickle "
                            "file (face_landmarks_detector_Nx3x256x256_onnx.pth) "
                            "from HuggingFace, which is a torch.fx.GraphModule "
                            "produced by onnx2torch. py-feat 0.7 dropped "
                            "onnx2torch from its dependencies in favor of the "
                            "TorchScript file (face_landmarks_detector.pt). "
                            "Either upgrade your py-feat to a version whose HF "
                            "revision pins the .pt file, or `pip install "
                            "onnx2torch` to use the legacy file."
                        ) from e
                    warnings.warn(
                        "mp_facemesh_v2: loading legacy pickled FX GraphModule via "
                        "weights_only=False; this requires onnx2torch importable and "
                        "executes arbitrary code at load time. Re-run with the new "
                        "face_landmarks_detector.pt TorchScript file (uploaded to "
                        "py-feat/mp_facemesh_v2 in v0.7).",
                        stacklevel=2,
                    )
                    self.landmark_detector = torch.load(
                        landmark_model_file,
                        map_location=self.device,
                        weights_only=False,
                    )
                self.landmark_detector.eval()
                self.landmark_detector.to(self.device)
            else:
                raise ValueError(f"{landmark_model} is not currently supported.")

        else:
            self.face_size = 112
            self.landmark_detector = None

        # Initialize AU Detector
        self.info["au_model"] = au_model
        if au_model is not None:
            if self.landmark_detector is not None:
                if au_model == "mp_blendshapes":
                    self.au_detector = MediaPipeBlendshapesMLPMixer()
                    au_model_path = hf_hub_download(
                        repo_id="py-feat/mp_blendshapes",
                        filename="face_blendshapes.pth",
                        cache_dir=get_resource_path(),
                    )
                    au_checkpoint = torch.load(
                        au_model_path, map_location=device, weights_only=True
                    )
                    self.au_detector.load_state_dict(au_checkpoint)
                    self.au_detector.eval()
                    self.au_detector.to(self.device)
                else:
                    raise ValueError(f"{au_model} is not currently supported.")
            else:
                raise ValueError(
                    f"Landmark Detector is required for AU Detection with {au_model}."
                )
        else:
            self.au_detector = None

        # Initialize FacePose Detector - will compute this from facemesh - skip for now.
        self.facepose_detector = None

        # Initialize Emotion Detector
        self.info["emotion_model"] = emotion_model
        if emotion_model is not None:
            if emotion_model == "resmasknet":
                emotion_config_file = hf_hub_download(
                    repo_id="py-feat/resmasknet",
                    filename="config.json",
                    cache_dir=get_resource_path(),
                )
                with open(emotion_config_file, "r") as f:
                    emotion_config = json.load(f)

                self.emotion_detector = ResMasking(
                    "", in_channels=emotion_config["in_channels"]
                )
                self.emotion_detector.fc = nn.Sequential(
                    nn.Dropout(0.4), nn.Linear(512, emotion_config["num_classes"])
                )
                emotion_model_file = hf_hub_download(
                    repo_id="py-feat/resmasknet",
                    filename="ResMaskNet_Z_resmasking_dropout1_rot30.pth",
                    cache_dir=get_resource_path(),
                )
                emotion_checkpoint = torch.load(
                    emotion_model_file, map_location=device, weights_only=True
                )["net"]
                self.emotion_detector.load_state_dict(emotion_checkpoint)
                self.emotion_detector.eval()
                self.emotion_detector.to(self.device)
                # self.emotion_detector = torch.compile(self.emotion_detector)
            elif emotion_model == "svm":
                if self.landmark_detector is not None:
                    self.emotion_detector = EmoSVMClassifier()
                    emotion_model_path = hf_hub_download(
                        repo_id="py-feat/svm_emo",
                        filename="svm_emo_classifier.skops",
                        cache_dir=get_resource_path(),
                    )
                    emotion_unknown_types = get_untrusted_types(file=emotion_model_path)
                    loaded_emotion_model = load(
                        emotion_model_path, trusted=emotion_unknown_types
                    )
                    self.emotion_detector.load_weights(
                        scaler_full=loaded_emotion_model.scaler_full,
                        pca_model_full=loaded_emotion_model.pca_model_full,
                        classifiers=loaded_emotion_model.classifiers,
                    )
                else:
                    raise ValueError(
                        "Landmark Detector is required for Emotion Detection with {emotion_model}."
                    )

            else:
                raise ValueError(f"{emotion_model} is not currently supported.")
        else:
            self.emotion_detector = None

        # Initialize Identity Detecctor -  facenet
        self.info["identity_model"] = identity_model
        if identity_model is not None:
            if identity_model == "facenet":
                self.identity_detector = InceptionResnetV1(
                    pretrained=None,
                    classify=False,
                    num_classes=None,
                    dropout_prob=0.6,
                    device=self.device,
                )
                self.identity_detector.logits = nn.Linear(512, 8631)
                identity_model_file = hf_hub_download(
                    repo_id="py-feat/facenet",
                    filename="facenet_20180402_114759_vggface2.pth",
                    cache_dir=get_resource_path(),
                )
                self.identity_detector.load_state_dict(
                    torch.load(
                        identity_model_file, map_location=device, weights_only=True
                    )
                )
                self.identity_detector.eval()
                self.identity_detector.to(self.device)
                # self.identity_detector = torch.compile(self.identity_detector)
            else:
                raise ValueError(f"{identity_model} is not currently supported.")
        else:
            self.identity_detector = None

    @torch.inference_mode()
    def detect_faces(self, images, face_size=256, face_detection_threshold=0.5):
        """
        detect faces and poses in a batch of images using img2pose

        Args:
            img (torch.Tensor): Tensor of shape (B, C, H, W) representing the images
            face_size (int): Output size to resize face after cropping.

        Returns:
            Fex: Prediction results dataframe
        """

        frames = convert_image_to_tensor(images, img_type="float32")
        B = frames.size(0)

        # Run the face detector once on the whole batch. The Retinaface
        # wrapper handles its own mean-subtraction and on-device NMS;
        # we just pass the [B, 3, H, W] tensor in [0, 255] units it expects
        # and get back per-image lists of [x1, y1, x2, y2, score].
        is_retinaface = self.info["face_model"] in _RETINAFACE_ALIASES
        if is_retinaface:
            rf_outputs = self.face_detector(frames.to(self.device))
        else:
            rf_outputs = [None] * B

        # Gather bboxes across the batch so face cropping runs as one
        # batched grid_sample instead of one per frame. Per-frame Python
        # loops over GPU ops were measured at ~1ms/frame of pure kernel-
        # launch overhead.
        bbox_chunks = []
        score_chunks = []
        n_per_frame = []
        wants_resmasknet = self.info["emotion_model"] == "resmasknet"
        for i in range(B):
            image_dets = rf_outputs[i] if is_retinaface else None
            if image_dets:
                arr = torch.tensor(
                    image_dets, dtype=torch.float32, device=self.device
                )
                bbox_chunks.append(arr[:, :4])
                score_chunks.append(arr[:, 4])
                n_per_frame.append(arr.shape[0])
            else:
                # No detection: contribute one zero-bbox placeholder so
                # downstream forward() always has >= 1 row per frame.
                bbox_chunks.append(torch.zeros((1, 4), device=self.device))
                score_chunks.append(torch.zeros((1,), device=self.device))
                n_per_frame.append(1)

        all_bboxes = torch.cat(bbox_chunks, dim=0)
        all_scores = torch.cat(score_chunks, dim=0)
        n_per_frame_t = torch.tensor(n_per_frame, device=self.device)
        all_frame_idx = torch.repeat_interleave(
            torch.arange(B, device=self.device), n_per_frame_t
        )

        frames_dev = frames.to(self.device)
        all_extracted, all_new_bboxes = extract_face_from_bbox_torch(
            frames_dev / 255.0,
            all_bboxes,
            face_size=face_size,
            expand_bbox=1.25,
            frame_idx=all_frame_idx,
        )

        # Zero out crops for placeholder (no-detection) entries so the
        # downstream models see the same input the per-frame loop produced.
        no_det_mask = all_scores == 0
        if no_det_mask.any():
            all_extracted = all_extracted.clone()
            all_extracted[no_det_mask] = 0
            # Original code wrote zero bboxes/new_bboxes for no-detection.
            all_new_bboxes = all_new_bboxes.clone()
            all_new_bboxes[no_det_mask] = 0

        if wants_resmasknet:
            resmasknet_all, _ = extract_face_from_bbox_torch(
                frames_dev,
                all_bboxes,
                face_size=224,
                expand_bbox=1.1,
                frame_idx=all_frame_idx,
            )
            resmasknet_all = resmasknet_all / 255.0
            if no_det_mask.any():
                resmasknet_all = resmasknet_all.clone()
                resmasknet_all[no_det_mask] = 0

        # Redistribute into per-frame dicts (preserves the public return
        # signature of detect_faces). This second pass is cheap — just
        # tensor slicing — and lets forward() stay unchanged.
        batch_results = []
        cursor = 0
        for i in range(B):
            n = n_per_frame[i]
            sl = slice(cursor, cursor + n)
            frame_results = {
                "face_id": i,
                "faces": all_extracted[sl],
                "boxes": all_bboxes[sl],
                "new_boxes": all_new_bboxes[sl],
                "scores": all_scores[sl],
            }
            if wants_resmasknet:
                frame_results["resmasknet_faces"] = resmasknet_all[sl]
            batch_results.append(frame_results)
            cursor += n

        return batch_results

    @torch.inference_mode()
    def forward(self, faces_data):
        """
        Run Model Inference on detected faces.

        Args:
            faces_data (list of dict): Detected faces and associated data from `detect_faces`.

        Returns:
            Fex: Prediction results dataframe
        """

        extracted_faces = torch.cat([face["faces"] for face in faces_data], dim=0)
        new_bboxes = torch.cat([face["new_boxes"] for face in faces_data], dim=0)
        n_faces = extracted_faces.shape[0]

        # Hoist CPU->device transfers out of per-detector branches: landmark
        # and identity detectors both consume the face crops, and previously
        # each branch issued its own `.to(self.device)` (each a fresh copy
        # since the source stays on CPU). Move once, reuse. The HOG-emotion
        # path below still uses the CPU-side `extracted_faces`.
        faces_dev = extracted_faces.to(self.device)
        new_bboxes_dev = new_bboxes.to(self.device)

        if self.landmark_detector is not None:
            landmarks = self.landmark_detector.forward(faces_dev)[0]

            # Project landmarks back onto original image. # only rescale X/Y Coordinates, leave Z in original scale
            landmarks_3d = landmarks.reshape(n_faces, 478, 3)
            img_size = (
                torch.tensor((1 / self.face_size, 1 / self.face_size))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            landmarks_2d = (
                landmarks_3d[:, :, :2] * img_size
            )  # Scale X/Y Coordinates to [0,1]
            rescaled_landmarks_2d = inverse_transform_landmarks_torch(
                landmarks_2d.reshape(n_faces, 478 * 2), new_bboxes_dev
            )
            new_landmarks = torch.cat(
                (
                    rescaled_landmarks_2d.reshape(n_faces, 478, 2),
                    landmarks_3d[:, :, 2].unsqueeze(2),
                ),
                dim=2,
            )  # leave Z in original scale

        else:
            # new_landmarks = torch.full((n_faces, 136), float('nan'))
            new_landmarks = torch.full((n_faces, 1434), float("nan"))

        if self.emotion_detector is not None:
            if self.info["emotion_model"] == "resmasknet":
                resmasknet_faces = torch.cat(
                    [face["resmasknet_faces"] for face in faces_data], dim=0
                )
                emotions = self.emotion_detector.forward(resmasknet_faces.to(self.device))
                emotions = torch.softmax(emotions, 1)
            elif self.info["emotion_model"] == "svm":
                hog_features, emo_new_landmarks = extract_hog_features(
                    extracted_faces, landmarks
                )
                emotions = self.emotion_detector.detect_emo(
                    frame=hog_features, landmarks=[emo_new_landmarks]
                )
                emotions = torch.tensor(emotions)
        else:
            emotions = torch.full((n_faces, 7), float("nan"))

        if self.identity_detector is not None:
            identity_embeddings = self.identity_detector.forward(faces_dev)
        else:
            identity_embeddings = torch.full((n_faces, 512), float("nan"))

        if self.au_detector is not None:
            aus = (
                self.au_detector(
                    landmarks.reshape(n_faces, 478, 3)[
                        :, MP_BLENDSHAPE_MODEL_LANDMARKS_SUBSET, :2
                    ].to(self.device)
                )
                .squeeze(2)
                .squeeze(2)
            )
        else:
            aus = torch.full((n_faces, 52), float("nan"))

        # Create Fex Output Representation
        bboxes = torch.cat(
            [
                convert_bbox_output(
                    face_output["new_boxes"].to(self.device),
                    face_output["scores"].to(self.device),
                )
                for face_output in faces_data
            ],
            dim=0,
        )
        feat_faceboxes = pd.DataFrame(
            bboxes.cpu().detach().numpy(),
            columns=FEAT_FACEBOX_COLUMNS,
        )

        # For now, we are running PnP outside of the forward call because pytorch inference_mode doesn't allow us to backprop
        poses = torch.full((n_faces, 6), float("nan"))
        feat_poses = pd.DataFrame(
            poses.cpu().detach().numpy(), columns=FEAT_FACEPOSE_COLUMNS_6D
        )

        feat_landmarks = pd.DataFrame(
            new_landmarks.reshape(n_faces, 478 * 3).cpu().detach().numpy(),
            columns=MP_LANDMARK_COLUMNS,
        )
        feat_aus = pd.DataFrame(aus.cpu().detach().numpy(), columns=MP_BLENDSHAPE_NAMES)

        feat_emotions = pd.DataFrame(
            emotions.cpu().detach().numpy(), columns=FEAT_EMOTION_COLUMNS
        )

        feat_identities = pd.DataFrame(
            identity_embeddings.cpu().detach().numpy(), columns=FEAT_IDENTITY_COLUMNS[1:]
        )

        return Fex(
            pd.concat(
                [
                    feat_faceboxes,
                    feat_landmarks,
                    feat_poses,
                    feat_aus,
                    feat_emotions,
                    feat_identities,
                ],
                axis=1,
            ),
            au_columns=AU_LANDMARK_MAP["Feat"],
            emotion_columns=FEAT_EMOTION_COLUMNS,
            facebox_columns=FEAT_FACEBOX_COLUMNS,
            landmark_columns=MP_LANDMARK_COLUMNS,
            facepose_columns=FEAT_FACEPOSE_COLUMNS_6D,
            gaze_columns=FEAT_GAZE_COLUMNS,
            identity_columns=FEAT_IDENTITY_COLUMNS[1:],
            detector="Feat",
            face_model=self.info["face_model"],
            landmark_model=self.info["landmark_model"],
            au_model=self.info["au_model"],
            emotion_model=self.info["emotion_model"],
            facepose_model=self.info["facepose_model"],
            identity_model=self.info["identity_model"],
        )

    def detect(
        self,
        inputs,
        data_type="image",
        output_size=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        face_identity_threshold=0.8,
        face_detection_threshold=0.5,
        skip_frames=None,
        progress_bar=True,
        **kwargs,
    ):
        """
        Detects FEX from one or more image files.

        Args:
            inputs (list of str, torch.Tensor): Path to a list of paths to image files or torch.Tensor of images (B, C, H, W)
            data_type (str): type of data to be processed; Default 'image' ['image', 'tensor', 'video']
            output_size (int): image size to rescale all image preserving aspect ratio.
            batch_size (int): how many batches of images you want to run at one shot.
            num_workers (int): how many subprocesses to use for data loading.
            pin_memory (bool): If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them.
            face_identity_threshold (float): value between 0-1 to determine similarity of person using face identity embeddings; Default >= 0.8
            face_detection_threshold (float): value between 0-1 to determine if a face was detected; Default >= 0.5
            skip_frames (int or None): number of frames to skip to speed up inference (video only); Default None
            progress_bar (bool): Whether to show the tqdm progress bar. Default is True.
            **kwargs: additional detector-specific kwargs

        Returns:
            pd.DataFrame: Concatenated results for all images in the batch
        """

        if data_type.lower() == "image":
            data_loader = DataLoader(
                ImageDataset(
                    inputs,
                    output_size=output_size,
                    preserve_aspect_ratio=True,
                    padding=True,
                ),
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=False,
            )
        elif data_type.lower() == "tensor":
            data_loader = DataLoader(
                TensorDataset(inputs),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        elif data_type.lower() == "video":
            dataset = VideoDataset(
                inputs, skip_frames=skip_frames, output_size=output_size
            )
            data_loader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=False,
            )

        data_iterator = tqdm(data_loader) if progress_bar else data_loader

        batch_output = []
        frame_counter = 0
        for batch_id, batch_data in enumerate(data_iterator):
            faces_data = self.detect_faces(
                batch_data["Image"],
                face_size=self.face_size,
                face_detection_threshold=face_detection_threshold,
            )
            batch_results = self.forward(faces_data)

            # Create metadata for each frame
            file_names = []
            frame_ids = []
            for i, face in enumerate(faces_data):
                n_faces = len(face["scores"])
                if data_type.lower() == "video":
                    current_frame_id = batch_data["Frame"].detach().numpy()[i]
                else:
                    current_frame_id = frame_counter + i
                frame_ids.append(np.repeat(current_frame_id, n_faces))
                file_names.append(np.repeat(batch_data["FileName"][i], n_faces))
            batch_results["input"] = np.concatenate(file_names)
            batch_results["frame"] = np.concatenate(frame_ids)

            # Invert the face boxes and landmarks based on the padded output size.
            invert_padding_to_results(batch_results, batch_data, n_landmarks=478)

            batch_output.append(batch_results)
            # Use the actual batch size (may be smaller than `batch_size` for the
            # last batch when len(dataset) is not divisible by batch_size).
            frame_counter += batch_data["Image"].shape[0]
        batch_output = pd.concat(batch_output)
        batch_output.reset_index(drop=True, inplace=True)
        if data_type.lower() == "video":
            batch_output["approx_time"] = [
                dataset.calc_approx_frame_time(x)
                for x in batch_output["frame"].to_numpy()
            ]

        # Compute Identities
        batch_output.compute_identities(threshold=face_identity_threshold, inplace=True)

        # Pose: closed-form rigid alignment of the observed mesh to MediaPipe's
        # canonical face model. Replaces the previous Adam-loop estimator that
        # minimized z-coordinates (the wrong objective). Pure-PyTorch, no
        # gradient required.
        landmarks_3d = convert_landmarks_3d(batch_output)
        R, t = estimate_face_pose_from_mesh(landmarks_3d, return_euler_angles=False)
        euler = rotation_matrix_to_euler_angles(R)
        batch_output.loc[:, FEAT_FACEPOSE_COLUMNS_6D] = (
            torch.cat((euler, t), dim=1).cpu().numpy()
        )

        # Gaze in head-centric frame (compensated by R from above) with iris
        # landmarks (478-pt mesh).
        gaze = estimate_gaze(landmarks_3d, R=R)
        py = gaze["combined_pitch_yaw"].cpu().numpy()
        batch_output["gaze_pitch"] = py[:, 0]
        batch_output["gaze_yaw"] = py[:, 1]
        # Backward-compat: keep `gaze_angle` as the angle from head-forward.
        combined_vec = gaze["combined_vector"].cpu().numpy()
        forward = np.array([0.0, 0.0, 1.0])
        cos_angle = np.clip((combined_vec * forward).sum(axis=1), -1.0, 1.0)
        batch_output["gaze_angle"] = np.arccos(cos_angle)

        return batch_output
