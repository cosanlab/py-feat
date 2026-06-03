"""
This is an experimental module written primarily by @ljchang porting over
Tensorflow's MediaPipe face mesh model to PyTorch for better real-time
performance. It is not currently recommended for use. See this (closed) PR
for more discussion: https://github.com/cosanlab/py-feat/pull/228
"""

import os
import json
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feat.data import Fex, ImageDataset, TensorDataset, VideoDataset
from skops.io import load, get_untrusted_types
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from feat.pretrained import AU_LANDMARK_MAP
from torch.utils.data import DataLoader
from PIL import Image
from feat.face_detectors.Retinaface.Retinaface_test import Retinaface
from feat.identity_detectors.arcface.arcface_model import ArcFace


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
    N_MEDIAPIPE_LANDMARKS,
    N_MEDIAPIPE_LANDMARKS_3D_FLAT,
    N_MEDIAPIPE_LANDMARKS_2D_FLAT,
)
from feat.utils.image_operations import (
    convert_image_to_tensor,
    extract_face_from_bbox_torch,
    inverse_transform_landmarks_torch,
    convert_bbox_output,
    per_face_padding_inversion_terms,
    HOGLayer,
)
from feat.utils.blendshape_to_au import pls_predict_batch, _load_pls_weights
from feat.utils.io import get_resource_path

# One-time guard: confirm the PLS regressor's au_columns are in
# AU_LANDMARK_MAP["Feat"] order before we relabel its output. Single-element
# list so forward() can flip it without `global`.
_pls_au_order_verified = [False]
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
    ).reshape(fex.shape[0], N_MEDIAPIPE_LANDMARKS, 3)
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
    gaze=False,
    gaze_color="yellow",
    gaze_linewidth=2,
):
    """Plots face landmarks on the given frame using specified styles for each part.

    Args:
        fex: DataFrame containing face landmarks (x, y coordinates).
        frame_idx: Index of the frame to plot.
        ax: Matplotlib axis to draw on. If None, a new axis is created.
        oval_color, tesselation_color, mouth_color, eye_color, iris_color: Colors for each face part.
        oval_linestyle, tesselation_linestyle, mouth_linestyle, eye_linestyle, iris_linestyle: Linestyle for each face part.
        oval_linewidth, tesselation_linewidth, mouth_linewidth, eye_linewidth, iris_linewidth: Linewidth for each face part.
        gaze (bool): Whether to draw L2CS gaze arrows from each face's bbox
            center. Requires ``gaze_pitch``/``gaze_yaw`` columns in ``fex``
            (populated by ``MPDetector(gaze_model='l2cs')`` or `'geometric'`).
        gaze_color, gaze_linewidth: gaze arrow style.
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

    # Face parts in draw order (matplotlib stacks later draws on top, so
    # tessellation goes first as a backdrop and the face oval last as a
    # solid outline). Iterating part-major preserves the legacy
    # draw order: all faces' tessellation, then all faces' mouths, etc.
    face_parts = (
        (FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
         tesselation_color, tesselation_linestyle, tesselation_linewidth),
        (FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
         mouth_color, mouth_linestyle, mouth_linewidth),
        (FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
         iris_color, iris_linestyle, iris_linewidth),
        (FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
         eye_color, eye_linestyle, eye_linewidth),
        (FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
         eye_color, eye_linestyle, eye_linewidth),
        (FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
         iris_color, iris_linestyle, iris_linewidth),
        (FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
         eye_color, eye_linestyle, eye_linewidth),
        (FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
         eye_color, eye_linestyle, eye_linewidth),
        (FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
         oval_color, oval_linestyle, oval_linewidth),
    )

    for connections, color, linestyle, linewidth in face_parts:
        for face in range(n_faces_frame):
            draw_connections(face, connections, color, linestyle, linewidth)

    # Draw L2CS gaze arrows from each face's bbox center (if requested).
    if gaze and "gaze_pitch" in fex_frame.columns and "gaze_yaw" in fex_frame.columns:
        from feat.plotting import draw_facegaze
        for _, row in fex_frame.iterrows():
            facebox = [row.get(c) for c in ("FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight")]
            if all(v is not None and not np.isnan(v) for v in facebox):
                draw_facegaze(
                    pitch_rad=row["gaze_pitch"],
                    yaw_rad=row["gaze_yaw"],
                    facebox=facebox,
                    ax=ax,
                    color=gaze_color,
                    linewidth=gaze_linewidth,
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
        identity_model="arcface",
        gaze_model="l2cs",
        device="cpu",
    ):
        super().__init__()

        self.info = dict(
            face_model=face_model,
            landmark_model=landmark_model,
            emotion_model=emotion_model,
            facepose_model=facepose_model,
            au_model=au_model,
            identity_model=identity_model,
            gaze_model=gaze_model,
        )
        self.device = set_torch_device(device)

        # Cache one HOGLayer per detector instance. See feat/detector.py for
        # the same pattern; mp_blendshapes doesn't use HOG but svm_au still
        # might via the same HOG-feature extraction path.
        self._hog_layer = HOGLayer(
            orientations=8,
            pixels_per_cell=8,
            cells_per_block=2,
            block_normalization="L2-Hys",
            feature_vector=True,
            device=self.device,
        ).to(self.device)

        # Initialize Face Detector. The v0.7 RetinaFace rebuild dropped the
        # MobileNet0.25 path in favor of the ResNet34 wrapper at
        # feat.face_detectors.Retinaface.Retinaface_test.Retinaface, which
        # batches its postprocess on-device and downloads weights from the
        # py-feat/retinaface_r34 HuggingFace repo (the repo retains the
        # backbone-tagged name; the kwarg is just 'retinaface').
        self.info["face_model"] = face_model
        if face_model is not None:
            if face_model == "retinaface":
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

        # Pre-built [1, 1, 2] tensor used in forward() to scale landmark
        # x/y from face-crop pixels to [0, 1]. Built once instead of
        # per-frame to skip the small allocation in the hot path.
        # Registered as a buffer so MPDetector.to(other_device) moves it
        # alongside self._hog_layer (which is a registered submodule).
        self.register_buffer(
            "_landmark_scale",
            torch.tensor(
                [[[1.0 / self.face_size, 1.0 / self.face_size]]],
                dtype=torch.float32,
                device=self.device,
            ),
        )

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
            elif identity_model in ("arcface", "arcface_r50"):
                # ArcFace ResNet50 (InsightFace buffalo_l). See
                # feat.identity_detectors.arcface.arcface_model and the
                # equivalent branch in feat.detector.Detector for rationale
                # and license notes.
                self.identity_detector = ArcFace(backbone="r50")
                arcface_path = os.environ.get("FEAT_ARCFACE_R50_PATH")
                if arcface_path is None:
                    arcface_path = hf_hub_download(
                        repo_id="py-feat/arcface_r50",
                        filename="arcface_r50.safetensors",
                        cache_dir=get_resource_path(),
                    )
                # See the equivalent branch in feat.detector.Detector for
                # the strict-vs-validation rationale.
                missing, unexpected = self.identity_detector.net.load_state_dict(
                    load_file(arcface_path), strict=False
                )
                real_missing = [k for k in missing if "num_batches_tracked" not in k]
                if real_missing or unexpected:
                    raise RuntimeError(
                        f"ArcFace weights at {arcface_path!r} are inconsistent "
                        f"with the architecture. Missing: {real_missing}; "
                        f"unexpected: {list(unexpected)}. Re-download from "
                        f"py-feat/arcface_r50 or re-run "
                        f"scripts/convert_arcface_onnx_to_safetensors.py."
                    )
                self.identity_detector.eval()
                self.identity_detector.to(self.device)
            else:
                raise ValueError(f"{identity_model} is not currently supported.")
        else:
            self.identity_detector = None

        # Initialize Gaze Detector. v0.7 switches the default from the
        # geometric iris-vector path (estimate_gaze, MediaPipe-only) to
        # L2CS-Net (Abdelrahman et al. 2022) — drops gaze MAE from ~10°
        # to ~4° on Gaze360 and fixes the >100° error on off-frontal
        # faces that the geometric path produced (see docstring of
        # feat/tests/test_mpdetector_gaze.py). The geometric path stays
        # available as gaze_model='geometric' for MPDetector users who
        # need MediaPipe-iris-only inference (e.g. fully offline / no
        # ResNet50 download).
        self.info["gaze_model"] = gaze_model
        if gaze_model is None:
            self.gaze_detector = None
        elif gaze_model == "l2cs":
            from feat.gaze_detectors.l2cs import load_l2cs_from_hf
            self.gaze_detector = load_l2cs_from_hf(device=self.device)
        elif gaze_model == "geometric":
            # No network model; estimate_gaze is called from detect()
            # using MediaPipe iris landmarks + the rotation matrix R.
            self.gaze_detector = None
        else:
            raise ValueError(
                f"gaze_model must be 'l2cs', 'geometric', or None; got {gaze_model!r}"
            )

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
        is_retinaface = self.info["face_model"] == "retinaface"
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
                # Drop detections below the score threshold (was ignored — every
                # detection was kept regardless of confidence). Mirrors Detector
                # / Detectorv2; the placeholder branch then treats a frame whose
                # detections are all sub-threshold as a no-detection frame.
                arr = arr[arr[:, 4] >= face_detection_threshold]
            else:
                arr = torch.empty((0, 5), device=self.device)
            if arr.shape[0] == 0:
                # No detection: contribute one zero-bbox placeholder so
                # downstream forward() always has >= 1 row per frame.
                bbox_chunks.append(torch.zeros((1, 4), device=self.device))
                score_chunks.append(torch.zeros((1,), device=self.device))
                n_per_frame.append(1)
            else:
                bbox_chunks.append(arr[:, :4])
                score_chunks.append(arr[:, 4])
                n_per_frame.append(arr.shape[0])

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
    def forward(self, faces_data, batch_data):
        """
        Run Model Inference on detected faces.

        Args:
            faces_data (list of dict): Detected faces and associated data from `detect_faces`.
            batch_data (dict): The DataLoader's batch dict for this call.
                Used to convert per-face bbox/landmark coordinates from
                the padded-image space the models operate in back to the
                original-frame space the caller expects, in a single
                vectorized tensor op (replacing the prior post-hoc
                `invert_padding_to_results` DataFrame mutation).

        Returns:
            pandas.DataFrame: per-face predictions in original-frame
            coordinates, including FrameHeight / FrameWidth columns.
            Wrapped into a Fex by `detect()` once at the end.
        """

        extracted_faces = torch.cat([face["faces"] for face in faces_data], dim=0)
        new_bboxes = torch.cat([face["new_boxes"] for face in faces_data], dim=0)
        n_faces = extracted_faces.shape[0]

        # Per-face mapping back to the source frame in the batch. Used
        # below to broadcast the DataLoader's per-frame Rescale
        # (Padding + Scale) parameters to per-face tensors so we can
        # convert bboxes / landmarks from padded-image space to
        # original-frame space without a post-hoc DataFrame walk.
        n_per_frame = [face["faces"].shape[0] for face in faces_data]
        frame_idx = torch.repeat_interleave(
            torch.arange(len(faces_data), device=self.device),
            torch.tensor(n_per_frame, device=self.device),
        )
        pad_left, pad_top, scale, frame_h, frame_w = per_face_padding_inversion_terms(
            batch_data, frame_idx, self.device
        )

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
            landmarks_3d = landmarks.reshape(n_faces, N_MEDIAPIPE_LANDMARKS, 3)
            landmarks_2d = (
                landmarks_3d[:, :, :2] * self._landmark_scale
            )  # Scale X/Y Coordinates to [0,1]
            rescaled_landmarks_2d = inverse_transform_landmarks_torch(
                landmarks_2d.reshape(n_faces, N_MEDIAPIPE_LANDMARKS_2D_FLAT),
                new_bboxes_dev,
            )
            new_landmarks = torch.cat(
                (
                    rescaled_landmarks_2d.reshape(n_faces, N_MEDIAPIPE_LANDMARKS, 2),
                    landmarks_3d[:, :, 2].unsqueeze(2),
                ),
                dim=2,
            )  # leave Z in original scale

        else:
            new_landmarks = torch.full(
                (n_faces, N_MEDIAPIPE_LANDMARKS_3D_FLAT), float("nan")
            )

        if self.emotion_detector is not None:
            if self.info["emotion_model"] == "resmasknet":
                resmasknet_faces = torch.cat(
                    [face["resmasknet_faces"] for face in faces_data], dim=0
                )
                emotions = self.emotion_detector.forward(resmasknet_faces.to(self.device))
                emotions = torch.softmax(emotions, 1)
            elif self.info["emotion_model"] == "svm":
                # MPDetector's landmark detector is mp_facemesh_v2, which
                # outputs 478 mediapipe landmarks. The HOG / SVM emotion path
                # needs 68 OpenFace-layout landmarks at align_face quality; the
                # DLIB68_FROM_MP478 subsample is dlib-layout but not validated
                # for HOG alignment, and MPDetector already surfaces AUs via the
                # blendshape->AU model and supports 3D mesh visualization, so we
                # don't wire SVM emotion here. Use Detector(emotion_model='svm')
                # for the HOG/SVM path, or MPDetector(emotion_model='resmasknet').
                raise NotImplementedError(
                    "MPDetector(emotion_model='svm') is not supported: the SVM "
                    "emotion classifier needs 68 OpenFace-layout landmarks at "
                    "HOG-alignment quality. Use Detector(emotion_model='svm') "
                    "instead, or MPDetector(emotion_model='resmasknet')."
                )
        else:
            emotions = torch.full((n_faces, 7), float("nan"))

        if self.identity_detector is not None:
            identity_embeddings = self.identity_detector.forward(faces_dev)
        else:
            identity_embeddings = torch.full((n_faces, 512), float("nan"))

        if self.au_detector is not None:
            aus = (
                self.au_detector(
                    landmarks.reshape(n_faces, N_MEDIAPIPE_LANDMARKS, 3)[
                        :, MP_BLENDSHAPE_MODEL_LANDMARKS_SUBSET, :2
                    ].to(self.device)
                )
                .squeeze(2)
                .squeeze(2)
            )
        else:
            aus = torch.full((n_faces, 52), float("nan"))

        # Create Fex Output Representation. Bboxes come out of
        # convert_bbox_output in PADDED-frame space (the same space the
        # face detector + landmark detector operated in). Subtract the
        # DataLoader's per-face padding and divide by its scale to land
        # in ORIGINAL-frame space — what the user expects in Fex.
        # In-place arithmetic: each axis reads + writes the same column,
        # axes are independent, and `bboxes` is freshly allocated by
        # torch.cat above and not referenced again after the DataFrame
        # is built. (Score column 4 is unchanged by Rescale inversion.)
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
        bboxes[:, 0] = (bboxes[:, 0] - pad_left) / scale
        bboxes[:, 1] = (bboxes[:, 1] - pad_top) / scale
        bboxes[:, 2] = bboxes[:, 2] / scale
        bboxes[:, 3] = bboxes[:, 3] / scale
        feat_faceboxes = pd.DataFrame(
            bboxes.cpu().detach().numpy(),
            columns=FEAT_FACEBOX_COLUMNS,
        )

        # For now, we are running PnP outside of the forward call because pytorch inference_mode doesn't allow us to backprop
        poses = torch.full((n_faces, 6), float("nan"))
        feat_poses = pd.DataFrame(
            poses.cpu().detach().numpy(), columns=FEAT_FACEPOSE_COLUMNS_6D
        )

        # Invert the DataLoader's Rescale on the 478 (x, y, z) landmark
        # triples. Z is depth in face-crop scale already (untouched by
        # the image-plane Rescale), so only X/Y need adjustment.
        # In-place is safe — `new_landmarks` is not used after the
        # DataFrame is built.
        landmarks_3d = new_landmarks.reshape(n_faces, N_MEDIAPIPE_LANDMARKS, 3)
        landmarks_3d[..., 0] = (
            landmarks_3d[..., 0] - pad_left[:, None]
        ) / scale[:, None]
        landmarks_3d[..., 1] = (
            landmarks_3d[..., 1] - pad_top[:, None]
        ) / scale[:, None]
        feat_landmarks = pd.DataFrame(
            landmarks_3d.reshape(n_faces, N_MEDIAPIPE_LANDMARKS_3D_FLAT)
            .cpu()
            .detach()
            .numpy(),
            columns=MP_LANDMARK_COLUMNS,
        )
        # MP's blendshape head outputs 52 ARKit-style coefficients. Keep these
        # under their canonical blendshape names so users can still inspect them
        # individually (e.g., for the AU-mesh viz model that consumes blendshapes).
        bs_array = aus.cpu().detach().numpy()
        feat_blendshapes = pd.DataFrame(bs_array, columns=MP_BLENDSHAPE_NAMES)

        # Predict 20 FACS AU intensities from the 52 blendshapes via the
        # Cheong-style PLS regressor (py-feat/bs_to_au on HuggingFace). This
        # gives MPDetector an AU-column output stream comparable to Detector's
        # xgb output, while retaining the blendshape columns alongside. The
        # one-time assertion guards against a future re-trained npz with a
        # shuffled au_columns order that would silently mislabel AU columns.
        if not _pls_au_order_verified[0]:
            _w = _load_pls_weights()
            if _w["au_columns"] != AU_LANDMARK_MAP["Feat"]:
                raise RuntimeError(
                    "BS→AU PLS au_columns drifted from AU_LANDMARK_MAP['Feat']. "
                    f"PLS: {_w['au_columns']}; canonical: {AU_LANDMARK_MAP['Feat']}. "
                    "Re-train or update the canonical AU list."
                )
            _pls_au_order_verified[0] = True
        feat_aus = pd.DataFrame(
            pls_predict_batch(bs_array), columns=AU_LANDMARK_MAP["Feat"],
        )

        feat_emotions = pd.DataFrame(
            emotions.cpu().detach().numpy(), columns=FEAT_EMOTION_COLUMNS
        )

        feat_identities = pd.DataFrame(
            identity_embeddings.cpu().detach().numpy(), columns=FEAT_IDENTITY_COLUMNS[1:]
        )

        # Gaze: L2CS-Net on the face crops already on device. The
        # geometric path (gaze_model='geometric') is computed later in
        # detect() from MediaPipe iris landmarks because it needs the
        # rotation matrix R, which isn't computed until after Fex
        # assembly. Both paths populate the same FEAT_GAZE_COLUMNS.
        if self.gaze_detector is not None and n_faces > 0:
            pitch_rad, yaw_rad = self.gaze_detector(faces_dev)
            cos_angle = np.clip(
                np.cos(pitch_rad) * np.cos(yaw_rad), -1.0, 1.0
            )
            gaze_angle = np.arccos(cos_angle)
            feat_gaze = pd.DataFrame(
                np.column_stack([pitch_rad, yaw_rad, gaze_angle]),
                columns=FEAT_GAZE_COLUMNS,
            )
        else:
            feat_gaze = pd.DataFrame(
                np.full((n_faces, len(FEAT_GAZE_COLUMNS)), np.nan),
                columns=FEAT_GAZE_COLUMNS,
            )

        # Frame metadata: original (pre-Rescale) frame dimensions per
        # face. Added here instead of in `invert_padding_to_results`
        # post-hoc so the entire DataFrame leaves forward() in
        # original-frame coords with no further mutation.
        feat_frame_meta = pd.DataFrame(
            {
                "FrameHeight": frame_h.cpu().detach().numpy().astype(np.float64),
                "FrameWidth": frame_w.cpu().detach().numpy().astype(np.float64),
            }
        )

        # No-detection rows carry a zero-bbox / zero-score placeholder (see
        # detect_faces). Blank the final predictions that ran on the zeroed crop
        # so they don't surface fabricated numbers. Landmarks are left finite
        # here because detect() recomputes pose/gaze from the mesh; landmarks,
        # pose and gaze are blanked together at the end of detect().
        scores = torch.cat([face["scores"] for face in faces_data], dim=0)
        no_det = (scores == 0).cpu().numpy()
        if no_det.any():
            for _df in (feat_blendshapes, feat_aus, feat_emotions,
                        feat_identities, feat_gaze):
                _df.loc[no_det, :] = np.nan

        # Return a plain pd.DataFrame; detect() wraps the concatenated
        # result in a single Fex at the end. Avoids Fex.__init__'s
        # O(n_columns) metadata loop (see Fex.__init__ in data.py) firing
        # once per batch, which dominated wall-time on long videos.
        return pd.concat(
            [
                feat_faceboxes,
                feat_landmarks,
                feat_poses,
                feat_blendshapes,
                feat_aus,
                feat_emotions,
                feat_identities,
                feat_gaze,
                feat_frame_meta,
            ],
            axis=1,
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
            batch_results = self.forward(faces_data, batch_data)

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

            # Padded->original-frame coordinate inversion now happens
            # inside forward() in tensor space; no post-hoc DataFrame walk
            # is needed here. (`invert_padding_to_results` is still
            # exported for any external caller that depended on it.)

            batch_output.append(batch_results)
            # Use the actual batch size (may be smaller than `batch_size` for the
            # last batch when len(dataset) is not divisible by batch_size).
            frame_counter += batch_data["Image"].shape[0]
        # Concatenate all batch DataFrames (cheap), then wrap in Fex once.
        # See forward() comment for why we don't wrap per batch.
        concat_df = pd.concat(batch_output).reset_index(drop=True)
        batch_output = Fex(
            concat_df,
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
            gaze_model=self.info["gaze_model"],
        )
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

        # Gaze: L2CS-Net path (the default) has already populated
        # `gaze_pitch`/`gaze_yaw`/`gaze_angle` per-batch in forward().
        # The geometric path needs the full-batch rotation matrix R
        # computed above, so we run it here when explicitly requested.
        if self.info["gaze_model"] == "geometric":
            gaze = estimate_gaze(landmarks_3d, R=R)
            py = gaze["combined_pitch_yaw"].cpu().numpy()
            batch_output["gaze_pitch"] = py[:, 0]
            batch_output["gaze_yaw"] = py[:, 1]
            # Backward-compat: keep `gaze_angle` as the angle from head-forward.
            combined_vec = gaze["combined_vector"].cpu().numpy()
            forward = np.array([0.0, 0.0, 1.0])
            cos_angle = np.clip((combined_vec * forward).sum(axis=1), -1.0, 1.0)
            batch_output["gaze_angle"] = np.arccos(cos_angle)

        # Pose / gaze / landmarks above were (re)computed from the no-detection
        # placeholder crop's mesh; blank them for those rows so empty frames
        # carry NaN throughout (the predictions were already blanked in
        # forward()). No-detection rows carry a zero FaceScore placeholder —
        # a detector-independent signal (AU columns may be absent if au_model
        # is None).
        no_det = (batch_output["FaceScore"].to_numpy() == 0)
        if no_det.any():
            blank_cols = [
                c for c in (list(FEAT_FACEPOSE_COLUMNS_6D)
                            + list(MP_LANDMARK_COLUMNS)
                            + list(FEAT_GAZE_COLUMNS))
                if c in batch_output.columns
            ]
            batch_output.loc[no_det, blank_cols] = np.nan

        return batch_output
