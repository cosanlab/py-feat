import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.optim import Adam
from feat.data import Fex, ImageDataset, TensorDataset, VideoDataset
from skops.io import load, get_untrusted_types
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from feat.pretrained import AU_LANDMARK_MAP
from torch.utils.data import DataLoader
from feat.face_detectors.Retinaface.Retinaface_model import (
    RetinaFace,
    postprocess_retinaface,
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
    FEAT_IDENTITY_COLUMNS,
    MP_LANDMARK_COLUMNS,
    MP_BLENDSHAPE_NAMES,
    MP_BLENDSHAPE_MODEL_LANDMARKS_SUBSET,
)
from feat.utils.image_operations import (
    convert_image_to_tensor,
    convert_color_vector_to_tensor,
    extract_face_from_bbox_torch,
    inverse_transform_landmarks_torch,
    extract_hog_features,
    convert_bbox_output,
    compute_original_image_size,
)
from feat.utils.io import get_resource_path


def get_camera_intrinsics(batch_hw_tensor, focal_length=None):
    """
    Computes the camera intrinsic matrix for a batch of images.

    Args:
        batch_hw_tensor (torch.Tensor): A tensor of shape [B, 2] where B is the batch size, and each entry contains [H, W] for the height and width of the images.
        focal_length (torch.Tensor, optional): A tensor of shape [B] representing the focal length for each image in the batch. If None, the focal length will default to the image width for each image.

    Returns:
        K (torch.Tensor): A tensor of shape [B, 3, 3] containing the camera intrinsic matrices for each image in the batch.
    """
    # Extract the batch size
    batch_size = batch_hw_tensor.shape[0]

    # Extract heights and widths
    heights = batch_hw_tensor[:, 0]
    widths = batch_hw_tensor[:, 1]

    # If focal_length is not provided, default to image width for each image
    if focal_length is None:
        focal_length = widths  # [B]

    # Initialize the camera intrinsic matrices
    K = torch.zeros((batch_size, 3, 3), dtype=torch.float32)

    # Populate the intrinsic matrices
    K[:, 0, 0] = focal_length  # fx
    K[:, 1, 1] = focal_length  # fy
    K[:, 0, 2] = widths / 2  # cx
    K[:, 1, 2] = heights / 2  # cy
    K[:, 2, 2] = 1.0  # The homogeneous coordinate

    return K


def convert_landmarks_3d(fex):
    """
    Converts facial landmarks from a feature extraction object into a 3D tensor.

    Args:
        fex (Fex): Fex DataFrame containing 478 3D landmark coordinates

    Returns:
        landmarks (torch.Tensor): A tensor of shape [batch_size, 478, 3] containing the 3D coordinates (x, y, z) of 478 facial landmarks for each instance in the batch.
    """

    return torch.tensor(fex.landmarks.astype(float).values).reshape(fex.shape[0], 478, 3)


def estimate_gaze_direction(fex, gaze_angle="combined", metric="radians"):
    """
    Estimates the gaze direction based on the 3D facial landmarks of the eyes and irises.

    NOTES: This could eventually be added as Fex Method

    Args:
        fex (Fex): Fex DataFrame containing 478 3D landmark coordinates
        gaze_angle (str): Specifies which gaze angle to calculate (default='combined')
        metric (str): Specifies the unit for the resulting gaze angle (default='radians'):

    Returns:
        angle (torch.Tensor): A tensor of shape [batch_size] containing the estimated gaze angles for each
            instance in the batch, in the specified metric (radians or degrees).
    """

    # Landmark roi locations
    left_eye_roi = torch.tensor(
        [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
        dtype=int,
    )
    right_eye_roi = torch.tensor(
        [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
        dtype=int,
    )
    left_iris_roi = torch.tensor([468, 469, 470, 471, 472], dtype=int)
    right_iris_roi = torch.tensor([473, 474, 475, 476, 477], dtype=int)

    # Extract ROIs
    landmarks = convert_landmarks_3d(fex.landmarks)
    left_eye_landmarks = landmarks[:, left_eye_roi, :]
    right_eye_landmarks = landmarks[:, right_eye_roi, :]
    left_iris_landmarks = landmarks[:, left_iris_roi, :]
    right_iris_landmarks = landmarks[:, right_iris_roi, :]

    # Calculate the centers of the left and right eyes for the batch
    left_eye_center = torch.mean(left_eye_landmarks, dim=1)  # [batch_size, 3]
    right_eye_center = torch.mean(right_eye_landmarks, dim=1)  # [batch_size, 3]

    # Calculate the centers of the left and right irises for the batch
    left_iris_center = torch.mean(left_iris_landmarks, dim=1)  # [batch_size, 3]
    right_iris_center = torch.mean(right_iris_landmarks, dim=1)  # [batch_size, 3]

    # Calculate the gaze vectors for the left and right eyes
    left_gaze_vector = F.normalize(
        left_iris_center - left_eye_center, dim=1
    )  # [batch_size, 3]
    right_gaze_vector = F.normalize(
        right_iris_center - right_eye_center, dim=1
    )  # [batch_size, 3]

    if gaze_angle.lower() == "combined":
        combined_gaze_vector = F.normalize(
            (left_gaze_vector + right_gaze_vector) / 2, dim=1
        )  # [batch_size, 3]

        # Assuming the forward vector is along the camera's z-axis, repeated for the batch
        forward_vector = (
            torch.tensor([0, 0, 1], dtype=combined_gaze_vector.dtype)
            .unsqueeze(0)
            .repeat(combined_gaze_vector.size(0), 1)
        )  # [batch_size, 3]

        gaze_angles = torch.acos(
            torch.sum(combined_gaze_vector * forward_vector, dim=1)
            / (
                torch.norm(combined_gaze_vector, dim=1)
                * torch.norm(forward_vector, dim=1)
            )
        )
    elif gaze_angle.lower() == "left":
        # Assuming the forward vector is along the camera's z-axis, repeated for the batch
        forward_vector = (
            torch.tensor([0, 0, 1], dtype=left_gaze_vector.dtype)
            .unsqueeze(0)
            .repeat(left_gaze_vector.size(0), 1)
        )  # [batch_size, 3]

        gaze_angles = torch.acos(
            torch.sum(left_gaze_vector * forward_vector, dim=1)
            / (torch.norm(left_gaze_vector, dim=1) * torch.norm(forward_vector, dim=1))
        )
    elif gaze_angle.lower() == "right":
        # Assuming the forward vector is along the camera's z-axis, repeated for the batch
        forward_vector = (
            torch.tensor([0, 0, 1], dtype=right_gaze_vector.dtype)
            .unsqueeze(0)
            .repeat(right_gaze_vector.size(0), 1)
        )  # [batch_size, 3]

        gaze_angles = torch.acos(
            torch.sum(right_gaze_vector * forward_vector, dim=1)
            / (torch.norm(right_gaze_vector, dim=1) * torch.norm(forward_vector, dim=1))
        )
    else:
        raise NotImplementedError(
            "Only ['combined', 'left', 'right'] gaze_angle are currently implemented"
        )

    if metric.lower() == "radians":
        return gaze_angles
    elif metric.lower() == "degrees":
        return torch.rad2deg(gaze_angles)
    else:
        raise NotImplementedError("metric can only be ['radians', 'degrees']")


def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (pitch, roll, yaw).

    Parameters:
    -----------
    R : torch.Tensor
        A tensor of shape [batch_size, 3, 3] containing rotation matrices.

    Returns:
    --------
    euler_angles : torch.Tensor
        A tensor of shape [batch_size, 3] containing the Euler angles (pitch, roll, yaw) in radians.
    """
    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)

    singular = sy < 1e-6

    pitch = torch.where(
        singular,
        torch.atan2(-R[:, 2, 1], R[:, 1, 1]),
        torch.atan2(R[:, 2, 1], R[:, 2, 2]),
    )
    roll = torch.atan2(-R[:, 2, 0], sy)
    yaw = torch.where(
        singular, torch.zeros_like(pitch), torch.atan2(R[:, 1, 0], R[:, 0, 0])
    )

    return torch.stack([pitch, roll, yaw], dim=1)


def estimate_face_pose(pts_3d, K, max_iter=100, lr=1e-3, return_euler_angles=True):
    """
    Estimate the face pose for a batch of 3D points using an iterative optimization approach.

    Args:
        pts_3d (torch.Tensor): A tensor of shape [batch_size, n_points, 3] representing the batch of 3D facial landmarks.
        K (torch.Tensor): A tensor of shape [batch_size, 3, 3] representing the camera intrinsic matrix for each image, or [3, 3] for a single shared intrinsic matrix.
        max_iter (int): The maximum number of iterations for the optimization loop. (default=100)
        lr (float): The learning rate for the Adam optimizer (default=1e-3)
        return_euler_angles (bool): If True, return 6 DOF (i.e., pitch, roll, and yaw angles) instead of the rotation matrix. (default=True)

    Returns:
        R_or_angles (torch.Tensor): If `return_euler_angles` is True, returns a tensor of shape [batch_size, 3] containing the Euler angles (pitch, roll, yaw). If `return_euler_angles` is False, returns a tensor of shape [batch_size, 3, 3] containing the rotation matrices.
        t (torch.Tensor): A tensor of shape [batch_size, 3] containing the estimated translation vectors.
    """

    # Ensure the dtype is consistent (e.g., float32)
    pts_3d = pts_3d.float()
    K = K.float()

    batch_size = pts_3d.size(0)

    # Check if K is a single matrix or a batch of matrices
    if K.dim() == 2:
        # If K is not batched, repeat it for each batch element
        K = K.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 3, 3]

    # Initial estimates for R and t (use identity and zeros for each batch)
    R = (
        torch.eye(3, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
        .requires_grad_(True)
    )  # [batch_size, 3, 3]
    t = torch.zeros(batch_size, 3, dtype=torch.float32).requires_grad_(
        True
    )  # [batch_size, 3]

    optimizer = Adam([R, t], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()

        # Rebuild the computation graph in every iteration
        pts_3d_proj = torch.bmm(pts_3d, R.transpose(1, 2)) + t.unsqueeze(
            1
        )  # [batch_size, n_points, 3]
        pts_2d_proj = torch.bmm(K, pts_3d_proj.transpose(1, 2)).transpose(
            1, 2
        )  # [batch_size, n_points, 3]

        # Normalize by the third coordinate
        pts_2d_proj = pts_2d_proj[:, :, :2] / pts_2d_proj[:, :, 2:].clamp(
            min=1e-7
        )  # [batch_size, n_points, 2]

        # Assuming directly facing camera means minimizing deviation from (x, y) plane
        loss = torch.mean(pts_3d_proj[:, :, 2] ** 2)  # Minimize z-coordinates to zero

        # Backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()

        # Normalize R to keep it a valid rotation matrix (optional step)
        with torch.no_grad():  # Detach the graph here
            U, _, V = torch.svd(R)
            R.copy_(torch.bmm(U, V.transpose(1, 2)))  # Copy the values back to R in-place

    if return_euler_angles:
        # Convert rotation matrices to Euler angles (pitch, roll, yaw)
        euler_angles = rotation_matrix_to_euler_angles(R)
        return euler_angles, t
    else:
        return R, t


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

        # Initialize Face Detector
        self.info["face_model"] = face_model
        if face_model is not None:
            if face_model == "retinaface":
                face_config_file = hf_hub_download(
                    repo_id="py-feat/retinaface",
                    filename="config.json",
                    cache_dir=get_resource_path(),
                )
                with open(face_config_file, "r") as f:
                    self.face_config = json.load(f)

                face_model_file = hf_hub_download(
                    repo_id="py-feat/retinaface",
                    filename="mobilenet0.25_Final.pth",
                    cache_dir=get_resource_path(),
                )
                face_checkpoint = torch.load(
                    face_model_file, map_location=self.device, weights_only=True
                )

                self.face_detector = RetinaFace(cfg=self.face_config, phase="test")
            else:
                raise ValueError("{face_model} is not currently supported.")

            self.face_detector.load_state_dict(face_checkpoint)
            self.face_detector.eval()
            self.face_detector.to(self.device)
            # self.face_detector = torch.compile(self.face_detector)
        else:
            self.face_detector = None

        # Initialize Landmark Detector
        self.info["landmark_model"] = landmark_model
        if landmark_model is not None:
            if landmark_model == "mp_facemesh_v2":
                self.face_size = 256
                landmark_model_file = hf_hub_download(
                    repo_id="py-feat/mp_facemesh_v2",
                    filename="face_landmarks_detector_Nx3x256x256_onnx.pth",
                    cache_dir=get_resource_path(),
                )
                self.landmark_detector = torch.load(
                    landmark_model_file, map_location=self.device, weights_only=False
                )
                self.landmark_detector.eval()
                self.landmark_detector.to(self.device)
                # self.landmark_detector = torch.compile(self.landmark_detector)
            else:
                raise ValueError("{landmark_model} is not currently supported.")

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
                    self.au_detector.to(self.device)
                else:
                    raise ValueError("{au_model} is not currently supported.")
            else:
                raise ValueError(
                    "Landmark Detector is required for AU Detection with {au_model}."
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
                raise ValueError("{emotion_model} is not currently supported.")
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
                raise ValueError("{identity_model} is not currently supported.")
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

        batch_results = []
        for i in range(frames.size(0)):
            frame = frames[i, ...].unsqueeze(0)  # Extract single image from batch

            if self.info["face_model"] == "retinaface":
                single_frame = torch.sub(
                    frame, convert_color_vector_to_tensor(np.array([123, 117, 104]))
                )

                predicted_locations, predicted_scores, predicted_landmarks = (
                    self.face_detector.forward(single_frame.to(self.device))
                )
                face_output = postprocess_retinaface(
                    predicted_locations,
                    predicted_scores,
                    predicted_landmarks,
                    self.face_config,
                    single_frame,
                    device=self.device,
                )

                bbox = face_output["boxes"]
                facescores = face_output["scores"]
                _ = face_output["landmarks"]

            # Extract faces from bbox
            if bbox.numel() != 0:
                extracted_faces, new_bbox = extract_face_from_bbox_torch(
                    frame / 255.0, bbox, face_size=face_size, expand_bbox=1.25
                )
            else:  # No Face Detected
                extracted_faces = torch.zeros((1, 3, face_size, face_size))
                bbox = torch.zeros((1, 4))
                new_bbox = torch.zeros((1, 4))
                facescores = torch.zeros((1))
                # poses = torch.zeros((1,6))

            frame_results = {
                "face_id": i,
                "faces": extracted_faces,
                "boxes": bbox,
                "new_boxes": new_bbox,
                "scores": facescores,
            }

            # Extract Faces separately for Resmasknet
            if self.info["emotion_model"] == "resmasknet":
                if torch.all(frame_results["scores"] == 0):  # No Face Detected
                    frame_results["resmasknet_faces"] = torch.zeros((1, 3, 224, 224))
                else:
                    resmasknet_faces, _ = extract_face_from_bbox_torch(
                        single_frame, bbox, expand_bbox=1.1, face_size=224
                    )
                    frame_results["resmasknet_faces"] = resmasknet_faces / 255.0

            batch_results.append(frame_results)

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

        if self.landmark_detector is not None:
            landmarks = self.landmark_detector.forward(extracted_faces.to(self.device))[0]

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
                landmarks_2d.reshape(n_faces, 478 * 2), new_bboxes.to(self.device)
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
            identity_embeddings = self.identity_detector.forward(
                extracted_faces.to(self.device)
            )
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

            # Invert the face boxes and landmarks based on the padded output size
            for j, frame_idx in enumerate(batch_results["frame"].unique()):
                batch_results.loc[
                    batch_results["frame"] == frame_idx, ["FrameHeight", "FrameWidth"]
                ] = (
                    compute_original_image_size(batch_data)[j, :]
                    .repeat(
                        len(
                            batch_results.loc[
                                batch_results["frame"] == frame_idx, "frame"
                            ]
                        ),
                        1,
                    )
                    .numpy()
                )
                batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectX"] = (
                    batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectX"]
                    - batch_data["Padding"]["Left"].detach().numpy()[j]
                ) / batch_data["Scale"].detach().numpy()[j]
                batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectY"] = (
                    batch_results.loc[batch_results["frame"] == frame_idx, "FaceRectY"]
                    - batch_data["Padding"]["Top"].detach().numpy()[j]
                ) / batch_data["Scale"].detach().numpy()[j]
                batch_results.loc[
                    batch_results["frame"] == frame_idx, "FaceRectWidth"
                ] = (
                    (
                        batch_results.loc[
                            batch_results["frame"] == frame_idx, "FaceRectWidth"
                        ]
                    )
                    / batch_data["Scale"].detach().numpy()[j]
                )
                batch_results.loc[
                    batch_results["frame"] == frame_idx, "FaceRectHeight"
                ] = (
                    (
                        batch_results.loc[
                            batch_results["frame"] == frame_idx, "FaceRectHeight"
                        ]
                    )
                    / batch_data["Scale"].detach().numpy()[j]
                )

                for i in range(478):
                    batch_results.loc[batch_results["frame"] == frame_idx, f"x_{i}"] = (
                        batch_results.loc[batch_results["frame"] == frame_idx, f"x_{i}"]
                        - batch_data["Padding"]["Left"].detach().numpy()[j]
                    ) / batch_data["Scale"].detach().numpy()[j]
                    batch_results.loc[batch_results["frame"] == frame_idx, f"y_{i}"] = (
                        batch_results.loc[batch_results["frame"] == frame_idx, f"y_{i}"]
                        - batch_data["Padding"]["Top"].detach().numpy()[j]
                    ) / batch_data["Scale"].detach().numpy()[j]
                    # batch_results.loc[batch_results['frame']==frame_idx, f'z_{i}'] = (batch_results.loc[batch_results['frame']==frame_idx, f'z_{i}'] - batch_data["Padding"]["Top"].detach().numpy()[j])/batch_data["Scale"].detach().numpy()[j]

            batch_output.append(batch_results)
            frame_counter += 1 * batch_size
        batch_output = pd.concat(batch_output)
        batch_output.reset_index(drop=True, inplace=True)
        if data_type.lower() == "video":
            batch_output["approx_time"] = [
                dataset.calc_approx_frame_time(x)
                for x in batch_output["frame"].to_numpy()
            ]

        # Compute Identities
        batch_output.compute_identities(threshold=face_identity_threshold, inplace=True)

        # Add Gaze
        batch_output["gaze_angle"] = estimate_gaze_direction(
            batch_output, metric="radians", gaze_angle="combined"
        )

        # Add Pose
        landmarks_3d = convert_landmarks_3d(batch_output)[
            :, :468, :
        ]  # Drop Irises - could also use restricted set (min 6) to speed up computation
        K = get_camera_intrinsics(
            torch.tensor(batch_output[["FrameHeight", "FrameWidth"]].values)
        )  # Camera intrinsic matrix
        with torch.enable_grad():  # Enable gradient tracking for pose estimation
            R, t = estimate_face_pose(landmarks_3d, K, return_euler_angles=True)
        batch_output.loc[:, FEAT_FACEPOSE_COLUMNS_6D] = (
            torch.cat((R, t), dim=1).detach().numpy()
        )

        return batch_output
