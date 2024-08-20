import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from collections import OrderedDict

from feat.emo_detectors.ResMaskNet.resmasknet_test import (
    ResMasking,
    resmasking_dropout1,
)
from feat.identity_detectors.facenet.facenet_model import InceptionResnetV1
from feat.facepose_detectors.img2pose.deps.models import (
    FasterDoFRCNN,
    postprocess_img2pose,
)
from feat.au_detectors.StatLearning.SL_test import XGBClassifier, SVMClassifier
from feat.emo_detectors.StatLearning.EmoSL_test import EmoSVMClassifier
from feat.landmark_detectors.mobilefacenet_test import MobileFaceNet

from feat.landmark_detectors.basenet_test import MobileNet_GDConv
from feat.landmark_detectors.pfld_compressed_test import PFLDInference
from feat.pretrained import load_model_weights, AU_LANDMARK_MAP
from feat.utils import (
    set_torch_device,
    openface_2d_landmark_columns,
    FEAT_EMOTION_COLUMNS,
    FEAT_FACEBOX_COLUMNS,
    # FEAT_FACEPOSE_COLUMNS_3D,
    FEAT_FACEPOSE_COLUMNS_6D,
    FEAT_TIME_COLUMNS,
    FEAT_IDENTITY_COLUMNS,
)
from feat.utils.io import get_resource_path
from feat.utils.image_operations import (
    extract_face_from_landmarks,
    convert_image_to_tensor,
    align_face,
    mask_image,
)
from feat.data import (
    Fex,
    ImageDataset,
    TensorDataset,
    VideoDataset
)
from skops.io import load, get_untrusted_types
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.utils import draw_keypoints, draw_bounding_boxes, make_grid
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize
from scipy.spatial import ConvexHull
from skimage.morphology.convex_hull import grid_points_in_poly
from skimage.feature import hog
import sys

sys.modules["__main__"].__dict__["XGBClassifier"] = XGBClassifier
sys.modules["__main__"].__dict__["SVMClassifier"] = SVMClassifier
sys.modules["__main__"].__dict__["EmoSVMClassifier"] = EmoSVMClassifier


def plot_frame(
    frame,
    boxes=None,
    landmarks=None,
    boxes_width=2,
    boxes_colors="cyan",
    landmarks_radius=2,
    landmarks_width=2,
    landmarks_colors="white",
):
    """
    Plot Torch Frames and py-feat output. If multiple frames will create a grid of images

    Args:
        frame (torch.Tensor): Tensor of shape (B, C, H, W) or (C, H, W)
        boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes
        landmarks (torch.Tensor): Tensor of shape (N, 136) containing flattened 68 point landmark keystones

    Returns:
        PILImage
    """

    if len(frame.shape) == 4:
        B, C, H, W = frame.shape
    elif len(frame.shape) == 3:
        C, H, W = frame.shape
    else:
        raise ValueError("Can only plot (B,C,H,W) or (C,H,W)")
    if B == 1:
        if boxes is not None:
            new_frame = draw_bounding_boxes(
                frame.squeeze(0), boxes, width=boxes_width, colors=boxes_colors
            )

            if landmarks is not None:
                new_frame = draw_keypoints(
                    new_frame,
                    landmarks.reshape(landmarks.shape[0], -1, 2),
                    radius=landmarks_radius,
                    width=landmarks_width,
                    colors=landmarks_colors,
                )
        else:
            if landmarks is not None:
                new_frame = draw_keypoints(
                    frame.squeeze(0),
                    landmarks.reshape(landmarks.shape[0], -1, 2),
                    radius=landmarks_radius,
                    width=landmarks_width,
                    colors=landmarks_colors,
                )
            else:
                new_frame = frame.squeeze(0)
        return transforms.ToPILImage()(new_frame.squeeze(0))
    else:
        if (boxes is not None) & (landmarks is None):
            new_frame = make_grid(
                torch.stack(
                    [
                        draw_bounding_boxes(
                            f, b.unsqueeze(0), width=boxes_width, colors=boxes_colors
                        )
                        for f, b in zip(frame.unbind(dim=0), boxes.unbind(dim=0))
                    ],
                    dim=0,
                )
            )
        elif (landmarks is not None) & (boxes is None):
            new_frame = make_grid(
                torch.stack(
                    [
                        draw_keypoints(
                            f,
                            l.unsqueeze(0),
                            radius=landmarks_radius,
                            width=landmarks_width,
                            colors=landmarks_colors,
                        )
                        for f, l in zip(
                            frame.unbind(dim=0),
                            landmarks.reshape(landmarks.shape[0], -1, 2).unbind(dim=0),
                        )
                    ],
                    dim=0,
                )
            )
        elif (boxes is not None) & (landmarks is not None):
            new_frame = make_grid(
                torch.stack(
                    [
                        draw_keypoints(
                            fr,
                            l.unsqueeze(0),
                            radius=landmarks_radius,
                            width=landmarks_width,
                            colors=landmarks_colors,
                        )
                        for fr, l in zip(
                            [
                                draw_bounding_boxes(
                                    f,
                                    b.unsqueeze(0),
                                    width=boxes_width,
                                    colors=boxes_colors,
                                )
                                for f, b in zip(
                                    frame.unbind(dim=0), boxes.unbind(dim=0)
                                )
                            ],
                            landmarks.reshape(landmarks.shape[0], -1, 2).unbind(dim=0),
                        )
                    ]
                )
            )
        else:
            new_frame = make_grid(frame)
        return transforms.ToPILImage()(new_frame)


def convert_bbox_output(img2pose_output):
    """Convert im2pose_output into Fex Format"""

    widths = (
        img2pose_output["boxes"][:, 2] - img2pose_output["boxes"][:, 0]
    )  # right - left
    heights = (
        img2pose_output["boxes"][:, 3] - img2pose_output["boxes"][:, 1]
    )  # bottom - top

    return torch.stack(
        (
            img2pose_output["boxes"][:, 0],
            img2pose_output["boxes"][:, 1],
            widths,
            heights,
            img2pose_output["scores"],
        ),
        dim=1,
    )


def extract_face_from_bbox_torch(frame, detected_faces, face_size=112, expand_bbox=1.2):
    """Extract face from image and resize using pytorch."""
    
    device = frame.device
    B, C, H, W = frame.shape
    N = detected_faces.shape[0]

    # Move detected_faces to the same device as frame
    detected_faces = detected_faces.to(device)

    # Extract the bounding box coordinates
    x1, y1, x2, y2 = (
        detected_faces[:, 0],
        detected_faces[:, 1],
        detected_faces[:, 2],
        detected_faces[:, 3],
    )
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = (x2 - x1) * expand_bbox
    height = (y2 - y1) * expand_bbox

    # Calculate expanded bounding box coordinates
    new_x1 = (center_x - width / 2).clamp(min=0)
    new_y1 = (center_y - height / 2).clamp(min=0)
    new_x2 = (center_x + width / 2).clamp(max=W)
    new_y2 = (center_y + height / 2).clamp(max=H)

    # Cast the bounding box coordinates to long for indexing
    new_bboxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1).long()

    # Create a mesh grid for the face size
    yy, xx = torch.meshgrid(
        torch.arange(face_size, device=device),
        torch.arange(face_size, device=device),
        indexing='ij'
    )
    yy = yy.float()
    xx = xx.float()

    # Calculate the normalized coordinates for the grid sampling
    grid_x = (xx + 0.5) / face_size * (new_x2 - new_x1).view(N, 1, 1) + new_x1.view(N, 1, 1)
    grid_y = (yy + 0.5) / face_size * (new_y2 - new_y1).view(N, 1, 1) + new_y1.view(N, 1, 1)

    # Normalize grid coordinates to the range [-1, 1]
    grid_x = 2 * grid_x / (W - 1) - 1
    grid_y = 2 * grid_y / (H - 1) - 1

    # Stack grid coordinates and reshape
    grid = torch.stack((grid_x, grid_y), dim=-1)  # Shape: (N, face_size, face_size, 2)

    # Ensure frame and grid are float32 for grid_sample
    frame = frame.float()
    grid = grid.float()

    # Calculate frame indices for each face, assuming faces are sequentially ordered
    face_indices = torch.arange(N, device=device) % B  # Repeat for each batch element
    frame_expanded = frame[face_indices]  # Select corresponding frame for each face

    # Use grid_sample to extract and resize faces
    cropped_faces = F.grid_sample(frame_expanded, grid, align_corners=False)

    # The output shape should be (N, C, face_size, face_size)
    return cropped_faces, new_bboxes

def inverse_transform_landmarks_torch(landmarks, boxes):
    """
    Transforms landmarks based on new bounding boxes.

    Args:
        landmarks (torch.Tensor): Tensor of shape (N, 136) representing 68 landmarks for N samples.
        new_bbox (torch.Tensor): Tensor of shape (N, 4) representing bounding boxes [x1, y1, x2, y2] for N samples.

    Returns:
        torch.Tensor: Transformed landmarks of shape (N, 136).
    """
    N = landmarks.shape[0]
    landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)

    # Extract bounding box coordinates
    left = boxes[:, 0]  # (N,)
    top = boxes[:, 1]  # (N,)
    right = boxes[:, 2]  # (N,)
    bottom = boxes[:, 3]  # (N,)

    # Calculate width and height of the bounding boxes
    width = right - left  # (N,)
    height = bottom - top  # (N,)

    # Rescale the landmarks
    transformed_landmarks = torch.zeros_like(landmarks)
    transformed_landmarks[:, :, 0] = landmarks[:, :, 0] * width.unsqueeze(
        1
    ) + left.unsqueeze(1)
    transformed_landmarks[:, :, 1] = landmarks[:, :, 1] * height.unsqueeze(
        1
    ) + top.unsqueeze(1)

    return transformed_landmarks.reshape(N, 136)

def extract_hog_features(extracted_faces, landmarks):
    """
    Helper function used in batch processing hog features

    Args:
        frames: a batch of extracted faces
        landmarks: a list of list of detected landmarks

    Returns:
        hog_features: a numpy array of hog features for each detected landmark
        landmarks: updated landmarks
    """
    n_faces = landmarks.shape[0]
    face_size = extracted_faces.shape[-1]
    extracted_faces_bboxes = torch.tensor([0, 0, face_size, face_size]).unsqueeze(0).repeat(n_faces,1)
    extracted_landmarks = inverse_transform_landmarks_torch(landmarks, extracted_faces_bboxes)
    hog_features = []
    au_new_landmarks = []
    for j in range(n_faces):
        convex_hull, new_landmark = extract_face_from_landmarks(extracted_faces[j, ...], extracted_landmarks[j, ...])
        hog_features.append(hog(
            transforms.ToPILImage()(convex_hull[0]),
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            channel_axis=-1,
        ).reshape(1, -1))
        au_new_landmarks.append(new_landmark)
    return np.concatenate(hog_features), au_new_landmarks
        

def extract_face_from_landmarks(frame, landmarks, face_size=112):
    """Extract a face in a frame with a convex hull of landmarks.

    This function extracts the faces of the frame with convex hulls and masks out the rest.

    Args:
        frame (array): The original image]
        detected_faces (list): face bounding box
        landmarks (list): the landmark information]
        align (bool): align face to standard position
        size_output (int, optional): [description]. Defaults to 112.

    Returns:
        resized_face_np: resized face as a numpy array
        new_landmarks: landmarks of aligned face
    """

    if not isinstance(frame, torch.Tensor):
        raise ValueError(f"image must be a tensor not {type(frame)}")

    if len(frame.shape) != 4:
        frame = frame.unsqueeze(0)

    landmarks = landmarks.detach().numpy()

    aligned_img, new_landmarks = align_face(
        frame,
        landmarks.flatten(),
        landmark_type=68,
        box_enlarge=2.5,
        img_size=face_size,
    )

    hull = ConvexHull(new_landmarks)
    mask = grid_points_in_poly(
        shape=aligned_img.shape[-2:],
        # for some reason verts need to be flipped
        verts=list(
            zip(
                new_landmarks[hull.vertices][:, 1],
                new_landmarks[hull.vertices][:, 0],
            )
        ),
    )
    mask[
        0 : np.min([new_landmarks[0][1], new_landmarks[16][1]]),
        new_landmarks[0][0] : new_landmarks[16][0],
    ] = True
    masked_image = mask_image(aligned_img, mask)

    return (masked_image, new_landmarks)


class FastDetector(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                landmark_model="mobilefacenet",
                au_model="xgb",
                emotion_model="resmasknet",
                identity_model="facenet",
                device="cpu"):
        super(FastDetector, self).__init__()

        self.info = dict(
            face_model='img2pose',
            landmark_model=None,
            emotion_model=None,
            facepose_model='img2pose',
            au_model=None,
            identity_model=None,
        )
        self.device = set_torch_device(device)
        
        # Load Model Configurations
        facepose_config_file = hf_hub_download(
            repo_id="py-feat/img2pose",
            filename="config.json",
            cache_dir=get_resource_path(),
        )
        with open(facepose_config_file, "r") as f:
            facepose_config = json.load(f)

        # Initialize img2pose
        backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=None)
        backbone.eval()
        backbone.to(self.device)
        self.facepose_detector = FasterDoFRCNN(
            backbone=backbone,
            num_classes=2,
            min_size=facepose_config["min_size"],
            max_size=facepose_config["max_size"],
            pose_mean=torch.tensor(facepose_config["pose_mean"]),
            pose_stddev=torch.tensor(facepose_config["pose_stddev"]),
            threed_68_points=torch.tensor(facepose_config["threed_points"]),
            rpn_pre_nms_top_n_test=facepose_config["rpn_pre_nms_top_n_test"],
            rpn_post_nms_top_n_test=facepose_config["rpn_post_nms_top_n_test"],
            bbox_x_factor=facepose_config["bbox_x_factor"],
            bbox_y_factor=facepose_config["bbox_y_factor"],
            expand_forehead=facepose_config["expand_forehead"],
        )
        facepose_model_file = hf_hub_download(
            repo_id="py-feat/img2pose",
            filename="model.safetensors",
            cache_dir=get_resource_path(),
        )
        facepose_checkpoint = load_file(facepose_model_file)
        self.facepose_detector.load_state_dict(facepose_checkpoint, load_model_weights)
        self.facepose_detector.eval()
        self.facepose_detector.to(self.device)
        # self.facepose_detector = torch.compile(self.facepose_detector)

        # Initialize Landmark Detector
        self.info["landmark_model"] = landmark_model
        if landmark_model is not None:
            if landmark_model == "mobilefacenet":
                self.face_size = 112
                self.landmark_detector = MobileFaceNet([self.face_size, self.face_size], 136, device=self.device)
                landmark_model_file = hf_hub_download(
                    repo_id="py-feat/mobilefacenet",
                    filename="mobilefacenet_model_best.pth.tar",
                    cache_dir=get_resource_path(),
                )
                landmark_state_dict = torch.load(landmark_model_file, map_location=self.device, weights_only=True)["state_dict"]                # Ensure Model weights are Float32 for MPS
                # mobilefacenet_state_dict =  torch.load(landmark_model_file, map_location=self.device, weights_only=True)["state_dict"]
                # mobilefacenet_state_dict = {k: v.float() for k, v in mobilefacenet_state_dict.items()} 
                # self.landmark_detector.load_state_dict(mobilefacenet_state_dict)
                # self.landmark_detector = torch.compile(self.landmark_detector)
            elif landmark_model == "mobilenet":
                self.face_size = 224
                self.landmark_detector = MobileNet_GDConv(136)
                landmark_model_file = hf_hub_download(
                    repo_id="py-feat/mobilenet",
                    filename="mobilenet_224_model_best_gdconv_external.pth.tar",
                    cache_dir=get_resource_path(),
                )
                mobilenet_state_dict = torch.load(landmark_model_file, map_location=self.device, weights_only=True)["state_dict"]                # Ensure Model weights are Float32 for MPS
                landmark_state_dict = OrderedDict()
                for k, v in mobilenet_state_dict.items():
                    if "module." in k:
                        k = k.replace("module.", "")
                    landmark_state_dict[k] = v
            elif landmark_model == "pfld":
                self.face_size = 112
                self.landmark_detector = PFLDInference()
                landmark_model_file = hf_hub_download(
                    repo_id="py-feat/pfld",
                    filename="pfld_model_best.pth.tar",
                    cache_dir=get_resource_path(),
                )
                landmark_state_dict = torch.load(landmark_model_file, map_location=self.device, weights_only=True)["state_dict"]                # Ensure Model weights are Float32 for MPS
            else:
                raise ValueError("{landmark_model} is not currently supported.")
            self.landmark_detector.load_state_dict(landmark_state_dict)
            self.landmark_detector.eval()
            self.landmark_detector.to(self.device)
        else:
            self.landmark_detector = None
            
        # Initialize AU Detector
        self.info['au_model'] = au_model
        if au_model is not None:
            if self.landmark_detector is not None:
                if au_model == "xgb":
                    self.au_detector = XGBClassifier()
                    au_model_path = hf_hub_download(
                        repo_id="py-feat/xgb_au",
                        filename="xgb_au_classifier.skops",
                        cache_dir=get_resource_path(),
                    )
        
                elif au_model == "svm":
                    self.au_detector = SVMClassifier()
                    au_model_path = hf_hub_download(
                        repo_id="py-feat/svm_au",
                        filename="svm_au_classifier.skops",
                        cache_dir=get_resource_path(),
                    )
                else:
                    raise ValueError("{au_model} is not currently supported.")
                
                au_unknown_types = get_untrusted_types(file=au_model_path)
                loaded_au_model = load(au_model_path, trusted=au_unknown_types)
                self.au_detector.load_weights(
                    scaler_upper=loaded_au_model.scaler_upper,
                    pca_model_upper=loaded_au_model.pca_model_upper,
                    scaler_lower=loaded_au_model.scaler_lower,
                    pca_model_lower=loaded_au_model.pca_model_lower,
                    scaler_full=loaded_au_model.scaler_full,
                    pca_model_full=loaded_au_model.pca_model_full,
                    classifiers=loaded_au_model.classifiers,
                )
            else:
                raise ValueError("Landmark Detector is required for AU Detection with {au_model}.")
        else:
            self.au_detector = None
            
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
            elif emotion_model == 'svm':
                if self.landmark_detector is not None:
                    self.emotion_detector = EmoSVMClassifier()
                    emotion_model_path = hf_hub_download(repo_id="py-feat/svm_emo", 
                                                        filename="svm_emo_classifier.skops", 
                                                        cache_dir=get_resource_path())
                    emotion_unknown_types = get_untrusted_types(file=emotion_model_path)
                    loaded_emotion_model = load(emotion_model_path, trusted=emotion_unknown_types)
                    self.emotion_detector.load_weights(scaler_full=loaded_emotion_model.scaler_full, 
                                            pca_model_full=loaded_emotion_model.pca_model_full, 
                                            classifiers=loaded_emotion_model.classifiers)
                else:
                    raise ValueError("Landmark Detector is required for Emotion Detection with {emotion_model}.")

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
                    torch.load(identity_model_file, map_location=device, weights_only=True)
                )
                self.identity_detector.eval()
                self.identity_detector.to(self.device)
                # self.identity_detector = torch.compile(self.identity_detector)
            else:
                raise ValueError("{identity_model} is not currently supported.")
        else:
            self.identity_detector = None

    @torch.inference_mode()
    def detect_faces(self, images, face_size=112, face_detection_threshold=0.5):
        """
        detect faces and poses in a batch of images using img2pose

        Args:
            img (torch.Tensor): Tensor of shape (B, C, H, W) representing the images
            face_size (int): Output size to resize face after cropping.

        Returns:
            Fex: Prediction results dataframe
        """

        # img2pose
        frames = convert_image_to_tensor(images, img_type="float32") / 255.0
        frames.to(self.device)
        
        batch_results = []
        for i in range(frames.size(0)):
            single_frame = frames[i, ...].unsqueeze(0)  # Extract single image from batch
            img2pose_output = self.facepose_detector(single_frame)
            img2pose_output = postprocess_img2pose(img2pose_output[0], detection_threshold=face_detection_threshold)
            bbox = img2pose_output["boxes"]
            poses = img2pose_output["dofs"]
            facescores = img2pose_output["scores"]

            # Extract faces from bbox
            extracted_faces, new_bbox = extract_face_from_bbox_torch(
                single_frame, bbox, face_size=face_size
            )
            
            frame_results = {
                "face_id": i,
                "faces": extracted_faces,
                "boxes": bbox,
                "new_bbox": new_bbox,
                "poses": poses,
                "scores": facescores,
            }
            
            # Extract Faces separately for Resmasknet
            if self.info['emotion_model'] == 'resmasknet':
                resmasknet_faces, _ = extract_face_from_bbox_torch(
                    single_frame, bbox, expand_bbox=1.1, face_size=224
                )
                frame_results["resmasknet_faces"] =  resmasknet_faces

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
        new_bboxes = torch.cat([face["new_bbox"] for face in faces_data], dim=0)
        n_faces = extracted_faces.shape[0]
        
        if self.landmark_detector is not None:
            if self.info["landmark_model"].lower() == "mobilenet":
                extracted_faces = Compose(
                    [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                )(extracted_faces)
            
            if self.info["landmark_model"].lower() == "mobilefacenet":
                landmarks = self.landmark_detector.forward(extracted_faces.to(self.device))[0]
            else:
                landmarks = self.landmark_detector.forward(extracted_faces.to(self.device))
            new_landmarks = inverse_transform_landmarks_torch(landmarks, new_bboxes)            
        else:
            new_landmarks = torch.full((n_faces, 136), float('nan'))
            
        if self.emotion_detector is not None:
            if self.info['emotion_model'] == 'resmasknet':
                resmasknet_faces = torch.cat([face["resmasknet_faces"] for face in faces_data], dim=0)
                emotions = self.emotion_detector.forward(resmasknet_faces)
                emotions = torch.softmax(emotions, 1)
            elif self.info['emotion_model'] == 'svm':
                hog_features, emo_new_landmarks = extract_hog_features(extracted_faces, landmarks)
                emotions = self.emotion_detector.detect_emo(frame=hog_features, landmarks=[emo_new_landmarks])
                emotions = torch.tensor(emotions)
        else:
            emotions = torch.full((n_faces, 7), float('nan'))

        if self.identity_detector is not None:
            identity_embeddings = self.identity_detector.forward(extracted_faces)
        else:
            identity_embeddings = torch.full((n_faces, 512), float('nan'))

        if self.au_detector is not None:    
            hog_features, au_new_landmarks = extract_hog_features(extracted_faces, landmarks)
            aus = self.au_detector.detect_au(
                frame=hog_features, landmarks=[au_new_landmarks]
            )
        else:
            aus = torch.full((n_faces, 20), float('nan'))

        # Create Fex Output Representation
        new_bboxes = torch.cat([face["new_bbox"] for face in faces_data], dim=0)

        bboxes = torch.cat([convert_bbox_output(face_output) for face_output in faces_data], dim=0)
        feat_faceboxes = pd.DataFrame(
            bboxes.detach().numpy(),
            columns=FEAT_FACEBOX_COLUMNS,
        )
        
        poses = torch.cat([face_output['poses'] for face_output in faces_data], dim=0)
        feat_poses = pd.DataFrame(
            poses.detach().numpy(), columns=FEAT_FACEPOSE_COLUMNS_6D
        )
        
        reshape_landmarks = new_landmarks.reshape(new_landmarks.shape[0], 68, 2)
        reordered_landmarks = torch.cat(
            [reshape_landmarks[:, :, 0], reshape_landmarks[:, :, 1]], dim=1
        )
        feat_landmarks = pd.DataFrame(
            reordered_landmarks.detach().numpy(), columns=openface_2d_landmark_columns
        )
        
        feat_aus = pd.DataFrame(aus, columns=AU_LANDMARK_MAP["Feat"])

        feat_emotions = pd.DataFrame(
            emotions.detach().numpy(), columns=FEAT_EMOTION_COLUMNS
        )
        
        feat_identities = pd.DataFrame(
            identity_embeddings.detach().numpy(), columns=FEAT_IDENTITY_COLUMNS[1:]
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
            landmark_columns=openface_2d_landmark_columns,
            facepose_columns=FEAT_FACEPOSE_COLUMNS_6D,
            identity_columns=FEAT_IDENTITY_COLUMNS[1:],
            detector="Feat",
            face_model=self.info["face_model"],
            landmark_model=self.info["landmark_model"],
            au_model=self.info["au_model"],
            emotion_model=self.info["emotion_model"],
            facepose_model=self.info["facepose_model"],
            identity_model=self.info["identity_model"]
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
            **kwargs: additional detector-specific kwargs

        Returns:
            pd.DataFrame: Concatenated results for all images in the batch
        """

        if data_type.lower() == 'image':
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
        elif data_type.lower() == 'tensor':
            data_loader = DataLoader(
                TensorDataset(inputs),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        elif data_type.lower() == 'video':
            dataset = VideoDataset(
                    inputs, 
                    skip_frames=skip_frames, 
                    output_size=output_size
            )
            data_loader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=False,
            )        
        
        batch_output = []
        frame_counter = 0
        for batch_id, batch_data in enumerate(tqdm(data_loader)):
            faces_data = self.detect_faces(batch_data["Image"], face_size=self.face_size, face_detection_threshold=face_detection_threshold)
            batch_results = self.forward(faces_data)

            # Create metadata for each frame                
            file_names = []
            frame_ids = []            
            for i,face in enumerate(faces_data):
                n_faces = len(face['scores'])
                if data_type.lower() == 'video':
                    current_frame_id = batch_data['Frame'].detach().numpy()[i]
                else:
                    current_frame_id = frame_counter + i
                frame_ids.append(np.repeat(current_frame_id, n_faces))
                file_names.append(np.repeat(batch_data['FileName'][i], n_faces))
            batch_results['input'] = np.concatenate(file_names)
            batch_results['frame'] = np.concatenate(frame_ids)

            # Invert the face boxes and landmarks based on the padded output size
            for j,frame_idx in enumerate(batch_results['frame'].unique()):
                batch_results.loc[batch_results['frame']==frame_idx, 'FaceRectX'] = (batch_results.loc[batch_results['frame']==frame_idx, 'FaceRectX'] - batch_data["Padding"]["Left"].detach().numpy()[j])/batch_data["Scale"].detach().numpy()[j]
                batch_results.loc[batch_results['frame']==frame_idx, 'FaceRectY'] = (batch_results.loc[batch_results['frame']==frame_idx, 'FaceRectY'] - batch_data["Padding"]["Top"].detach().numpy()[j])/batch_data["Scale"].detach().numpy()[j]
                batch_results.loc[batch_results['frame']==frame_idx, 'FaceRectWidth'] = (batch_results.loc[batch_results['frame']==frame_idx, 'FaceRectWidth'])/batch_data["Scale"].detach().numpy()[j]
                batch_results.loc[batch_results['frame']==frame_idx, 'FaceRectHeight'] = (batch_results.loc[batch_results['frame']==frame_idx, 'FaceRectHeight'])/batch_data["Scale"].detach().numpy()[j]
            
                for i in range(68):
                    batch_results.loc[batch_results['frame']==frame_idx, f'x_{i}'] = (batch_results.loc[batch_results['frame']==frame_idx, f'x_{i}'] - batch_data["Padding"]["Left"].detach().numpy()[j])/batch_data["Scale"].detach().numpy()[j]
                    batch_results.loc[batch_results['frame']==frame_idx, f'y_{i}'] = (batch_results.loc[batch_results['frame']==frame_idx, f'y_{i}'] - batch_data["Padding"]["Top"].detach().numpy()[j])/batch_data["Scale"].detach().numpy()[j]
            
            batch_output.append(batch_results)
            frame_counter += 1 * batch_size
        batch_output = pd.concat(batch_output)
        batch_output.reset_index(drop=True, inplace=True)
        if data_type.lower() == 'video':
            batch_output["approx_time"] = [dataset.calc_approx_frame_time(x) for x in batch_output["frame"].to_numpy()]
        batch_output.compute_identities(
            threshold=face_identity_threshold, inplace=True
        )
        return batch_output
    