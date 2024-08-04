import os 
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from feat.face_detectors.FaceBoxes.FaceBoxes_test import FaceBoxes
from feat.face_detectors.Retinaface.Retinaface_test import Retinaface
from feat.face_detectors.MTCNN.MTCNN_test import MTCNN
from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMasking, resmasking_dropout1
from feat.identity_detectors.facenet.facenet_model import InceptionResnetV1
from feat.facepose_detectors.img2pose.img2pose_model import img2poseModel, WrappedModel
from feat.facepose_detectors.img2pose.deps.models import FasterDoFRCNN, postprocess_img2pose
from feat.au_detectors.StatLearning.SL_test import XGBClassifier, SVMClassifier
from feat.emo_detectors.StatLearning.EmoSL_test import EmoSVMClassifier
from feat.landmark_detectors.mobilefacenet_test import MobileFaceNet
from feat.landmark_detectors.basenet_test import MobileNet_GDConv
from feat.landmark_detectors.pfld_compressed_test import PFLDInference
from feat.pretrained import load_model_weights, AU_LANDMARK_MAP
from feat.utils import (
    openface_2d_landmark_columns,
    FEAT_EMOTION_COLUMNS,
    FEAT_FACEBOX_COLUMNS,
    FEAT_FACEPOSE_COLUMNS_3D,
    FEAT_FACEPOSE_COLUMNS_6D,
    FEAT_TIME_COLUMNS,
    FEAT_IDENTITY_COLUMNS,
)
from feat.utils.io import get_resource_path, get_test_data_path
from feat.utils.image_operations import (
                                            extract_face_from_landmarks,
                                            convert_image_to_tensor,
                                            align_face,
                                            mask_image
                                        )
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.utils import draw_keypoints, draw_bounding_boxes, make_grid
from scipy.spatial import ConvexHull
from skimage.morphology.convex_hull import grid_points_in_poly
from torch.utils.data import DataLoader
from feat.data import (
    Fex,
    ImageDataset,
    VideoDataset,
    _inverse_face_transform,
    _inverse_landmark_transform,
)
from skimage.feature import hog
import torchvision.transforms as transforms
from feat.utils import set_torch_device
from feat.utils.image_operations import (
    extract_face_from_landmarks,
    convert_image_to_tensor,
    BBox,
)

def plot_frame(frame, boxes=None, landmarks=None, boxes_width=2, boxes_colors='cyan', landmarks_radius=2, landmarks_width=2, landmarks_colors='white'):
    ''' 
    Plot Torch Frames and py-feat output. If multiple frames will create a grid of images

    Args:
        frame (torch.Tensor): Tensor of shape (B, C, H, W) or (C, H, W)
        boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes
        landmarks (torch.Tensor): Tensor of shape (N, 136) containing flattened 68 point landmark keystones
        
    Returns:
        PILImage
    '''
    
    if len(frame.shape) == 4:
        B, C, H, W = frame.shape
    elif len(frame.shape) == 3:
        C, H, W = frame.shape
    else:
        raise ValueError('Can only plot (B,C,H,W) or (C,H,W)')
    if B == 1:
        if boxes is not None:
            new_frame = draw_bounding_boxes(frame.squeeze(0), boxes, width=boxes_width, colors=boxes_colors)
            
            if landmarks is not None:
                new_frame = draw_keypoints(new_frame, landmarks.reshape(landmarks.shape[0], -1, 2), radius=landmarks_radius, width=landmarks_width, colors=landmarks_colors)
        else:
            if landmarks is not None:
                new_frame = draw_keypoints(frame.squeeze(0), landmarks.reshape(landmarks.shape[0], -1, 2), radius=landmarks_radius, width=landmarks_width, colors=landmarks_colors)        
            else:
                new_frame = frame.squeeze(0)
        return transforms.ToPILImage()(new_frame.squeeze(0))
    else:
        if (boxes is not None) & (landmarks is None):
            new_frame = make_grid(torch.stack([draw_bounding_boxes(f, b.unsqueeze(0), width=boxes_width, colors=boxes_colors) for f,b in zip(frame.unbind(dim=0), boxes.unbind(dim=0))], dim=0))
        elif (landmarks is not None) & (boxes is None):
            new_frame = make_grid(torch.stack([draw_keypoints(f, l.unsqueeze(0), radius=landmarks_radius, width=landmarks_width, colors=landmarks_colors) for f,l in zip(frame.unbind(dim=0), landmarks.reshape(landmarks.shape[0], -1, 2).unbind(dim=0))], dim=0))
        elif (boxes is not None) & (landmarks is not None):
            new_frame = make_grid(torch.stack([draw_keypoints(fr, l.unsqueeze(0), radius=landmarks_radius, width=landmarks_width, colors=landmarks_colors) for fr,l in zip([draw_bounding_boxes(f, b.unsqueeze(0), width=boxes_width, colors=boxes_colors) for f,b in zip(frame.unbind(dim=0), boxes.unbind(dim=0))], 
                                                  landmarks.reshape(landmarks.shape[0], -1, 2).unbind(dim=0))]))
        else:
            new_frame = make_grid(frame)
        return transforms.ToPILImage()(new_frame)

def convert_bbox_output(img2pose_output):
    '''Convert im2pose_output into Fex Format'''
    
    widths = img2pose_output['boxes'][:, 2] - img2pose_output['boxes'][:, 0]  # right - left
    heights = img2pose_output['boxes'][:, 3] - img2pose_output['boxes'][:, 1] # bottom - top
    
    return torch.stack((img2pose_output['boxes'][:, 0], img2pose_output['boxes'][:, 1], widths, heights, img2pose_output['scores']), dim=1)

def extract_face_from_bbox_torch(frame, detected_faces, face_size=112, expand_bbox=1.2):
    """Extract face from image and resize using pytorch.

    Args:
        frame (torch.Tensor): Batch of images with shape (B, C, H, W).
        detected_faces (torch.Tensor): Tensor of shape (N, 4) representing bounding boxes [x1, y1, x2, y2].
        face_size (int): Output size to resize face after cropping.
        expand_bbox (float): Amount to expand bbox before cropping.

    Returns:
        cropped_faces (torch.Tensor): Tensor of extracted faces of shape (N, C, face_size, face_size).
        new_bboxes (torch.Tensor): Tensor of new bounding boxes with shape (N, 4).
    """
    B, C, H, W = frame.shape
    N = detected_faces.shape[0]

    # Extract the bounding box coordinates
    x1, y1, x2, y2 = detected_faces[:, 0], detected_faces[:, 1], detected_faces[:, 2], detected_faces[:, 3]
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
    yy, xx = torch.meshgrid(torch.arange(face_size, device=frame.device), torch.arange(face_size, device=frame.device))
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
    face_indices = torch.arange(N) % B  # Repeat for each batch element
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
    left = boxes[:, 0]   # (N,)
    top = boxes[:, 1]    # (N,)
    right = boxes[:, 2]  # (N,)
    bottom = boxes[:, 3] # (N,)

    # Calculate width and height of the bounding boxes
    width = right - left  # (N,)
    height = bottom - top # (N,)

    # Rescale the landmarks
    transformed_landmarks = torch.zeros_like(landmarks)
    transformed_landmarks[:, :, 0] = landmarks[:, :, 0] * width.unsqueeze(1) + left.unsqueeze(1)
    transformed_landmarks[:, :, 1] = landmarks[:, :, 1] * height.unsqueeze(1) + top.unsqueeze(1)

    return transformed_landmarks.reshape(N, 136)

def _batch_hog(frame, landmarks):
    """
    Helper function used in batch processing hog features

    Args:
        frames: a batch of frames
        landmarks: a list of list of detected landmarks

    Returns:
        hog_features: a numpy array of hog features for each detected landmark
        landmarks: updated landmarks
    """

    n_faces = landmarks.shape[0]
    batches = frame.shape[0]
    
    hog_features = []
    new_landmark_frames = []
    for i in range(batches):
        if len(landmarks) != 0:        
            new_landmarks_faces = []
            for j in range(n_faces):
                convex_hull, new_landmark = extract_face_from_landmarks(
                    frame=frame[i,:,:,:],
                    landmarks=landmarks[j, :],
                    face_size=112,
                )
        
                hog_features.append(
                    hog(
                        transforms.ToPILImage()(convex_hull[0] / 255.0),
                        orientations=8,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        visualize=False,
                        channel_axis=-1,
                    ).reshape(1, -1)
                )
        
                new_landmarks_faces.append(new_landmark)
            new_landmark_frames.append(new_landmarks_faces)
        else:
            hog_features.append(
                np.zeros((1, 5408))
            )  # LC: Need to confirm this size is fixed.
            new_landmark_frames.append([np.zeros((68, 2))])

    hog_features = np.concatenate(hog_features)

    return (hog_features, new_landmarks_faces)

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
    def __init__(self, device="cpu"):
        super(FastDetector, self).__init__()

        self.device = set_torch_device(device)

        # Load Model Configurations
        with open(os.path.join(get_resource_path(), 'model_config.json'), 'r') as file:
            self.model_configs = json.load(file)
        
        # Initialize img2pose
        backbone = resnet_fpn_backbone(backbone_name=f"resnet{self.model_configs['img2pose']['depth']}", weights=None)
        backbone.eval()
        backbone.to(self.device)
        self.facepose_detector = FasterDoFRCNN(backbone=backbone,
                                    num_classes=2,
                                    min_size=self.model_configs['img2pose']['min_size'],
                                    max_size=self.model_configs['img2pose']['max_size'],
                                    pose_mean=torch.tensor(self.model_configs['img2pose']['pose_mean']),
                                    pose_stddev=torch.tensor(self.model_configs['img2pose']['pose_stddev']),
                                    threed_68_points=torch.tensor(self.model_configs['img2pose']['threed_points']),
                                    rpn_pre_nms_top_n_test=self.model_configs['img2pose']['rpn_pre_nms_top_n_test'],
                                    rpn_post_nms_top_n_test=self.model_configs['img2pose']['rpn_post_nms_top_n_test'],
                                    bbox_x_factor=self.model_configs['img2pose']['bbox_x_factor'],
                                    bbox_y_factor=self.model_configs['img2pose']['bbox_y_factor'],
                                    expand_forehead=self.model_configs['img2pose']['expand_forehead'])
        facepose_model_file = hf_hub_download(repo_id= "py-feat/img2pose", filename="model.safetensors", cache_dir=get_resource_path())
        facepose_checkpoint = load_file(facepose_model_file)
        self.facepose_detector.load_state_dict(facepose_checkpoint)
        self.facepose_detector.eval()
        self.facepose_detector.to(self.device)

        # Initialize mobilefacenet
        self.landmark_detector = MobileFaceNet([112, 112], 136)
        self.landmark_detector.from_pretrained('py-feat/mobilefacenet', cache_dir=get_resource_path())
        self.landmark_detector.eval()
        self.landmark_detector.to(self.device)

        # Initialize xgb_au
        self.au_detector = XGBClassifier()
        au_weights = load_model_weights(model_type='au', model='xgb', location='huggingface')
        self.au_detector.load_weights(au_weights['scaler_upper'], au_weights['pca_model_upper'], au_weights['scaler_lower'], au_weights['pca_model_lower'], au_weights['scaler_full'], au_weights['pca_model_full'], au_weights['au_classifiers'])

        # Initialize resmasknet
        self.emotion_detector = ResMasking("", in_channels=3)
        self.emotion_detector.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 7))
        self.emotion_detector.from_pretrained("py-feat/resmasknet", cache_dir=get_resource_path())
        self.emotion_detector.eval()
        self.emotion_detector.to(self.device)

        # Initialize facenet
        self.identity_detector = InceptionResnetV1(
            pretrained=None,
            classify=False,
            num_classes=None,
            dropout_prob=0.6,
            device=self.device,
        )
        self.identity_detector.from_pretrained("py-feat/facenet", cache_dir=get_resource_path())
        self.identity_detector.eval()
        self.identity_detector.to(self.device)

    def forward(self, img, face_size=112):

        #########################
        # img2pose
        # Preprocess
        # frame = convert_image_to_tensor(img, img_type="float32") / 255.0
        frame = convert_image_to_tensor(img, img_type="float32").to(self.device) / 255.0

        # Forward
        img2pose_output = self.facepose_detector(frame)
        
        # Postprocess
        img2pose_output = postprocess_img2pose(img2pose_output[0])
        faceboxes = img2pose_output['boxes']
        poses = img2pose_output['dofs']
        facescores = img2pose_output['scores']
        
        #########################
        # mobilefacenet
        # Preprocess
        extracted_faces, new_bbox = extract_face_from_bbox_torch(frame, faceboxes, face_size=face_size)

        # Forward
        landmarks = self.landmark_detector.forward(extracted_faces)[0]

        # Postprocess
        new_landmarks = inverse_transform_landmarks_torch(landmarks.reshape(landmarks.shape[0], -1, 2), new_bbox)
        
        #########################
        # resmasknet
        # Preprocess
        resmasknet_faces, resmasknet_bbox = extract_face_from_bbox_torch(frame, img2pose_output['boxes'], expand_bbox=1.1, face_size = self.model_configs['Resmasknet']['image_size'])

        # Forward
        emotions = self.emotion_detector.forward(resmasknet_faces)
        
        # Postprocessing
        emotion_probabilities = torch.softmax(emotions, 1)

        #########################
        # facenet
        identity_embeddings = self.identity_detector.forward(extracted_faces)

        #########################
        # xgb_au
        # Preprocess
        frame_au = convert_image_to_tensor(img, img_type="float32")
        hog_features, new_landmarks = _batch_hog(frame=convert_image_to_tensor(frame_au, img_type="float32"), landmarks=landmarks)

        # Forward
        aus = self.au_detector.detect_au(frame=hog_features, landmarks=[new_landmarks])

        # print(f'img: {frame.shape}, facebox: {faceboxes.shape}, pose: {poses.shape}, facescore: {facescores.shape}, extracted_face: {extracted_faces.shape}, resmasknet_face:{resmasknet_faces.shape}, emotion:{emotions.shape}')

        # Create Fex Output Representation
        feat_faceboxes = pd.DataFrame(img2pose_output['boxes'].detach().numpy(), columns=FEAT_FACEBOX_COLUMNS[:-1])
        feat_faceboxes[FEAT_FACEBOX_COLUMNS[-1]] = img2pose_output['scores'].detach().numpy()
        feat_poses = pd.DataFrame(img2pose_output['dofs'].detach().numpy(), columns=FEAT_FACEPOSE_COLUMNS_3D)
        feat_landmarks = pd.DataFrame(landmarks.detach().numpy(), columns=openface_2d_landmark_columns)
        feat_aus = pd.DataFrame(aus, columns=AU_LANDMARK_MAP['Feat'])
        feat_emotions = pd.DataFrame(emotion_probabilities.detach().numpy(), columns=FEAT_EMOTION_COLUMNS)
        feat_identities = pd.DataFrame(identity_embeddings.detach().numpy(), columns=FEAT_IDENTITY_COLUMNS[1:])
        frame = 0
        
        return Fex(pd.concat([feat_faceboxes, feat_landmarks, feat_aus, feat_emotions, feat_identities], axis=1),
                au_columns=AU_LANDMARK_MAP['Feat'],
                emotion_columns=FEAT_EMOTION_COLUMNS,
                facebox_columns=FEAT_FACEBOX_COLUMNS,
                landmark_columns=openface_2d_landmark_columns,
                facepose_columns=FEAT_FACEPOSE_COLUMNS_3D,
                identity_columns=FEAT_IDENTITY_COLUMNS[1:],
                detector="Feat",
                face_model='img2pose',
                landmark_model='mobilefacenet',
                au_model='xgb_au',
                emotion_model='resmasknet',
                facepose_model='img2pose',
                identity_model='facenet')    
        
        
def run_detection():
    
    # Initialize Detector
    detector = FastDetector()

    # Define Parameters
    output_size=500
    batch_size=1
    num_workers=0
    pin_memory=False
    frame_counter=0
    face_detection_threshold=0.5
    face_identity_threshold=0.8
    
    # Load Data
    multi_face = os.path.join(get_test_data_path(), "multi_face.jpg")
    single_face = os.path.join(get_test_data_path(), "single_face.jpg")
    input_file_list = [multi_face]*50 + [single_face]*50
    input_file_list = [multi_face, single_face]
    
    data_loader = DataLoader(
        ImageDataset(
            input_file_list,
            output_size=output_size,
            preserve_aspect_ratio=True,
            padding=True,
        ),
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=False,
    )

    # Run Detector
    frame_id = 0
    batch_output = []
    for batch_id, batch_data in enumerate(tqdm(data_loader)):
        fex_data = detector.forward(batch_data['Image'])
        fex_data['input'] = batch_data['FileNames'][0]
        fex_data['frame'] = frame_id
        frame_id += 1
        batch_output.append(fex_data)
            #     faces = _inverse_face_transform(faces, batch_data)
            # landmarks = _inverse_landmark_transform(landmarks, batch_data)
    batch_output = pd.concat(batch_output)
    batch_output.compute_identities(threshold=0.8, inplace=True)

    return batch_output

if __name__ == "__main__":
    output = run_detection()
