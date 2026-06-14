"""
py-feat utility and helper functions for performing operations on images.
"""

import os
from .io import get_resource_path
import math
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import PILToTensor, Compose
import PIL
from feat.utils.geometry import (
    warp_affine,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    euler_from_quaternion,
)
from skimage.morphology.convex_hull import grid_points_in_poly
from feat.transforms import Rescale
from feat.utils import set_torch_device
from copy import deepcopy
from skimage import draw
import logging
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

__all__ = [
    "neutral",
    "registration",
    "convert68to49",
    "extract_face_from_landmarks",
    "extract_face_from_bbox",
    "convert68to49",
    "align_face",
    "BBox",
    "reverse_color_order",
    "expand_img_dimensions",
    "convert_image_to_tensor",
    "convert_color_vector_to_tensor",
    "mask_image",
    "convert_to_euler",
    "py_cpu_nms",
    "decode",
    "extract_face_from_bbox_torch",
    "inverse_transform_landmarks_torch",
    "extract_hog_features",
    "convert_bbox_output",
    "compute_original_image_size",
    "invert_padding_to_results",
]

# Neutral face coordinates
neutral = pd.read_csv(
    os.path.join(get_resource_path(), "neutral_face_coordinates.csv"), index_col=False
)


def registration(face_lms, neutral=neutral, method="fullface"):
    """Register faces to a neutral face.

    Affine registration of face landmarks to neutral face.

    Args:
        face_lms(array): face landmarks to register with shape (n,136). Columns 0~67 are x coordinates and 68~136 are y coordinates
        neutral(array): target neutral face array that face_lm will be registered
        method(str or list): If string, register to all landmarks ('fullface', default), or inner parts of face nose,mouth,eyes, and brows ('inner'). If list, pass landmarks to register to e.g. [27, 28, 29, 30, 36, 39, 42, 45]

    Return:
        registered_lms: registered landmarks in shape (n,136)
    """
    assert isinstance(face_lms, np.ndarray), TypeError("face_lms must be type np.ndarray")
    assert face_lms.ndim == 2, ValueError("face_lms must be shape (n, 136)")
    assert face_lms.shape[1] == 136, ValueError("Must have 136 landmarks")
    registered_lms = []
    for row in face_lms:
        face = [row[:68], row[68:]]
        face = np.array(face).T
        #   Rotate face
        primary = np.array(face)
        secondary = np.array(neutral)
        _ = primary.shape[0]
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        X1, Y1 = pad(primary), pad(secondary)
        if isinstance(method, str):
            if method == "fullface":
                A, res, rank, s = np.linalg.lstsq(X1, Y1, rcond=None)
            elif method == "inner":
                A, res, rank, s = np.linalg.lstsq(X1[17:, :], Y1[17:, :], rcond=None)
            else:
                raise ValueError("method is either 'fullface' or 'inner'")
        elif isinstance(method, list):
            A, res, rank, s = np.linalg.lstsq(X1[method], Y1[method], rcond=None)
        else:
            raise TypeError("method is string ('fullface','inner') or list of landmarks")
        transform = lambda x: unpad(np.dot(pad(x), A))
        registered_lms.append(transform(primary).T.reshape(1, 136).ravel())
    return np.array(registered_lms)


def extract_face_from_landmarks(frame, landmarks, face_size=112):
    """Extract a face from a frame using a convex hull around its landmarks.

    Aligns the face by 68-landmark transform, masks pixels outside the
    convex hull of the landmarks, and returns the cropped face.

    Args:
        frame (torch.Tensor): image tensor of shape `[C, H, W]` or `[B, C, H, W]`.
        landmarks (torch.Tensor): 68 landmark coordinates as a flat or
            `(68, 2)` tensor on any device (moved to CPU/numpy internally).
        face_size (int): output crop size in pixels (default 112).

    Returns:
        masked_image: aligned cropped face with non-face pixels masked out.
        new_landmarks: landmark coordinates in the aligned crop's frame.
    """

    if not isinstance(frame, torch.Tensor):
        raise ValueError(f"image must be a tensor not {type(frame)}")

    if len(frame.shape) != 4:
        frame = frame.unsqueeze(0)

    landmarks = landmarks.detach().cpu().numpy().copy()

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


def extract_face_from_bbox(frame, detected_faces, face_size=112, expand_bbox=1.2):
    """Extract face from image and resize

    Args:
        frame (torch.tensor): img with faces
        detected_faces (list): list of lists of face bounding boxes from detect_face()
        face_size (int): output size to resize face after cropping
        expand_bbox (float): amount to expand bbox before cropping

    Returns:
        cropped_face (torch.Tensor): Tensor of extracted faces of shape=face_size
        new_bbox (list): list of new bounding boxes that correspond to cropped face
    """

    length_index = [len(ama) for ama in detected_faces]
    length_cumu = np.cumsum(length_index)

    flat_faces = [
        item for sublist in detected_faces for item in sublist
    ]  # Flatten the faces

    im_height, im_width = frame.shape[-2:]

    bbox_list = []
    cropped_faces = []
    for k, face in enumerate(flat_faces):
        frame_assignment = np.where(k < length_cumu)[0][0]  # which frame is it?
        bbox = BBox(
            face[:-1], bottom_boundary=im_height, right_boundary=im_width
        ).expand_by_factor(expand_bbox)
        cropped = bbox.extract_from_image(frame[frame_assignment])
        logging.info(
            f"RESCALING WARNING: image_operations.extract_face_from_bbox() is rescaling cropped img with shape {cropped.shape} to {face_size}"
        )
        transform = Compose(
            [Rescale(output_size=face_size, preserve_aspect_ratio=True, padding=True)]
        )
        cropped_faces.append(transform(cropped))
        bbox_list.append(bbox)

        faces = torch.cat(
            tuple([convert_image_to_tensor(x["Image"]) for x in cropped_faces]), 0
        )

    return (faces, bbox_list)


def convert68to49(landmarks):
    """Convert landmark from 68 to 49 points

    Function modified from https://github.com/D-X-Y/landmark-detection/blob/7bc7a5dbdbda314653124a4596f3feaf071e8589/SAN/lib/datasets/dataset_utils.py#L169 to fit pytorch tensors. Converts 68 point landmarks to 49 point landmarks

    Args:
        landmarks: landmark points of shape (2,68)

    Return:
        converted landmarks: converted 49 landmark points of shape (2,49)
    """

    if landmarks.shape != (68, 2):
        if landmarks.shape[::-1] == (68, 2):
            landmarks = landmarks.shape[::-1]
        else:
            raise ValueError("landmarks should be a numpy array of (68,2)")

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.clone()
        out = torch.ones((68,), dtype=torch.bool)
    elif isinstance(landmarks, (np.ndarray, tuple)):
        landmarks = landmarks.copy()
        out = np.ones((68,)).astype("bool")
    else:
        raise ValueError(
            f"landmarks should be a numpy array or torch.Tensor not {type(landmarks)}"
        )

    out[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 60, 64]] = False

    return landmarks[out]


def align_face(img, landmarks, landmark_type=68, box_enlarge=2.5, img_size=112):
    """Performs affine transformation to align the images by eyes.

    Performs affine alignment based on eyes.

    Args:
        img: gray or RGB
        landmark_type (int): Landmark system (68, 49)
        landmarks: 68 system flattened landmarks, shape:(136)
        box_enlarge: relative size of face on the image. Smaller value indicate larger proportion
        img_size = output image size

    Returns:
        aligned_img: aligned image
        new_landmarks: aligned landmarks
    """

    if landmark_type == 68:
        left_eye0 = (
            float(
                landmarks[2 * 36]
                + landmarks[2 * 37]
                + landmarks[2 * 38]
                + landmarks[2 * 39]
                + landmarks[2 * 40]
                + landmarks[2 * 41]
            )
            / 6.0
        )
        left_eye1 = (
            float(
                landmarks[2 * 36 + 1]
                + landmarks[2 * 37 + 1]
                + landmarks[2 * 38 + 1]
                + landmarks[2 * 39 + 1]
                + landmarks[2 * 40 + 1]
                + landmarks[2 * 41 + 1]
            )
            / 6.0
        )
        right_eye0 = (
            float(
                landmarks[2 * 42]
                + landmarks[2 * 43]
                + landmarks[2 * 44]
                + landmarks[2 * 45]
                + landmarks[2 * 46]
                + landmarks[2 * 47]
            )
            / 6.0
        )
        right_eye1 = (
            float(
                landmarks[2 * 42 + 1]
                + landmarks[2 * 43 + 1]
                + landmarks[2 * 44 + 1]
                + landmarks[2 * 45 + 1]
                + landmarks[2 * 46 + 1]
                + landmarks[2 * 47 + 1]
            )
            / 6.0
        )

        mat2 = np.array(
            [
                [left_eye0, left_eye1, 1],
                [right_eye0, right_eye1, 1],
                [float(landmarks[2 * 30]), float(landmarks[2 * 30 + 1]), 1.0],
                [float(landmarks[2 * 48]), float(landmarks[2 * 48 + 1]), 1.0],
                [float(landmarks[2 * 54]), float(landmarks[2 * 54 + 1]), 1.0],
            ],
            dtype=float,
        )
    elif landmark_type == 49:
        left_eye0 = (
            float(
                landmarks[2 * 19]
                + landmarks[2 * 20]
                + landmarks[2 * 21]
                + landmarks[2 * 22]
                + landmarks[2 * 23]
                + landmarks[2 * 24]
            )
            / 6.0
        )
        left_eye1 = (
            float(
                landmarks[2 * 19 + 1]
                + landmarks[2 * 20 + 1]
                + landmarks[2 * 21 + 1]
                + landmarks[2 * 22 + 1]
                + landmarks[2 * 23 + 1]
                + landmarks[2 * 24 + 1]
            )
            / 6.0
        )
        right_eye0 = (
            float(
                landmarks[2 * 25]
                + landmarks[2 * 26]
                + landmarks[2 * 27]
                + landmarks[2 * 28]
                + landmarks[2 * 29]
                + landmarks[2 * 30]
            )
            / 6.0
        )
        right_eye1 = (
            float(
                landmarks[2 * 25 + 1]
                + landmarks[2 * 26 + 1]
                + landmarks[2 * 27 + 1]
                + landmarks[2 * 28 + 1]
                + landmarks[2 * 29 + 1]
                + landmarks[2 * 30 + 1]
            )
            / 6.0
        )

        mat2 = np.array(
            [
                [left_eye0, left_eye1, 1],
                [right_eye0, right_eye1, 1],
                [float(landmarks[2 * 13]), float(landmarks[2 * 13 + 1]), 1.0],
                [float(landmarks[2 * 31]), float(landmarks[2 * 31 + 1]), 1.0],
                [float(landmarks[2 * 37]), float(landmarks[2 * 37 + 1]), 1.0],
            ],
            dtype=float,
        )
    else:
        raise ValueError("landmark_type must be (68,49).")

    delta_x = right_eye0 - left_eye0
    delta_y = right_eye1 - left_eye1

    l = math.sqrt(delta_x**2 + delta_y**2)
    sin_val = delta_y / l
    cos_val = delta_x / l
    mat1 = np.array(
        [[cos_val, sin_val, 0.0], [-sin_val, cos_val, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    mat2 = (mat1 @ mat2.T).T

    center_x = (mat2[:, 0].max() + mat2[:, 0].min()) / 2.0
    center_y = (mat2[:, 1].max() + mat2[:, 1].min()) / 2.0

    if (mat2[:, 0].max() - mat2[:, 0].min()) > (mat2[:, 1].max() - mat2[:, 1].min()):
        half_size = 0.5 * box_enlarge * (mat2[:, 0].max() - mat2[:, 0].min())
    else:
        half_size = 0.5 * box_enlarge * (mat2[:, 1].max() - mat2[:, 1].min())

    scale = (img_size - 1) / 2.0 / half_size

    mat3 = np.array(
        [
            [scale, 0.0, scale * (half_size - center_x)],
            [0.0, scale, scale * (half_size - center_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    mat = mat3 @ mat1
    affine_matrix = torch.tensor(mat[0:2, :]).type(torch.float32).unsqueeze(0)

    # warp_affine expects [batch, channel, height, width]
    if img.ndim == 3:
        img = img[None, :]

    # warp_affine internally builds a sampling grid that must live on
    # the same device as the input. When img is on mps/cuda this raises
    # grid_sampler device mismatch unless the matrix is moved too.
    affine_matrix = affine_matrix.to(img.device)

    aligned_img = warp_affine(
        img,
        affine_matrix,
        (img_size, img_size),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
        fill_value=(128, 128, 128),
    )

    land_3d = np.ones((len(landmarks) // 2, 3))
    land_3d[:, 0:2] = np.reshape(np.array(landmarks), (len(landmarks) // 2, 2))
    new_landmarks = (mat @ land_3d.T).T
    new_landmarks = np.array(list(zip(new_landmarks[:, 0], new_landmarks[:, 1]))).astype(
        int
    )

    return (aligned_img, new_landmarks)


class BBox(object):
    def __init__(
        self,
        bbox,
        order=None,
        left_boundary=0,
        top_boundary=0,
        right_boundary=None,
        bottom_boundary=None,
    ):
        """Class to work with Bounding Box

        Args:
            bbox: (list): values
            order (list): order of values (e.g., ['left', 'top', 'right', 'bottom'])
            left optional (float): boundary of left (default 0)
            right toptional (float): boundary of right border (e.g., width of image)
            top optional (float): boundary of top border (default 0)
            bottom optional(float): boundary of right border (e.g., height of image)

        """
        if order is None:
            self.order = ["left", "top", "right", "bottom"]
        else:
            if not isinstance(order, list):
                raise ValueError("order must be a list")
            self.order = [x.lower() for x in order]

        if len(bbox) != 4:
            raise ValueError("bbox must contain 4 values")

        self.left = bbox[self.order.index("left")]
        self.right = bbox[self.order.index("right")]
        self.top = bbox[self.order.index("top")]
        self.bottom = bbox[self.order.index("bottom")]
        self.center_x = (self.right + self.left) // 2
        self.center_y = (self.top + self.bottom) // 2
        self.width = self.right - self.left
        self.height = self.bottom - self.top

        self = self.set_boundary(
            left=left_boundary,
            right=right_boundary,
            top=top_boundary,
            bottom=bottom_boundary,
            apply_boundary=True,
        )

    def __repr__(self):
        return f"'height': {self.height}, 'width': {self.width}"

    def __mul__(self, bbox2):
        """Create a new BBox based on the intersection between two BBox instances (AND operation)"""

        if isinstance(bbox2, (BBox)):
            return BBox(
                [
                    np.max([self.left, bbox2.left]),
                    np.max([self.top, bbox2.top]),
                    np.min([self.right, bbox2.right]),
                    np.min([self.bottom, bbox2.bottom]),
                ]
            )
        else:
            raise NotImplementedError(
                "Multiplication is currently only supported between two BBox instances"
            )

    def __add__(self, bbox2):
        """Create a new BBox based on the intersection between two BBox instances (OR Operation)"""

        if isinstance(bbox2, (BBox)):
            return BBox(
                [
                    np.min([self.left, bbox2.left]),
                    np.min([self.top, bbox2.top]),
                    np.max([self.right, bbox2.right]),
                    np.max([self.bottom, bbox2.bottom]),
                ]
            )
        else:
            raise NotImplementedError(
                "Addition is currently only supported between two BBox instances"
            )

    def expand_by_factor(self, factor, symmetric=True):
        """Expand box by factor

        Args:
            factor (float): factor to expand.
            symmetric (bool): if symmetric then expand equally based on largest side
        """

        if symmetric:
            new_size = max([self.width, self.height]) * factor
            self.width = new_size
            self.height = new_size

        else:
            self.width *= factor
            self.height *= factor

        self.left = self.center_x - (self.width // 2)
        self.right = self.center_x + (self.width // 2)
        self.top = self.center_y - (self.height // 2)
        self.bottom = self.center_y + (self.height // 2)

        self._apply_boundary()

        return self

    def set_boundary(self, left=0, right=None, top=0, bottom=None, apply_boundary=True):
        """Set maximum boundary of bounding box such as the edge of the original image

        Use _apply_boundary() method to update the bounding box

        Args:
            left (float): boundary of left (default 0)
            right (float): boundary of right border (e.g., width of image)
            top (float): boundary of top border (default 0)
            bottom (float): boundary of right border (e.g., height of image)
            apply (bool): apply boundary to BBox

        """

        left = max(left, 0)
        top = max(top, 0)

        (
            self.left_boundary,
            self.top_boundary,
            self.right_boundary,
            self.bottom_boundary,
        ) = (left, top, right, bottom)

        if apply_boundary:
            self._apply_boundary()
        return self

    def _apply_boundary(self):
        """Helper function to apply stored boundaries to BBox

        Currently does not update stored width/height or center values
        """

        if self.left_boundary is not None:
            if self.left_boundary > self.left:
                self.left = self.left_boundary

        if self.right_boundary is not None:
            if self.right_boundary < self.right:
                self.right = self.right_boundary

        if self.top_boundary is not None:
            if self.top_boundary > self.top:
                self.top = self.top_boundary

        if self.bottom_boundary is not None:
            if self.bottom_boundary < self.bottom:
                self.bottom = self.bottom_boundary

        return

    def extract_from_image(self, img):
        """Crop Image using Bounding Box

        Args:
            img (np.array, torch.tensor): image (B, C, H, W) or (C, H, W) or (H,W)

        Returns:
            cropped (np.array, torch.tensor)"""

        if not isinstance(img, (np.ndarray, torch.Tensor)):
            raise ValueError("images must be (np.array, torch.tensor)")

        if len(img.shape) == 2:
            return img[int(self.top) : int(self.bottom), int(self.left) : int(self.right)]
        elif len(img.shape) == 3:
            return img[
                :, int(self.top) : int(self.bottom), int(self.left) : int(self.right)
            ]
        elif len(img.shape) == 4:
            return img[
                :, :, int(self.top) : int(self.bottom), int(self.left) : int(self.right)
            ]
        else:
            raise ValueError("Not a valid image size")

    def to_dict(self):
        """bounding box coordinates as a dictionary"""
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }

    def to_list(self):
        """Output bounding box coordinates to list"""
        return [self.to_dict()[x] for x in self.order]

    def transform_landmark(self, landmark):
        """Scale Landmarks to be within a 1 unit box (e.g., [0,1])

        based on https://github.com/cunjian/pytorch_face_landmark/

        Args:

        Returns:
            scaled landmarks
        """

        landmark_ = np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            landmark_[i] = (
                (point[0] - self.left) / self.width,
                (point[1] - self.top) / self.height,
            )
        return landmark_

    def inverse_transform_landmark(self, landmark):
        """Re-scale landmarks from unit scaling back into BBox

        based on  https://github.com/cunjian/pytorch_face_landmark/

        Args:
            landmarks: (np.array): landmarks

        Returns:
            re-scaled landmarks
        """

        landmark_ = np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            x = point[0] * self.width + self.left
            y = point[1] * self.height + self.top
            landmark_[i] = (x, y)
        return landmark_

    def area(self):
        """Compute the area of the bounding box"""
        return self.height * self.width

    def overlap(self, bbox2):
        """Compute the percent overlap between BBox with another BBox"""
        overlap_bbox = self * bbox2
        if (overlap_bbox.height < 0) or (overlap_bbox.width < 0):
            return 0
        else:
            return (self * bbox2).area() / self.area()

    def plot(self, ax=None, fill=False, linewidth=2, **kwargs):
        """Plot bounding box

        Args:
            ax: matplotlib axis
            fill (bool): fill rectangle
        """

        if ax is None:
            fig, ax = plt.subplots()
            ax.plot()

        ax.add_patch(
            Rectangle(
                (self.left, self.top),
                self.width,
                self.height,
                fill=fill,
                linewidth=linewidth,
                **kwargs,
            )
        )
        return ax


def reverse_color_order(img):
    """Convert BGR OpenCV image to RGB format"""

    if not isinstance(img, (np.ndarray)):
        raise ValueError(f"Image must be a numpy array, not a {type(img)}")

    if len(img.shape) != 3:
        raise ValueError(
            f"Image must be a 3D numpy array (Height, Width, Color), currently {img.shape}"
        )
    return img[:, :, [2, 1, 0]]


def expand_img_dimensions(img):
    """Expand image dimensions to 4 dimensions"""

    if img.ndim == 4:
        return img
    elif img.ndim == 3:
        return np.expand_dims(img, 0)
    else:
        raise ValueError(
            f"Image with {img.ndim} not currently supported (must be 3 or 4)"
        )


def convert_image_to_tensor(img, img_type=None):
    """Convert Image data (PIL, cv2, TV) to Tensor"""

    if isinstance(img, (np.ndarray)):  # numpy array
        img = torch.from_numpy(
            expand_img_dimensions(reverse_color_order(img)).transpose(0, 3, 1, 2)
        )
    elif isinstance(img, PIL.Image.Image):
        transform = Compose([PILToTensor()])
        img = transform(img)
        img = img.expand(1, -1, -1, -1)
    elif isinstance(img, torch.Tensor):
        if len(img.shape) == 3:
            img = img.expand(1, -1, -1, -1)
    else:
        raise ValueError(
            f"{type(img)} is not currently supported please use CV2, PIL, or TorchVision to load image"
        )

    if img_type is not None:
        torch_types = [
            "int",
            "int8",
            "int16",
            "int32",
            "int16",
            "float",
            "float16",
            "float32",
            "float64",
        ]
        if img_type not in torch_types:
            raise ValueError(
                f"img_type {img_type} is not supported, please try {torch_types}"
            )
        img = img.type(eval(f"torch.{img_type}"))

    return img


def convert_color_vector_to_tensor(vector):
    """Convert a color vector into a tensor (1,3,1,1)"""
    return torch.from_numpy(vector).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def mask_image(img, mask):
    """Apply numpy mask of (h,w) to pytorch image (b,c,h,w)"""
    # if ~isinstance(img, torch.Tensor) & ~isinstance(mask, np.ndarray):
    #     raise ValueError(
    #         f"img must be pytorch tensor, not {type(img)} and mask must be np array not {type(mask)}"
    #     )
    mask_t = torch.tensor(mask).to(torch.float32).to(img.device)
    return torch.sgn(mask_t).unsqueeze(0).unsqueeze(0) * img


def convert_to_euler(rotvec, is_rotvec=True):
    """
    Converts the rotation vector or matrix (the standard output for head pose models) into euler angles in the form
    of a ([pitch, roll, yaw]) vector. Adapted from https://github.com/vitoralbiero/img2pose.

    Args:
        rotvec: The rotation vector produced by the headpose model
        is_rotvec:

    Returns:
        np.ndarray: euler angles ([pitch, roll, yaw])
    """
    if is_rotvec:
        rotvec = Rotation.from_rotvec(rotvec).as_matrix()
    rot_mat_2 = np.transpose(rotvec)
    angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
    return [angle[0], -angle[2], -angle[1]]  # pitch, roll, yaw


def rotvec_to_euler_angles(rotation_vector):
    """
    Convert a rotation vector to Euler angles using Kornia in 'xyz'

    Args:
        rotation_vector (torch.Tensor): Tensor of shape (N, 3) representing the rotation vectors.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) representing the Euler angles.
    """

    if rotation_vector.dim() == 1:
        rotation_vector = rotation_vector.unsqueeze(0)
    rotation_matrix = axis_angle_to_rotation_matrix(rotation_vector)
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    euler_angles = euler_from_quaternion(
        quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    )
    return torch.stack(euler_angles, dim=-1)


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline

    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------

    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.

    Adapted from https://github.com/Hakuyume/chainer-ssd

    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes

    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class HOGLayer(torch.nn.Module):
    def __init__(
        self,
        orientations=10,
        pixels_per_cell=8,
        cells_per_block=2,
        transform_sqrt=False,
        block_normalization="L2",
        feature_vector=True,
        device="auto",
    ):
        """PyTorch HOG feature extractor matching skimage.feature.hog output.

        Args:
            orientations (int): Number of orientation bins.
            pixels_per_cell (int): Size in pixels of a square cell. skimage's
                ``pixels_per_cell`` tuple is collapsed to a scalar here since
                we only support square cells.
            cells_per_block (int): Block side length in cells (square blocks).
            transform_sqrt (bool): Apply power-law compression (per-channel
                ``sqrt``) before computing gradients. Image values must be
                non-negative if enabled.
            block_normalization (str): One of ``"L1"``, ``"L1-sqrt"``, ``"L2"``,
                ``"L2-Hys"``. Matches the skimage option of the same name.
            feature_vector (bool): If True, flatten the output in skimage's
                ``(blocks_row, blocks_col, b_row, b_col, orientations)`` order;
                if False, return the unflattened array in the same layout.
            device (str): one of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``.
        """
        super().__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.transform_sqrt = transform_sqrt
        self.device = set_torch_device(device)
        self.feature_vector = feature_vector
        self.isfit = False

        if block_normalization is not None:
            self.block_normalization = block_normalization.lower()
        else:
            self.block_normalization = block_normalization

        # Centered finite-difference kernels matching skimage._hog._hog_channel_gradient:
        #   g_col[r, c] = I[r, c+1] - I[r, c-1]   (gx)
        #   g_row[r, c] = I[r+1, c] - I[r-1, c]   (gy)
        # F.conv2d implements cross-correlation, so for output[r, c] = sum_k k*input[r+offset_k]
        # we need kernel weights [-1, 0, 1] to produce input[+1] - input[-1].
        gx_kernel = torch.tensor(
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        )
        gy_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        weight = torch.stack((gx_kernel, gy_kernel), dim=0).unsqueeze(1)  # [2, 1, 3, 3]
        self.register_buffer("weight", weight)
        # skimage averages magnitudes over the cell (cell_hog returns total / (cell_rows*cell_columns)).
        self.cell_pooler = nn.AvgPool2d(
            pixels_per_cell,
            stride=pixels_per_cell,
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
        )

    def forward(self, img):
        with torch.no_grad():
            img = img.to(self.device)

            if self.transform_sqrt:
                img = img.sqrt()

            n, c, h, w = img.shape

            # Per-channel gradients with per-pixel max-magnitude channel selection.
            # Mirrors skimage._hog with channel_axis=-1: compute (gx, gy) for each
            # channel, then for each pixel pick the channel whose gradient magnitude
            # is largest and use its (gx, gy).
            img_flat = img.reshape(n * c, 1, h, w)
            gxy_flat = F.conv2d(
                img_flat,
                self.weight,
                bias=None,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
            )
            _, _, hp, wp = gxy_flat.shape
            gxy = gxy_flat.reshape(n, c, 2, hp, wp)

            # Zero out gradient at edges to match skimage:
            #   g_row[0, :] = g_row[-1, :] = 0  (gy zero in top/bottom rows)
            #   g_col[:, 0] = g_col[:, -1] = 0  (gx zero in left/right cols)
            gxy[:, :, 0, :, 0] = 0  # gx, leftmost col
            gxy[:, :, 0, :, -1] = 0  # gx, rightmost col
            gxy[:, :, 1, 0, :] = 0  # gy, top row
            gxy[:, :, 1, -1, :] = 0  # gy, bottom row

            if c == 1:
                gx = gxy[:, 0, 0, :, :]
                gy = gxy[:, 0, 1, :, :]
            else:
                # Per-pixel: pick the channel with the largest gradient magnitude.
                mag_per_chan = (gxy[:, :, 0] ** 2 + gxy[:, :, 1] ** 2).sqrt()
                best_chan = mag_per_chan.argmax(dim=1, keepdim=True)  # [N, 1, H, W]
                # Gather (gx, gy) at the chosen channel for each pixel.
                idx = best_chan.unsqueeze(2).expand(-1, -1, 2, -1, -1)
                gxy = gxy.gather(1, idx).squeeze(1)  # [N, 2, H, W]
                gx = gxy[:, 0]
                gy = gxy[:, 1]

            mag = (gx ** 2 + gy ** 2).sqrt()  # [N, H, W]

            # Orientation in degrees, wrapped to [0, 180).
            # skimage: rad2deg(arctan2(g_rows, g_cols)) % 180
            phase_deg = torch.rad2deg(torch.atan2(gy, gx)) % 180.0

            # Hard binning (skimage uses a half-open interval per bin, no linear interp).
            # bin_idx = floor(phase_deg / (180 / orientations)), clamped to [0, orientations-1].
            bin_width_deg = 180.0 / self.orientations
            bin_idx = (phase_deg / bin_width_deg).long().clamp(0, self.orientations - 1)

            # Per-pixel orientation histogram: scatter magnitude into the chosen bin.
            out = torch.zeros(
                (n, self.orientations, hp, wp), dtype=mag.dtype, device=mag.device
            )
            out.scatter_add_(1, bin_idx.unsqueeze(1), mag.unsqueeze(1))

            # Cell aggregation. skimage's cell_hog returns total / (cell_rows*cell_cols),
            # which is exactly what AvgPool2d computes.
            out = self.cell_pooler(out)
            self.orientation_histogram = deepcopy(out)
            self.isfit = True
            self.img_shape = img.shape

            # Block normalization. Block shape after unfold is
            # (cells_per_block, cells_per_block) along dims 4 and 5; orientation is dim 1.
            # The norm sums over (orientation, block_row, block_col) per block.
            if self.block_normalization is not None:
                eps = 1e-5
                out = out.unfold(2, self.cells_per_block, 1).unfold(
                    3, self.cells_per_block, 1
                )
                # out shape: [N, orientations, n_blocks_row, n_blocks_col, b_row, b_col]
                if self.block_normalization == "l1":
                    norm = out.abs().sum(dim=(1, 4, 5), keepdim=True) + eps
                    out = out / norm
                elif self.block_normalization == "l1-sqrt":
                    norm = out.abs().sum(dim=(1, 4, 5), keepdim=True) + eps
                    out = (out / norm).sqrt()
                elif self.block_normalization == "l2":
                    norm = (out.pow(2).sum(dim=(1, 4, 5), keepdim=True) + eps ** 2).sqrt()
                    out = out / norm
                elif self.block_normalization == "l2-hys":
                    norm = (out.pow(2).sum(dim=(1, 4, 5), keepdim=True) + eps ** 2).sqrt()
                    out = out / norm
                    out = out.clamp(max=0.2)
                    norm = (out.pow(2).sum(dim=(1, 4, 5), keepdim=True) + eps ** 2).sqrt()
                    out = out / norm
                else:
                    raise ValueError(
                        'Selected block normalization method is invalid. '
                        'Use ["l1", "l1-sqrt", "l2", "l2-hys"].'
                    )

            # Permute to skimage's layout regardless of `feature_vector`:
            # (batch, blocks_row, blocks_col, b_row, b_col, orientations).
            # The flat output (`feature_vector=True`) is then a row-major
            # flatten of skimage's `normalized_blocks`. The unflattened
            # output preserves the same axis order so callers see the same
            # tensor shape skimage's `feature_vector=False` mode produces.
            out = out.permute(0, 2, 3, 4, 5, 1).contiguous()
            if self.feature_vector:
                return out.flatten(start_dim=1)
            return out

    def plot(self):
        """Visualize the hog feature representation. Creates numpy matrix for each image.

        Based on skimage.feature._hog
        """
        if not self.isfit:
            raise ValueError(
                "HOG Feature Extractor has not been run yet. Nothing to plot."
            )

        n_batch, _, s_row, s_col = self.img_shape
        c_row, c_col = [self.pixels_per_cell] * 2
        n_cells_row = int(s_row // c_row)
        n_cells_col = int(s_col // c_col)

        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(self.orientations)
        orientation_bin_midpoints = np.pi * (orientations_arr + 0.5) / self.orientations

        # sin/cos appear to be flipped compared to skimage.feature.hog
        dr_arr = radius * np.cos(orientation_bin_midpoints)
        dc_arr = radius * np.sin(orientation_bin_midpoints)
        hog_image = np.zeros((n_batch, s_row, s_col), dtype=float)
        for i in range(n_batch):
            for r in range(n_cells_row):
                for c in range(n_cells_col):
                    for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                        center = tuple([r * c_row + c_row // 2, c * c_col + c_col // 2])
                        rr, cc = draw.line(
                            int(center[0] - dc),
                            int(center[1] + dr),
                            int(center[0] + dc),
                            int(center[1] - dr),
                        )
                        hog_image[i, rr, cc] += self.orientation_histogram[
                            i, o, r, c
                        ].numpy()
        return hog_image


def extract_face_from_bbox_torch(
    frame, detected_faces, face_size=112, expand_bbox=1.2, frame_idx=None
):
    """Extract face from image and resize using pytorch.

    Args:
        frame: ``[B, C, H, W]`` tensor of source frames.
        detected_faces: ``[N, 4]`` tensor of bboxes in ``[x1, y1, x2, y2]``
            format. ``N`` need not equal ``B`` — multiple faces per frame
            are supported via ``frame_idx``.
        face_size: output spatial size; crops are returned at
            ``[N, C, face_size, face_size]``.
        expand_bbox: multiplier on bbox width/height before clipping to
            the source frame; lets the crop carry context around the face.
        frame_idx: optional ``[N]`` long tensor mapping each face to the
            frame it came from. Required whenever ``B > 1`` and faces
            aren't striped one-per-frame across the batch. When ``None``,
            falls back to ``arange(N) % B`` for backwards compatibility
            with the legacy single-frame call sites (``B == 1``).

    Returns:
        cropped_faces: ``[N, C, face_size, face_size]`` tensor.
        new_bboxes: ``[N, 4]`` clipped/expanded bboxes (long).
    """

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

    # Cast the bounding box coordinates to long for indexing. Round (not
    # truncate): plain .long() floors, so a sub-pixel-jittering box snaps
    # between e.g. 213 and 214 across frames, adding a ±1px step to the
    # crop + mesh transform. Rounding removes that discrete flicker.
    new_bboxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1).round().long()

    # Create a mesh grid for the face size
    yy, xx = torch.meshgrid(
        torch.arange(face_size, device=device),
        torch.arange(face_size, device=device),
        indexing="ij",
    )
    yy = yy.float()
    xx = xx.float()

    # Calculate the normalized coordinates for the grid sampling
    grid_x = (xx + 0.5) / face_size * (new_x2 - new_x1).view(N, 1, 1) + new_x1.view(
        N, 1, 1
    )
    grid_y = (yy + 0.5) / face_size * (new_y2 - new_y1).view(N, 1, 1) + new_y1.view(
        N, 1, 1
    )

    # Normalize grid coordinates to the range [-1, 1]
    grid_x = 2 * grid_x / (W - 1) - 1
    grid_y = 2 * grid_y / (H - 1) - 1

    # Stack grid coordinates and reshape
    grid = torch.stack((grid_x, grid_y), dim=-1)  # Shape: (N, face_size, face_size, 2)

    # Ensure frame and grid are float32 for grid_sample
    frame = frame.float()
    grid = grid.float()

    # Map each face to its source frame. Callers with multi-frame batches
    # and variable face counts pass an explicit `frame_idx`; legacy
    # single-frame callers (B == 1) use the default `arange(N) % B`.
    if frame_idx is None:
        face_indices = torch.arange(N, device=device) % B
    else:
        face_indices = frame_idx.to(device=device, dtype=torch.long)
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
        boxes (torch.Tensor): Tensor of shape (N, 4) representing bounding boxes [x1, y1, x2, y2] for N samples.

    Returns:
        torch.Tensor: Transformed landmarks of shape (N, 136).
    """
    # Ensure both tensors are on the same device
    device = landmarks.device
    boxes = boxes.to(device)

    N, N_landmarks = landmarks.shape

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

    return transformed_landmarks.reshape(N, N_landmarks)


def procrustes_align_2d_batched(coords, anchor_idx, ref_anchors):
    """Batched 2D Umeyama similarity alignment to a fixed reference frame.

    For each face, finds the rigid + isotropic-scale transform that best maps
    its anchor subset to ``ref_anchors`` (least-squares), then applies that
    transform to all of the face's landmarks. Mirrors the helper used at
    py-feat training time so inference and training canonicalize landmarks
    in the same frame.

    Used by the dlib-68 → MP-478 bridge in ``feat.plotting`` and any other
    consumer that needs to align landmarks to a saved reference (e.g.,
    cross-detector landmark normalization). Numpy / float64 internally for
    SVD stability; output cast to float32.

    Args:
        coords: ``(n, K, 2)`` raw 2-D landmarks per face. ``K`` is whatever
            number of points the caller carries (e.g., 68 for dlib).
        anchor_idx: ``(k,)`` int indices selecting stable anchor points
            from each face's K landmarks.
        ref_anchors: ``(k, 2)`` reference anchor positions in the target
            frame (e.g., a population-mean canonical pose).

    Returns:
        ``(n, K, 2)`` aligned landmarks in the reference frame. Same dtype
        as the input (after promotion to float32).

    Raises:
        ValueError: if ``coords`` is not 3-D with last-dim 2, or if
            ``anchor_idx`` does not match ``ref_anchors`` length.
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(
            f"coords must have shape (n, K, 2); got {coords.shape}"
        )
    anchor_idx = np.asarray(anchor_idx, dtype=np.int64)
    ref_anchors = np.asarray(ref_anchors, dtype=np.float32)
    if ref_anchors.shape != (anchor_idx.shape[0], 2):
        raise ValueError(
            f"ref_anchors shape {ref_anchors.shape} must be "
            f"({anchor_idx.shape[0]}, 2) to match anchor_idx length."
        )
    N = coords.shape[0]
    P = coords[:, anchor_idx]  # (N, k, 2)
    P_mean = P.mean(axis=1, keepdims=True)
    Q_mean = ref_anchors.mean(axis=0)
    P_c = P - P_mean
    Q_c = ref_anchors - Q_mean
    # SVD in float64 for numerical stability; small (2x2) so cheap regardless.
    H = np.einsum("ki,nkj->nij", Q_c.astype(np.float64), P_c.astype(np.float64))
    U, S, Vt = np.linalg.svd(H)
    UVt = U @ Vt
    det = np.linalg.det(UVt)
    d = np.where(det < 0, -1.0, 1.0)
    D = np.zeros((N, 2, 2), dtype=np.float64)
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = d
    R = U @ D @ Vt
    var_P = (P_c.astype(np.float64) ** 2).sum(axis=(1, 2))
    s_num = (S * np.stack([np.ones(N), d], axis=1)).sum(axis=1)
    s = s_num / np.maximum(var_P, 1e-12)
    P_mean_sq = P_mean.squeeze(1).astype(np.float64)
    Rp = np.einsum("nij,nj->ni", R, P_mean_sq)
    t = Q_mean.astype(np.float64)[None] - s[:, None] * Rp
    coords_64 = coords.astype(np.float64)
    aligned = (
        s[:, None, None] * np.einsum("nvi,nji->nvj", coords_64, R)
        + t[:, None, :]
    )
    return aligned.astype(np.float32)


def procrustes_similarity_torch(
    src_landmarks: torch.Tensor, ref_template: torch.Tensor
) -> torch.Tensor:
    """Batched closed-form 2D similarity transform (Umeyama / Procrustes).

    For each face's landmarks, finds the rotation + uniform scale +
    translation that best maps src → ref_template in the least-squares
    sense. Returns the forward-pixel-coord ``[B, 2, 3]`` matrix that
    ``feat.utils.geometry.warp_affine`` consumes.

    Pure torch, fully batched, GPU-friendly. No CPU detour — keeps the
    forward graph intact, so this can be wired anywhere downstream of
    the landmark detector without breaking batched inference.

    Args:
        src_landmarks: ``[B, K, 2]`` per-face landmark coordinates in
            face-crop pixel space.
        ref_template: ``[K, 2]`` canonical template landmark positions
            (the target frame all faces align to).

    Returns:
        ``[B, 2, 3]`` affine matrices in pixel coordinates (src->dst),
        ready for ``warp_affine(face_crops, M, dsize=...)``.
    """
    if src_landmarks.dim() != 3 or src_landmarks.shape[-1] != 2:
        raise ValueError(
            f"src_landmarks must be [B, K, 2]; got {tuple(src_landmarks.shape)}"
        )
    if ref_template.shape != (src_landmarks.shape[1], 2):
        raise ValueError(
            f"ref_template shape {tuple(ref_template.shape)} must be "
            f"({src_landmarks.shape[1]}, 2)"
        )

    B = src_landmarks.shape[0]
    device = src_landmarks.device
    dtype = src_landmarks.dtype
    P = src_landmarks  # [B, K, 2]
    Q = ref_template.to(device=device, dtype=dtype)  # [K, 2]

    P_mean = P.mean(dim=1, keepdim=True)  # [B, 1, 2]
    Q_mean = Q.mean(dim=0)  # [2]
    P_c = P - P_mean  # [B, K, 2]
    Q_c = Q - Q_mean  # [K, 2]

    # Cross-covariance H = Q_c^T @ P_c (target first, source second).
    # Matches the convention of the existing numpy procrustes_align_2d_batched
    # so the SVD-based R below comes out with the correct orientation.
    H = torch.einsum("ki,bkj->bij", Q_c, P_c)  # [B, 2, 2]

    # SVD (batched). [B, 2, 2], [B, 2], [B, 2, 2]
    U, S, Vh = torch.linalg.svd(H)
    UVt = U @ Vh
    det = torch.linalg.det(UVt)
    # Reflection correction: ensures R is a proper rotation (det = +1).
    d = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    D = torch.zeros(B, 2, 2, dtype=dtype, device=device)
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = d
    R = U @ D @ Vh  # [B, 2, 2]

    var_P = (P_c.pow(2).sum(dim=(1, 2))).clamp(min=1e-12)  # [B]
    s_num = (S * torch.stack([torch.ones_like(d), d], dim=1)).sum(dim=1)  # [B]
    s = s_num / var_P  # [B]

    # Translation: t = Q_mean - s * R @ P_mean
    Rp = torch.einsum("bij,bj->bi", R, P_mean.squeeze(1))  # [B, 2]
    t = Q_mean.unsqueeze(0) - s.unsqueeze(-1) * Rp  # [B, 2]

    # Build [B, 2, 3]: M = [s*R | t]
    sR = s.view(B, 1, 1) * R  # [B, 2, 2]
    M = torch.cat([sR, t.unsqueeze(-1)], dim=-1)  # [B, 2, 3]
    return M


def procrustes_warp_face_crops(
    face_crops: torch.Tensor,
    face_landmarks: torch.Tensor,
    ref_template: torch.Tensor,
    out_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Warp face crops so their landmarks align with a canonical template.

    Combines ``procrustes_similarity_torch`` (per-face similarity transform
    from landmarks → template) with ``warp_affine`` (apply to images).
    Output crops are shape-normalized: same anatomical features land on
    the same pixel coordinates across faces.

    Args:
        face_crops: ``[B, C, H, W]`` source face crops.
        face_landmarks: ``[B, K, 2]`` per-face landmarks in source pixel coords.
        ref_template: ``[K, 2]`` canonical landmark template in target pixel coords.
        out_size: ``(out_h, out_w)`` output size. Defaults to source ``(H, W)``.

    Returns:
        ``[B, C, out_h, out_w]`` shape-normalized face crops.
    """
    from feat.utils.geometry import warp_affine

    if out_size is None:
        out_size = (face_crops.shape[-2], face_crops.shape[-1])
    M = procrustes_similarity_torch(face_landmarks, ref_template)
    return warp_affine(face_crops, M, dsize=out_size, mode="bilinear")


def extract_hog_features(extracted_faces, landmarks, hog_layer=None):
    """Extract HOG features for AU classification using torch-native HOGLayer.

    Replaces the prior per-face skimage call which round-tripped each face
    through tensor -> PIL -> numpy -> CPU HOG -> numpy. HOGLayer keeps the
    whole batch on the input device and matches skimage.feature.hog to ~5e-8
    absolute tolerance (verified by test_HOGLayer_matches_skimage); the
    trained AU classifier needs no retraining.

    Args:
        extracted_faces: [N, C, H, W] face crops, float32 in [0, 1].
        landmarks: [N, n_landmarks*2] flattened (x, y) landmark coordinates
            in image space.
        hog_layer: optional pre-built HOGLayer to reuse across calls.
            Detectorv1 and MPDetector cache one in __init__ so repeated
            detect() calls don't pay the per-call construction cost.
            If None, a fresh layer is built (backward-compat for direct
            external callers).

    Returns:
        hog_features: numpy array of shape [N, n_features].
        au_new_landmarks: list of per-face landmark arrays in the
            face-aligned crop's coordinates.
    """
    n_faces = landmarks.shape[0]
    if n_faces == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    face_size = extracted_faces.shape[-1]
    extracted_faces_bboxes = (
        torch.tensor([0, 0, face_size, face_size]).unsqueeze(0).repeat(n_faces, 1)
    )
    extracted_landmarks = inverse_transform_landmarks_torch(
        landmarks, extracted_faces_bboxes
    )

    convex_hulls = []
    au_new_landmarks = []
    for j in range(n_faces):
        convex_hull, new_landmark = extract_face_from_landmarks(
            extracted_faces[j, ...], extracted_landmarks[j, ...]
        )
        convex_hulls.append(convex_hull[0])  # [C, H, W]
        au_new_landmarks.append(new_landmark)

    if not convex_hulls:
        return np.zeros((0, 0), dtype=np.float32), au_new_landmarks

    batch = torch.stack(convex_hulls, dim=0)  # [N, C, H, W]
    if hog_layer is None:
        hog_layer = HOGLayer(
            orientations=8,
            pixels_per_cell=8,
            cells_per_block=2,
            block_normalization="L2-Hys",
            feature_vector=True,
            device=batch.device,
        ).to(batch.device)
    with torch.inference_mode():
        features = hog_layer(batch).cpu().numpy()
    return features, au_new_landmarks


def convert_bbox_output(boxes, scores):
    """Convert im2pose_output into Fex Format"""

    widths = boxes[:, 2] - boxes[:, 0]  # right - left
    heights = boxes[:, 3] - boxes[:, 1]  # bottom - top

    return torch.stack(
        (boxes[:, 0], boxes[:, 1], widths, heights, scores),
        dim=1,
    )


def compute_original_image_size(batch_data):
    """
    Computes the original image size before padding and scaling for a batch of images.

    Args:
        batch_data (dict): batch_data from data loader containing 'Image', 'Padding', and 'Scale' tensors.

    Returns:
        original_height_width (torch.Tensor): A tensor of shape [batch_size, 2] representing the original heights and widths of the images.
    """

    # Extract the batch size and dimensions from the input tensors
    batch_size, _, scaled_height, scaled_width = batch_data["Image"].shape

    # Calculate the height and width after scaling but before padding
    height_after_scaling = (
        scaled_height - batch_data["Padding"]["Top"] - batch_data["Padding"]["Bottom"]
    )
    width_after_scaling = (
        scaled_width - batch_data["Padding"]["Left"] - batch_data["Padding"]["Right"]
    )

    # Reverse scaling to get the original height and width before scaling
    original_height = height_after_scaling / batch_data["Scale"]
    original_width = width_after_scaling / batch_data["Scale"]

    # Stack the original height and width into a single tensor of shape [B, 2]
    original_height_width = torch.stack((original_height, original_width), dim=1)

    return original_height_width


def per_face_padding_inversion_terms(batch_data, frame_idx, device):
    """Look up per-face DataLoader-Rescale inversion terms.

    The DataLoader's ``Rescale`` transform pads + scales each frame to a
    uniform shape so the batch can collate. ``forward()`` consumes the
    padded frames directly, so any coordinates it produces (face bboxes,
    landmarks) are in *padded-frame* space. To convert back to the
    *original-frame* coordinates the user expects, we need per-frame
    ``pad_left``, ``pad_top``, ``scale`` values plus the original
    ``frame_h`` / ``frame_w``.

    This helper expands those per-frame quantities to per-face tensors
    via the ``frame_idx`` mapping (which face came from which frame in
    the batch).

    Args:
        batch_data: dict from the DataLoader. Must contain ``"Padding"``
            (with ``"Left"`` / ``"Top"``), ``"Scale"``, and ``"Image"``.
        frame_idx: ``[N]`` long tensor mapping each face to its source
            frame in the batch. Build via
            ``torch.repeat_interleave(arange(B), n_faces_per_frame)``.
        device: device the returned tensors should live on (typically
            the model's device, so the inversion math runs on-device
            without an extra round-trip).

    Returns:
        tuple of five ``[N]`` float tensors on ``device``:
        ``(pad_left, pad_top, scale, frame_h, frame_w)``.
    """
    # Cast to float32 BEFORE moving to device. MPS doesn't accept
    # float64 tensors via `.to('mps')`, and the DataLoader's Padding /
    # Scale tensors come back as float64 by default. Casting first
    # avoids the round-trip.
    pad_left = batch_data["Padding"]["Left"].float().to(device)[frame_idx]
    pad_top = batch_data["Padding"]["Top"].float().to(device)[frame_idx]
    scale = batch_data["Scale"].float().to(device)[frame_idx]
    original_hw = compute_original_image_size(batch_data).float().to(device)
    frame_h = original_hw[frame_idx, 0]
    frame_w = original_hw[frame_idx, 1]
    return pad_left, pad_top, scale, frame_h, frame_w


def invert_padding_to_results(batch_results, batch_data, n_landmarks):
    """Vectorized inversion of dataloader padding/scaling on a batch of detector outputs.

    Replaces a per-frame loop that previously called ``compute_original_image_size``,
    extracted ``Padding`` and ``Scale`` numpy arrays, and rewrote
    ``FrameHeight``, ``FrameWidth``, the four ``FaceRect*`` columns, and
    ``x_i`` / ``y_i`` for every landmark *inside the loop*. The pandas
    ``.loc[mask, col]`` rewrites scaled quadratically with batch size and
    landmark count.

    The replacement still does O(rows × landmarks) work overall (you
    have to read and write every cell), but as a constant number of
    vectorized numpy ops instead of ``n_frames × 2 × n_landmarks``
    boolean-mask-then-write pandas operations. For a 60-frame batch
    with 478 landmarks the prior loop did ~57k mask writes; the new
    helper does 7 column assignments. By steps:

    1. Mapping each row's ``frame`` value to its position in the batch.
    2. Looking up per-row ``pad_left``, ``pad_top``, ``scale``,
       ``frame_height``, ``frame_width`` once.
    3. Doing all column updates with broadcasted numpy ops.

    Args:
        batch_results: pandas DataFrame (or Fex). Modified in place.
        batch_data: dict from the DataLoader for the current batch with
            ``Image``, ``Padding``, ``Scale`` tensors.
        n_landmarks: number of (x_i, y_i) landmark pairs to invert
            (68 for img2pose / mobilefacenet pipelines, 478 for MediaPipe).

    Returns:
        batch_results: the same object passed in (returned for chaining).
    """
    if len(batch_results) == 0:
        return batch_results

    # Once-per-batch numpy extractions (was once-per-frame in the old loop).
    # Cast to float64 to match the dtype the legacy `df.loc[mask, col] = ...`
    # path produced (pandas float64 columns dominated the float32 input).
    original_hw = compute_original_image_size(batch_data).numpy().astype(np.float64)
    pad_left_arr = batch_data["Padding"]["Left"].detach().numpy().astype(np.float64)
    pad_top_arr = batch_data["Padding"]["Top"].detach().numpy().astype(np.float64)
    scale_arr = batch_data["Scale"].detach().numpy().astype(np.float64)

    # Each unique frame in batch_results corresponds to a position in the
    # current batch. Preserve the order of first appearance to match the
    # j = enumerate(unique_frames) convention the prior loop used.
    unique_frames = batch_results["frame"].drop_duplicates().to_numpy()
    frame_to_j = {f: j for j, f in enumerate(unique_frames)}
    j_per_row = batch_results["frame"].map(frame_to_j).to_numpy()  # [n_rows]

    pad_left = pad_left_arr[j_per_row]  # [n_rows]
    pad_top = pad_top_arr[j_per_row]
    scale = scale_arr[j_per_row]
    frame_h = original_hw[j_per_row, 0]
    frame_w = original_hw[j_per_row, 1]

    batch_results["FrameHeight"] = frame_h
    batch_results["FrameWidth"] = frame_w

    batch_results["FaceRectX"] = (
        batch_results["FaceRectX"].to_numpy() - pad_left
    ) / scale
    batch_results["FaceRectY"] = (
        batch_results["FaceRectY"].to_numpy() - pad_top
    ) / scale
    batch_results["FaceRectWidth"] = batch_results["FaceRectWidth"].to_numpy() / scale
    batch_results["FaceRectHeight"] = batch_results["FaceRectHeight"].to_numpy() / scale

    x_cols = [f"x_{i}" for i in range(n_landmarks)]
    y_cols = [f"y_{i}" for i in range(n_landmarks)]
    x_vals = batch_results[x_cols].to_numpy()  # [n_rows, n_landmarks]
    y_vals = batch_results[y_cols].to_numpy()
    # iloc-based column assignment is ~30x faster than the equivalent
    # `batch_results[x_cols] = arr` form. The label-based form goes through
    # pandas's `_iset_split_block` path once per column (956 ops on a
    # MediaPipe-shape DataFrame, 136 ops on a 68-landmark one). iloc with a
    # positional index lets pandas write the whole 2D slice through a
    # single block update.
    x_idx = batch_results.columns.get_indexer(x_cols)
    y_idx = batch_results.columns.get_indexer(y_cols)
    batch_results.iloc[:, x_idx] = (x_vals - pad_left[:, None]) / scale[:, None]
    batch_results.iloc[:, y_idx] = (y_vals - pad_top[:, None]) / scale[:, None]

    return batch_results
