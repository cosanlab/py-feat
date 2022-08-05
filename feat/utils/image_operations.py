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
from torchvision.transforms import PILToTensor, Compose
import PIL
from kornia.geometry.transform import warp_affine
from skimage.morphology.convex_hull import grid_points_in_poly
from feat.transforms import Rescale

__all__ = [
    "neutral",
    "face_rect_to_coords",
    "registration",
    "convert68to49",
    "extract_face_from_landmark",
    "extract_face_from_bbox",
    "convert68to49",
    "align_face_68pts",
    "align_face_49pts",
    "BBox",
    "round_vals",
    "reverse_color_order",
    "expand_img_dimensions",
    "convert_image_to_tensor",
    "convert_color_vector_to_tensor",
    "mask_image",
    "convert_to_euler",
    "py_cpu_nms",
    "decode",
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
    assert type(face_lms) == np.ndarray, TypeError("face_lms must be type np.ndarray")
    assert face_lms.ndim == 2, ValueError("face_lms must be shape (n, 136)")
    assert face_lms.shape[1] == 136, ValueError("Must have 136 landmarks")
    registered_lms = []
    for row in face_lms:
        face = [row[:68], row[68:]]
        face = np.array(face).T
        #   Rotate face
        primary = np.array(face)
        secondary = np.array(neutral)
        n = primary.shape[0]
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        X1, Y1 = pad(primary), pad(secondary)
        if type(method) == str:
            if method == "fullface":
                A, res, rank, s = np.linalg.lstsq(X1, Y1, rcond=None)
            elif method == "inner":
                A, res, rank, s = np.linalg.lstsq(X1[17:, :], Y1[17:, :], rcond=None)
            else:
                raise ValueError("method is either 'fullface' or 'inner'")
        elif type(method) == list:
            A, res, rank, s = np.linalg.lstsq(X1[method], Y1[method], rcond=None)
        else:
            raise TypeError(
                "method is string ('fullface','inner') or list of landmarks"
            )
        transform = lambda x: unpad(np.dot(pad(x), A))
        registered_lms.append(transform(primary).T.reshape(1, 136).ravel())
    return np.array(registered_lms)


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

    landmarks = np.array(landmarks).copy()

    aligned_img, new_landmarks = align_face_68pts(
        frame, landmarks.flatten(), 2.5, img_size=face_size
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
        frame_assignment = np.where(k <= length_cumu)[0][0]  # which frame is it?
        bbox = BBox(
            face[:-1], bottom_boundary=im_height, right_boundary=im_width
        ).expand_by_factor(expand_bbox)
        cropped = bbox.extract_from_image(frame[frame_assignment])
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


def align_face_68pts(img, img_land, box_enlarge, img_size=112):
    """Performs affine transformation to align the images by eyes.

    Performs affine alignment including eyes.

    This can probably be merged with align_face_49pts

    Args:
        img: gray or RGB
        img_land: 68 system flattened landmarks, shape:(136)
        box_enlarge: relative size of face on the image. Smaller value indicate larger proportion
        img_size = output image size
    Return:
        aligned_img: the aligned image
        new_land: the new landmarks
    """
    leftEye0 = (
        img_land[2 * 36]
        + img_land[2 * 37]
        + img_land[2 * 38]
        + img_land[2 * 39]
        + img_land[2 * 40]
        + img_land[2 * 41]
    ) / 6.0
    leftEye1 = (
        img_land[2 * 36 + 1]
        + img_land[2 * 37 + 1]
        + img_land[2 * 38 + 1]
        + img_land[2 * 39 + 1]
        + img_land[2 * 40 + 1]
        + img_land[2 * 41 + 1]
    ) / 6.0
    rightEye0 = (
        img_land[2 * 42]
        + img_land[2 * 43]
        + img_land[2 * 44]
        + img_land[2 * 45]
        + img_land[2 * 46]
        + img_land[2 * 47]
    ) / 6.0
    rightEye1 = (
        img_land[2 * 42 + 1]
        + img_land[2 * 43 + 1]
        + img_land[2 * 44 + 1]
        + img_land[2 * 45 + 1]
        + img_land[2 * 46 + 1]
        + img_land[2 * 47 + 1]
    ) / 6.0
    deltaX = rightEye0 - leftEye0
    deltaY = rightEye1 - leftEye1
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])
    mat2 = np.mat(
        [
            [leftEye0, leftEye1, 1],
            [rightEye0, rightEye1, 1],
            [img_land[2 * 30], img_land[2 * 30 + 1], 1],
            [img_land[2 * 48], img_land[2 * 48 + 1], 1],
            [img_land[2 * 54], img_land[2 * 54 + 1], 1],
        ]
    )
    mat2 = (mat1 * mat2.T).T
    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5
    if float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(
        max(mat2[:, 1]) - min(mat2[:, 1])
    ):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))
    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat(
        [
            [scale, 0, scale * (halfSize - cx)],
            [0, scale, scale * (halfSize - cy)],
            [0, 0, 1],
        ]
    )
    mat = mat3 * mat1

    affine_matrix = torch.tensor(mat[0:2, :]).type(torch.float32).unsqueeze(0)
    aligned_img = warp_affine(
        img,
        affine_matrix,
        (img_size, img_size),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
        fill_value=(128, 128, 128),
    )

    land_3d = np.ones((int(len(img_land) / 2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land) / 2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.array(list(zip(new_land[:, 0], new_land[:, 1]))).astype(int)

    return aligned_img, new_land


def align_face_49pts(img, img_land, box_enlarge=2.9, img_size=200):
    """Align face using 49 landmarks

    based on https://github.com/ZhiwenShao/PyTorch-JAANet/blob/master/dataset/face_transform.py

    Args:
        img (torch.Tensor): image represented as tensor (B,C,H,W)
        img_land: facial landmarks (49 points)
        box_enlarge (float): enlarge factor for the face transform, centered at face
        img_size (int): size of the desired output image

    Return:
        aligned_img: aligned images
        aligned_landmarks: transformed landmarks
        binocular: binocular distance
    """
    leftEye0 = (
        img_land[2 * 19]
        + img_land[2 * 20]
        + img_land[2 * 21]
        + img_land[2 * 22]
        + img_land[2 * 23]
        + img_land[2 * 24]
    ) / 6.0
    leftEye1 = (
        img_land[2 * 19 + 1]
        + img_land[2 * 20 + 1]
        + img_land[2 * 21 + 1]
        + img_land[2 * 22 + 1]
        + img_land[2 * 23 + 1]
        + img_land[2 * 24 + 1]
    ) / 6.0
    rightEye0 = (
        img_land[2 * 25]
        + img_land[2 * 26]
        + img_land[2 * 27]
        + img_land[2 * 28]
        + img_land[2 * 29]
        + img_land[2 * 30]
    ) / 6.0
    rightEye1 = (
        img_land[2 * 25 + 1]
        + img_land[2 * 26 + 1]
        + img_land[2 * 27 + 1]
        + img_land[2 * 28 + 1]
        + img_land[2 * 29 + 1]
        + img_land[2 * 30 + 1]
    ) / 6.0
    deltaX = rightEye0 - leftEye0
    deltaY = rightEye1 - leftEye1
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])

    mat2 = np.mat(
        [
            [leftEye0, leftEye1, 1],
            [rightEye0, rightEye1, 1],
            [img_land[2 * 13], img_land[2 * 13 + 1], 1],
            [img_land[2 * 31], img_land[2 * 31 + 1], 1],
            [img_land[2 * 37], img_land[2 * 37 + 1], 1],
        ]
    )

    mat2 = (mat1 * mat2.T).T

    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5

    if float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(
        max(mat2[:, 1]) - min(mat2[:, 1])
    ):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))

    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat(
        [
            [scale, 0, scale * (halfSize - cx)],
            [0, scale, scale * (halfSize - cy)],
            [0, 0, 1],
        ]
    )
    mat = mat3 * mat1

    affine_matrix = torch.tensor(mat[0:2, :]).type(torch.float32).unsqueeze(0)
    aligned_img = warp_affine(
        img,
        affine_matrix,
        (img_size, img_size),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
        fill_value=(128, 128, 128),
    )

    land_3d = np.ones((int(len(img_land) / 2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land) / 2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.reshape(new_land[:, 0:2], len(img_land))

    return aligned_img, new_land


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
        self.height = self.top - self.bottom

        self = self.set_boundary(
            left=left_boundary,
            right=right_boundary,
            top=top_boundary,
            bottom=bottom_boundary,
            apply_boundary=True,
        )

    def __repr__(self):
        return f"'height': {self.height}, 'width': {self.width}"

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
        self.bottom = self.center_y - (self.height // 2)
        self.top = self.center_y + (self.height // 2)

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
            return img[
                int(self.bottom) : int(self.top), int(self.left) : int(self.right)
            ]
        elif len(img.shape) == 3:
            return img[
                :, int(self.bottom) : int(self.top), int(self.left) : int(self.right)
            ]
        elif len(img.shape) == 4:
            return img[
                :, :, int(self.bottom) : int(self.top), int(self.left) : int(self.right)
            ]
        else:
            raise ValueError("Not a valid image size")

    def to_dict(self):
        """bounding box coordinates as a dictionary"""
        return {
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
            "top": self.top,
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
            y = point[1] * self.height + self.bottom
            landmark_[i] = (x, y)
        return landmark_


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
    return (
        torch.sgn(torch.tensor(mask).to(torch.float32)).unsqueeze(0).unsqueeze(0) * img
    )


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
    return np.array([angle[0], -angle[2], -angle[1]])  # pitch, roll, yaw


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


class HOGLayer(nn.Module):
    def __init__(
        self,
        nbins=10,
        pixels_per_cell=8,
        cells_per_block=2,
        max_angle=math.pi,
        stride=1,
        padding=1,
        dilation=1,
        transform_sqrt=False,
        block_normalization=None,
        feature_vector=True,
        device="auto",
    ):
        """Pytorch Model to extract HOG features. Designed to be similar to skimage.feature.hog.

        Based on https://gist.github.com/etienne87/b79c6b4aa0ceb2cff554c32a7079fa5a

        Args:
            orientations (int): Number of orientation bins.
            pixels_per_cell (int, int): Size (in pixels) of a cell.
            transform_sqrt (bool): Apply power law compression to normalize the image before processing.
                                    DO NOT use this if the image contains negative values.
            block_normalization (str): Block normalization method:
                                    ``L1``
                                       Normalization using L1-norm.
                                    ``L1-sqrt``
                                       Normalization using L1-norm, followed by square root.
                                    ``L2``
                                       Normalization using L2-norm.
                                    ``L2-Hys``
                                       Normalization using L2-norm, followed by limiting the
                                       maximum values to 0.2 (`Hys` stands for `hysteresis`) and
                                       renormalization using L2-norm. (default)
            feature_vector (bool): Return as a feature vector
            device (str): device to execute code. can be ['auto', 'cpu', 'cuda', 'mps']

        """

        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.max_angle = max_angle
        self.transform_sqrt = transform_sqrt
        self.device = set_torch_device(device)
        self.feature_vector = feature_vector
        if block_normalization is not None:
            self.block_normalization = block_normalization.lower()
        else:
            self.block_normalization = block_normalization

        # Construct a Sobel Filter
        mat = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:, None, :, :])
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

            # 1. Global Normalization. The first stage applies an optional global
            # image normalization equalisation that is designed to reduce the influence
            # of illuminationeffects. In practice we use gamma (power law) compression,
            # either computing the square root or the log of each color channel.
            # Image texture strength is typically proportional to the local surface
            # illumination so this compression helps to reduce the effects of local
            # shadowing and illumination variations.
            if self.transform_sqrt:
                img = img.sqrt()

            # 2. Compute Gradients. The second stage computes first order image gradients.
            # These capture contour, silhouette and some texture information,
            # while providing further resistance to illumination variations.
            gxy = F.conv2d(
                img,
                self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,
            )

            # 3. Binning Mag with linear interpolation. The third stage aims to produce
            # an encoding that is sensitive to local image content while remaining
            # resistant to small changes in pose or appearance. The adopted method pools
            # gradient orientation information locally in the same way as the SIFT
            # [Lowe 2004] feature. The image window is divided into small spatial regions,
            # called "cells". For each cell we accumulate a local 1-D histogram of gradient
            # or edge orientations over all the pixels in the cell. This combined
            # cell-level 1-D histogram forms the basic "orientation histogram" representation.
            # Each orientation histogram divides the gradient angle range into a fixed
            # number of predetermined bins. The gradient magnitudes of the pixels in the
            # cell are used to vote into the orientation histogram.
            mag = gxy.norm(dim=1)
            norm = mag[:, None, :, :]
            phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])
            phase_int = phase / self.max_angle * self.nbins
            phase_int = phase_int[:, None, :, :]

            n, c, h, w = gxy.shape
            out = torch.zeros(
                (n, self.nbins, h, w), dtype=torch.float, device=self.device
            )
            out.scatter_(1, phase_int.floor().long() % self.nbins, norm)
            out.scatter_add_(1, phase_int.ceil().long() % self.nbins, 1 - norm)
            out = self.cell_pooler(out)

            # 4. Compute Normalization. The fourth stage computes normalization,
            # which takes local groups of cells and contrast normalizes their overall
            # responses before passing to next stage. Normalization introduces better
            # invariance to illumination, shadowing, and edge contrast. It is performed
            # by accumulating a measure of local histogram "energy" over local groups
            # of cells that we call "blocks". The result is used to normalize each cell
            # in the block. Typically each individual cell is shared between several
            # blocks, but its normalizations are block dependent and thus different.
            # The cell thus appears several times in the final output vector with
            # different normalizations. This may seem redundant but it improves the
            # performance. We refer to the normalized block descriptors as Histogram
            # of Oriented Gradient (HOG) descriptors.
            #
            # Note: we can probably find a way to vectorize this loop at some point
            if self.block_normalization is not None:
                n_batch, n_channel, s_row, s_col = img.shape
                c_row, c_col = [self.pixels_per_cell] * 2
                b_row, b_col = [self.cells_per_block] * 2
                n_cells_row = int(s_row // c_row)  # number of cells along row-axis
                n_cells_col = int(s_col // c_col)  # number of cells along col-axis
                n_blocks_row = (
                    n_cells_row - b_row
                ) + 1  # number of blocks along row-axis
                n_blocks_col = (
                    n_cells_col - b_col
                ) + 1  # number of blocks along col-axis

                hog_out = torch.zeros(
                    (
                        n_batch,
                        nbins,
                        cells_per_block,
                        cells_per_block,
                        n_blocks_row,
                        n_blocks_col,
                    ),
                    dtype=torch.float,
                    device=self.device,
                )
                for r in range(n_blocks_row):
                    for c in range(n_blocks_col):
                        hog_out[:, :, :, :, r, c] = self._normalize_block(
                            out[:, :, r : r + b_row, c : c + b_col],
                            self.block_normalization,
                        )
                out = hog_out

            if self.feature_vector:
                return out.flatten(start_dim=1)
            else:
                return out

    def _normalize_block(self, block, method, eps=1e-5):
        """helper function to perform normalization"""

        if method == "l1":
            out = torch.divide(block, block.abs().sum() + eps)
        elif method == "l1-sqrt":
            out = torch.divide(block, block.abs().sum() + eps).sqrt()
        elif method == "l2":
            out = torch.divide(block, (torch.square(block).sum() + eps**2).sqrt())
        elif method == "l2-hys":
            out = torch.divide(block, (torch.square(block).sum() + eps**2).sqrt())
            out = torch.minimum(out, torch.ones(out.shape) * 0.2)
            out = torch.divide(out, (torch.square(out).sum() + eps**2).sqrt())
        else:
            raise ValueError(
                'Selected block normalization method is invalid. Use ["l1","l1-sqrt","l2","l2-hys"]'
            )
        return out
