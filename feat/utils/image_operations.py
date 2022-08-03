"""
py-feat utility and helper functions for performing operations on images. 
"""

from .io import get_resource_path
import math

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import torch
from torchvision.transforms import PILToTensor, Compose
import PIL
from kornia.geometry.transform import warp_affine
from skimage.morphology.convex_hull import grid_points_in_poly

__all__ = [
    "neutral",
    "face_rect_to_coords",
    "registration",
    "convert68to49",
    "extract_face",
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
]

# Neutral face coordinates
neutral = pd.read_csv(
    os.path.join(get_resource_path(), "neutral_face_coordinates.csv"), index_col=False
)

# Replace with BBox
def face_rect_to_coords(rectangle):
    """
    Takes in a (x, y, w, h) array and transforms it into (x, y, x2, y2)
    """
    return [
        rectangle[0],
        rectangle[1],
        rectangle[0] + rectangle[2],
        rectangle[1] + rectangle[3],
    ]


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


def extract_face(frame, landmarks, size_output=112):
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
        frame, landmarks.flatten(), 2.5, img_size=size_output
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


def convert68to49(points):
    """Convert landmark form 68 to 49

    Function slightly modified from https://github.com/D-X-Y/landmark-detection/blob/7bc7a5dbdbda314653124a4596f3feaf071e8589/SAN/lib/datasets/dataset_utils.py#L169 to fit pytorch tensors. Converts 68 point landmarks to 49 point landmarks

    Args:
        points: landmark points of shape (2,68) or (3,68)

    Return:
        cpoints: converted 49 landmark points of shape (2,49)
    """
    assert (
        len(points.shape) == 2
        and (points.shape[0] == 3 or points.shape[0] == 2)
        and points.shape[1] == 68
    ), "The shape of points is not right : {}".format(points.shape)

    if isinstance(points, torch.Tensor):
        points = points.clone()
        out = torch.ones((68,), dtype=torch.bool)
    elif type(points) is np.ndarray:
        points = points.copy()
        out = np.ones((68,)).astype("bool")

    out[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 60, 64]] = False
    cpoints = points[:, out]

    assert len(cpoints.shape) == 2 and cpoints.shape[1] == 49
    return cpoints


def align_face_68pts(img, img_land, box_enlarge, img_size=112):
    """Performs affine transformation to align the images by eyes.

    Performs affine alignment including eyes.

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
    """
    code from:
    https://github.com/ZhiwenShao/PyTorch-JAANet/blob/master/dataset/face_transform.py
    Did some small modifications to fit into our program.
    The function performs preproecessing transformations on pictures.
    Args:
        img: iamges loaded by cv2. Shape: (3,H,W)
        img_land: landmark file for the img. Shape()
        box_enlarge: englarge factor for the face transform, centered at face
        img_size: size of the desired output image
    Return:
        aligned_img: aligned images by cv2
        new_land: transformed landmarks
        biocular: biocular distancxe
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

    # aligned_img = cv2.warpAffine(
    #     img,
    #     mat[0:2, :],
    #     (img_size, img_size),
    #     cv2.INTER_LINEAR,
    #     borderValue=(128, 128, 128),
    # )

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
    def __init__(self, bbox, order=None):
        """Class to work with Bounding Box

        Args:
            bbox: (list): values
            order (list): order of values (e.g., ['left', 'bottom', 'right', 'top'])
        """
        if order is None:
            self.order = ["left", "bottom", "right", "top"]
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

    def __repr__(self):
        return f"'height': {self.height}, 'width': {self.width}"

    def expand_by_factor(self, factor):
        """Expand box by factor

        Args:
            factor (float, tuple): factor to expand. if float equal in all directions, if tuple (height, width)
        """

        if not isinstance(factor, (int, float, tuple)):
            raise ValueError("factor must be float or tuple(height, width)")

        if isinstance(factor, tuple):
            if len(factor) != 2:
                raise ValueError("factor must include (height,width)")
            else:
                self.height *= factor[0]
                self.width *= factor[1]
        else:
            self.height *= factor
            self.width *= factor
        self.left = max(self.center_x - self.width, 0)
        self.right = self.center_x + self.width
        self.bottom = max(self.center_x - self.height, 0)
        self.top = self.center_x + self.height
        return self

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

    def project_landmark(self, landmark):
        """scale to [0,1] from https://github.com/cunjian/pytorch_face_landmark/"""
        landmark_ = np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            landmark_[i] = (
                (point[0] - self.left) / self.width,
                (point[1] - self.top) / self.height,
            )
        return landmark_

    def reproject_landmark(self, landmark):
        """landmark of (5L, 2L) from [0,1] to real range from https://github.com/cunjian/pytorch_face_landmark/"""
        landmark_ = np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            x = point[0] * self.width + self.left
            y = point[1] * self.height + self.bottom
            landmark_[i] = (x, y)
        return landmark_


def round_vals(list_of_arrays, ndigits):
    list_of_arrays2 = list_of_arrays.copy()
    for i, arr0 in enumerate(list_of_arrays):
        for j, arr1 in enumerate(list_of_arrays):
            list_of_arrays2[i][j] = np.around(list_of_arrays[i][j], ndigits)
    return list_of_arrays2


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
