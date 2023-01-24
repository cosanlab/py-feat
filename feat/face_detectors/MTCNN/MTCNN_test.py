"""
The codes in this file comes from the original codes at:
    https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py
The original paper on MTCNN is:
K. Zhang, Z. Zhang, Z. Li and Y. Qiao. Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks, IEEE Signal Processing Letters, 2016
"""
import numpy as np
import torch
from PIL import Image
from feat.face_detectors.MTCNN.MTCNN_model import PNet, RNet, ONet
from feat.face_detectors.MTCNN.MTCNN_utils import detect_face
from feat.utils import set_torch_device

import torch.nn as nn


class MTCNN(nn.Module):
    """MTCNN face detection module.
    This class loads pretrained P-, R-, and O-nets and returns images cropped to include the face
    only, given raw input images of one of the following types:
        - PIL image or list of PIL images
        - numpy.ndarray (uint8) representing either a single image (3D) or a batch of images (4D).
    Cropped faces can optionally be saved to file
    also.

    Keyword Arguments:
        image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size (this is a bug in davidsandberg/facenet).
            (default: {0})
        min_face_size {int} -- Minimum face size to search for. (default: {20})
        thresholds {list} -- MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
        detection_threshold (float): threshold for detectiong faces (default=0.5). Will override the last stage of thresholds
        factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
        post_process {bool} -- Whether or not to post process images tensors before returning.
            (default: {True})
        select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
            If False, the face with the highest detection probability is returned.
            (default: {True})
        selection_method {string} -- Which heuristic to use for selection. Default None. If
            specified, will override select_largest:
                    "probability": highest probability selected
                    "largest": largest box selected
                    "largest_over_threshold": largest box over a certain probability selected
                    "center_weighted_size": box size minus weighted squared offset from image center
                (default: {None})
        keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
            select_largest parameter.
            (default: {False})
        device {torch.device} -- The device on which to run neural net passes. Image tensors and
            models are copied to this device before running forward passes. (default: 'auto')
    """

    def __init__(
        self,
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        detection_threshold=0.5,
        factor=0.709,
        post_process=True,
        select_largest=True,
        selection_method=None,
        keep_all=True,
        device="auto",
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.thresholds[-1] = detection_threshold
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.selection_method = selection_method

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        self.device = set_torch_device(device)
        self.to(self.device)

        if not self.selection_method:
            self.selection_method = "largest" if self.select_largest else "probability"

    def forward(self, img, save_path=None, return_prob=False):
        """Run MTCNN face detection on a PIL image or numpy array. This method performs both
        detection and extraction of faces, returning tensors representing detected faces rather
        than the bounding boxes. To access bounding boxes, see the MTCNN.detect() method below.

        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
            save_path {str} -- An optional save path for the cropped image. Note that when
                self.post_process=True, although the returned tensor is post processed, the saved
                face image is not, so it is a true representation of the face in the input image.
                If `img` is a list of images, `save_path` should be a list of equal length.
                (default: {None})
            return_prob {bool} -- Whether or not to return the detection probability.
                (default: {False})

        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. Optionally, the probability that a
                face was detected. If self.keep_all is True, n detected faces are returned in an
                n x 3 x image_size x image_size tensor with an optional list of detection
                probabilities. If `img` is a list of images, the item(s) returned have an extra
                dimension (batch) as the first dimension.
        Example:
        >>> from facenet_pytorch import MTCNN
        >>> mtcnn = MTCNN()
        >>> face_tensor, prob = mtcnn(img, save_path='face.png', return_prob=True)
        """

        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)

        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes,
                batch_probs,
                batch_points,
                img,
                method=self.selection_method,
            )
        # Extract faces
        faces = self.extract(img, batch_boxes, save_path)

        if return_prob:
            return faces, batch_probs
        else:
            return faces

    def __call__(self, img, landmarks=False):
        """Detect all faces in PIL image and return bounding boxes and optional facial landmarks.
        This method is used by the forward method and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this method
        followed by the extract_face() function.

        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.
        Keyword Arguments:
            landmarks {bool} -- Whether to return facial landmarks in addition to bounding boxes.
                (default: {False})

        Returns:
            tuple(numpy.ndarray, list) -- For N detected faces, a tuple containing an
                Nx4 array of bounding boxes and a length N list of detection probabilities.
                Returned boxes will be sorted in descending order by detection probability if
                self.select_largest=False, otherwise the largest face will be returned first.
                If `img` is a list of images, the items returned have an extra dimension
                (batch) as the first dimension. Optionally, a third item, the facial landmarks,
                are returned if `landmarks=True`.
        Example:
        >>> from PIL import Image, ImageDraw
        >>> from facenet_pytorch import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        >>> img_draw.save('annotated_faces.png')
        """

        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img,
                self.min_face_size,
                self.pnet,
                self.rnet,
                self.onet,
                self.thresholds,
                self.factor,
                self.device,
            )

        boxes, points = [], []
        for box, point in zip(batch_boxes, batch_points):
            if len(box) == 0:
                boxes.append(None)
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort(
                    (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
                )[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box.tolist())
                points.append(point)
            else:
                boxes.append(box.tolist())
                points.append(point)

        if (
            not isinstance(img, (list, tuple))
            and not (isinstance(img, np.ndarray) and len(img.shape) == 4)
            and not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            boxes = boxes[0]
            points = points[0]

        if landmarks:
            return boxes, points

        return boxes
