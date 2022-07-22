import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import Compose, Pad
from feat.transforms import Rescale
from .img2pose_model import img2poseModel
from feat.utils import get_resource_path, convert_image_to_tensor, set_torch_device
from feat.face_detectors.Retinaface.Retinaface_utils import py_cpu_nms
from ..utils import convert_to_euler

# from feat.data import Rescale


class Img2Pose:
    def __init__(
        self,
        device="auto",
        constrained=True,
        detection_threshold=0.5,
        nms_threshold=0.6,
        nms_inclusion_threshold=0.05,
        top_k=5000,
        keep_top_k=750,
        BORDER_SIZE=100,
        DEPTH=18,
        MAX_SIZE=1400,
        MIN_SIZE=400,
        POSE_MEAN=os.path.join(get_resource_path(), "WIDER_train_pose_mean_v1.npy"),
        POSE_STDDEV=os.path.join(get_resource_path(), "WIDER_train_pose_stddev_v1.npy"),
        THREED_FACE_MODEL=os.path.join(
            get_resource_path(), "reference_3d_68_points_trans.npy"
        ),
    ):
        """Creates an img2pose model. Constrained model is optimized for face detection/ pose estimation for
        front-facing faces ( [-90, 90] degree range) only. Unconstrained model can detect faces and poses at any angle,
        but shows slightly dampened performance on face pose estimation.

        Args:
            device (str): device to execute code. can be ['auto', 'cpu', 'cuda', 'mps']
            contrained (bool): whether to run constrained (default) or unconstrained mode

        Returns:
            Img2Pose object

        """

        self.device = set_torch_device(device)

        pose_mean = np.load(POSE_MEAN, allow_pickle=True)
        pose_stddev = np.load(POSE_STDDEV, allow_pickle=True)
        threed_points = np.load(THREED_FACE_MODEL, allow_pickle=True)

        self.model = img2poseModel(
            DEPTH,
            MIN_SIZE,
            MAX_SIZE,
            pose_mean=pose_mean,
            pose_stddev=pose_stddev,
            threed_68_points=threed_points,
            device=self.device,
        )
        # self.transform = transforms.Compose([transforms.ToTensor()])

        # Load the constrained model
        model_file = "img2pose_v1_ft_300w_lp.pth" if constrained else "img2pose_v1.pth"
        self.load_model(os.path.join(get_resource_path(), model_file))
        self.model.evaluate()

        # Set threshold score for bounding box detection
        (
            self.detection_threshold,
            self.nms_threshold,
            self.nms_inclusion_threshold,
            self.top_k,
            self.keep_top_k,
            self.MIN_SIZE,
            self.MAX_SIZE,
            self.BORDER_SIZE,
        ) = (
            detection_threshold,
            nms_threshold,
            nms_inclusion_threshold,
            top_k,
            keep_top_k,
            MIN_SIZE,
            MAX_SIZE,
            BORDER_SIZE,
        )

    def load_model(self, model_path, optimizer=None):
        """Loads model weights for the img2pose model
        Args:
            model_path (str): file path to saved model weights
            optimizer (torch.optim.Optimizer): An optimizer to load (pass an optimizer when model_path also contains a
                                               saved optimizer)
            cpu_mode (bool): whether or not to use CPU (True) or GPU (False)

        Returns:
            None
        """

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model.fpn_model.load_state_dict(checkpoint["fpn_model"])

        if "optimizer" in checkpoint and optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif optimizer:
            print("Optimizer not found in model path - cannot be loaded")

    def __call__(self, img):
        """Runs scale_and_predict on each image in the passed image list

        Args:
            img_ (np.ndarray): (B,H,W,C), B is batch number, H is image height, W is width and C is channel.

        Returns:
            tuple: (faces, poses) - 3D lists (B, F, bbox) or (B, F, face pose) where B is batch/ image number and
                                    F is face number
        """

        img = convert_image_to_tensor(img)
        img = img.type(torch.float32)
        img = img.to(self.device)

        preds = self.scale_and_predict(img)
        # faces = []
        # poses = []
        # for img in img_:
        #     preds = self.scale_and_predict(img)
        #     faces.append(preds["boxes"])
        #     poses.append(preds["poses"])

        return preds["boxes"], preds["poses"]

    def scale_and_predict(self, img, euler=True):
        """Runs a prediction on the passed image. Returns detected faces and associates poses.
        Args:
            img (np.ndarray): A cv2 image
            euler (bool): set to True to obtain euler angles, False to obtain rotation vector

        Returns:
            dict: key 'pose' contains array - [yaw, pitch, roll], key 'boxes' contains 2D array of bboxes
        """

        # Transform image to improve model performance. Resize the image so that both dimensions are in the range [MIN_SIZE, MAX_SIZE]
        scale = 1
        border_size = 0
        if min(img.shape[-2:]) < self.MIN_SIZE or max(img.shape[-2:]) > self.MAX_SIZE:
            transform = Compose([Rescale(self.MAX_SIZE, preserve_aspect_ratio=True)])
            transformed_img = transform(img)
            img = transformed_img["Image"]
            scale = transformed_img["Scale"]

        # Predict
        preds = self.predict(img, border_size=border_size, scale=scale, euler=euler)

        # If the prediction is unsuccessful, try adding a white border to the image. This can improve bounding box
        # performance on images where face takes up entire frame, and images located at edge of frame.
        if len(preds["boxes"]) == 0:
            WHITE = 255
            border_size = self.BORDER_SIZE
            transform = Compose([Pad(border_size, fill=WHITE)])
            img = transform(img)
            preds = self.predict(img, border_size=border_size, scale=scale, euler=euler)

        return preds

    def predict(self, img, border_size=0, scale=1.0, euler=True):
        """Runs the img2pose model on the passed image and returns bboxes and face poses.

        Args:
            img (np.ndarray): A cv2 image
            border_size (int): if the cv2 image has a border, the width of the border (in pixels)
            scale (float): if the image was resized, the scale factor used to perform resizing
            euler (bool): set to True to obtain euler angles, False to obtain rotation vector

        Returns:
            dict: A dictionary of bboxes and poses

        """
        # # img2pose expects RGB form
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Obtain prediction
        pred = self.model.predict(img)[0]
        # pred = self.model.predict([self.transform(img)])[0]
        boxes = pred["boxes"].cpu().numpy().astype("float")
        scores = pred["scores"].cpu().numpy().astype("float")
        dofs = pred["dofs"].cpu().numpy().astype("float")

        # Obtain boxes sorted by score
        inds = np.where(scores > self.nms_inclusion_threshold)[0]
        boxes, scores, dofs = boxes[inds], scores[inds], dofs[inds]
        order = scores.argsort()[::-1][: self.top_k]
        boxes, scores, dofs = boxes[order], scores[order], dofs[order]

        # Perform NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)

        # Prepare predictions
        det_bboxes = []
        det_dofs = []
        for i in keep:
            bbox = dets[i]

            # Remove added image borders
            bbox[0] = max(bbox[0] - border_size, 0) // scale
            bbox[1] = max(bbox[1] - border_size, 0) // scale
            bbox[2] = (bbox[2] - border_size) // scale
            bbox[3] = (bbox[3] - border_size) // scale

            # Keep bboxes with sufficiently high scores
            score = bbox[4]
            if score > self.detection_threshold:
                det_bboxes.append(list(bbox))
                det_dofs.append(dofs[i])

        # Obtain pitch, roll, yaw estimates
        det_pose = []
        for pose_pred in det_dofs:
            if euler:  # Convert rotation vector into euler angles
                pose_pred[:3] = convert_to_euler(pose_pred[:3])

            three_dof_pose = pose_pred[:3]  # pitch, roll, yaw (when euler=True)
            three_dof_pose = three_dof_pose.reshape(1, -1)
            det_pose.append(three_dof_pose)

        return {"boxes": det_bboxes, "poses": det_pose}

    def set_threshold(self, threshold):
        """Alter the threshold for face detection.

        Args:
            threshold (float): A number representing the face detection score threshold to use

        Returns:
            None
        """
        self.detection_threshold = threshold
