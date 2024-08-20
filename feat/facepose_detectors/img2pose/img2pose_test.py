import os
import torch
import numpy as np
from torchvision.transforms import Compose, Pad
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from feat.transforms import Rescale
from .img2pose_model import img2poseModel, WrappedModel
from .deps.models import FasterDoFRCNN
from feat.utils import set_torch_device
from feat.utils.io import get_resource_path
from feat.utils.image_operations import convert_to_euler, py_cpu_nms
import logging
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

model_config = {}
model_config['img2pose'] = {'rpn_pre_nms_top_n_test':6000,
                            'rpn_post_nms_top_n_test':1000,
                            'bbox_x_factor':1.1,
                            'bbox_y_factor':1.1,
                            'expand_forehead':0.3,
                            'depth':18,
                            'max_size':1400,
                            'min_size':400,
                            'constrained':True,
                            'pose_mean':torch.tensor([-0.0238,  0.0275, -0.0144,  0.0664,  0.2380,  3.4813]),
                            'pose_stddev':torch.tensor([0.2353, 0.5395, 0.1767, 0.1320, 0.1358, 0.3663]),
                            'threed_points':torch.tensor([[-0.7425, -0.3662,  0.4207],
                            [-0.7400, -0.1836,  0.5642],
                            [-0.6339,  0.0051,  0.1404],
                            [-0.5988,  0.1618, -0.0176],
                            [-0.5455,  0.3358, -0.0198],
                            [-0.4669,  0.4768, -0.1059],
                            [-0.3721,  0.5836, -0.1078],
                            [-0.2199,  0.6593, -0.3520],
                            [-0.0184,  0.7019, -0.4312],
                            [ 0.1829,  0.6588, -0.4117],
                            [ 0.3413,  0.5932, -0.2251],
                            [ 0.4535,  0.5002, -0.1201],
                            [ 0.5530,  0.3364, -0.0101],
                            [ 0.6051,  0.1617,  0.0017],
                            [ 0.6010,  0.0050,  0.2182],
                            [ 0.7230, -0.1830,  0.5235],
                            [ 0.7264, -0.3669,  0.3882],
                            [-0.5741, -0.5247, -0.1624],
                            [-0.4902, -0.6011, -0.3335],
                            [-0.3766, -0.6216, -0.4337],
                            [-0.2890, -0.6006, -0.4818],
                            [-0.1981, -0.5750, -0.5065],
                            [ 0.1583, -0.5989, -0.5168],
                            [ 0.2487, -0.6201, -0.4938],
                            [ 0.3631, -0.6215, -0.4385],
                            [ 0.4734, -0.6011, -0.3499],
                            [ 0.5571, -0.5475, -0.1870],
                            [-0.0182, -0.3929, -0.5284],
                            [ 0.0050, -0.2602, -0.6295],
                            [-0.0181, -0.1509, -0.7110],
                            [-0.0181, -0.0620, -0.7463],
                            [-0.1305,  0.0272, -0.5205],
                            [-0.0647,  0.0506, -0.5580],
                            [ 0.0049,  0.0500, -0.5902],
                            [ 0.0480,  0.0504, -0.5732],
                            [ 0.1149,  0.0275, -0.5329],
                            [-0.4233, -0.3598, -0.2748],
                            [-0.3783, -0.4226, -0.3739],
                            [-0.2903, -0.4217, -0.3799],
                            [-0.2001, -0.3991, -0.3561],
                            [-0.2667, -0.3545, -0.3658],
                            [-0.3764, -0.3536, -0.3441],
                            [ 0.1835, -0.3995, -0.3551],
                            [ 0.2501, -0.4219, -0.3741],
                            [ 0.3411, -0.4223, -0.3760],
                            [ 0.4082, -0.3987, -0.3338],
                            [ 0.3410, -0.3550, -0.3626],
                            [ 0.2488, -0.3763, -0.3652],
                            [-0.2374,  0.2695, -0.4086],
                            [-0.1736,  0.2257, -0.5026],
                            [-0.0644,  0.1823, -0.5703],
                            [ 0.0049,  0.2052, -0.5784],
                            [ 0.0479,  0.1826, -0.5739],
                            [ 0.1563,  0.2245, -0.5130],
                            [ 0.2441,  0.2697, -0.4012],
                            [ 0.1572,  0.3153, -0.4905],
                            [ 0.0713,  0.3393, -0.5457],
                            [ 0.0050,  0.3398, -0.5557],
                            [-0.0846,  0.3391, -0.5393],
                            [-0.1505,  0.3151, -0.4926],
                            [-0.2374,  0.2695, -0.4086],
                            [-0.0845,  0.2493, -0.5288],
                            [ 0.0050,  0.2489, -0.5514],
                            [ 0.0711,  0.2489, -0.5354],
                            [ 0.2245,  0.2698, -0.4106],
                            [ 0.0711,  0.2489, -0.5354],
                            [ 0.0050,  0.2489, -0.5514],
                            [-0.0645,  0.2489, -0.5364]]),
                            'nms_threshold':0.6,
                            'nms_inclusion_threshold':0.05,
                            'top_k':5000,
                            'keep_top_k':750,
                            'border_size':100,
                            'return_dim':3,
                            'device':'cpu'}

class Img2Pose:
    def __init__(
        self,
        cfg=model_config['img2pose'],
        pretrained='huggingface',
        device="auto",
        detection_threshold=0.5,
        # nms_threshold=0.6,
        # nms_inclusion_threshold=0.05,
        # top_k=5000,
        # keep_top_k=750,
        # BORDER_SIZE=100,
        # DEPTH=18,
        # MAX_SIZE=1400,
        # MIN_SIZE=400,
        # RETURN_DIM=3,
        # POSE_MEAN=os.path.join(get_resource_path(), "WIDER_train_pose_mean_v1.npy"),
        # POSE_STDDEV=os.path.join(get_resource_path(), "WIDER_train_pose_stddev_v1.npy"),
        # THREED_FACE_MODEL=os.path.join(
        #     get_resource_path(), "reference_3d_68_points_trans.npy"
        # ),
        **kwargs,
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

        if pretrained == 'huggingface':
            backbone = resnet_fpn_backbone(backbone_name=f"resnet{cfg['depth']}", weights=None)
            self.model = FasterDoFRCNN(backbone=backbone,
                                    num_classes=2,
                                    min_size=cfg['min_size'],
                                    max_size=cfg['max_size'],
                                    pose_mean=cfg['pose_mean'],
                                    pose_stddev=cfg['pose_stddev'],
                                    threed_68_points=cfg['threed_points'],
                                    rpn_pre_nms_top_n_test=cfg['rpn_pre_nms_top_n_test'],
                                    rpn_post_nms_top_n_test=cfg['rpn_post_nms_top_n_test'],
                                    bbox_x_factor=cfg['bbox_x_factor'],
                                    bbox_y_factor=cfg['bbox_y_factor'],
                                    expand_forehead=cfg['expand_forehead'])
            # self.model = WrappedModel(self.model)
            # self.model.from_pretrained('py-feat/img2pose')
            # Download the model file
            model_file = hf_hub_download(repo_id= "py-feat/img2pose", filename="model.safetensors")

            # Load the model state dict from the SafeTensors file
            model_state_dict = load_file( model_file)

            # Initialize the model
            self.model.load_state_dict(model_state_dict)
            self.model.eval()
        else:
            self.model = img2poseModel(
                cfg['depth'],
                cfg['min_size'],
                cfg['max_size'],
                pose_mean=cfg['pose_mean'],
                pose_stddev=cfg['pose_stddev'],
                threed_68_points=cfg['threed_points'],
                device=self.device,
                **kwargs,
            )

            # Load the constrained model
            model_file = "img2pose_v1_ft_300w_lp.pth" if cfg['constrained'] else "img2pose_v1.pth"
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
            self.RETURN_DIM,
        ) = (
            detection_threshold,
            cfg['nms_threshold'],
            cfg['nms_inclusion_threshold'],
            cfg['top_k'],
            cfg['keep_top_k'],
            cfg['min_size'],
            cfg['max_size'],
            cfg['border_size'],
            cfg['return_dim'],
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
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['fpn_model'].items()}
        self.model.fpn_model.load_state_dict(state_dict)

        if "optimizer" in checkpoint and optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif optimizer:
            print("Optimizer not found in model path - cannot be loaded")

    def __call__(self, img_):
        """Runs scale_and_predict on each image in the passed image list

        Args:
            img_ (np.ndarray): (B,C,H,W), B is batch number, H is image height, W is width and C is channel.

        Returns:
            tuple: (faces, poses) - 3D lists (B, F, bbox) or (B, F, face pose) where B is batch/ image number and
                                    F is face number
        """

        # Notes: vectorized version runs, but only returns results from a single image. Switching back to list version for now.
        # preds = self.scale_and_predict(img_)
        # return preds["boxes"], preds["poses"]
        faces = []
        poses = []
        for img in img_:
            preds = self.scale_and_predict(img)
            faces.append(preds["boxes"])
            poses.append(preds["poses"])

        return faces, poses

    def scale_and_predict(self, img, euler=True):
        """Runs a prediction on the passed image. Returns detected faces and associates poses.
        Args:
            img (tensor): A torch tensor image
            euler (bool): set to True to obtain euler angles, False to obtain rotation vector

        Returns:
            dict: key 'pose' contains array - [yaw, pitch, roll], key 'boxes' contains 2D array of bboxes
        """

        # Transform image to improve model performance. Resize the image so that both dimensions are in the range [MIN_SIZE, MAX_SIZE]
        scale = 1
        border_size = 0
        if min(img.shape[-2:]) < self.MIN_SIZE or max(img.shape[-2:]) > self.MAX_SIZE:
            logging.info(
                f"img2pose: RESCALING WARNING: img2pose has a min img size of {self.MIN_SIZE} and a max img size of {self.MAX_SIZE} but checked value is {img.shape[-2:]}."
            )
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
        # For device='mps'
        # Uncommenting this line at least gets img2pose running but errors with
        # Error: command buffer exited with error status.
        # The Metal Performance Shaders operations encoded on it may not have completed.

        # img = img.to(self.device)

        # Obtain prediction
        with torch.no_grad():
            pred = self.model([img])[0]
            # pred = self.model.predict([img])[0]
        # pred = self.model.predict(img)[0]
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

            if self.RETURN_DIM == 3:
                dof_pose = pose_pred[:3]  # pitch, roll, yaw (when euler=True)
            else:
                dof_pose = pose_pred[:]  # pitch, roll, yaw, x, y, z

            dof_pose = dof_pose.reshape(1, -1)
            det_pose.append(list(dof_pose.flatten()))

        return {"boxes": det_bboxes, "poses": det_pose}

    def set_threshold(self, threshold):
        """Alter the threshold for face detection.

        Args:
            threshold (float): A number representing the face detection score threshold to use

        Returns:
            None
        """
        self.detection_threshold = threshold
