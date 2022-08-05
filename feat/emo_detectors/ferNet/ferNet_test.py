from hmac import trans_36
from feat.emo_detectors.ferNet.ferNet_model import fer_net
import torch
import torch.nn as nn
import numpy as np
from feat.transforms import Rescale
from feat.utils import set_torch_device
from feat.utils.io import get_resource_path
from feat.utils.image_operations import (
    convert68to49,
    align_face_49pts,
)
import os
from torchvision.transforms import Compose, Resize
import torch


class ferNetModule(nn.Module):
    def __init__(self, device="auto", img_size=200) -> None:
        """
        Initialize model. Loads model weights
        """
        super(ferNetModule, self).__init__()

        self.device = set_torch_device(device)
        self.img_size = img_size
        self.net = fer_net(in_chs=3, num_classes=7, img_size=img_size)
        self.net.load_state_dict(
            torch.load(
                os.path.join(get_resource_path(), "best_ferModel.pth"),
                map_location=self.device,
            )
        )
        self.net.eval()

    def detect_emo(self, img, landmarks):
        """
        This documentation needs to be updated. Not accurate. Image size does not seem to matter.
        Also takes landmarks and not face boxes

        Our model is trained on black/white fer data. So the function
        first converts imgs to grayscale, crops imgs to detected face,
        and resizes imgs to 48x48 (the training size).
        Args:
            imgs: processed image (using JAANet pipeline)
            detected_face: the bounding box of detected face used for
                            cropping
            img_w: the image width after resizing
            img_h: the image height after resizing
        Returns:
            pred_emo_softmax: probablilities for each emotion class.
        """

        length_index = [len(x) for x in landmarks]
        length_cumu = np.cumsum(length_index)

        flat_landmarks = np.array([item for sublist in landmarks for item in sublist])

        aligned_imgs = None
        new_landmark_list = []
        for i in range(flat_landmarks.shape[0]):

            frame_assignment = np.where(i <= length_cumu)[0][0]  # which frame is it?

            landmark_49 = convert68to49(flat_landmarks[i]).flatten()
            new_img, new_landmarks = align_face_49pts(
                img[frame_assignment].unsqueeze(0), landmark_49, img_size=self.img_size
            )

            new_landmark_list.append(new_landmarks)
            if aligned_imgs is None:
                aligned_imgs = new_img
            else:
                aligned_imgs = torch.cat((aligned_imgs, new_img), 0)

        new_landmark_list = torch.from_numpy(np.concatenate(new_landmark_list)).type(
            torch.float32
        )

        pred_emo_softmax = (
            nn.functional.softmax(self.net(aligned_imgs), dim=1)
            .cpu()
            .float()
            .data.numpy()
        )

        return pred_emo_softmax
