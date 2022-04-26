from feat.emo_detectors.ferNet.ferNet_model import fer_net
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from feat.utils import get_resource_path, face_rect_to_coords, convert68to49
import os
from torchvision import transforms
import math
import cv2
import torch
from torchvision.utils import save_image


class ferNetModule(nn.Module):
    def __init__(self) -> None:

        """
        Initialize model. Loads model weights
        """
        super(ferNetModule, self).__init__()

        self.pretrained_path = os.path.join(get_resource_path(), "best_ferModel.pth")

        self.net0 = fer_net(in_chs=3, num_classes=7, img_size=200)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.net0 = self.net0.cuda()
        if self.use_gpu:
            self.net0.load_state_dict(torch.load(self.pretrained_path))
        else:
            self.net0.load_state_dict(
                torch.load(self.pretrained_path, map_location={"cuda:0": "cpu"})
            )

    def align_face_49pts(self, img, img_land, box_enlarge=2.9, img_size=200):
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

        aligned_img = cv2.warpAffine(
            img,
            mat[0:2, :],
            (img_size, img_size),
            cv2.INTER_LINEAR,
            borderValue=(128, 128, 128),
        )

        land_3d = np.ones((int(len(img_land) / 2), 3))
        land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land) / 2), 2))
        mat_land_3d = np.mat(land_3d)
        new_land = np.array((mat * mat_land_3d.T).T)
        new_land = np.reshape(new_land[:, 0:2], len(img_land))

        return aligned_img, new_land

    def detect_emo(self, imgs, land_data, img_w=48, img_h=48, greyscale=True):
        """
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

        lenth_index = [len(ama) for ama in land_data]
        lenth_cumu = np.cumsum(lenth_index)

        flat_faces = np.array(
            [item for sublist in land_data for item in sublist]
        )  # Flatten the faces
        flat_faces = flat_faces.transpose(0, 2, 1)
        pt49_array = None

        img_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        input_torch = None
        land_torch = None
        for i in range(flat_faces.shape[0]):

            frame_assignment = np.where(i <= lenth_cumu)[0][0]  # which frame is it?

            land_convert = convert68to49(flat_faces[i]).T
            new_land_data = land_convert.flatten()
            # print("new land:",new_land_data)
            new_img, new_land = self.align_face_49pts(
                imgs[frame_assignment], new_land_data, img_size=200
            )
            # print("New Land:",new_land)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(new_img)
            input = img_transforms(im_pil)
            if len(input.shape) < 4:
                input.unsqueeze_(0)
            new_land = torch.from_numpy(new_land)

            if input_torch is None:
                input_torch = input
            else:
                input_torch = torch.cat((input_torch, input), 0)
            if land_torch is None:
                land_torch = new_land
            else:
                land_torch = torch.cat((land_torch, new_land), 0)

        self.net0.eval()

        if self.use_gpu:
            input_torch = input_torch.cuda()

        pred_emo = self.net0(input_torch)
        pred_emo_softmax = (
            nn.functional.softmax(pred_emo, dim=1).cpu().float().data.numpy()
        )

        return pred_emo_softmax
