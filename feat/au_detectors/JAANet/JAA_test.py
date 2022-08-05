from feat.utils import set_torch_device
import numpy as np
import torch
import pandas as pd
import feat.au_detectors.JAANet.JAANet_model as network
import torch.nn as nn
from torchvision.transforms import Compose, Normalize
from feat.utils import set_torch_device
from feat.utils.io import get_resource_path
from feat.utils.image_operations import convert68to49, align_face_49pts
import os


class JAANet(nn.Module):
    def __init__(self, device="auto") -> None:
        """
        Initialize.

        Args:
            img_data: numpy array image data files of shape (N,3,W,H)
            land_data: numpy array landmark data of shape (N, 49*2)
        """
        # self.imgs = img_data
        # self.land_data = land_data
        super(JAANet, self).__init__()

        self.device = set_torch_device(device)

        self.params = {
            "config_unit_dim": 8,
            "config_crop_size": 176,
            "config_map_size": 44,
            "config_au_num": 12,
            "config_land_num": 49,
            "config_fill_coeff": 0.56,
            "config_write_path_prefix": get_resource_path(),
        }

        self.region_learning = network.network_dict["HMRegionLearning"](
            input_dim=3, unit_dim=self.params["config_unit_dim"]
        )
        self.align_net = network.network_dict["AlignNet"](
            crop_size=self.params["config_crop_size"],
            map_size=self.params["config_map_size"],
            au_num=self.params["config_au_num"],
            land_num=self.params["config_land_num"],
            input_dim=self.params["config_unit_dim"] * 8,
            fill_coeff=self.params["config_fill_coeff"],
        )
        self.local_attention_refine = network.network_dict["LocalAttentionRefine"](
            au_num=self.params["config_au_num"], unit_dim=self.params["config_au_num"]
        )
        self.local_au_net = network.network_dict["LocalAUNetv2"](
            au_num=self.params["config_au_num"],
            input_dim=self.params["config_unit_dim"] * 8,
            unit_dim=self.params["config_au_num"],
        )
        self.global_au_feat = network.network_dict["HLFeatExtractor"](
            input_dim=self.params["config_unit_dim"] * 8,
            unit_dim=self.params["config_unit_dim"],
        )
        self.au_net = network.network_dict["AUNet"](
            au_num=self.params["config_au_num"],
            input_dim=12000,
            unit_dim=self.params["config_unit_dim"],
        )

        self.region_learning.load_state_dict(
            torch.load(
                os.path.join(
                    self.params["config_write_path_prefix"], "region_learning.pth"
                ),
                map_location=self.device,
            )
        )
        self.align_net.load_state_dict(
            torch.load(
                os.path.join(self.params["config_write_path_prefix"], "align_net.pth"),
                map_location=self.device,
            )
        )
        self.local_attention_refine.load_state_dict(
            torch.load(
                os.path.join(
                    self.params["config_write_path_prefix"],
                    "local_attention_refine.pth",
                ),
                map_location=self.device,
            )
        )
        self.local_au_net.load_state_dict(
            torch.load(
                os.path.join(
                    self.params["config_write_path_prefix"], "local_au_net.pth"
                ),
                map_location=self.device,
            )
        )
        self.global_au_feat.load_state_dict(
            torch.load(
                os.path.join(
                    self.params["config_write_path_prefix"], "global_au_feat.pth"
                ),
                map_location=self.device,
            )
        )
        self.au_net.load_state_dict(
            torch.load(
                os.path.join(self.params["config_write_path_prefix"], "au_net.pth"),
                map_location=self.device,
            )
        )

        self.region_learning.eval()
        self.align_net.eval()
        self.local_attention_refine.eval()
        self.local_au_net.eval()
        self.global_au_feat.eval()
        self.au_net.eval()

    def detect_au(self, img, landmarks):

        transforms = Compose(
            [
                transforms.CenterCrop(self.params["config_crop_size"]),
                # transforms.ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        img = transforms(img / 255.0)

        length_index = [len(x) for x in landmarks]
        length_cumu = np.cumsum(length_index)
        flat_landmarks = np.array([item for sublist in landmarks for item in sublist])

        aligned_imgs = None
        new_landmark_list = []
        for i in range(flat_landmarks.shape[0]):
            frame_assignment = np.where(i <= length_cumu)[0][0]  # which frame is it?

            landmark_49 = convert68to49(flat_landmarks[i]).flatten()
            new_img, new_landmarks = align_face_49pts(
                img[frame_assignment].unsqueeze(0), landmark_49
            )

            new_landmark_list.append(new_landmarks)

            if aligned_imgs is None:
                aligned_imgs = new_img
            else:
                aligned_imgs = torch.cat((aligned_imgs, new_img), 0)

        print(aligned_imgs.shape, aligned_imgs.type())

        region_feat = self.region_learning(aligned_imgs)
        align_feat, align_output, aus_map = self.align_net(region_feat)

        output_aus_map = self.local_attention_refine(aus_map.detach())
        local_au_out_feat, local_aus_output = self.local_au_net(
            region_feat, output_aus_map
        )
        local_aus_output = (local_aus_output[:, 1, :]).exp()
        global_au_out_feat = self.global_au_feat(region_feat)
        concat_au_feat = torch.cat(
            (align_feat, global_au_out_feat, local_au_out_feat.detach()), 1
        )
        aus_output = self.au_net(concat_au_feat)
        aus_output = (aus_output[:, 1, :]).exp()
        all_output = aus_output.data.cpu().float()
        AUoccur_pred_prob = all_output.data.numpy()
        return AUoccur_pred_prob
