import torch
import pandas as pd
import numpy as np
from feat.au_detectors.DRML.DRML_model import DRML_net
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from feat.utils import get_resource_path
import cv2
import feat
import os

pretrained_path = os.path.join(get_resource_path(), 'DRMLNetParams.pth')

class DRMLNet(nn.Module):
    def __init__(self) -> None:
        """
        Initialize.
        Args:
            img_data: numpy array image data files of shape (N,3,W,H)
            land_data: numpy array landmark data of shape (N, 49*2)
        """
        #self.imgs = img_data
        #self.land_data = land_data
        self.params = {
                "config_au_num" : 12,
                "config_write_path_prefix" : get_resource_path()
        }

    def detect_au(self, imgs, landmarks=None):

        use_gpu = torch.cuda.is_available()
        config_write_path_prefix = self.params["config_write_path_prefix"]
        config_au_num = self.params["config_au_num"]

        drml_net = DRML_net(AU_num=config_au_num)

        if use_gpu:
            drml_net = drml_net.cuda()

        # Load parameters
        if use_gpu:
            drml_net.load_state_dict(torch.load(pretrained_path))
        else:
            drml_net.load_state_dict(torch.load(pretrained_path,map_location={'cuda:0': 'cpu'}))
        drml_net.eval()

        img_transforms = transforms.Compose([
                transforms.CenterCrop(170),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        ])
        imgs = Image.fromarray(imgs)
        input = img_transforms(imgs)
        if len(input.shape) < 4:
            input.unsqueeze_(0)

        if use_gpu:
            input = input.cuda()

        pred_au = drml_net(input)
        all_pred_au = pred_au.data.cpu().float()
        all_pred_au = (all_pred_au[:,1,:]).exp()
        
        all_pred_au[all_pred_au<0.5]=0
        all_pred_au[all_pred_au>=0.5]=1
        #all_pred_au = all_pred_au.transpose((1,0))
        all_pred_au = all_pred_au.data.numpy()
        return all_pred_au