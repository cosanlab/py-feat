import torch
from feat.au_detectors.DRML.DRML_model import DRML_net
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from feat.utils import get_resource_path
import os


class DRMLNet(nn.Module):
    def __init__(self) -> None:
        """
        Initialize.
        """
        super(DRMLNet, self).__init__()

        self.params = {
            "config_au_num": 12,
            "config_write_path_prefix": os.path.join(
                get_resource_path(), "DRMLNetParams.pth"
            ),
        }
        self.use_gpu = torch.cuda.is_available()
        self.drml_net = DRML_net(AU_num=self.params["config_au_num"])
        if self.use_gpu:
            self.drml_net.load_state_dict(torch.load(
                self.params["config_write_path_prefix"]))
            self.drml_net = self.drml_net.cuda()
        else:
            self.drml_net.load_state_dict(
                torch.load(self.params["config_write_path_prefix"], map_location={
                           "cuda:0": "cpu"})
            )
        self.drml_net.eval()

    def detect_au(self, imgs, landmarks=None):
        """
        Wrapper function that takes in imgs and produces AU occurence predictions.
        Args:
            imgs: processed images type 4d numpy array (BATCH, )
        Return:
            all_pred_au: AU occurence predictions for all AU classes
        """
        # Load parameters
        img_transforms = transforms.Compose(
            [
                transforms.CenterCrop(170),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]

        )
        if len(imgs.shape) < 4:
            imgs = np.expand_dims(imgs, 0)

        img_concat_tensor = None

        for batch in imgs.shape[0]:
            img = Image.fromarray(imgs[batch])
            input = img_transforms(imgs)
            input.unsqueeze_(0)
            if img_concat_tensor is None:
                img_concat_tensor = input
            else:
                img_concat_tensor = torch.cat((img_concat_tensor, input), 0)

        if self.use_gpu:
            img_concat_tensor = img_concat_tensor.cuda()

        pred_au = self.drml_net(img_concat_tensor)
        all_pred_au = pred_au.data.cpu().float()
        all_pred_au = all_pred_au.data.numpy()
        return all_pred_au
