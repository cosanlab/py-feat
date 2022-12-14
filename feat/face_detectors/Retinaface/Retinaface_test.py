from __future__ import print_function
import os
import torch
import numpy as np

# import time
# import feat
from feat.face_detectors.Retinaface.Retinaface_model import PriorBox, RetinaFace
from feat.face_detectors.Retinaface.Retinaface_utils import decode_landm
from feat.utils import set_torch_device
from feat.utils.io import get_resource_path
from feat.utils.image_operations import (
    convert_color_vector_to_tensor,
    py_cpu_nms,
    decode,
)


class Retinaface:
    def __init__(
        self,
        device="auto",
        resize=1,
        vis_threshold=0.5,
        nms_threshold=0.4,
        keep_top_k=750,
        top_k=5000,
        confidence_threshold=0.02,
    ):
        """
        Function to perform inference with RetinaFace

        Args:
            device: (str)
            timer_flag: (bool)
            resize: (int)
            vis_threshold: (float)
            nms_threshold: (float)
            keep_top_k: (float)
            top_k: (float)
            confidence_threshold: (float)

        """

        torch.set_grad_enabled(False)
        self.cfg = {
            "name": "mobilenet0.25",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
            "loc_weight": 2.0,
            "gpu_train": True,
            "batch_size": 32,
            "ngpu": 1,
            "epoch": 250,
            "decay1": 190,
            "decay2": 220,
            "image_size": 640,
            "pretrain": False,
            "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
            "in_channel": 32,
            "out_channel": 64,
        }

        # net and model
        self.device = set_torch_device(device=device)

        net = RetinaFace(cfg=self.cfg, phase="test")
        pretrained_dict = torch.load(
            os.path.join(get_resource_path(), f"mobilenet0.25_Final.pth"),
            map_location=self.device,
        )
        net.load_state_dict(pretrained_dict, strict=False)
        net = net.to(self.device)
        self.net = net.eval()

        # Set cutoff parameters
        (
            self.resize,
            self.vis_threshold,
            self.nms_threshold,
            self.keep_top_k,
            self.top_k,
            self.confidence_threshold,
        ) = (
            resize,
            vis_threshold,
            nms_threshold,
            keep_top_k,
            top_k,
            confidence_threshold,
        )

    def __call__(self, img):
        """
        forward function

        Args:
            img: (B,C,H,W), B is batch number, C is channel, H is image height, and W is width
        """

        img = torch.sub(img, convert_color_vector_to_tensor(np.array([123, 117, 104])))

        im_height, im_width = img.shape[-2:]
        scale = torch.Tensor([im_height, im_width, im_height, im_width])
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass
        total_boxes = []
        for i in range(loc.shape[0]):
            tmp_box = self._calculate_boxinfo(
                im_height=im_height,
                im_width=im_width,
                loc=loc[i],
                conf=conf[i],
                landms=landms[i],
                scale=scale,
                img=img,
            )
            total_boxes.append(tmp_box)

        return total_boxes

    def _calculate_boxinfo(self, im_height, im_width, loc, conf, landms, scale, img):
        """
        helper function to calculate deep learning results
        """

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg["variance"])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), priors.data, self.cfg["variance"])
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[: self.keep_top_k, :]

        # filter using vis_thres - rescale box size to be proportional to image size
        scale_x, scale_y = (im_width / im_height, im_height / im_width)
        det_bboxes = []
        for b in dets:
            if b[4] > self.vis_threshold:
                xmin, ymin, xmax, ymax, score = b
                det_bboxes.append(
                    [
                        xmin * scale_x,
                        ymin * scale_y,
                        xmax * scale_x,
                        ymax * scale_y,
                        score,
                    ]
                )

        return det_bboxes
