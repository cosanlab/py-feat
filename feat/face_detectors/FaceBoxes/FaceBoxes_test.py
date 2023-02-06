# MIT License

# Copyright (c) 2017 Max deGroot, Ellis Brown
# Copyright (c) 2019 Zisian Wong, Shifeng Zhang
# Copyright (c) 2020 Jianzhu Guo, in Center for Biometrics and Security Research (CBSR)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Please check raw code at https://github.com/cleardusk/3DDFA_V2
# coding: utf-8

import torch
import numpy as np

# import cv2
from feat.utils import set_torch_device
import os
from feat.face_detectors.FaceBoxes.FaceBoxes_model import FaceBoxesNet, PriorBox

# from feat.face_detectors.FaceBoxes.FaceBoxes_utils import (
#     load_model,
#     nms,
# )
from feat.utils.io import get_resource_path
from feat.utils.image_operations import (
    convert_color_vector_to_tensor,
    decode,
    py_cpu_nms,
)


class FaceBoxes:
    def __init__(
        self,
        confidence_threshold=0.05,
        top_k=5000,
        keep_top_k=750,
        nms_threshold=0.3,
        detection_threshold=0.5,
        resize=1,
        device="auto",
    ):
        self.cfg = {
            "name": "FaceBoxes",
            "min_sizes": [[32, 64, 128], [256], [512]],
            "steps": [32, 64, 128],
            "variance": [0.1, 0.2],
            "clip": False,
        }

        torch.set_grad_enabled(False)
        self.device = set_torch_device(device)

        # initialize detector
        net = FaceBoxesNet(phase="test", size=None, num_classes=2)
        pretrained_dict = torch.load(
            os.path.join(get_resource_path(), "FaceBoxesProd.pth"),
            map_location=self.device,
        )
        net.load_state_dict(pretrained_dict, strict=False)
        net = net.to(self.device)
        self.net = net.eval()

        (
            self.confidence_threshold,
            self.top_k,
            self.keep_top_k,
            self.nms_threshold,
            self.detection_threshold,
            self.resize,
        ) = (
            confidence_threshold,
            top_k,
            keep_top_k,
            nms_threshold,
            detection_threshold,
            resize,
        )

    def __call__(self, img):
        """
        img is of shape BxCxHxW --
        """

        img = torch.sub(img, convert_color_vector_to_tensor(np.array([104, 117, 123])))

        im_height, im_width = img.shape[-2:]

        scale = torch.Tensor([im_height, im_width, im_height, im_width])
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf = self.net(img)  # forward pass

        total_boxes = []
        for i in range(loc.shape[0]):
            tmp_box = self._calculate_boxinfo(
                im_height=im_height,
                im_width=im_width,
                loc=loc[i],
                conf=conf[i],
                scale=scale,
            )
            total_boxes.append(tmp_box)

        return total_boxes

    def _calculate_boxinfo(self, im_height, im_width, loc, conf, scale):

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg["variance"])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[: self.keep_top_k, :]

        # filter using detection_threshold - rescale box size to be proportional to image size
        scale_x, scale_y = (im_width / im_height, im_height / im_width)
        det_bboxes = []
        for b in dets:
            if b[4] > self.detection_threshold:
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
