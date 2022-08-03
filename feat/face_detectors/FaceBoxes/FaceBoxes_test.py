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
from feat.face_detectors.FaceBoxes.FaceBoxes_model import FaceBoxesNet
from feat.face_detectors.FaceBoxes.FaceBoxes_utils import (
    PriorBox,
    decode,
    load_model,
    nms,
)
from feat.utils.io import get_resource_path
from feat.utils.image_operations import convert_color_vector_to_tensor


class FaceBoxes:
    def __init__(
        self,
        confidence_threshold=0.05,
        top_k=5000,
        keep_top_k=750,
        nms_threshold=0.3,
        vis_threshold=0.5,
        resize=1,
        HEIGHT=720,
        WIDTH=1080,
        scale=1,
        pretrained_path=os.path.join(get_resource_path(), "FaceBoxesProd.pth"),
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
        self.net = load_model(net, pretrained_path=pretrained_path, device=self.device)
        self.net.eval()

        (
            self.confidence_threshold,
            self.top_k,
            self.keep_top_k,
            self.nms_threshold,
            self.vis_threshold,
            self.resize,
            self.HEIGHT,
            self.WIDTH,
            self.scale,
        ) = (
            confidence_threshold,
            top_k,
            keep_top_k,
            nms_threshold,
            vis_threshold,
            resize,
            HEIGHT,
            WIDTH,
            scale,
        )

    def __call__(self, img):
        """
        img is of shape BxCxHxW --
        img is of shape BxHxWxC - old
        """

        img = torch.sub(img, convert_color_vector_to_tensor(np.array([104, 117, 123])))

        # scaling to speed up - this is now done in image loader
        # scale = 1
        # if scale_flag:
        #     h, w = img_raw.shape[1:3]
        #     if h > HEIGHT:
        #         scale = HEIGHT / h
        #     if w * scale > WIDTH:
        #         scale *= WIDTH / (w * scale)
        #     # print(scale)
        #     if scale == 1:
        #         img_raw_scale = img_raw
        #     else:
        #         h_s = int(scale * h)
        #         w_s = int(scale * w)
        #         # print(h_s, w_s)
        #         img_raw_scale = np.zeros((img_raw.shape[0], h_s, w_s, img_raw.shape[3]))
        #         for i in range(img_raw.shape[0]):
        #             img_raw_scale[i] = cv2.resize(img_raw[i, :, :, :], dsize=(w_s, h_s))
        #         # img_raw_scale = cv2.resize(img_raw, dsize=(w_s, h_s))
        #         # print(img_raw_scale.shape)

        #     img = np.float32(img_raw_scale)
        # else:
        #     img = np.float32(img_raw)

        # forward

        im_height, im_width = img.shape[-2:]
        scale = torch.Tensor([im_height, im_width, im_height, im_width])
        img = img.to(self.device)
        scale_bbox = scale.to(self.device)

        loc, conf = self.net(img)  # forward pass

        total_boxes = []
        for i in range(loc.shape[0]):
            tmp_box = self._calculate_boxinfo(
                im_height=im_height,
                im_width=im_width,
                loc=loc[i],
                conf=conf[i],
                scale=self.scale,
                img=img,
                scale_bbox=scale_bbox,
            )
            total_boxes.append(tmp_box)

        return total_boxes

    def _calculate_boxinfo(
        self, im_height, im_width, loc, conf, scale, img, scale_bbox
    ):

        priorbox = PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])

        boxes = boxes * scale_bbox / self.resize

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
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, self.nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[: self.keep_top_k, :]

        # filter using vis_thres
        det_bboxes = []
        for b in dets:
            if b[4] > self.vis_threshold:
                xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                bbox = [xmin, ymin, xmax, ymax, score]
                det_bboxes.append(bbox)

        return det_bboxes
