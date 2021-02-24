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
import cv2
import os
from feat.face_detectors.FaceBoxes.FaceBoxes_model import FaceBoxesNet
from feat.face_detectors.FaceBoxes.FaceBoxes_utils import (
    PriorBox,
    decode,
    Timer,
    load_model,
    nms,
)
from feat.utils import get_resource_path

# some global configs
confidence_threshold = 0.05
top_k = 5000
keep_top_k = 750
nms_threshold = 0.3
vis_thres = 0.5
resize = 1
scale_flag = True
HEIGHT, WIDTH = 720, 1080
pretrained_path = os.path.join(get_resource_path(), "FaceBoxesProd.pth")
cfg = {
    "name": "FaceBoxes",
    "min_sizes": [[32, 64, 128], [256], [512]],
    "steps": [32, 64, 128],
    "variance": [0.1, 0.2],
    "clip": False,
}


def viz_bbox(img, dets, wfp="out.jpg"):
    # show
    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imwrite(wfp, img)
    print(f"Viz bbox to {wfp}")


class FaceBoxes:
    def __init__(self, timer_flag=False):
        torch.set_grad_enabled(False)

        net = FaceBoxesNet(
            phase="test", size=None, num_classes=2
        )  # initialize detector
        self.net = load_model(net, pretrained_path=pretrained_path, load_to_cpu=True)
        self.net.eval()
        # print('Finished loading model!')

        self.timer_flag = timer_flag

    def __call__(self, img_):
        img_raw = img_.copy()

        # scaling to speed up
        scale = 1
        if scale_flag:
            h, w = img_raw.shape[:2]
            if h > HEIGHT:
                scale = HEIGHT / h
            if w * scale > WIDTH:
                scale *= WIDTH / (w * scale)
            # print(scale)
            if scale == 1:
                img_raw_scale = img_raw
            else:
                h_s = int(scale * h)
                w_s = int(scale * w)
                # print(h_s, w_s)
                img_raw_scale = cv2.resize(img_raw, dsize=(w_s, h_s))
                # print(img_raw_scale.shape)

            img = np.float32(img_raw_scale)
        else:
            img = np.float32(img_raw)

        # forward
        _t = {"forward_pass": Timer(), "misc": Timer()}
        im_height, im_width, _ = img.shape
        scale_bbox = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        )
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        _t["forward_pass"].tic()
        loc, conf = self.net(img)  # forward pass
        _t["forward_pass"].toc()
        _t["misc"].tic()
        priorbox = PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg["variance"])
        if scale_flag:
            boxes = boxes * scale_bbox / scale / resize
        else:
            boxes = boxes * scale_bbox / resize

        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        _t["misc"].toc()

        if self.timer_flag:
            print(
                "Detection: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s".format(
                    1, 1, _t["forward_pass"].average_time, _t["misc"].average_time
                )
            )

        # filter using vis_thres
        det_bboxes = []
        for b in dets:
            if b[4] > vis_thres:
                xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                bbox = [xmin, ymin, xmax, ymax, score]
                det_bboxes.append(bbox)

        return det_bboxes


# if __name__ == '__main__':
# face_boxes = FaceBoxes(timer_flag=True)
# fn = 'trump_hillary.jpg'
# img_fp = f'../examples/inputs/{fn}'
# img = cv2.imread(img_fp)
# dets = face_boxes(img)  # xmin, ymin, w, h
# # print(dets)
# wfn = fn.replace('.jpg', '_det.jpg')
# wfp = osp.join('../examples/results', wfn)
# viz_bbox(img, dets, wfp)
