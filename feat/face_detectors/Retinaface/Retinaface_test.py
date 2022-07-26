from __future__ import print_function
import os
import torch
import numpy as np

# import time
# import feat
from feat.face_detectors.Retinaface.Retinaface_model import PriorBox, RetinaFace
from feat.face_detectors.Retinaface.Retinaface_utils import (
    py_cpu_nms,
    decode,
    decode_landm,
)
from feat.utils import (
    get_resource_path,
    convert_image_to_tensor,
    convert_color_vector_to_tensor,
    set_torch_device,
)


class Retinaface:
    def __init__(
        self,
        device="auto",
        timer_flag=False,
        resize=1,
        vis_threshold=0.5,
        nms_threshold=0.3,
        keep_top_k=750,
        top_k=5000,
        confidence_threshold=0.05,
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
            self.timer_flag,
            self.resize,
            self.vis_threshold,
            self.nms_threshold,
            self.keep_top_k,
            self.top_k,
            self.confidence_threshold,
        ) = (
            timer_flag,
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
            img: (B,C, H,W,C), B is batch number, C is channel, H is image height, and W is width
        """
        img = convert_image_to_tensor(img)
        img = img.type(torch.float32)

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
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
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
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[: self.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        if self.timer_flag:
            print(
                "Detection: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s".format(
                    1, 1, _t["forward_pass"].average_time, _t["misc"].average_time
                )
            )

        # filter using vis_threshold
        det_bboxes = []
        for b in dets:
            if b[4] > self.vis_threshold:
                xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                bbox = [xmin, ymin, xmax, ymax, score]
                det_bboxes.append(bbox)

        return det_bboxes
