# import os
import torch

# import json
from itertools import product as product
from math import ceil
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
import warnings
from huggingface_hub import PyTorchModelHubMixin

# with open(os.path.join(get_resource_path(), "model_config.json"), "r") as f:
# model_config = json.load(f)


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(
            out_channel // 4, out_channel // 4, stride=1, leaky=leaky
        )
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(
            in_channels_list[0], out_channels, stride=1, leaky=leaky
        )
        self.output2 = conv_bn1X1(
            in_channels_list[1], out_channels, stride=1, leaky=leaky
        )
        self.output3 = conv_bn1X1(
            in_channels_list[2], out_channels, stride=1, leaky=leaky
        )

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(
            output3, size=[output2.size(2), output2.size(3)], mode="nearest"
        )
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(
            output2, size=[output1.size(2), output1.size(3)], mode="nearest"
        )
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(
            inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module, PyTorchModelHubMixin):
    def __init__(self, cfg=None, phase="train"):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        backbone = None
        if cfg["name"] == "mobilenet0.25":
            backbone = MobileNetV1()
            if cfg["pretrain"]:
                checkpoint = torch.load(
                    "./weights/mobilenetV1X0.25_pretrain.tar",
                    map_location=torch.device("cpu"),
                )
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in checkpoint["state_dict"].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg["name"] == "Resnet50":
            import torchvision.models as models

            # TODO: Update to handle deprecation warning:
            # UserWarning: Arguments other than a weight enum or `None` for 'weights'
            # are deprecated since 0.13 and may be removed in the future. The current
            # behavior is equivalent to passing
            # `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use
            # `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                backbone = models.resnet50(weights=cfg["pretrain"])

        # TODO: Update to handle deprecation warning:
        # UserWarning: Using 'backbone_name' as positional parameter(s) is deprecated
        # since 0.13 and may be removed in the future. Please use keyword parameter(s)
        # instead.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.body = _utils.IntermediateLayerGetter(backbone, cfg["return_layers"])

        in_channels_stage2 = cfg["in_channel"]
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg["out_channel"]
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.LandmarkHead = self._make_landmark_head(
            fpn_num=3, inchannels=cfg["out_channel"]
        )

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1
        )

        if self.phase == "train":
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (
                bbox_regressions,
                F.softmax(classifications, dim=-1),
                ldm_regressions,
            )
        return output


def generate_prior_boxes(min_sizes, steps, clip, image_size, device="cpu"):
    """
    Generates prior boxes (anchors) based on the configuration and image size.

    Args:
        min_sizes (list): List of minimum sizes for anchors at each feature map level.
        steps (list): List of step sizes corresponding to feature map levels.
        clip (bool): Whether to clip the anchor values between 0 and 1.
        image_size (tuple): Image size in the format (height, width).
        device (str): Device to store the prior boxes (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor containing the prior boxes, shape: [num_priors, 4].
    """
    feature_maps = [
        [
            int(torch.ceil(torch.tensor(image_size[0] / step))),
            int(torch.ceil(torch.tensor(image_size[1] / step))),
        ]
        for step in steps
    ]

    anchors = []
    for k, f in enumerate(feature_maps):
        min_sizes_k = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes_k:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                cx = (j + 0.5) * steps[k] / image_size[1]
                cy = (i + 0.5) * steps[k] / image_size[0]
                anchors.append([cx, cy, s_kx, s_ky])

    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)

    if clip:
        anchors.clamp_(max=1, min=0)

    return anchors


def decode_boxes(loc, priors, variances):
    """
    Decodes bounding box predictions using priors and variances.

    Args:
        loc (torch.Tensor): Location predictions with shape [batch_size, num_priors, 4].
        priors (torch.Tensor): Prior boxes with shape [num_priors, 4].
        variances (list): List of variances for bounding box regression.

    Returns:
        torch.Tensor: Decoded bounding boxes with shape [batch_size, num_priors, 4].
    """
    boxes = torch.cat(
        (
            priors[:, :2].unsqueeze(0)
            + loc[:, :, :2] * variances[0] * priors[:, 2:].unsqueeze(0),
            priors[:, 2:].unsqueeze(0) * torch.exp(loc[:, :, 2:] * variances[1]),
        ),
        dim=2,
    )
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def decode_landmarks(pre, priors, variances):
    """
    Decodes landmark predictions using priors and variances.

    Args:
        pre (torch.Tensor): Landmark predictions with shape [batch_size, num_priors, 10].
        priors (torch.Tensor): Prior boxes with shape [num_priors, 4].
        variances (list): List of variances for landmark regression.

    Returns:
        torch.Tensor: Decoded landmarks with shape [batch_size, num_priors, 10].
    """
    priors_cxcy = priors[:, :2].unsqueeze(0).unsqueeze(2)  # shape: [1, num_priors, 1, 2]
    landm_deltas = pre.view(
        pre.size(0), -1, 5, 2
    )  # shape: [batch_size, num_priors, 5, 2]
    landms = priors_cxcy + landm_deltas * variances[0] * priors[:, 2:].unsqueeze(
        0
    ).unsqueeze(2)
    return landms.view(pre.size(0), -1, 10)


def batched_nms(boxes, scores, landmarks, batch_size, nms_threshold, keep_top_k):
    """
    Applies Non-Maximum Suppression (NMS) in a batched manner to the bounding boxes.

    Args:
        boxes (torch.Tensor): Bounding boxes with shape [batch_size, num_boxes, 4].
        scores (torch.Tensor): Confidence scores with shape [batch_size, num_boxes].
        landmarks (torch.Tensor): Landmarks with shape [batch_size, num_boxes, 10].
        batch_size (int): Number of batches.
        nms_threshold (float): NMS IoU threshold.
        keep_top_k (int): Maximum number of boxes to keep after NMS.

    Returns:
        tuple: (final_boxes, final_scores, final_landmarks) after NMS.
    """
    final_boxes, final_scores, final_landmarks = [], [], []

    for i in range(batch_size):
        dets = torch.cat([boxes[i], scores[i].unsqueeze(1)], dim=1)
        keep = torch.ops.torchvision.nms(dets[:, :4], dets[:, 4], nms_threshold)
        keep = keep[:keep_top_k]

        final_boxes.append(dets[keep, :5])
        final_scores.append(dets[keep, 4])
        final_landmarks.append(landmarks[i][keep])

    return final_boxes, final_scores, final_landmarks


def rescale_boxes_to_image_size(boxes, im_width, im_height):
    """
    Rescales the bounding boxes to match the original image size.

    Args:
        boxes (torch.Tensor): Bounding boxes with shape [num_boxes, 5].
        im_width (int): Width of the original image.
        im_height (int): Height of the original image.

    Returns:
        torch.Tensor: Rescaled bounding boxes with shape [num_boxes, 5].
    """
    scale_factors = torch.tensor(
        [
            im_width / im_height,
            im_height / im_width,
            im_width / im_height,
            im_height / im_width,
        ],
        device=boxes.device,
    )
    boxes[:, :4] *= scale_factors
    return boxes


def postprocess_retinaface(
    predicted_locations,
    predicted_scores,
    predicted_landmarks,
    face_config,
    img,
    device="cpu",
):
    """
    Postprocesses the RetinaFace model outputs by decoding the predictions, applying NMS, and rescaling boxes.

    Args:
        predicted_locations (torch.Tensor): Predicted location (bbox) outputs from the model.
        predicted_scores (torch.Tensor): Predicted confidence scores from the model.
        predicted_landmarks (torch.Tensor): Predicted landmarks from the model.
        face_config (dict): Configuration settings for face detection (e.g., thresholds, variances).
        img (torch.Tensor): The input image tensor.
        device (str): Device for computation (e.g., 'cuda', 'cpu', 'mps').

    Returns:
        dict: Dictionary containing 'boxes', 'scores', and 'landmarks' after postprocessing.
    """
    im_height, im_width = img.shape[-2:]

    # Move scale tensor to the specified device
    scale = torch.Tensor([im_height, im_width, im_height, im_width]).to(device)

    batch_size = predicted_locations.size(0)

    # Generate prior boxes
    priors = generate_prior_boxes(
        face_config["min_sizes"],
        face_config["steps"],
        face_config["clip"],
        image_size=(im_height, im_width),
        device=device,
    )

    # Decode boxes and landmarks
    boxes = decode_boxes(predicted_locations, priors, face_config["variance"]).to(device)
    boxes = boxes * scale / face_config["resize"]

    scores = predicted_scores[:, :, 1]  # Positive class scores
    landmarks = decode_landmarks(predicted_landmarks, priors, face_config["variance"]).to(
        device
    )

    # Move scale1 tensor to the specified device
    scale1 = torch.tensor([img.shape[3], img.shape[2]] * 5, device=device)
    landmarks = landmarks * scale1 / face_config["resize"]

    # Filter by confidence threshold
    mask = scores > face_config["confidence_threshold"]
    boxes = boxes[mask].view(batch_size, -1, 4)
    scores = scores[mask].view(batch_size, -1)
    landmarks = landmarks[mask].view(batch_size, -1, 10)

    # Keep top-K before NMS
    top_k_inds = torch.argsort(scores, dim=1, descending=True)[:, : face_config["top_k"]]
    boxes = torch.gather(boxes, 1, top_k_inds.unsqueeze(-1).expand(-1, -1, 4))
    scores = torch.gather(scores, 1, top_k_inds)
    landmarks = torch.gather(landmarks, 1, top_k_inds.unsqueeze(-1).expand(-1, -1, 10))

    # Apply NMS
    final_boxes, final_scores, final_landmarks = batched_nms(
        boxes,
        scores,
        landmarks,
        batch_size,
        face_config["nms_threshold"],
        face_config["keep_top_k"],
    )

    # Rescale boxes
    final_boxes_tensor = torch.cat(final_boxes, dim=0)
    final_rescaled_boxes = rescale_boxes_to_image_size(
        final_boxes_tensor, im_width, im_height
    )

    return {
        "boxes": final_rescaled_boxes,
        "scores": torch.cat(final_scores, dim=0),
        "landmarks": torch.cat(final_landmarks, dim=0),
    }


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase="train"):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
