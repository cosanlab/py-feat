from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision.models.detection._utils as det_utils
from torch import nn
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from .generalized_rcnn import GeneralizedRCNN
from .rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from .pose_operations import transform_pose_global_project_bbox
from huggingface_hub import PyTorchModelHubMixin
from torchvision.ops import nms
from feat.utils.image_operations import rotvec_to_euler_angles


class FastRCNNDoFPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNDoFPredictor, self).__init__()
        hidden_layer = 256
        self.dof_pred = nn.Sequential(
            nn.Linear(in_channels, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, num_classes * 6),
        )

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        dof = self.dof_pred(x)

        return dof


class FastRCNNClassPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNClassPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        return scores


class FasterDoFRCNN(GeneralizedRCNN, PyTorchModelHubMixin):
    def __init__(
        self,
        backbone=None,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=6000,
        rpn_pre_nms_top_n_test=6000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.4,
        rpn_fg_iou_thresh=0.5,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=1000,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        pose_mean=None,
        pose_stddev=None,
        threed_68_points=None,
        threed_5_points=None,
        bbox_x_factor=1.1,
        bbox_y_factor=1.1,
        expand_forehead=0.3,
    ):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    "num_classes should be None when box_predictor is specified"
                )
        else:
            if box_predictor is None:
                raise ValueError(
                    "num_classes should not be None when box_predictor "
                    "is not specified"
                )

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = {
            "training": rpn_pre_nms_top_n_train,
            "testing": rpn_pre_nms_top_n_test,
        }
        rpn_post_nms_top_n = {
            "training": rpn_post_nms_top_n_train,
            "testing": rpn_post_nms_top_n_test,
        }

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNDoFPredictor(representation_size, num_classes)

        roi_heads = DOFRoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            out_channels,
            pose_mean=pose_mean,
            pose_stddev=pose_stddev,
            threed_68_points=threed_68_points,
            threed_5_points=threed_5_points,
            bbox_x_factor=bbox_x_factor,
            bbox_y_factor=bbox_y_factor,
            expand_forehead=expand_forehead,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterDoFRCNN, self).__init__(backbone, rpn, roi_heads, transform)

    def set_max_min_size(self, max_size, min_size):
        self.min_size = (min_size,)
        self.max_size = max_size

        self.transform.min_size = self.min_size
        self.transform.max_size = self.max_size


class DOFRoIHeads(RoIHeads):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        out_channels,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        pose_mean=None,
        pose_stddev=None,
        threed_68_points=None,
        threed_5_points=None,
        bbox_x_factor=1.1,
        bbox_y_factor=1.1,
        expand_forehead=0.3,
    ):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        num_classes = 2
        self.class_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        self.class_head = TwoMLPHead(out_channels * resolution**2, representation_size)
        self.class_predictor = FastRCNNClassPredictor(representation_size, num_classes)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

        self.pose_mean = pose_mean
        self.pose_stddev = pose_stddev
        self.threed_68_points = threed_68_points
        self.threed_5_points = threed_5_points

        self.bbox_x_factor = bbox_x_factor
        self.bbox_y_factor = bbox_y_factor
        self.expand_forehead = expand_forehead

    def postprocess_detections(
        self,
        class_logits,  # type: torch.Tensor
        dof_regression,  # type: torch.Tensor
        proposals,  # type: List[torch.Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        device = class_logits.device

        # Move proposals to the correct device
        proposals = [p.to(device) for p in proposals]

        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = torch.cat(proposals, dim=0)
        N = dof_regression.shape[0]
        pred_boxes = pred_boxes.reshape(N, -1, 4)
        pred_dofs = dof_regression.reshape(N, -1, 6)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_dofs_list = pred_dofs.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_dofs = []
        for boxes, dofs, scores, image_shape in zip(
            pred_boxes_list, pred_dofs_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            dofs = dofs[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            dofs = dofs.reshape(-1, 6)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, dofs, scores, labels = (
                boxes[inds],
                dofs[inds],
                scores[inds],
                labels[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, dofs, scores, labels = (
                boxes[keep],
                dofs[keep],
                scores[keep],
                labels[keep],
            )

            # create boxes from the predicted poses
            boxes, dofs = transform_pose_global_project_bbox(
                boxes,
                dofs,
                self.pose_mean,
                self.pose_stddev,
                image_shape,
                self.threed_68_points,
                bbox_x_factor=self.bbox_x_factor,
                bbox_y_factor=self.bbox_y_factor,
                expand_forehead=self.expand_forehead,
            )

            # Ensure all tensors are on the correct device
            boxes = boxes.to(device)
            scores = scores.to(device)
            labels = labels.to(device)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]

            boxes, dofs, scores, labels = (
                boxes[keep],
                dofs[keep],
                scores[keep],
                labels[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_dofs.append(dofs)

        return all_boxes, all_dofs, all_scores, all_labels

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        dof_regression = self.box_predictor(box_features)
        class_features = self.class_roi_pool(features, proposals, image_shapes)
        class_features = self.class_head(class_features)
        class_logits = self.class_predictor(class_features)
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])

        losses = {}
        boxes, dofs, scores, labels = self.postprocess_detections(
            class_logits, dof_regression, proposals, image_shapes
        )
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                    "dofs": dofs[i],
                }
            )

        return result, losses


def postprocess_img2pose(
    img2pose_output,
    nms_inclusion_threshold=0.05,
    top_k=5000,
    nms_threshold=0.3,
    detection_threshold=0.5,
):
    """Post-process output from img2pose model to threshold and convert dof to angles"""

    # Sort boxes by score
    include = img2pose_output["scores"] > nms_inclusion_threshold
    boxes = img2pose_output["boxes"][include]
    scores = img2pose_output["scores"][include]
    dofs = img2pose_output["dofs"][include]
    _, order = torch.sort(scores, descending=True)[:top_k]
    boxes, scores, dofs = boxes[order], scores[order], dofs[order]

    # Perform Non-Maximum Suppression
    keep = nms(boxes, scores, nms_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    dofs = dofs[keep]

    # Threshold
    boxes = boxes[scores >= detection_threshold]
    dofs = dofs[
        scores >= detection_threshold
    ]  # return 6 rotation and translation parameters
    # dofs = dofs[scores >= detection_threshold][:, :3] # Only returning xyz for now not translation
    scores = scores[scores >= detection_threshold]

    # Convert Rotation Vector to Euler (Radians)
    dofs = torch.cat((rotvec_to_euler_angles(dofs[:, :3]), dofs[:, 3:]), dim=1)
    return {"boxes": boxes, "dofs": dofs, "scores": scores}
    # return {'boxes':[list(t.detach().cpu().numpy()) for t in list(torch.unbind(boxes, dim=0))],
    # 'dofs':[list(t.detach().cpu().numpy()) for t in list(torch.unbind(dofs, dim=0))],
    # 'scores':list(img2pose_output['scores'].detach().cpu().numpy())}
