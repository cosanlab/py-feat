# Import xgboost FIRST, before any module that pulls in torch. On
# Python 3.13 + macOS, if torch's OpenMP/MKL runtime loads before
# xgboost's, subsequent xgboost.Booster.__setstate__ calls during
# skops.io.load(...) crash the process with SIGSEGV. The first OpenMP
# runtime wins; loading xgboost first puts its runtime in charge and
# the skops load path stays stable. Verified empirically: with this
# order the load succeeds, with `import torch` ahead of it the load
# is exit 139.
import xgboost  # noqa: F401  (must come before torch-using imports)

import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from collections import OrderedDict

from feat.emo_detectors.ResMaskNet.resmasknet_test import (
    ResMasking,
)
from feat.identity_detectors.facenet.facenet_model import InceptionResnetV1
from feat.identity_detectors.arcface.arcface_model import ArcFace
from feat.facepose_detectors.img2pose.deps.models import (
    FasterDoFRCNN,
    postprocess_img2pose,
)
from feat.au_detectors.StatLearning.SL_test import XGBClassifier, SVMClassifier
from feat.emo_detectors.StatLearning.EmoSL_test import EmoSVMClassifier
from feat.face_detectors.Retinaface.Retinaface_test import Retinaface
from feat.landmark_detectors.mobilefacenet_test import MobileFaceNet
from feat.landmark_detectors.basenet_test import MobileNet_GDConv
from feat.landmark_detectors.pfld_compressed_test import PFLDInference
from feat.pretrained import load_model_weights, AU_LANDMARK_MAP
from feat.utils import (
    set_torch_device,
    openface_2d_landmark_columns,
    FEAT_EMOTION_COLUMNS,
    FEAT_FACEBOX_COLUMNS,
    FEAT_FACEPOSE_COLUMNS_6D,
    FEAT_GAZE_COLUMNS,
    FEAT_IDENTITY_COLUMNS,
    hf_hub_download_with_fallback,
    N_OPENFACE_LANDMARKS,
    N_OPENFACE_LANDMARKS_2D_FLAT,
)
from feat.utils.io import get_resource_path
from feat.utils.image_operations import (
    convert_image_to_tensor,
    extract_face_from_bbox_torch,
    inverse_transform_landmarks_torch,
    convert_bbox_output,
    compute_original_image_size,
    invert_padding_to_results,
    per_face_padding_inversion_terms,
    HOGLayer,
)
from feat.utils.face_mask import extract_hog_features_batched
from feat.data import Fex, ImageDataset, TensorDataset, VideoDataset
from skops.io import load, get_untrusted_types
from safetensors.torch import load_file
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import Compose, Normalize
import sys
import warnings
from pathlib import Path

sys.modules["__main__"].__dict__["XGBClassifier"] = XGBClassifier
sys.modules["__main__"].__dict__["SVMClassifier"] = SVMClassifier
sys.modules["__main__"].__dict__["EmoSVMClassifier"] = EmoSVMClassifier


def _patch_xgboost_setstate_for_skops():
    """Defend xgboost.Booster.__setstate__ against a bytearray-lifetime bug.

    skops's load flow calls ``Booster.__setstate__({"handle": bytearray, ...})``.
    xgboost then does ``ptr = (c_char * len(buf)).from_buffer(buf)`` and passes
    ``ptr`` into a C call. On Python 3.13 the bytearray can be reallocated
    mid-call, producing intermittent SIGSEGV. Copying the buffer into a
    freshly-allocated bytearray narrows the window. The underlying race is
    upstream and not fully closed - this is best-effort.

    Applied unconditionally because skops's load flow always routes through
    ``Booster.__setstate__`` regardless of which model file format the
    Booster's serialized state is in. Idempotent.
    """
    import xgboost.core

    if getattr(xgboost.core.Booster.__setstate__, "_pyfeat_patched", False):
        return
    _orig = xgboost.core.Booster.__setstate__

    def _patched(self, state):
        if state.get("handle") is not None:
            state = {**state, "handle": bytearray(state["handle"])}
        return _orig(self, state)

    _patched._pyfeat_patched = True
    xgboost.core.Booster.__setstate__ = _patched


_patch_xgboost_setstate_for_skops()

# Suppress sklearn warning about pickled estimators and diff sklearn versions
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


class Detector(nn.Module, PyTorchModelHubMixin):
    _SUPPORTED_FACE_MODELS = ("img2pose", "retinaface")

    def __init__(
        self,
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        identity_model="arcface",
        gaze_model="l2cs",
        device="cpu",
    ):
        # v0.7 swaps the default face detector from img2pose to retinaface
        # (ResNet34, py-feat/retinaface_r34): 88.9% WIDERFACE Hard AP vs
        # img2pose's 55.5% (per Cheong et al. 2023), and ~5× faster at
        # batch 32. Pose accuracy is preserved via the landmarks-to-pose
        # MLP distilled from img2pose (~5° avg MAE vs img2pose) — see
        # feat.utils.face_pose_mlp. Users who need bit-identical img2pose
        # pose can still pass `face_model='img2pose'`.
        super().__init__()

        if face_model not in self._SUPPORTED_FACE_MODELS:
            raise ValueError(
                f"face_model must be one of {self._SUPPORTED_FACE_MODELS}; "
                f"got {face_model!r}"
            )

        self.info = dict(
            face_model=face_model,
            landmark_model=None,
            emotion_model=None,
            # facepose_model tracks where 6DoF pose comes from. img2pose
            # regresses pose natively; retinaface derives pose via DLT-PnP
            # from the 68 landmarks (see feat.utils.face_pose_pnp).
            facepose_model="pnp_dlt" if face_model == "retinaface" else "img2pose",
            au_model=None,
            identity_model=None,
            gaze_model=None,
        )
        self.device = set_torch_device(device)

        # Cache one HOGLayer per Detector instance. Building it allocates
        # the Sobel buffers and the AvgPool2d module; doing it inside the
        # HOG-feature extractor means paying that cost twice per detect()
        # call (once for emotion=svm, once for au=xgb). The layer carries
        # no state across calls, so reusing is safe.
        self._hog_layer = HOGLayer(
            orientations=8,
            pixels_per_cell=8,
            cells_per_block=2,
            block_normalization="L2-Hys",
            feature_vector=True,
            device=self.device,
        ).to(self.device)

        if face_model == "img2pose":
            # Load Model Configurations
            facepose_config_file = hf_hub_download(
                repo_id="py-feat/img2pose",
                filename="config.json",
                cache_dir=get_resource_path(),
            )
            with open(facepose_config_file, "r") as f:
                facepose_config = json.load(f)

            # Initialize img2pose
            backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=None)
            backbone.eval()
            backbone.to(self.device)
            self.facepose_detector = FasterDoFRCNN(
                backbone=backbone,
                num_classes=2,
                min_size=facepose_config["min_size"],
                max_size=facepose_config["max_size"],
                pose_mean=torch.tensor(facepose_config["pose_mean"]),
                pose_stddev=torch.tensor(facepose_config["pose_stddev"]),
                threed_68_points=torch.tensor(facepose_config["threed_points"]),
                rpn_pre_nms_top_n_test=facepose_config["rpn_pre_nms_top_n_test"],
                rpn_post_nms_top_n_test=facepose_config["rpn_post_nms_top_n_test"],
                bbox_x_factor=facepose_config["bbox_x_factor"],
                bbox_y_factor=facepose_config["bbox_y_factor"],
                expand_forehead=facepose_config["expand_forehead"],
            )
            facepose_model_file = hf_hub_download(
                repo_id="py-feat/img2pose",
                filename="model.safetensors",
                cache_dir=get_resource_path(),
            )
            facepose_checkpoint = load_file(facepose_model_file)
            self.facepose_detector.load_state_dict(facepose_checkpoint, load_model_weights)
            self.facepose_detector.eval()
            self.facepose_detector.to(self.device)
        else:  # retinaface
            # RetinaFace-R34: 88.9% WIDERFACE Hard AP (per yakhyo upstream),
            # 15-20x faster per-image than img2pose at batch 16+ on MPS.
            # No 6DoF head pose - pose columns are populated as NaN.
            self.facepose_detector = Retinaface(device=self.device)
            warnings.warn(
                "face_model='retinaface' does not regress 6DoF head pose. "
                "Pose columns are populated via the landmarks-to-pose MLP "
                "(distilled from img2pose on CelebV-HQ, ~5° avg MAE vs "
                "img2pose). PnP-DLT is used as a fallback when the MLP "
                "weights aren't available. Use face_model='img2pose' for "
                "the slowest, highest-accuracy path. See "
                "feat.utils.face_pose_mlp for details.",
                stacklevel=2,
            )

        # Initialize Landmark Detector
        self.info["landmark_model"] = landmark_model
        if landmark_model is not None:
            if landmark_model == "mobilefacenet":
                self.face_size = 112
                self.landmark_detector = MobileFaceNet(
                    [self.face_size, self.face_size],
                    N_OPENFACE_LANDMARKS_2D_FLAT,
                    device=self.device,
                )
                landmark_model_file = hf_hub_download(
                    repo_id="py-feat/mobilefacenet",
                    filename="mobilefacenet_model_best.pth.tar",
                    cache_dir=get_resource_path(),
                )
                landmark_state_dict = torch.load(
                    landmark_model_file, map_location=self.device, weights_only=True
                )["state_dict"]  # Ensure Model weights are Float32 for MPS
            elif landmark_model == "mobilenet":
                self.face_size = 224
                self.landmark_detector = MobileNet_GDConv(N_OPENFACE_LANDMARKS_2D_FLAT)
                landmark_model_file = hf_hub_download(
                    repo_id="py-feat/mobilenet",
                    filename="mobilenet_224_model_best_gdconv_external.pth.tar",
                    cache_dir=get_resource_path(),
                )
                mobilenet_state_dict = torch.load(
                    landmark_model_file, map_location=self.device, weights_only=True
                )["state_dict"]  # Ensure Model weights are Float32 for MPS
                landmark_state_dict = OrderedDict()
                for k, v in mobilenet_state_dict.items():
                    if "module." in k:
                        k = k.replace("module.", "")
                    landmark_state_dict[k] = v
            elif landmark_model == "pfld":
                self.face_size = 112
                self.landmark_detector = PFLDInference()
                landmark_model_file = hf_hub_download(
                    repo_id="py-feat/pfld",
                    filename="pfld_model_best.pth.tar",
                    cache_dir=get_resource_path(),
                )
                landmark_state_dict = torch.load(
                    landmark_model_file, map_location=self.device, weights_only=True
                )["state_dict"]  # Ensure Model weights are Float32 for MPS
            else:
                raise ValueError(f"{landmark_model} is not currently supported.")
            self.landmark_detector.load_state_dict(landmark_state_dict)
            self.landmark_detector.eval()
            self.landmark_detector.to(self.device)
            # self.landmark_detector = torch.compile(self.landmark_detector)
        else:
            self.landmark_detector = None

        # Initialize AU Detector
        self.info["au_model"] = au_model
        if au_model is not None:
            if self.landmark_detector is not None:
                if au_model == "xgb":
                    self.au_detector = XGBClassifier()
                    # _v2 file references the real wrapper class path (not __main__)
                    # and embeds Booster buffers in xgboost's modern UBJ format.
                    # Fall back to v1 if v2 isn't on the hub yet (e.g. fresh
                    # py-feat install during the upload window between
                    # code-release and HF-upload). Once v2 is uploaded the
                    # fallback path is dead.
                    au_model_path = hf_hub_download_with_fallback(
                        repo_id="py-feat/xgb_au",
                        filename="xgb_au_classifier_v2.skops",
                        fallback_filename="xgb_au_classifier.skops",
                        cache_dir=get_resource_path(),
                    )

                elif au_model == "svm":
                    self.au_detector = SVMClassifier()
                    au_model_path = hf_hub_download_with_fallback(
                        repo_id="py-feat/svm_au",
                        filename="svm_au_classifier_v2.skops",
                        fallback_filename="svm_au_classifier.skops",
                        cache_dir=get_resource_path(),
                    )
                else:
                    raise ValueError(f"{au_model} is not currently supported.")

                au_unknown_types = get_untrusted_types(file=au_model_path)
                loaded_au_model = load(au_model_path, trusted=au_unknown_types)
                self.au_detector.load_weights(
                    scaler_upper=loaded_au_model.scaler_upper,
                    pca_model_upper=loaded_au_model.pca_model_upper,
                    scaler_lower=loaded_au_model.scaler_lower,
                    pca_model_lower=loaded_au_model.pca_model_lower,
                    scaler_full=loaded_au_model.scaler_full,
                    pca_model_full=loaded_au_model.pca_model_full,
                    classifiers=loaded_au_model.classifiers,
                )
            else:
                raise ValueError(
                    f"Landmark Detector is required for AU Detection with {au_model}."
                )
        else:
            self.au_detector = None

        # Initialize Emotion Detector
        self.info["emotion_model"] = emotion_model
        if emotion_model is not None:
            if emotion_model == "resmasknet":
                emotion_config_file = hf_hub_download(
                    repo_id="py-feat/resmasknet",
                    filename="config.json",
                    cache_dir=get_resource_path(),
                )
                with open(emotion_config_file, "r") as f:
                    emotion_config = json.load(f)

                self.emotion_detector = ResMasking(
                    "", in_channels=emotion_config["in_channels"]
                )
                self.emotion_detector.fc = nn.Sequential(
                    nn.Dropout(0.4), nn.Linear(512, emotion_config["num_classes"])
                )
                emotion_model_file = hf_hub_download(
                    repo_id="py-feat/resmasknet",
                    filename="ResMaskNet_Z_resmasking_dropout1_rot30.pth",
                    cache_dir=get_resource_path(),
                )
                emotion_checkpoint = torch.load(
                    emotion_model_file, map_location=device, weights_only=True
                )["net"]
                self.emotion_detector.load_state_dict(emotion_checkpoint)
                self.emotion_detector.eval()
                self.emotion_detector.to(self.device)
                # self.emotion_detector = torch.compile(self.emotion_detector)
            elif emotion_model == "svm":
                if self.landmark_detector is not None:
                    self.emotion_detector = EmoSVMClassifier()
                    emotion_model_path = hf_hub_download(
                        repo_id="py-feat/svm_emo",
                        filename="svm_emo_classifier.skops",
                        cache_dir=get_resource_path(),
                    )
                    emotion_unknown_types = get_untrusted_types(file=emotion_model_path)
                    loaded_emotion_model = load(
                        emotion_model_path, trusted=emotion_unknown_types
                    )
                    self.emotion_detector.load_weights(
                        scaler_full=loaded_emotion_model.scaler_full,
                        pca_model_full=loaded_emotion_model.pca_model_full,
                        classifiers=loaded_emotion_model.classifiers,
                    )
                else:
                    raise ValueError(
                        "Landmark Detector is required for Emotion Detection with {emotion_model}."
                    )

            else:
                raise ValueError(f"{emotion_model} is not currently supported.")
        else:
            self.emotion_detector = None

        # Initialize Identity Detecctor -  facenet
        self.info["identity_model"] = identity_model
        if identity_model is not None:
            if identity_model == "facenet":
                self.identity_detector = InceptionResnetV1(
                    pretrained=None,
                    classify=False,
                    num_classes=None,
                    dropout_prob=0.6,
                    device=self.device,
                )
                self.identity_detector.logits = nn.Linear(512, 8631)
                identity_model_file = hf_hub_download(
                    repo_id="py-feat/facenet",
                    filename="facenet_20180402_114759_vggface2.pth",
                    cache_dir=get_resource_path(),
                )
                self.identity_detector.load_state_dict(
                    torch.load(
                        identity_model_file, map_location=device, weights_only=True
                    )
                )
                self.identity_detector.eval()
                self.identity_detector.to(self.device)
                # self.identity_detector = torch.compile(self.identity_detector)
            elif identity_model in ("arcface", "arcface_r50"):
                # ArcFace ResNet50 trained on WebFace600K (InsightFace's
                # buffalo_l recognition model, converted from ONNX to
                # PyTorch via scripts/convert_arcface_onnx_to_safetensors.py).
                # Embeddings are angular-margin-trained — they disentangle
                # identity from pose and expression much better than
                # facenet's triplet-loss embeddings. See model card on
                # https://huggingface.co/py-feat/arcface_r50 for license.
                self.identity_detector = ArcFace(backbone="r50")
                arcface_path = os.environ.get("FEAT_ARCFACE_R50_PATH")
                if arcface_path is None:
                    arcface_path = hf_hub_download(
                        repo_id="py-feat/arcface_r50",
                        filename="arcface_r50.safetensors",
                        cache_dir=get_resource_path(),
                    )
                # strict=False because BatchNorm's `num_batches_tracked`
                # buffer isn't in the converted safetensors (it's not in
                # the source ONNX). Validate the missing/unexpected keys
                # ourselves so a wrong-file or empty-file load fails
                # loudly rather than silently producing garbage embeddings.
                missing, unexpected = self.identity_detector.net.load_state_dict(
                    load_file(arcface_path), strict=False
                )
                real_missing = [k for k in missing if "num_batches_tracked" not in k]
                if real_missing or unexpected:
                    raise RuntimeError(
                        f"ArcFace weights at {arcface_path!r} are inconsistent "
                        f"with the architecture. Missing: {real_missing}; "
                        f"unexpected: {list(unexpected)}. Re-download from "
                        f"py-feat/arcface_r50 or re-run "
                        f"scripts/convert_arcface_onnx_to_safetensors.py."
                    )
                self.identity_detector.eval()
                self.identity_detector.to(self.device)
            else:
                raise ValueError(f"{identity_model} is not currently supported.")
        else:
            self.identity_detector = None

        # Initialize Gaze Detector. L2CS-Net (Abdelrahman et al. 2022)
        # regresses (pitch, yaw) from the face crop via a 90-bin
        # classification head per axis; reported ~3.92° MAE on Gaze360,
        # ~4.16° on MPIIFaceGaze. Replaces the geometric iris-vector
        # path that was previously available only in MPDetector and had
        # known >100° errors on off-frontal faces.
        self.info["gaze_model"] = gaze_model
        if gaze_model is None:
            self.gaze_detector = None
        elif gaze_model == "l2cs":
            from feat.gaze_detectors.l2cs import load_l2cs_from_hf
            self.gaze_detector = load_l2cs_from_hf(device=self.device)
        else:
            raise ValueError(
                f"gaze_model must be 'l2cs' or None for Detector; got {gaze_model!r}. "
                f"The geometric path requires MediaPipe iris landmarks and is "
                f"only available on MPDetector."
            )

    def __repr__(self):
        return (
            f"Detector(face_model={self.info['face_model']}, "
            f"landmark_model={self.info['landmark_model']}, "
            f"au_model={self.info['au_model']}, "
            f"emotion_model={self.info['emotion_model']}, "
            f"facepose_model={self.info['facepose_model']}, "
            f"identity_model={self.info['identity_model']}, "
            f"gaze_model={self.info['gaze_model']})"
        )

    @torch.inference_mode()
    def detect_faces(self, images, face_size=112, face_detection_threshold=0.5):
        """
        Detect faces and (with img2pose) 6DoF head pose in a batch of images.

        Args:
            images (torch.Tensor): Tensor of shape (B, C, H, W) representing the images
            face_size (int): Output size to resize face after cropping.
            face_detection_threshold (float): Score threshold for keeping detections.

        Returns:
            list of per-image dicts with keys: faces, boxes, new_boxes, poses,
            scores, face_id (and resmasknet_faces if emotion_model='resmasknet').
            Pose columns are NaN-filled when face_model='retinaface'.
        """

        # img2pose / RetinaFace both accept a batched [B, C, H, W] tensor and
        # return per-image detections. img2pose ingests pixel values in [0, 1];
        # RetinaFace ingests pixel values in [0, 255]. The wrapper handles its
        # own preprocessing so we keep the unscaled tensor for it.
        frames_unit = convert_image_to_tensor(images, img_type="float32") / 255.0
        frames_unit = frames_unit.to(self.device)

        if self.info["face_model"] == "img2pose":
            img2pose_outputs = self.facepose_detector(frames_unit)
            per_image_dets = []
            for img2pose_output in img2pose_outputs:
                processed = postprocess_img2pose(
                    img2pose_output, detection_threshold=face_detection_threshold
                )
                per_image_dets.append({
                    "boxes": processed["boxes"],
                    "scores": processed["scores"],
                    "poses": processed["dofs"],  # [N, 6]
                })
        else:  # retinaface: takes [0, 255] floats; returns list of [x1,y1,x2,y2,score]
            frames_px = convert_image_to_tensor(images, img_type="float32").to(self.device)
            rf_outputs = self.facepose_detector(frames_px)
            per_image_dets = []
            for image_dets in rf_outputs:
                if image_dets:
                    arr = torch.tensor(image_dets, dtype=torch.float32, device=self.device)
                    boxes = arr[:, :4]
                    scores = arr[:, 4]
                    keep = scores >= face_detection_threshold
                    boxes = boxes[keep]
                    scores = scores[keep]
                else:
                    boxes = torch.empty((0, 4), device=self.device)
                    scores = torch.empty((0,), device=self.device)
                per_image_dets.append({
                    "boxes": boxes,
                    "scores": scores,
                    "poses": torch.full(
                        (boxes.shape[0], 6), float("nan"), device=self.device
                    ),
                })

        # Gather bboxes across the batch so face cropping runs as one
        # batched grid_sample call instead of one per frame. The per-frame
        # Python loop over GPU ops cost ~1ms/frame of pure kernel-launch
        # overhead. No-detection frames contribute one NaN-bbox placeholder
        # so downstream forward() sees >= 1 row per frame.
        B = len(per_image_dets)
        wants_resmasknet = self.info["emotion_model"] == "resmasknet"

        bbox_chunks = []
        score_chunks = []
        pose_chunks = []
        no_det_per_frame = []
        n_per_frame = []
        for det in per_image_dets:
            if det["boxes"].numel() != 0:
                bbox_chunks.append(det["boxes"])
                score_chunks.append(det["scores"])
                pose_chunks.append(det["poses"])
                n_per_frame.append(det["boxes"].shape[0])
                no_det_per_frame.append(False)
            else:
                bbox_chunks.append(torch.full((1, 4), float("nan"), device=self.device))
                score_chunks.append(torch.zeros((1,), device=self.device))
                pose_chunks.append(torch.full((1, 6), float("nan"), device=self.device))
                n_per_frame.append(1)
                no_det_per_frame.append(True)

        all_bboxes = torch.cat(bbox_chunks, dim=0)
        all_scores = torch.cat(score_chunks, dim=0)
        all_poses = torch.cat(pose_chunks, dim=0)
        n_per_frame_t = torch.tensor(n_per_frame, device=self.device)
        all_frame_idx = torch.repeat_interleave(
            torch.arange(B, device=self.device), n_per_frame_t
        )

        # Replace NaN bboxes with zero so grid_sample doesn't propagate
        # NaNs into the crops; we restore the no-detection signal via
        # `all_bboxes` (kept NaN) and `extracted` masking below.
        bboxes_for_extract = torch.where(
            torch.isnan(all_bboxes), torch.zeros_like(all_bboxes), all_bboxes
        )
        all_extracted, all_new_bboxes = extract_face_from_bbox_torch(
            frames_unit,
            bboxes_for_extract,
            face_size=face_size,
            frame_idx=all_frame_idx,
        )

        no_det_mask = torch.isnan(all_bboxes).any(dim=1)
        if no_det_mask.any():
            all_extracted = all_extracted.clone()
            all_extracted[no_det_mask] = 0
            all_new_bboxes = all_new_bboxes.clone().to(torch.float32)
            all_new_bboxes[no_det_mask] = float("nan")

        if wants_resmasknet:
            resmasknet_all, _ = extract_face_from_bbox_torch(
                frames_unit,
                bboxes_for_extract,
                expand_bbox=1.1,
                face_size=224,
                frame_idx=all_frame_idx,
            )
            if no_det_mask.any():
                resmasknet_all = resmasknet_all.clone()
                resmasknet_all[no_det_mask] = float("nan")

        # Redistribute into per-frame dicts (preserves the public return
        # signature of detect_faces). This second pass is cheap — just
        # tensor slicing — and lets forward() stay unchanged.
        image_size = tuple(frames_unit.shape[-2:])
        batch_results = []
        cursor = 0
        for i in range(B):
            n = n_per_frame[i]
            sl = slice(cursor, cursor + n)
            frame_results = {
                "face_id": i,
                "faces": all_extracted[sl],
                "boxes": all_bboxes[sl],
                "new_boxes": all_new_bboxes[sl],
                "poses": all_poses[sl],
                "scores": all_scores[sl],
                "image_size": image_size,
            }
            if wants_resmasknet:
                frame_results["resmasknet_faces"] = resmasknet_all[sl]
            batch_results.append(frame_results)
            cursor += n

        return batch_results

    @torch.inference_mode()
    def forward(self, faces_data, batch_data):
        """
        Run Model Inference on detected faces.

        Args:
            faces_data (list of dict): Detected faces and associated data from `detect_faces`.
            batch_data (dict): The DataLoader's batch dict for this call.
                Used to convert per-face bbox/landmark coordinates from the
                padded-image space the models operate in back to the
                original-frame space the caller expects, in a single
                vectorized tensor op (replacing the prior post-hoc
                `invert_padding_to_results` DataFrame mutation).

        Returns:
            pandas.DataFrame: per-face predictions in original-frame
            coordinates, including FrameHeight / FrameWidth columns.
            Wrapped into a Fex by `detect()` once at the end.
        """

        extracted_faces = torch.cat([face["faces"] for face in faces_data], dim=0)
        new_bboxes = torch.cat([face["new_boxes"] for face in faces_data], dim=0)
        n_faces = extracted_faces.shape[0]

        # Per-face mapping back to the source frame in the batch. Used
        # below to broadcast the DataLoader's per-frame Rescale
        # (Padding + Scale) parameters to per-face tensors so we can
        # convert bboxes / landmarks from padded-image space to
        # original-frame space without a post-hoc DataFrame walk.
        n_per_frame = [face["faces"].shape[0] for face in faces_data]
        frame_idx = torch.repeat_interleave(
            torch.arange(len(faces_data), device=self.device),
            torch.tensor(n_per_frame, device=self.device),
        )
        pad_left, pad_top, scale, frame_h, frame_w = per_face_padding_inversion_terms(
            batch_data, frame_idx, self.device
        )

        # Hoist CPU->device transfers out of per-detector branches: landmark
        # and identity detectors both consume the face crops, and previously
        # each branch issued its own `.to(self.device)` (each a fresh copy
        # since the source stays on CPU). Move once, reuse. The HOG-based
        # AU and SVM-emotion paths below still use the CPU-side
        # `extracted_faces`.
        faces_dev = extracted_faces.to(self.device)

        if self.landmark_detector is not None:
            if self.info["landmark_model"].lower() == "mobilenet":
                # Normalize must run on whichever copy will be passed in;
                # apply on CPU then transfer once.
                extracted_faces = Compose(
                    [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                )(extracted_faces)
                faces_dev = extracted_faces.to(self.device)
                landmarks = self.landmark_detector.forward(faces_dev)
            elif self.info["landmark_model"].lower() == "mobilefacenet":
                landmarks = self.landmark_detector.forward(faces_dev)[0]
            else:
                landmarks = self.landmark_detector.forward(faces_dev)
            new_landmarks = inverse_transform_landmarks_torch(landmarks, new_bboxes)
        else:
            new_landmarks = torch.full(
                (n_faces, N_OPENFACE_LANDMARKS_2D_FLAT), float("nan")
            )

        if self.emotion_detector is not None:
            if self.info["emotion_model"] == "resmasknet":
                resmasknet_faces = torch.cat(
                    [face["resmasknet_faces"] for face in faces_data], dim=0
                )
                emotions = self.emotion_detector.forward(resmasknet_faces.to(self.device))
                emotions = torch.softmax(emotions, 1)
            elif self.info["emotion_model"] == "svm":
                hog_features, emo_new_landmarks = extract_hog_features_batched(
                    extracted_faces, landmarks, hog_layer=self._hog_layer
                )
                emotions = self.emotion_detector.detect_emo(
                    frame=hog_features, landmarks=[emo_new_landmarks]
                )
                emotions = torch.tensor(emotions)
        else:
            emotions = torch.full((n_faces, 7), float("nan"))

        if self.identity_detector is not None:
            identity_embeddings = self.identity_detector.forward(faces_dev)
        else:
            identity_embeddings = torch.full((n_faces, 512), float("nan"))

        if self.au_detector is not None:
            hog_features, au_new_landmarks = extract_hog_features_batched(
                extracted_faces, landmarks, hog_layer=self._hog_layer
            )
            aus = self.au_detector.detect_au(
                frame=hog_features, landmarks=[au_new_landmarks]
            )
        else:
            aus = torch.full((n_faces, 20), float("nan"))

        # Create Fex Output Representation. Bboxes come out of
        # convert_bbox_output in PADDED-frame space (the same space the
        # face detector + landmark detector operated in). Subtract the
        # DataLoader's per-face padding and divide by its scale to land
        # in ORIGINAL-frame space — what the user expects in Fex.
        # In-place arithmetic: each axis reads + writes the same column,
        # axes are independent, and `bboxes` is freshly allocated by
        # torch.cat above and not referenced again after the DataFrame
        # is built. (Score column 4 is unchanged by Rescale inversion.)
        bboxes = torch.cat(
            [
                convert_bbox_output(
                    face_output["new_boxes"].to(self.device),
                    face_output["scores"].to(self.device),
                )
                for face_output in faces_data
            ],
            dim=0,
        )
        bboxes[:, 0] = (bboxes[:, 0] - pad_left) / scale
        bboxes[:, 1] = (bboxes[:, 1] - pad_top) / scale
        bboxes[:, 2] = bboxes[:, 2] / scale
        bboxes[:, 3] = bboxes[:, 3] / scale
        feat_faceboxes = pd.DataFrame(
            bboxes.cpu().detach().numpy(),
            columns=FEAT_FACEBOX_COLUMNS,
        )

        poses = torch.cat(
            [face_output["poses"].to(self.device) for face_output in faces_data], dim=0
        )

        # When face_model='retinaface' (or any future detector that
        # doesn't natively regress 6DoF pose), the per-frame `poses` tensors
        # are NaN-padded. Replace with PnP-derived pose from the 68 landmarks
        # we just computed, using img2pose's published 3D template (so the
        # output lives in the same head-centric coordinate frame).
        #
        # All frames in a single detect() share the same H/W (collated by
        # DataLoader), so default intrinsics are constant across the batch
        # and PnP can be one batched call across every face in every frame.
        if (
            self.info["face_model"] != "img2pose"
            and self.landmark_detector is not None
            and torch.isnan(poses).any()
        ):
            # Skip faces with NaN landmarks (no detection in that frame).
            valid = ~torch.isnan(new_landmarks).any(dim=1)
            if valid.any():
                lmk = (
                    new_landmarks[valid]
                    .reshape(-1, N_OPENFACE_LANDMARKS, 2)
                    .to(self.device)
                )
                # Pose-MLP path (replaces PnP-DLT default). The MLP was
                # distilled from img2pose on CelebV-HQ (~570k frames) and
                # matches img2pose's coordinate frame. It's bbox-relative
                # rather than image-relative, so it doesn't suffer from
                # PnP-DLT's "intrinsics from full image" bug — pose stays
                # sensible on multi-face wide-angle images.
                from feat.utils.face_pose_mlp import pose_from_landmarks_mlp

                # Bbox-free: MLP normalizes landmarks by their own
                # centroid + inter-eye distance, so it doesn't matter
                # whether bbox conventions match the training-time
                # detector (img2pose) or not.
                mlp_pose = pose_from_landmarks_mlp(lmk)

                if mlp_pose is not None:
                    self.info["facepose_model"] = "pose_mlp"
                    poses[valid] = mlp_pose
                else:
                    # MLP weights not available — fall back to PnP-DLT.
                    from feat.utils.face_pose_pnp import pose_from_landmarks_2d
                    image_size = faces_data[0]["image_size"]
                    pnp_pose = pose_from_landmarks_2d(lmk, image_size)
                    poses[valid] = pnp_pose

        feat_poses = pd.DataFrame(
            poses.cpu().detach().numpy(), columns=FEAT_FACEPOSE_COLUMNS_6D
        )

        # Invert the DataLoader's Rescale on the 68 (x, y) landmark pairs.
        # The PnP block above (when active) already consumed the
        # padded-frame landmarks; invert here for the user-visible
        # output. In-place is safe — `new_landmarks` is not used after
        # the DataFrame is built. NaN landmarks (no-detection rows)
        # propagate as NaN through the arithmetic, which is what we want.
        reshape_landmarks = new_landmarks.reshape(
            new_landmarks.shape[0], N_OPENFACE_LANDMARKS, 2
        )
        reshape_landmarks[..., 0] = (
            reshape_landmarks[..., 0] - pad_left[:, None]
        ) / scale[:, None]
        reshape_landmarks[..., 1] = (
            reshape_landmarks[..., 1] - pad_top[:, None]
        ) / scale[:, None]
        reordered_landmarks = torch.cat(
            [reshape_landmarks[:, :, 0], reshape_landmarks[:, :, 1]], dim=1
        )
        feat_landmarks = pd.DataFrame(
            reordered_landmarks.cpu().detach().numpy(),
            columns=openface_2d_landmark_columns,
        )

        feat_aus = pd.DataFrame(aus, columns=AU_LANDMARK_MAP["Feat"])

        feat_emotions = pd.DataFrame(
            emotions.cpu().detach().numpy(), columns=FEAT_EMOTION_COLUMNS
        )

        feat_identities = pd.DataFrame(
            identity_embeddings.cpu().detach().numpy(), columns=FEAT_IDENTITY_COLUMNS[1:]
        )

        # Gaze: L2CS-Net on the face crops already on device. Returns
        # head-centric (pitch, yaw) in radians; combined gaze_angle is
        # the spherical-distance from straight-ahead.
        if self.gaze_detector is not None and n_faces > 0:
            pitch_rad, yaw_rad = self.gaze_detector(faces_dev)
            # Angle from straight-ahead: arccos(cos(pitch) * cos(yaw)).
            cos_angle = np.clip(
                np.cos(pitch_rad) * np.cos(yaw_rad), -1.0, 1.0
            )
            gaze_angle = np.arccos(cos_angle)
            feat_gaze = pd.DataFrame(
                np.column_stack([pitch_rad, yaw_rad, gaze_angle]),
                columns=FEAT_GAZE_COLUMNS,
            )
        else:
            feat_gaze = pd.DataFrame(
                np.full((n_faces, len(FEAT_GAZE_COLUMNS)), np.nan),
                columns=FEAT_GAZE_COLUMNS,
            )

        # Frame metadata: original (pre-Rescale) frame dimensions per
        # face. Added here instead of in `invert_padding_to_results`
        # post-hoc so the entire DataFrame leaves forward() in
        # original-frame coords with no further mutation.
        feat_frame_meta = pd.DataFrame(
            {
                "FrameHeight": frame_h.cpu().detach().numpy().astype(np.float64),
                "FrameWidth": frame_w.cpu().detach().numpy().astype(np.float64),
            }
        )

        # Return a plain pd.DataFrame; detect() wraps the concatenated
        # result in a single Fex at the end. Avoids Fex.__init__'s
        # O(n_columns) metadata loop (see Fex.__init__ in data.py) firing
        # once per batch, which dominated wall-time on long videos.
        return pd.concat(
            [
                feat_faceboxes,
                feat_landmarks,
                feat_poses,
                feat_aus,
                feat_emotions,
                feat_identities,
                feat_gaze,
                feat_frame_meta,
            ],
            axis=1,
        )

    def detect(
        self,
        inputs,
        data_type="image",
        output_size=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        face_identity_threshold=0.8,
        face_detection_threshold=0.5,
        skip_frames=None,
        progress_bar=True,
        save=None,
        **kwargs,
    ):
        """
        Detects FEX from one or more imagathe files.

        Args:
            inputs (list of str, torch.Tensor): Path to a list of paths to image files or torch.Tensor of images (B, C, H, W)
            data_type (str): type of data to be processed; Default 'image' ['image', 'tensor', 'video']
            output_size (int): image size to rescale all image preserving aspect ratio.
            batch_size (int): how many batches of images you want to run at one shot.
            num_workers (int): how many subprocesses to use for data loading.
            pin_memory (bool): If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them.
            face_identity_threshold (float): value between 0-1 to determine similarity of person using face identity embeddings; Default >= 0.8
            face_detection_threshold (float): value between 0-1 to determine if a face was detected; Default >= 0.5
            skip_frames (int or None): number of frames to skip to speed up inference (video only); Default None
            progress_bar (bool): Whether to show the tqdm progress bar. Default is True.
            **kwargs: additional detector-specific kwargs
            save (None or str or Path): if immediately append detections to a csv file at with the given name after processing each batch, which can be useful to interrupted/resuming jobs and saving memory/RAM

        Returns:
            pd.DataFrame: Concatenated results for all images in the batch
        """

        save = Path(save) if save else None

        if data_type.lower() == "image":
            data_loader = DataLoader(
                ImageDataset(
                    inputs,
                    output_size=output_size,
                    preserve_aspect_ratio=True,
                    padding=True,
                ),
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=False,
            )
        elif data_type.lower() == "tensor":
            data_loader = DataLoader(
                TensorDataset(inputs),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        elif data_type.lower() == "video":
            dataset = VideoDataset(
                inputs,
                skip_frames=skip_frames,
                output_size=output_size,
            )
            data_loader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=False,
            )

        data_iterator = tqdm(data_loader) if progress_bar else data_loader

        batch_output = []
        frame_counter = 0

        try:
            _ = next(enumerate(tqdm(data_loader)))
        except RuntimeError as e:
            raise ValueError(
                f"When using `batch_size > 1`, all images must either have the same dimension or `output_size` should be something other than `None` to pad images prior to processing\n{e}"
            )

        for batch_id, batch_data in enumerate(data_iterator):
            faces_data = self.detect_faces(
                batch_data["Image"],
                face_size=self.face_size if hasattr(self, "face_size") else 112,
                face_detection_threshold=face_detection_threshold,
            )
            batch_results = self.forward(faces_data, batch_data)

            # Create metadata for each frame
            file_names = []
            frame_ids = []
            for i, face in enumerate(faces_data):
                n_faces = len(face["scores"])
                if data_type.lower() == "video":
                    current_frame_id = batch_data["Frame"].detach().numpy()[i]
                else:
                    current_frame_id = frame_counter + i
                frame_ids.append(np.repeat(current_frame_id, n_faces))
                file_names.append(np.repeat(batch_data["FileName"][i], n_faces))
            batch_results["input"] = np.concatenate(file_names)
            batch_results["frame"] = np.concatenate(frame_ids)

            # Padded->original-frame coordinate inversion now happens
            # inside forward() in tensor space; no post-hoc DataFrame walk
            # is needed here. (`invert_padding_to_results` is still
            # exported for any external caller that depended on it.)

            if save:
                # First batch truncates any pre-existing file; later batches
                # append. mode="a" on every batch would let stale data from
                # a previous detect() call survive — and the new run's
                # header row would then be appended as a real row, which
                # pd.read_csv later parses as a data row with column-name
                # strings as values (poisoning compute_identities).
                batch_results.to_csv(
                    save,
                    mode="w" if batch_id == 0 else "a",
                    index=False,
                    header=batch_id == 0,
                )
            else:
                batch_output.append(batch_results)
            # Use the actual batch size (may be smaller than `batch_size` for the
            # last batch when len(dataset) is not divisible by batch_size).
            frame_counter += batch_data["Image"].shape[0]

        # Build a single Fex once: either from the streamed CSV (save
        # mode) or by concatenating the per-batch DataFrames forward()
        # returned. See forward() comment for why we don't wrap per batch.
        if save:
            concat_df = pd.read_csv(save)
        else:
            concat_df = pd.concat(batch_output).reset_index(drop=True)
        batch_output = Fex(
            concat_df,
            au_columns=AU_LANDMARK_MAP["Feat"],
            emotion_columns=FEAT_EMOTION_COLUMNS,
            facebox_columns=FEAT_FACEBOX_COLUMNS,
            landmark_columns=openface_2d_landmark_columns,
            facepose_columns=FEAT_FACEPOSE_COLUMNS_6D,
            gaze_columns=FEAT_GAZE_COLUMNS,
            identity_columns=FEAT_IDENTITY_COLUMNS[1:],
            detector="Feat",
            face_model=self.info["face_model"],
            landmark_model=self.info["landmark_model"],
            au_model=self.info["au_model"],
            emotion_model=self.info["emotion_model"],
            facepose_model=self.info["facepose_model"],
            identity_model=self.info["identity_model"],
            gaze_model=self.info["gaze_model"],
        )
        if data_type.lower() == "video":
            batch_output["approx_time"] = [
                dataset.calc_approx_frame_time(x)
                for x in batch_output["frame"].to_numpy()
            ]
        batch_output.compute_identities(threshold=face_identity_threshold, inplace=True)
        # Overwrite with approx_time and identity columns
        if save:
            batch_output.to_csv(save, mode="w", index=False)
        return batch_output
