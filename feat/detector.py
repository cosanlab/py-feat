"""
Main Detector class. The Detector class wraps other pre-trained models
(e.g. face detector, au detector) and provides a high-level API to make it easier to
perform detection
"""

import os
import numpy as np
import pandas as pd
from skimage.feature import hog
from feat.utils import (
    openface_2d_landmark_columns,
    FEAT_EMOTION_COLUMNS,
    FEAT_FACEBOX_COLUMNS,
    FEAT_FACEPOSE_COLUMNS_3D,
    FEAT_FACEPOSE_COLUMNS_6D,
    FEAT_TIME_COLUMNS,
    FEAT_IDENTITY_COLUMNS,
    set_torch_device,
    is_list_of_lists_empty,
)
from feat.utils.io import get_resource_path
from feat.utils.image_operations import (
    extract_face_from_landmarks,
    extract_face_from_bbox,
    convert_image_to_tensor,
    BBox,
)
from feat.utils.stats import cluster_identities
from feat.pretrained import get_pretrained_models, fetch_model, AU_LANDMARK_MAP, load_model_weights
from feat.data import (
    Fex,
    ImageDataset,
    VideoDataset,
    _inverse_face_transform,
    _inverse_landmark_transform,
)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Grayscale, ToTensor
import torchvision.transforms as transforms
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from feat.facepose_detectors.img2pose.deps.models import postprocess_img2pose
import logging
import warnings
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json

# Supress sklearn warning about pickled estimators and diff sklearn versions
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class Detector(object):
    def __init__(
        self,
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        facepose_model="img2pose",
        identity_model="facenet",
        device="cpu",
        n_jobs=1,
        verbose=False,
        **kwargs,
    ):
        """Detector class to detect FEX from images or videos.

        Detector is a class used to detect faces, facial landmarks, emotions, and action units from images and videos.

        Args:
            n_jobs (int, default=1): Number of processes to use for extraction.
            device (str): specify device to process data (default='cpu'), can be
            ['auto', 'cpu', 'cuda', 'mps']
            verbose (bool): print logging and debug messages during operation
            **kwargs: you can pass each detector specific kwargs using a dictionary
            like: `face_model_kwargs = {...}, au_model_kwargs={...}, ...`

        Attributes:
            info (dict):
                n_jobs (int): Number of jobs to be used in parallel.
                face_model (str, default=retinaface): Name of face detection model
                landmark_model (str, default=mobilenet): Nam eof landmark model
                au_model (str, default=svm): Name of Action Unit detection model
                emotion_model (str, default=resmasknet): Path to emotion detection model.
                facepose_model (str, default=img2pose): Name of headpose detection model.
                identity_model (str, default=facenet): Name of identity detection model.
                face_detection_columns (list): Column names for face detection ouput (x, y, w, h)
                face_landmark_columns (list): Column names for face landmark output (x0, y0, x1, y1, ...)
                emotion_model_columns (list): Column names for emotion model output
                emotion_model_columns (list): Column names for emotion model output
                mapper (dict): Class names for emotion model output by index.
                input_shape (dict)

            face_detector: face detector object
            face_landmark: face_landmark object
            emotion_model: emotion_model object

        Examples:
            >> detector = Detector(n_jobs=1)
            >> detector.detect_image(["input.jpg"])
            >> detector.detect_video("input.mp4")
        """

        # Initial info dict with model names only
        self.info = dict(
            face_model=None,
            landmark_model=None,
            emotion_model=None,
            facepose_model=None,
            au_model=None,
            identity_model=None,
            n_jobs=n_jobs,
        )
        self.verbose = verbose
        # Setup verbosity
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
            logging.info("Verbose logging enabled")

        # Setup device
        self.device = set_torch_device(device)

        # Load Model Configs
        with open(os.path.join(get_resource_path(), 'model_config.json'), 'r') as file:
            self.model_configs = json.load(file)
        # Verify model names and download if necessary
        face, landmark, au, emotion, facepose, identity = get_pretrained_models(
            face_model,
            landmark_model,
            au_model,
            emotion_model,
            facepose_model,
            identity_model,
            verbose,
        )

        self._init_detectors(
            face,
            landmark,
            au,
            emotion,
            facepose,
            identity,
            openface_2d_landmark_columns,
            **kwargs,
        )

    def __repr__(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}(device={self.device}; face_model={self.info['face_model']}, landmark_model={self.info['landmark_model']}, au_model={self.info['au_model']}, emotion_model={self.info['emotion_model']}, facepose_model={self.info['facepose_model']}, identity_model={self.info['identity_model']})"

    def __getitem__(self, i):
        return self.info[i]

    def _init_detectors(
        self,
        face,
        landmark,
        au,
        emotion,
        facepose,
        identity,
        openface_2d_landmark_columns,
        **kwargs,
    ):
        """Helper function called by __init__ and change_model to (re)initialize one of
        the supported detectors"""

        # Keyword arguments than can be passed to the underlying models
        face_model_kwargs = kwargs.pop("face_model_kwargs", dict())
        landmark_model_kwargs = kwargs.pop("landmark_model_kwargs", dict())
        au_model_kwargs = kwargs.pop("au_model_kwargs", dict())
        emotion_model_kwargs = kwargs.pop("emotion_model_kwargs", dict())
        facepose_model_kwargs = kwargs.pop("facepose_model_kwargs", dict())
        identity_model_kwargs = kwargs.pop("identity_model_kwargs", dict())

        # Initialize model instances and any additional post init setup
        # Only initialize a model if the currently initialized model is diff than the
        # requested one. Lets us re-use this with .change_model

        # FACE MODEL
        if self.info["face_model"] != face:
            logging.info(f"Loading Face model: {face}")
            self.face_detector = fetch_model("face_model", face)
            self.info["face_model"] = face
            self.info["face_detection_columns"] = FEAT_FACEBOX_COLUMNS
            predictions = np.full_like(np.atleast_2d(FEAT_FACEBOX_COLUMNS), np.nan)
            empty_facebox = pd.DataFrame(predictions, columns=FEAT_FACEBOX_COLUMNS)
            self._empty_facebox = empty_facebox
            if self.face_detector is not None:
                if "img2pose" in face:
                    self.face_detector = self.face_detector(
                        constrained="img2pose-c" == face,
                        device=self.device,
                        **face_model_kwargs,
                    )
                else:
                    self.face_detector = self.face_detector(
                        device=self.device, **face_model_kwargs
                    )

        # LANDMARK MODEL
        if self.info["landmark_model"] != landmark:
            logging.info(f"Loading Facial Landmark model: {landmark}")
            self.landmark_detector = fetch_model("landmark_model", landmark)
            if self.landmark_detector is not None:
                if landmark == "mobilenet":
                    self.landmark_detector = self.landmark_detector(
                        136, **landmark_model_kwargs
                    )
                    self.landmark_detector.from_pretrained(f'py-feat/{landmark}', cache_dir=get_resource_path())

                    # checkpoint = torch.load(
                    #     os.path.join(
                    #         get_resource_path(),
                    #         "mobilenet_224_model_best_gdconv_external.pth.tar",
                    #     ),
                    #     map_location=self.device,
                    # )
                    # ##################################
                    # state_dict = checkpoint["state_dict"]
                    # from collections import OrderedDict

                    # new_state_dict = OrderedDict()
                    # for k, v in state_dict.items():
                    #     if "module." in k:
                    #         k = k.replace("module.", "")
                    #     new_state_dict[k] = v
                    # self.landmark_detector.load_state_dict(new_state_dict)
                    # #####################################

                elif landmark == "pfld":
                    self.landmark_detector = self.landmark_detector(
                        **landmark_model_kwargs
                    )
                    self.landmark_detector.from_pretrained(f'py-feat/{landmark}', cache_dir=get_resource_path())

                    # checkpoint = torch.load(
                    #     os.path.join(get_resource_path(), "pfld_model_best.pth.tar"),
                    #     map_location=self.device,
                    # )
                    # self.landmark_detector.load_state_dict(checkpoint["state_dict"])
                elif landmark == "mobilefacenet":
                    self.landmark_detector = self.landmark_detector(
                        [112, 112], 136, **landmark_model_kwargs
                    )
                    self.landmark_detector.from_pretrained(f'py-feat/{landmark}', cache_dir=get_resource_path())

                    # checkpoint = torch.load(
                    #     os.path.join(
                    #         get_resource_path(), "mobilefacenet_model_best.pth.tar"
                    #     ),
                    #     map_location=self.device,
                    # )
                    # self.landmark_detector.load_state_dict(checkpoint["state_dict"])
            self.landmark_detector.eval()
            self.landmark_detector.to(self.device)
            
            self.info["landmark_model"] = landmark
            self.info["mapper"] = openface_2d_landmark_columns
            self.info["face_landmark_columns"] = openface_2d_landmark_columns
            predictions = np.full_like(
                np.atleast_2d(openface_2d_landmark_columns), np.nan
            )
            empty_landmarks = pd.DataFrame(
                predictions, columns=openface_2d_landmark_columns
            )
            self._empty_landmark = empty_landmarks

        # FACEPOSE MODEL
        if self.info["facepose_model"] != facepose:
            logging.info(f"Loading facepose model: {facepose}")
            self.facepose_detector = fetch_model("facepose_model", facepose)
            if "img2pose" in facepose:
                backbone = resnet_fpn_backbone(backbone_name=f"resnet{self.model_configs['img2pose']['depth']}", weights=None)
                self.facepose_detector = self.facepose_detector(backbone=backbone,
                        num_classes=2,
                        min_size=self.model_configs['img2pose']['min_size'],
                        max_size=self.model_configs['img2pose']['max_size'],
                        pose_mean=torch.tensor(self.model_configs['img2pose']['pose_mean']),
                        pose_stddev=torch.tensor(self.model_configs['img2pose']['pose_stddev']),
                        threed_68_points=torch.tensor(self.model_configs['img2pose']['threed_points']),
                        rpn_pre_nms_top_n_test=self.model_configs['img2pose']['rpn_pre_nms_top_n_test'],
                        rpn_post_nms_top_n_test=self.model_configs['img2pose']['rpn_post_nms_top_n_test'],
                        bbox_x_factor=self.model_configs['img2pose']['bbox_x_factor'],
                        bbox_y_factor=self.model_configs['img2pose']['bbox_y_factor'],
                        expand_forehead=self.model_configs['img2pose']['expand_forehead'],
                        **facepose_model_kwargs)
                
                # self.facepose_detector = self.facepose_detector(
                #     constrained="img2pose-c" == face,
                #     device=self.device,
                #     **facepose_model_kwargs,
                # )
                facepose_model_file = hf_hub_download(repo_id= "py-feat/img2pose", filename="model.safetensors", cache_dir=get_resource_path())
                facepose_checkpoint = load_file(facepose_model_file)
                self.facepose_detector.load_state_dict(facepose_checkpoint)
                self.facepose_detector.eval()
                self.facepose_detector.to(self.device)
            else:
                self.facepose_detector = self.facepose_detector(**facepose_model_kwargs)
            self.info["facepose_model"] = facepose

            pose_dof = facepose_model_kwargs.get("RETURN_DIM", 3)
            self.info["facepose_model_columns"] = (
                FEAT_FACEPOSE_COLUMNS_3D if pose_dof == 3 else FEAT_FACEPOSE_COLUMNS_6D
            )
            predictions = np.full_like(
                np.atleast_2d(self.info["facepose_model_columns"]), np.nan
            )
            empty_facepose = pd.DataFrame(
                predictions, columns=self.info["facepose_model_columns"]
            )
            self._empty_facepose = empty_facepose

        # AU MODEL
        if self.info["au_model"] != au:
            logging.info(f"Loading AU model: {au}")
            self.au_model = fetch_model("au_model", au)
            self.info["au_model"] = au
            if self.info["au_model"] in ["svm", "xgb"]:
                self.info["au_presence_columns"] = AU_LANDMARK_MAP["Feat"]
            else:
                self.info["au_presence_columns"] = AU_LANDMARK_MAP[
                    self.info["au_model"]
                ]
            if self.au_model is not None:
                self.au_model = self.au_model(**au_model_kwargs)
                au_weights = load_model_weights(model_type='au', model=au, location='huggingface')
                self.au_model.load_weights(au_weights['scaler_upper'], au_weights['pca_model_upper'], au_weights['scaler_lower'], au_weights['pca_model_lower'], au_weights['scaler_full'], au_weights['pca_model_full'], au_weights['au_classifiers'])

                predictions = np.full_like(
                    np.atleast_2d(self.info["au_presence_columns"]), np.nan
                )
                empty_au_occurs = pd.DataFrame(
                    predictions, columns=self.info["au_presence_columns"]
                )
                self._empty_auoccurence = empty_au_occurs

        # EMOTION MODEL
        if self.info["emotion_model"] != emotion:
            logging.info(f"Loading emotion model: {emotion}")
            self.emotion_model = fetch_model("emotion_model", emotion)
            self.info["emotion_model"] = emotion
            if self.emotion_model is not None:
                if emotion == "resmasknet":
                    self.emotion_model = self.emotion_model(
                        device=self.device, **emotion_model_kwargs
                    )
                elif emotion == "svm":
                    self.emotion_model = self.emotion_model(**emotion_model_kwargs)
                    emo_weights = load_model_weights(model_type='emotion', model=emotion, location='huggingface')
                    self.emotion_model.load_weights(emo_weights['scaler_full'], emo_weights['pca_model_full'], emo_weights['emo_classifiers'])

                self.info["emotion_model_columns"] = FEAT_EMOTION_COLUMNS
                predictions = np.full_like(np.atleast_2d(FEAT_EMOTION_COLUMNS), np.nan)
                empty_emotion = pd.DataFrame(predictions, columns=FEAT_EMOTION_COLUMNS)
                self._empty_emotion = empty_emotion

        # IDENTITY MODEL
        if self.info["identity_model"] != identity:
            logging.info(f"Loading Identity model: {identity}")
            self.identity_model = fetch_model("identity_model", identity)
            self.info["identity_model"] = identity
            self.info["identity_model_columns"] = FEAT_IDENTITY_COLUMNS
            predictions = np.full_like(np.atleast_2d(FEAT_IDENTITY_COLUMNS), np.nan)
            empty_identity = pd.DataFrame(predictions, columns=FEAT_IDENTITY_COLUMNS)
            self._empty_identity = empty_identity
            if self.identity_model is not None:
                self.identity_model = self.identity_model(
                    device=self.device, **identity_model_kwargs
                )

        self.info["output_columns"] = (
            FEAT_TIME_COLUMNS
            + self.info["face_detection_columns"]
            + self.info["face_landmark_columns"]
            + self.info["au_presence_columns"]
            + self.info["facepose_model_columns"]
            + self.info["emotion_model_columns"]
            + self.info["identity_model_columns"]
            + ["input"]
        )

    def change_model(self, **kwargs):
        """Swap one or more pre-trained detector models for another one. Just pass in
        the the new models to use as kwargs, e.g. emotion_model='svm'"""

        face_model = kwargs.get("face_model", self.info["face_model"])
        landmark_model = kwargs.get("landmark_model", self.info["landmark_model"])
        au_model = kwargs.get("au_model", self.info["au_model"])
        emotion_model = kwargs.get("emotion_model", self.info["emotion_model"])
        facepose_model = kwargs.get("facepose_model", self.info["facepose_model"])
        identity_model = kwargs.get("identity_model", self.info["identity_model"])

        # Verify model names and download if necessary
        face, landmark, au, emotion, facepose, identity = get_pretrained_models(
            face_model,
            landmark_model,
            au_model,
            emotion_model,
            facepose_model,
            identity_model,
            self.verbose,
        )
        for requested, current_name in zip(
            [face, landmark, au, emotion, facepose, identity],
            [
                "face_model",
                "landmark_model",
                "au_model",
                "emotion_model",
                "facepose_model",
                "identity_model",
            ],
        ):
            if requested != self.info[current_name]:
                print(
                    f"Changing {current_name} from {self.info[current_name]} -> {requested}"
                )

        self._init_detectors(
            face,
            landmark,
            au,
            emotion,
            facepose,
            identity,
            openface_2d_landmark_columns,
        )

    def detect_faces(self, frame, threshold=0.5, **face_model_kwargs):
        """Detect faces from image or video frame

        Args:
            frame (np.ndarray): 3d (single) or 4d (multiple) image array
            threshold (float): threshold for detectiong faces (default=0.5)

        Returns:
            list: list of lists with the same length as the number of frames. Each list
            item is a list containing the (x1, y1, x2, y2) coordinates of each detected
            face in that frame.

        """

        logging.info("detecting faces...")

        frame = convert_image_to_tensor(frame, img_type="float32")

        if "img2pose" in self.info["face_model"]:
            frame = frame / 255
            # faces, poses = self.face_detector(frame, **face_model_kwargs)
            img2pose_output = postprocess_img2pose(self.facepose_detector(frame, **face_model_kwargs))
            faces = img2pose_output['boxes']
            poses = img2pose_output['dofs'][:, :3] # Only returning xyz for now not translation
        else:
            faces = self.face_detector(frame, **face_model_kwargs)

        if is_list_of_lists_empty(faces):
            logging.warning("Warning: NO FACE is detected")

        thresholded_face = []
        for fframe in faces:  # first level is each frame
            fframe_x = []
            for fface in fframe:  # second level is each face within a frame
                if fface[4] >= threshold:  # set thresholds
                    fframe_x.append(fface)
            thresholded_face.append(fframe_x)

        return thresholded_face

    def detect_landmarks(self, frame, detected_faces, **landmark_model_kwargs):
        """Detect landmarks from image or video frame

        Args:
            frame (np.ndarray): 3d (single) or 4d (multiple) image array
            detected_faces (array):

        Returns:
            list: x and y landmark coordinates (1,68,2)

        Examples:
            >>> from feat import Detector
            >>> from feat.utils import read_pictures
            >>> img_data = read_pictures(['my_image.jpg'])
            >>> detector = Detector()
            >>> detected_faces = detector.detect_faces(frame)
            >>> detector.detect_landmarks(frame, detected_faces)
        """

        logging.info("detecting landmarks...")
        frame = convert_image_to_tensor(frame)

        if is_list_of_lists_empty(detected_faces):
            list_concat = detected_faces
        else:
            if self.info["landmark_model"]:
                if self.info["landmark_model"].lower() == "mobilenet":
                    out_size = 224
                else:
                    out_size = 112

            extracted_faces, new_bbox = extract_face_from_bbox(
                frame, detected_faces, face_size=out_size
            )

            extracted_faces = extracted_faces / 255.0

            if self.info["landmark_model"].lower() == "mobilenet":
                extracted_faces = Compose(
                    [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                )(extracted_faces)

            # Run Landmark Model
            if self.info["landmark_model"].lower() == "mobilefacenet":
                landmark = (
                    self.landmark_detector(extracted_faces, **landmark_model_kwargs)[0]
                    .cpu()
                    .data.numpy()
                )
            else:
                landmark = (
                    self.landmark_detector(extracted_faces, **landmark_model_kwargs)
                    .cpu()
                    .data.numpy()
                )

            landmark = landmark.reshape(landmark.shape[0], -1, 2)

            landmark_results = []
            for ik in range(landmark.shape[0]):
                landmark_results.append(
                    new_bbox[ik].inverse_transform_landmark(landmark[ik, :, :])
                )

            length_index = [len(x) for x in detected_faces]
            new_lens = np.insert(np.cumsum(length_index), 0, 0)
            list_concat = []
            for ij in range(len(length_index)):
                list_concat.append(landmark_results[new_lens[ij] : new_lens[ij + 1]])

        return list_concat

    def detect_facepose(self, frame, landmarks=None, **facepose_model_kwargs):
        """Detect facepose from image or video frame.

        When used with img2pose, returns *all* detected poses, and facebox and landmarks
        are ignored. Use `detect_face` method in order to obtain bounding boxes
        corresponding to the detected poses returned by this method.

        Args:
            frame (np.ndarray): list of images
            landmarks (np.ndarray | None, optional): (num_images, num_faces, 68, 2)
            landmarks for the faces contained in list of images; Default None and
            ignored for img2pose and img2pose-c detectors

        Returns:
            list: poses (num_images, num_faces, [pitch, roll, yaw]) - Euler angles (in
            degrees) for each face within in each image}

        """

        logging.info("detecting poses...")
        # Normalize Data
        frame = convert_image_to_tensor(frame, img_type="float32") / 255

        output = {}
        if "img2pose" in self.info["facepose_model"]:
            img2pose_output = self.facepose_detector(frame, **facepose_model_kwargs)
            img2pose_output = postprocess_img2pose(img2pose_output[0])
            output["faces"] = img2pose_output['boxes']
            output["poses"] = img2pose_output['dofs'] # Only returning xyz for now not translation
        else:
            output["poses"] = self.facepose_detector(
                frame, landmarks, **facepose_model_kwargs
            )

        return output

    def detect_aus(self, frame, landmarks, **au_model_kwargs):
        """Detect Action Units from image or video frame

        Args:
            frame (np.ndarray): image loaded in array format (n, m, 3)
            landmarks (array): 68 landmarks used to localize face.

        Returns:
            array: Action Unit predictions

        Examples:
            >>> from feat import Detector
            >>> from feat.utils import read_pictures
            >>> frame = read_pictures(['my_image.jpg'])
            >>> detector = Detector()
            >>> detector.detect_aus(frame)
        """

        logging.info("detecting aus...")
        frame = convert_image_to_tensor(frame, img_type="float32")

        if is_list_of_lists_empty(landmarks):
            return landmarks
        else:
            if self["au_model"].lower() in ["svm", "xgb"]:
                # transform = Grayscale(3)
                # frame = transform(frame)
                hog_features, new_landmarks = self._batch_hog(
                    frames=frame, landmarks=landmarks
                )
                au_predictions = self.au_model.detect_au(
                    frame=hog_features, landmarks=new_landmarks, **au_model_kwargs
                )
            else:
                au_predictions = self.au_model.detect_au(
                    frame, landmarks=landmarks, **au_model_kwargs
                )

            return self._convert_detector_output(landmarks, au_predictions)

    def _batch_hog(self, frames, landmarks):
        """
        Helper function used in batch processing hog features

        Args:
            frames: a batch of frames
            landmarks: a list of list of detected landmarks

        Returns:
            hog_features: a numpy array of hog features for each detected landmark
            landmarks: updated landmarks
        """

        hog_features = []
        new_landmark_frames = []
        for i, frame_landmark in enumerate(landmarks):
            if len(frame_landmark) != 0:
                new_landmarks_faces = []
                for j in range(len(frame_landmark)):
                    convex_hull, new_landmark = extract_face_from_landmarks(
                        frame=frames[i],
                        landmarks=frame_landmark[j],
                        face_size=112,
                    )

                    hog_features.append(
                        hog(
                            transforms.ToPILImage()(convex_hull[0] / 255.0),
                            orientations=8,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            visualize=False,
                            channel_axis=-1,
                        ).reshape(1, -1)
                    )

                    new_landmarks_faces.append(new_landmark)
                new_landmark_frames.append(new_landmarks_faces)
            else:
                hog_features.append(
                    np.zeros((1, 5408))
                )  # LC: Need to confirm this size is fixed.
                new_landmark_frames.append([np.zeros((68, 2))])

        hog_features = np.concatenate(hog_features)

        return (hog_features, new_landmark_frames)

    def detect_emotions(self, frame, facebox, landmarks, **emotion_model_kwargs):
        """Detect emotions from image or video frame

        Args:
            frame ([type]): [description]
            facebox ([type]): [description]
            landmarks ([type]): [description]

        Returns:
            array: Action Unit predictions

        Examples:
            >>> from feat import Detector
            >>> from feat.utils import read_pictures
            >>> img_data = read_pictures(['my_image.jpg'])
            >>> detector = Detector()
            >>> detected_faces = detector.detect_faces(frame)
            >>> detected_landmarks = detector.detect_landmarks(frame, detected_faces)
            >>> detector.detect_emotions(frame, detected_faces, detected_landmarks)
        """

        logging.info("detecting emotions...")
        frame = convert_image_to_tensor(frame, img_type="float32")

        if is_list_of_lists_empty(facebox):
            return facebox
        else:
            if self.info["emotion_model"].lower() == "resmasknet":
                return self._convert_detector_output(
                    facebox,
                    self.emotion_model.detect_emo(
                        frame, facebox, **emotion_model_kwargs
                    ),
                )

            elif self.info["emotion_model"].lower() == "svm":
                hog_features, new_landmarks = self._batch_hog(
                    frames=frame, landmarks=landmarks
                )
                return self._convert_detector_output(
                    landmarks,
                    self.emotion_model.detect_emo(
                        frame=hog_features,
                        landmarks=new_landmarks,
                        **emotion_model_kwargs,
                    ),
                )

            else:
                raise ValueError(
                    "Cannot recognize input emo model! Please try to re-type emotion model"
                )

    def detect_identity(self, frame, facebox, **identity_model_kwargs):
        """Detects identity of faces from image or video frame using face representation embeddings

        Args:
            frame (np.ndarray): 3d (single) or 4d (multiple) image array
            threshold (float): threshold for matching identity (default=0.8)

        Returns:
            list: list of lists with the same length as the number of frames. Each list
            item is a list containing the (x1, y1, x2, y2) coordinates of each detected
            face in that frame.

        """

        logging.info("detecting identity...")

        frame = convert_image_to_tensor(frame, img_type="float32") / 255

        if is_list_of_lists_empty(facebox):
            return facebox
        else:
            extracted_faces, new_bbox = extract_face_from_bbox(frame, facebox)
            face_embeddings = self.identity_model(
                extracted_faces, **identity_model_kwargs
            )
        return self._convert_detector_output(facebox, face_embeddings.detach().numpy())

    def _run_detection_waterfall(
        self,
        batch_data,
        face_detection_threshold,
        face_model_kwargs,
        landmark_model_kwargs,
        facepose_model_kwargs,
        emotion_model_kwargs,
        au_model_kwargs,
        identity_model_kwargs,
        suppress_torchvision_warnings=True,
    ):
        """
        Main detection "waterfall." Calls each individual detector in the sequence
        required to support any interactions between detections. Called
        behind-the-scenes by .detect_image() and .detect_video()

        Args:
            batch_data (dict): singleton item from iterating over the output of a DataLoader
            face_detection_threshold (float): value between 0-1
            face_model_kwargs (dict): face model kwargs
            landmark_model_kwargs (dict): landmark model kwargs
            facepose_model_kwargs (dict): facepose model kwargs
            emotion_model_kwargs (dict): emotion model kwargs
            au_model_kwargs (dict): au model kwargs
            identity_model_kwargs (dict): identity model kwargs

        Returns:
            tuple: faces, landmarks, poses, aus, emotions, identities
        """

        # Reset warnings
        warnings.filterwarnings("default", category=UserWarning, module="torchvision")

        if suppress_torchvision_warnings:
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="torchvision"
            )

        faces = self.detect_faces(
            batch_data["Image"],
            threshold=face_detection_threshold,
            **face_model_kwargs,
        )

        landmarks = self.detect_landmarks(
            batch_data["Image"],
            detected_faces=faces,
            **landmark_model_kwargs,
        )
        
        poses_dict = self.detect_facepose(
            batch_data["Image"], landmarks, **facepose_model_kwargs
        )

        aus = self.detect_aus(batch_data["Image"], landmarks, **au_model_kwargs)

        emotions = self.detect_emotions(
            batch_data["Image"], faces, landmarks, **emotion_model_kwargs
        )

        identities = self.detect_identity(
            batch_data["Image"],
            faces,
            **identity_model_kwargs,
        )

        faces = _inverse_face_transform(faces, batch_data)
        landmarks = _inverse_landmark_transform(landmarks, batch_data)

        # match faces to poses - sometimes face detector finds different faces than pose detector.
        faces, poses = self._match_faces_to_poses(
            faces, poses_dict["faces"], poses_dict["poses"]
        )

        return faces, landmarks, poses, aus, emotions, identities

    def detect_image(
        self,
        input_file_list,
        output_size=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        frame_counter=0,
        face_detection_threshold=0.5,
        face_identity_threshold=0.8,
        **kwargs,
    ):
        """
        Detects FEX from one or more image files. If you want to speed up detection you
        can process multiple images in batches by setting `batch_size > 1`. However, all
        images must have **the same dimensions** to be processed in batches. Py-feat can
        automatically adjust image sizes by using the `output_size=int`. Common
        output-sizes include 256 and 512.

        **NOTE: Currently batch processing images gives slightly different AU detection results due to the way that py-feat integrates the underlying models. You can examine the degree of tolerance by checking out the results of `test_detection_and_batching_with_diff_img_sizes` in our test-suite**

        Args:
            input_file_list (list of str): Path to a list of paths to image files.
            output_size (int): image size to rescale all image preserving aspect ratio.
                                Will raise an error if not set and batch_size > 1 but images are not the same size
            batch_size (int): how many batches of images you want to run at one shot.
                                Larger gives faster speed but is more memory-consuming. Images must be the
            same size to be run in batches!
            num_workers (int): how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
            pin_memory (bool): If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them.  If your data elements are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type
            frame_counter (int): starting value to count frames
            face_detection_threshold (float): value between 0-1 to report a detection based on the
                                confidence of the face detector; Default >= 0.5
            face_identity_threshold (float): value between 0-1 to determine similarity of person using face identity embeddings; Default >= 0.8
            **kwargs: you can pass each detector specific kwargs using a dictionary
                                like: `face_model_kwargs = {...}, au_model_kwargs={...}, ...`

        Returns:
            Fex: Prediction results dataframe
        """

        # Keyword arguments than can be passed to the underlying models
        face_model_kwargs = kwargs.pop("face_model_kwargs", dict())
        landmark_model_kwargs = kwargs.pop("landmark_model_kwargs", dict())
        au_model_kwargs = kwargs.pop("au_model_kwargs", dict())
        emotion_model_kwargs = kwargs.pop("emotion_model_kwargs", dict())
        facepose_model_kwargs = kwargs.pop("facepose_model_kwargs", dict())
        identity_model_kwargs = kwargs.pop("identity_model_kwargs", dict())

        data_loader = DataLoader(
            ImageDataset(
                input_file_list,
                output_size=output_size,
                preserve_aspect_ratio=True,
                padding=True,
            ),
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=False,
        )

        if self.info["landmark_model"] == "mobilenet" and batch_size > 1:
            warnings.warn(
                "Currently using mobilenet for landmark detection with batch_size > 1 may lead to erroneous detections. We recommend either setting batch_size=1 or using mobilefacenet as the landmark detection model. You can follow this issue for more: https://github.com/cosanlab/py-feat/issues/151"
            )

        try:
            batch_output = []

            for batch_id, batch_data in enumerate(tqdm(data_loader)):
                (
                    faces,
                    landmarks,
                    poses,
                    aus,
                    emotions,
                    identities,
                ) = self._run_detection_waterfall(
                    batch_data,
                    face_detection_threshold,
                    face_model_kwargs,
                    landmark_model_kwargs,
                    facepose_model_kwargs,
                    emotion_model_kwargs,
                    au_model_kwargs,
                    identity_model_kwargs,
                )

                output = self._create_fex(
                    faces,
                    landmarks,
                    poses,
                    aus,
                    emotions,
                    identities,
                    batch_data["FileNames"],
                    frame_counter,
                )
                batch_output.append(output)
                frame_counter += 1 * batch_size

            batch_output = pd.concat(batch_output)
            batch_output.reset_index(drop=True, inplace=True)
            batch_output.compute_identities(
                threshold=face_identity_threshold, inplace=True
            )
            return batch_output
        except RuntimeError as e:
            raise ValueError(
                f"when using a batch_size > 1 all images must have the same dimensions or output_size must not be None so py-feat can rescale images to output_size. See pytorch error: \n{e}"
            )

    def detect_video(
        self,
        video_path,
        skip_frames=None,
        output_size=700,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        face_detection_threshold=0.5,
        face_identity_threshold=0.8,
        **kwargs,
    ):
        """Detects FEX from a video file.

        Args:
            video_path (str): Path to a video file.
            skip_frames (int or None): number of frames to skip (speeds up inference,
            but less temporal information); Default None
            output_size (int): image size to rescale all imagee preserving aspect ratio
            batch_size (int): how many batches of images you want to run at one shot. Larger gives faster speed but is more memory-consuming
            num_workers (int): how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
            pin_memory (bool): If ``True``, the data loader will copy Tensors
                                into CUDA pinned memory before returning them.  If your data elements
                                are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type
            face_detection_threshold (float): value between 0-1 to report a detection based on the
                                confidence of the face detector; Default >= 0.5
            face_identity_threshold (float): value between 0-1 to determine similarity of person using face identity embeddings; Default >= 0.8

        Returns:
            Fex: Prediction results dataframe
        """

        # Keyword arguments than can be passed to the underlying models
        face_model_kwargs = kwargs.pop("face_model_kwargs", dict())
        landmark_model_kwargs = kwargs.pop("landmark_model_kwargs", dict())
        au_model_kwargs = kwargs.pop("au_model_kwargs", dict())
        emotion_model_kwargs = kwargs.pop("emotion_model_kwargs", dict())
        facepose_model_kwargs = kwargs.pop("facepose_model_kwargs", dict())
        identity_model_kwargs = kwargs.pop("identity_model_kwargs", dict())

        dataset = VideoDataset(
            video_path, skip_frames=skip_frames, output_size=output_size
        )

        data_loader = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=False,
        )

        batch_output = []

        for batch_data in tqdm(data_loader):
            (
                faces,
                landmarks,
                poses,
                aus,
                emotions,
                identities,
            ) = self._run_detection_waterfall(
                batch_data,
                face_detection_threshold,
                face_model_kwargs,
                landmark_model_kwargs,
                facepose_model_kwargs,
                emotion_model_kwargs,
                au_model_kwargs,
                identity_model_kwargs,
            )

            frames = list(batch_data["Frame"].numpy())

            output = self._create_fex(
                faces,
                landmarks,
                poses,
                aus,
                emotions,
                identities,
                batch_data["FileName"],
                frames,
            )

            batch_output.append(output)

        batch_output = pd.concat(batch_output)
        batch_output.reset_index(drop=True, inplace=True)
        batch_output["approx_time"] = [
            dataset.calc_approx_frame_time(x) for x in batch_output["frame"].to_numpy()
        ]
        batch_output.compute_identities(threshold=face_identity_threshold, inplace=True)

        return batch_output.set_index("frame", drop=False)

    def _create_fex(
        self,
        faces,
        landmarks,
        poses,
        aus,
        emotions,
        identities,
        file_names,
        frame_counter,
    ):
        """Helper function to create a Fex instance using detector output

        Args:
            faces: output of detect_faces()
            landmarks: output of detect_landmarks()
            poses: output of dectect_facepose()
            aus: output of detect_aus()
            emotions: output of detect_emotions()
            identities: output of detect_identities()
            file_names: file name of input image
            frame_counter: starting value for frame counter, useful for integrating batches

        Returns:
            Fex object
        """

        logging.info("creating fex output...")

        out = []
        for i, frame in enumerate(faces):
            if not frame:
                facebox_df = pd.DataFrame(
                    {x: np.nan for x in self.info["face_detection_columns"]},
                    columns=self.info["face_detection_columns"],
                    index=[i],
                )
                facepose_df = pd.DataFrame(
                    {x: np.nan for x in self.info["facepose_model_columns"]},
                    columns=self.info["facepose_model_columns"],
                    index=[i],
                )
                landmarks_df = pd.DataFrame(
                    {x: np.nan for x in self.info["face_landmark_columns"]},
                    columns=self.info["face_landmark_columns"],
                    index=[i],
                )
                aus_df = pd.DataFrame(
                    {x: np.nan for x in self.info["au_presence_columns"]},
                    columns=self.info["au_presence_columns"],
                    index=[i],
                )
                emotions_df = pd.DataFrame(
                    {x: np.nan for x in self.info["emotion_model_columns"]},
                    columns=self.info["emotion_model_columns"],
                    index=[i],
                )
                identity_df = pd.DataFrame(
                    {x: np.nan for x in self.info["identity_model_columns"]},
                    columns=self.info["identity_model_columns"],
                    index=[i],
                )
                input_df = pd.DataFrame(file_names[i], columns=["input"], index=[i])
                tmp_df = pd.concat(
                    [
                        facebox_df,
                        landmarks_df,
                        facepose_df,
                        aus_df,
                        emotions_df,
                        identity_df,
                        input_df,
                    ],
                    axis=1,
                )
                if isinstance(frame_counter, (list)):
                    tmp_df[FEAT_TIME_COLUMNS] = frame_counter[i]
                else:
                    tmp_df[FEAT_TIME_COLUMNS] = frame_counter + i
                out.append(tmp_df)

            for j, face_in_frame in enumerate(frame):
                facebox_df = pd.DataFrame(
                    [
                        [
                            face_in_frame[0],
                            face_in_frame[1],
                            face_in_frame[2] - face_in_frame[0],
                            face_in_frame[3] - face_in_frame[1],
                            face_in_frame[4],
                        ]
                    ],
                    columns=self.info["face_detection_columns"],
                    index=[j],
                )

                facepose_df = pd.DataFrame(
                    [poses[i][j]],
                    columns=self.info["facepose_model_columns"],
                    index=[j],
                )

                landmarks_df = pd.DataFrame(
                    [landmarks[i][j].flatten(order="F")],
                    columns=self.info["face_landmark_columns"],
                    index=[j],
                )

                aus_df = pd.DataFrame(
                    aus[i][j, :].reshape(1, len(self["au_presence_columns"])),
                    columns=self.info["au_presence_columns"],
                    index=[j],
                )

                emotions_df = pd.DataFrame(
                    emotions[i][j, :].reshape(
                        1, len(self.info["emotion_model_columns"])
                    ),
                    columns=self.info["emotion_model_columns"],
                    index=[j],
                )

                identity_df = pd.DataFrame(
                    np.hstack([np.nan, identities[i][j]]).reshape(-1, 1).T,
                    columns=self.info["identity_model_columns"],
                    index=[j],
                )

                input_df = pd.DataFrame(
                    file_names[i],
                    columns=["input"],
                    index=[j],
                )

                tmp_df = pd.concat(
                    [
                        facebox_df,
                        landmarks_df,
                        facepose_df,
                        aus_df,
                        emotions_df,
                        identity_df,
                        input_df,
                    ],
                    axis=1,
                )

                if isinstance(frame_counter, (list)):
                    tmp_df[FEAT_TIME_COLUMNS] = frame_counter[i]
                else:
                    tmp_df[FEAT_TIME_COLUMNS] = frame_counter + i
                out.append(tmp_df)

        out = pd.concat(out)
        out.reset_index(drop=True, inplace=True)

        # TODO: Add in support for gaze_columns
        return Fex(
            out,
            au_columns=self.info["au_presence_columns"],
            emotion_columns=self.info["emotion_model_columns"],
            facebox_columns=self.info["face_detection_columns"],
            landmark_columns=self.info["face_landmark_columns"],
            facepose_columns=self.info["facepose_model_columns"],
            identity_columns=self.info["identity_model_columns"],
            detector="Feat",
            face_model=self.info["face_model"],
            landmark_model=self.info["landmark_model"],
            au_model=self.info["au_model"],
            emotion_model=self.info["emotion_model"],
            facepose_model=self.info["facepose_model"],
            identity_model=self.info["identity_model"],
        )

    @staticmethod
    def _convert_detector_output(detected_faces, detector_results):
        """
        Helper function to convert AU/Emotion detector output into frame by face list of lists.
        Either face or landmark detector list of list outputs can be used.

        Args:
            detected_faces (list): list of lists output from face/landmark detector
            au_results (np.array):, results from au/emotion detectors

        Returns:
            list_concat: (list of list). The list which contains the number of faces. for example
            if you process 2 frames and each frame contains 4 faces, it will return:
                [[xxx,xxx,xxx,xxx],[xxx,xxx,xxx,xxx]]
        """

        length_index = [len(x) for x in detected_faces]

        list_concat = []
        new_lens = np.insert(np.cumsum(length_index), 0, 0)
        for ij in range(len(length_index)):
            list_concat.append(detector_results[new_lens[ij] : new_lens[ij + 1], :])
        return list_concat

    @staticmethod
    def _match_faces_to_poses(faces, faces_pose, poses):
        """Helper function to match list of lists of faces and poses based on overlap in bounding boxes.

        Sometimes the face detector finds different faces than the pose detector unless the user
        is using the same detector (i.e., img2pose).

        This function will match the faces and poses and will return nans if more faces are detected then poses.
        Will only return poses that match faces even if more faces are detected by pose detector.

        Args:
            faces (list): list of lists of face bounding boxes from face detector
            faces_pose (list): list of lists of face bounding boxes from pose detector
            poses (list): list of lists of poses from pose detector

        Returns:
            faces (list): list of list of faces that have been matched to poses
            poses (list): list of list of poses that have been matched to faces
        """

        if len(faces) != len(faces_pose):
            raise ValueError(
                "Make sure the number of batches in faces and poses is the same."
            )

        if is_list_of_lists_empty(faces):
            # Currently assuming no faces if no face is detected. Not running pose
            return (faces, poses)

        else:
            overlap_faces = []
            overlap_poses = []
            for frame_face, frame_face_pose, frame_pose in zip(
                faces, faces_pose, poses
            ):
                if not frame_face:
                    n_faces = 0
                elif isinstance(frame_face[0], list):
                    n_faces = len(frame_face)
                else:
                    n_faces = 1

                if not frame_face_pose:
                    n_poses = 0
                elif isinstance(frame_face_pose[0], list):
                    n_poses = len(frame_face_pose)
                else:
                    n_poses = 1

                frame_overlap = np.zeros([n_faces, n_poses])

                if n_faces == 0:
                    overlap_faces.append([])
                    overlap_poses.append([])

                elif (n_faces == 1) & (n_poses > 1):
                    b1 = BBox(frame_face[0][:-1])

                    for pose_idx in range(n_poses):
                        b2 = BBox(frame_face_pose[pose_idx][:-1])
                        frame_overlap[0, pose_idx] = b1.overlap(b2)
                    matched_pose_index = np.where(
                        frame_overlap[0, :] == frame_overlap[0, :].max()
                    )[0][0]
                    overlap_faces.append(frame_face)
                    overlap_poses.append([frame_pose[matched_pose_index]])

                elif (n_faces > 1) & (n_poses == 1):
                    b2 = BBox(frame_face_pose[0][:-1])
                    for face_idx in range(n_faces):
                        b1 = BBox(frame_face[face_idx][:-1])
                        frame_overlap[face_idx, 0] = b1.overlap(b2)
                    matched_face_index = np.where(
                        frame_overlap[:, 0] == frame_overlap[:, 0].max()
                    )[0][0]
                    new_poses = []
                    for f_idx in range(n_faces):
                        if f_idx == matched_face_index:
                            new_poses.append(frame_pose[0])
                        else:
                            new_poses.append(np.ones(3) * np.nan)
                    overlap_faces.append(frame_face)
                    overlap_poses.append(new_poses)

                else:
                    for face_idx in range(n_faces):
                        b1 = BBox(frame_face[face_idx][:-1])
                        for pose_idx in range(n_poses):
                            b2 = BBox(frame_face_pose[pose_idx][:-1])
                            frame_overlap[face_idx, pose_idx] = b1.overlap(b2)

                    overlap_faces_frame = []
                    overlap_poses_frame = []
                    if n_faces < n_poses:
                        for face_idx in range(n_faces):
                            pose_idx = np.where(
                                frame_overlap[face_idx, :]
                                == frame_overlap[face_idx, :].max()
                            )[0][0]
                            overlap_faces_frame.append(frame_face[face_idx])
                            overlap_poses_frame.append(frame_pose[pose_idx])
                    elif n_faces > n_poses:
                        matched_pose_index = []
                        for pose_idx in range(n_poses):
                            matched_pose_index.append(
                                np.where(
                                    frame_overlap[:, pose_idx]
                                    == frame_overlap[:, pose_idx].max()
                                )[0][0]
                            )
                        for face_idx in range(n_faces):
                            overlap_faces_frame.append(frame_face[face_idx])
                            if face_idx in matched_pose_index:
                                overlap_poses_frame.append(
                                    frame_pose[
                                        np.where(
                                            frame_overlap[face_idx, :]
                                            == frame_overlap[face_idx, :].max()
                                        )[0][0]
                                    ]
                                )
                            else:
                                overlap_poses_frame.append(np.ones(3) * np.nan)
                    elif n_faces == n_poses:
                        overlap_faces_frame = frame_face
                        overlap_poses_frame = frame_pose

                    overlap_faces.append(overlap_faces_frame)
                    overlap_poses.append(overlap_poses_frame)

            return (overlap_faces, overlap_poses)
