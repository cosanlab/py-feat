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
    FACET_FACEPOSE_COLUMNS,
    FEAT_TIME_COLUMNS,
    FACET_TIME_COLUMNS,
    set_torch_device,
)
from feat.utils.io import get_resource_path
from feat.utils.image_operations import (
    extract_face_from_landmarks,
    extract_face_from_bbox,
    convert_image_to_tensor,
)
from feat.pretrained import get_pretrained_models, fetch_model, AU_LANDMARK_MAP
from feat.data import (
    Fex,
    ImageDataset,
    VideoDataset,
    _inverse_face_transform,
    _inverse_landmark_transform,
)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Grayscale
import logging
import warnings
from tqdm import tqdm

# Supress sklearn warning about pickled estimators and diff sklearn versions
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class Detector(object):
    def __init__(
        self,
        face_model="retinaface",
        landmark_model="mobilenet",
        au_model="svm",
        emotion_model="resmasknet",
        facepose_model="img2pose",
        device="auto",
        n_jobs=1,
        verbose=False,
    ):
        """Detector class to detect FEX from images or videos.

        Detector is a class used to detect faces, facial landmarks, emotions, and action units from images and videos.

        Args:
            n_jobs (int, default=1): Number of processes to use for extraction.
            device (str): specify device to process data (default='auto'), can be ['auto', 'cpu', 'cuda', 'mps']

        Attributes:
            info (dict):
                n_jobs (int): Number of jobs to be used in parallel.
                face_model (str, default=retinaface): Name of face detection model
                landmark_model (str, default=mobilenet): Nam eof landmark model
                au_model (str, default=rf): Name of Action Unit detection model
                emotion_model (str, default=resmasknet): Path to emotion detection model.
                facepose_model (str, default=pnp): Name of headpose detection model.
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
            n_jobs=n_jobs,
        )
        # Setup verbosity
        self.logger = logging.getLogger("Detector")
        self.verbose = verbose
        if self.verbose:
            log_level = logging.INFO
        else:
            log_level = logging.WARNING
        self.logger.setLevel(log_level)

        # Setup device
        self.device = set_torch_device(device)

        # Verify model names and download if necessary
        face, landmark, au, emotion, facepose = get_pretrained_models(
            face_model, landmark_model, au_model, emotion_model, facepose_model, verbose
        )

        self._init_detectors(
            face,
            landmark,
            au,
            emotion,
            facepose,
            openface_2d_landmark_columns,
        )

    def __repr__(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}(face_model={self.info['face_model']}, landmark_model={self.info['landmark_model']}, au_model={self.info['au_model']}, emotion_model={self.info['emotion_model']}, facepose_model={self.info['facepose_model']})"

    def __getitem__(self, i):
        return self.info[i]

    def _init_detectors(
        self,
        face,
        landmark,
        au,
        emotion,
        facepose,
        openface_2d_landmark_columns,
    ):
        """Helper function called by __init__ and change_model to (re)initialize one of
        the supported detectors"""

        # Initialize model instances and any additional post init setup
        # Only initialize a model if the currently initialized model is diff than the
        # requested one. Lets us re-use this with .change_model

        # FACE MODEL
        if self.info["face_model"] != face:
            self.logger.info(f"Loading Face model: {face}")
            self.face_detector = fetch_model("face_model", face)
            self.info["face_model"] = face
            self.info["face_detection_columns"] = FEAT_FACEBOX_COLUMNS
            predictions = np.full_like(np.atleast_2d(FEAT_FACEBOX_COLUMNS), np.nan)
            empty_facebox = pd.DataFrame(predictions, columns=FEAT_FACEBOX_COLUMNS)
            self._empty_facebox = empty_facebox
            if self.face_detector is not None:
                if "img2pose" in face:
                    self.face_detector = self.face_detector(
                        constrained="img2pose-c" == face, device=self.device
                    )
                else:
                    self.face_detector = self.face_detector(device=self.device)

        # LANDMARK MODEL
        if self.info["landmark_model"] != landmark:
            self.logger.info(f"Loading Facial Landmark model: {landmark}")
            self.landmark_detector = fetch_model("landmark_model", landmark)
            if self.landmark_detector is not None:
                if landmark == "mobilenet":
                    self.landmark_detector = self.landmark_detector(136)
                    self.landmark_detector = torch.nn.DataParallel(
                        self.landmark_detector
                    )
                    checkpoint = torch.load(
                        os.path.join(
                            get_resource_path(),
                            "mobilenet_224_model_best_gdconv_external.pth.tar",
                        ),
                        map_location=self.device,
                    )
                    self.landmark_detector.load_state_dict(checkpoint["state_dict"])
                elif landmark == "pfld":
                    self.landmark_detector = self.landmark_detector()
                    checkpoint = torch.load(
                        os.path.join(get_resource_path(), "pfld_model_best.pth.tar"),
                        map_location=self.device,
                    )
                    self.landmark_detector.load_state_dict(checkpoint["state_dict"])
                elif landmark == "mobilefacenet":
                    self.landmark_detector = self.landmark_detector([112, 112], 136)
                    checkpoint = torch.load(
                        os.path.join(
                            get_resource_path(), "mobilefacenet_model_best.pth.tar"
                        ),
                        map_location=self.device,
                    )
                    self.landmark_detector.load_state_dict(checkpoint["state_dict"])
            self.landmark_detector.eval()

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
            self.logger.info("Loading facepose model: ", facepose)
            self.facepose_detector = fetch_model("facepose_model", facepose)
            if "img2pose" in facepose:
                self.facepose_detector = self.facepose_detector(
                    constrained="img2pose-c" == face, device=self.device
                )
            else:
                self.facepose_detector = self.facepose_detector()
            self.info["facepose_model"] = facepose

            self.info["facepose_model_columns"] = FACET_FACEPOSE_COLUMNS
            predictions = np.full_like(np.atleast_2d(FACET_FACEPOSE_COLUMNS), np.nan)
            empty_facepose = pd.DataFrame(predictions, columns=FACET_FACEPOSE_COLUMNS)
            self._empty_facepose = empty_facepose

        # AU MODEL
        if self.info["au_model"] != au:
            self.logger.info(f"Loading AU model: {au}")
            self.au_model = fetch_model("au_model", au)
            self.info["au_model"] = au
            if self.info["au_model"] in ["svm", "logistic"]:
                self.info["au_presence_columns"] = AU_LANDMARK_MAP["Feat"]
            else:
                self.info["au_presence_columns"] = AU_LANDMARK_MAP[
                    self.info["au_model"]
                ]
            if self.au_model is not None:
                self.au_model = self.au_model()
                predictions = np.full_like(
                    np.atleast_2d(self.info["au_presence_columns"]), np.nan
                )
                empty_au_occurs = pd.DataFrame(
                    predictions, columns=self.info["au_presence_columns"]
                )
                self._empty_auoccurence = empty_au_occurs

        # EMOTION MODEL
        if self.info["emotion_model"] != emotion:
            self.logger.info("Loading emotion model: ", emotion)
            self.emotion_model = fetch_model("emotion_model", emotion)
            self.info["emotion_model"] = emotion
            if self.emotion_model is not None:
                self.emotion_model = self.emotion_model(device=self.device)
                self.info["emotion_model_columns"] = FEAT_EMOTION_COLUMNS
                predictions = np.full_like(np.atleast_2d(FEAT_EMOTION_COLUMNS), np.nan)
                empty_emotion = pd.DataFrame(predictions, columns=FEAT_EMOTION_COLUMNS)
                self._empty_emotion = empty_emotion

        self.info["output_columns"] = (
            FEAT_TIME_COLUMNS
            + FEAT_FACEBOX_COLUMNS
            + openface_2d_landmark_columns
            + self.info["au_presence_columns"]
            + FACET_FACEPOSE_COLUMNS
            + FEAT_EMOTION_COLUMNS
            + ["input"]
        )

    def change_model(self, **kwargs):
        """Swap one or more pre-trained detector models for another one. Just pass in
        the the new models to use as kwargs, e.g. emotion_model='rf'"""

        face_model = kwargs.get("face_model", self.info["face_model"])
        landmark_model = kwargs.get("landmark_model", self.info["landmark_model"])
        au_model = kwargs.get("au_model", self.info["au_model"])
        emotion_model = kwargs.get("emotion_model", self.info["emotion_model"])
        facepose_model = kwargs.get("facepose_model", self.info["facepose_model"])

        # Verify model names and download if necessary
        face, landmark, au, emotion, facepose = get_pretrained_models(
            face_model,
            landmark_model,
            au_model,
            emotion_model,
            facepose_model,
            self.verbose,
        )
        for requested, current_name in zip(
            [face, landmark, au, emotion, facepose],
            [
                "face_model",
                "landmark_model",
                "au_model",
                "emotion_model",
                "facepose_model",
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
            openface_2d_landmark_columns,
        )

    def detect_faces(self, frame):
        """Detect faces from image or video frame

        Args:
            frame (np.ndarray): 3d (single) or 4d (multiple) image array

        Returns:
            list: list of lists with the same length as the number of frames. Each list
            item is a list containing the (x1, y1, x2, y2) coordinates of each detected
            face in that frame.

        Examples:
            >>> from feat import Detector
            >>> from feat.utils import read_pictures
            >>> img_data = read_pictures(['my_image.jpg'])
            >>> detector = Detector()
            >>> detector.detect_faces(frame)
        """

        frame = convert_image_to_tensor(frame, img_type="float32")

        if "img2pose" in self.info["face_model"]:
            frame = frame / 255
            faces, poses = self.face_detector(frame)
        else:
            faces = self.face_detector(frame)

        if len(faces) == 0:
            self.logger.warning("Warning: NO FACE is detected")
        return faces

    def detect_landmarks(self, frame, detected_faces):
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

        frame = convert_image_to_tensor(frame)

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
            landmark = self.landmark_detector(extracted_faces)[0].cpu().data.numpy()
        else:
            landmark = self.landmark_detector(extracted_faces).cpu().data.numpy()

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

    def detect_facepose(self, frame, landmarks=None):
        """Detect facepose from image or video frame.

        When used with img2pose, returns *all* detected poses, and facebox and landmarks
        are ignored. Use `detect_face` method in order to obtain bounding boxes
        corresponding to the detected poses returned by this method.

        When used with pnp model, 'facebox' param is ignored, and the passed 2D
        landmarks are used to compute the head pose for the single face associated with
        the passed landmarks.

        Args:
            frame (np.ndarray): list of cv2 images
            landmarks (np.ndarray): (num_images, num_faces, 68, 2) landmarks for the faces contained in list of images

        Returns:
            dict: {"faces": list of face bounding boxes, "poses": (num_images, num_faces, [pitch, roll, yaw]) - Euler angles (in
            degrees) for each face within in each image}

        Examples:
            >>> from feat import Detector
            >>> from feat.utils import read_pictures
            >>> frame = read_pictures(['my_image.jpg'])

            >>> # Imgpose detector
            >>> imgpose_detector = Detector(face_model='imgpose', facepose_model='img2pose')
            >>> imgpose_detector.detect_facepose(frame) # one shot computation

            >>> # Retina face detector
            >>> retinaface_detector = Detector(face_model='retinaface', landmark_model='mobilefacenet', facepose_model='pnp')
            >>> faces = retinaface_detector.detect_faces(frame)
            >>> landmarks = retinaface_detector.detect_landmarks(detected_faces=faces)
            >>> retinaface_detector.detect_facepose(frame=frame, landmarks=landmarks) # detect pose for all faces
        """
        # Normalize Data
        frame = convert_image_to_tensor(frame, img_type="float32") / 255

        if "img2pose" in self.info["facepose_model"]:
            faces, poses = self.facepose_detector(frame)
        else:
            poses = self.facepose_detector(frame, landmarks)
            # faces = detected_faces

        # return {"faces": faces, "poses": poses}
        return poses

    def detect_aus(self, frame, landmarks):
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

        frame = convert_image_to_tensor(frame, img_type="float32")

        if self["au_model"].lower() in ["logistic", "svm"]:
            transform = Grayscale(3)
            frame = transform(frame)
            hog_arr, new_lands = self._batch_hog(frames=frame, landmarks=landmarks)
            au_predictions = self.au_model.detect_au(frame=hog_arr, landmarks=new_lands)
        else:
            au_predictions = self.au_model.detect_au(frame, landmarks=landmarks)

        return self._convert_detector_output(landmarks, au_predictions)

    def _batch_hog(self, frames, landmarks):
        """
        Helper function used in batch processing hog features
        frames is a batch of frames
        """

        len_index = [len(aa) for aa in landmarks]
        lenth_cumu = np.cumsum(len_index)
        lenth_cumu2 = np.insert(lenth_cumu, 0, 0)
        new_lands_list = []
        flat_land = [item for sublist in landmarks for item in sublist]
        hogs_arr = None

        for i in range(len(flat_land)):

            frame_assignment = np.where(i < lenth_cumu)[0][0]

            convex_hull, new_lands = extract_face_from_landmarks(
                frame=frames[frame_assignment],
                landmarks=flat_land[i],
                face_size=112,
            )

            hogs = hog(
                convex_hull.squeeze().permute(1, 2, 0).type(torch.int).numpy(),
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=False,
                channel_axis=-1,
            ).reshape(1, -1)

            if hogs_arr is None:
                hogs_arr = hogs
            else:
                hogs_arr = np.concatenate([hogs_arr, hogs], 0)

            new_lands_list.append(new_lands)

        new_lands = []
        for i in range(len(lenth_cumu)):
            new_lands.append(new_lands_list[lenth_cumu2[i] : (lenth_cumu2[i + 1])])

        return (hogs_arr, new_lands)

    def detect_emotions(self, frame, facebox, landmarks):
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

        frame = convert_image_to_tensor(frame, img_type="float32")

        if self.info["emotion_model"].lower() == "fer":
            transform = Grayscale(3)
            frame = transform(frame)

            return self._convert_detector_output(
                landmarks, self.emotion_model.detect_emo(frame, landmarks)
            )

        elif self.info["emotion_model"].lower() == "resmasknet":
            return self._convert_detector_output(
                facebox, self.emotion_model.detect_emo(frame, facebox)
            )

        elif self.info["emotion_model"].lower() in ["svm", "rf"]:
            return self._convert_detector_output(
                landmarks, self.emotion_model.detect_emo(frame, landmarks)
            )

        else:
            raise ValueError(
                "Cannot recognize input emo model! Please try to re-type emotion model"
            )

    def detect_image(
        self,
        input_file_list,
        output_size=256,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        frame_counter=0,
    ):
        """Detects FEX from an image file.

        Args:
            input_file_list (list of str): Path to a list of paths to image files.
            output_size (int): image size to rescale all imagee preserving aspect ratio
            batch_size (int): how many batches of images you want to run at one shot. Larger gives faster speed but is more memory-consuming
            num_workers (int): how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
            pin_memory (bool): If ``True``, the data loader will copy Tensors
                                into CUDA pinned memory before returning them.  If your data elements
                                are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type
            frame_counter (int): starting value to count frames

        Returns:
            Fex: Prediction results dataframe
        """

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

        batch_output = []
        for batch_id, batch_data in enumerate(tqdm(data_loader)):
            frame_counter += frame_counter + batch_id * batch_size
            faces = self.detect_faces(batch_data["Image"])
            landmarks = self.detect_landmarks(batch_data["Image"], detected_faces=faces)
            poses = self.detect_facepose(batch_data["Image"])
            aus = self.detect_aus(batch_data["Image"], landmarks)
            emotions = self.detect_emotions(batch_data["Image"], faces, landmarks)
            faces = _inverse_face_transform(faces, batch_data)
            landmarks = _inverse_landmark_transform(landmarks, batch_data)
            output = self._create_fex(
                faces,
                landmarks,
                poses,
                aus,
                emotions,
                batch_data["FileNames"],
                frame_counter,
            )
            batch_output.append(output)

        batch_output = pd.concat(batch_output)
        batch_output.reset_index(drop=True, inplace=True)

        return batch_output

    def detect_video(
        self,
        video_path,
        skip_frames=0,
        output_size=700,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        """Detects FEX from a video file.

        Args:
            video_path (str): Path to a video file.
            skip_frames (int): number of frames to skip (speeds up inference, but less temporal information)
            output_size (int): image size to rescale all imagee preserving aspect ratio
            batch_size (int): how many batches of images you want to run at one shot. Larger gives faster speed but is more memory-consuming
            num_workers (int): how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
            pin_memory (bool): If ``True``, the data loader will copy Tensors
                                into CUDA pinned memory before returning them.  If your data elements
                                are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type

        Returns:
            Fex: Prediction results dataframe
        """

        data_loader = DataLoader(
            VideoDataset(video_path, skip_frames=skip_frames, output_size=output_size),
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=False,
        )

        batch_output = []
        for batch_data in tqdm(data_loader):
            faces = self.detect_faces(batch_data["Image"])
            landmarks = self.detect_landmarks(batch_data["Image"], detected_faces=faces)
            poses = self.detect_facepose(batch_data["Image"])
            aus = self.detect_aus(batch_data["Image"], landmarks)
            emotions = self.detect_emotions(batch_data["Image"], faces, landmarks)
            frames = list(batch_data["Frame"].numpy())
            output = self._create_fex(
                faces, landmarks, poses, aus, emotions, batch_data["FileName"], frames
            )
            batch_output.append(output)

        batch_output = pd.concat(batch_output)
        batch_output.reset_index(drop=True, inplace=True)
        return batch_output

    def _convert_detector_output(detected_faces, detector_results):
        """Helper function to convert AU/Emotion detector output into frame by face list of lists.

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

    def _create_fex(
        self, faces, landmarks, poses, aus, emotions, file_names, frame_counter
    ):
        """Helper function to create a Fex instance using detector output

        Args:
            faces: output of detect_faces()
            landmarks: output of detect_landmarks()
            poses: output of dectect_facepose()
            aus: output of detect_aus()
            emotions: output of detect_emotions()
            file_names: file name of input image
            frame_counter: starting value for frame counter, useful for integrating batches

        Returns:
            Fex object
        """
        files = [[f] * n for f, n in zip(file_names, [len(x) for x in faces])]

        # Convert to Pandas Format
        out = []
        for i, frame in enumerate(faces):
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
                    columns=self["face_detection_columns"],
                    index=[j],
                )

                facepose_df = pd.DataFrame(
                    [poses[i][j].flatten(order="F")],
                    columns=self["facepose_model_columns"],
                    index=[j],
                )

                landmarks_df = pd.DataFrame(
                    [landmarks[i][j].flatten(order="F")],
                    columns=self["face_landmark_columns"],
                    index=[j],
                )

                aus_df = pd.DataFrame(
                    aus[i][j, :].reshape(1, len(self["au_presence_columns"])),
                    columns=self["au_presence_columns"],
                    index=[j],
                )

                emotions_df = pd.DataFrame(
                    emotions[i][j, :].reshape(1, len(FEAT_EMOTION_COLUMNS)),
                    columns=FEAT_EMOTION_COLUMNS,
                    index=[j],
                )

                input_df = pd.DataFrame(
                    files[i][j],
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

        return Fex(
            out,
            au_columns=self["au_presence_columns"],
            emotion_columns=FEAT_EMOTION_COLUMNS,
            facebox_columns=FEAT_FACEBOX_COLUMNS,
            landmark_columns=openface_2d_landmark_columns,
            facepose_columns=FACET_FACEPOSE_COLUMNS,
            time_columns=FACET_TIME_COLUMNS,
            detector="Feat",
            face_model=self.info["face_model"],
            landmark_model=self.info["landmark_model"],
            au_model=self.info["au_model"],
            emotion_model=self.info["emotion_model"],
            facepose_model=self.info["facepose_model"],
        )

    def _convert_detector_output(self, detected_faces, detector_results):
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
