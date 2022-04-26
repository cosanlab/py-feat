"""
Main Detector class. The Detector class wraps other pre-trained models
(e.g. face detector, au detector) and provides a high-level API to make it easier to
perform detection
"""

import traceback  # REMOVE LATER
import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.morphology.convex_hull import grid_points_in_poly
from skimage.feature import hog
import cv2
from feat.data import Fex
from feat.utils import (
    get_resource_path,
    openface_2d_landmark_columns,
    jaanet_AU_presence,
    RF_AU_presence,
    FEAT_EMOTION_COLUMNS,
    FEAT_FACEBOX_COLUMNS,
    FACET_FACEPOSE_COLUMNS,
    FEAT_TIME_COLUMNS,
    FACET_TIME_COLUMNS,
    BBox,
    align_face_68pts,
    FaceDetectionError,
    validate_input,
    read_pictures,
)
from feat.pretrained import get_pretrained_models, fetch_model
import torch
import logging
import warnings

# Supress sklearn warning about pickled estimators and diff sklearn versions
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class Detector(object):
    def __init__(
        self,
        face_model="retinaface",
        landmark_model="mobilenet",
        au_model="rf",
        emotion_model="resmasknet",
        facepose_model="pnp",
        n_jobs=1,
        verbose=False,
    ):
        """Detector class to detect FEX from images or videos.

        Detector is a class used to detect faces, facial landmarks, emotions, and action units from images and videos.

        Args:
            n_jobs (int, default=1): Number of processes to use for extraction.

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
        # Setup verbosity and CUDA
        self.logger = logging.getLogger("Detector")
        self.verbose = verbose
        if self.verbose:
            log_level = logging.INFO
        else:
            log_level = logging.WARNING
        self.logger.setLevel(log_level)

        if torch.cuda.is_available():
            self.map_location = lambda storage, loc: storage.cuda()
        else:
            self.map_location = "cpu"

        # Verify model names and download if necessary
        face, landmark, au, emotion, facepose = get_pretrained_models(
            face_model, landmark_model, au_model, emotion_model, facepose_model, verbose
        )
        if (au is None) or (au in ["jaanet", "drml"]):
            auoccur_columns = jaanet_AU_presence
        else:
            auoccur_columns = RF_AU_presence

        self._init_detectors(
            face,
            landmark,
            au,
            emotion,
            facepose,
            auoccur_columns,
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
        auoccur_columns,
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
                # img2pose is used both as face detector and pose estimator
                if "img2pose" in face:
                    self.face_detector = self.face_detector(
                        cpu_mode=self.map_location == "cpu",
                        constrained="img2pose-c" == face,
                    )
                    self.facepose_detector = self.face_detector
                else:
                    self.face_detector = self.face_detector()

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
                        map_location=self.map_location,
                    )
                    self.landmark_detector.load_state_dict(checkpoint["state_dict"])
                elif landmark == "pfld":
                    self.landmark_detector = self.landmark_detector()
                    checkpoint = torch.load(
                        os.path.join(get_resource_path(), "pfld_model_best.pth.tar"),
                        map_location=self.map_location,
                    )
                    self.landmark_detector.load_state_dict(checkpoint["state_dict"])
                elif landmark == "mobilefacenet":
                    self.landmark_detector = self.landmark_detector([112, 112], 136)
                    checkpoint = torch.load(
                        os.path.join(
                            get_resource_path(), "mobilefacenet_model_best.pth.tar"
                        ),
                        map_location=self.map_location,
                    )
                    self.landmark_detector.load_state_dict(checkpoint["state_dict"])

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

        # AU MODEL
        if self.info["au_model"] != au:
            self.logger.info(f"Loading AU model: {au}")
            self.au_model = fetch_model("au_model", au)
            self.info["au_model"] = au
            if self.au_model is not None:
                self.au_model = self.au_model()
                self.info["au_presence_columns"] = auoccur_columns
                predictions = np.full_like(np.atleast_2d(auoccur_columns), np.nan)
                empty_au_occurs = pd.DataFrame(predictions, columns=auoccur_columns)
                self._empty_auoccurence = empty_au_occurs

        # EMOTION MODEL
        if self.info["emotion_model"] != emotion:
            self.logger.info("Loading emotion model: ", emotion)
            self.emotion_model = fetch_model("emotion_model", emotion)
            self.info["emotion_model"] = emotion
            if self.emotion_model is not None:
                self.emotion_model = self.emotion_model()
                self.info["emotion_model_columns"] = FEAT_EMOTION_COLUMNS
                predictions = np.full_like(np.atleast_2d(FEAT_EMOTION_COLUMNS), np.nan)
                empty_emotion = pd.DataFrame(predictions, columns=FEAT_EMOTION_COLUMNS)
                self._empty_emotion = empty_emotion

        # FACEPOSE MODEL
        if self.info["facepose_model"] != facepose:
            self.logger.info("Loading facepose model: ", facepose)
            # Only assign it if it's not img2pose, otherwise it's already set by face model
            _m = fetch_model("facepose_model", facepose)
            if _m is not None:
                self.facepose_detector = _m()
            self.info["facepose_model"] = facepose

            self.info["facepose_model_columns"] = FACET_FACEPOSE_COLUMNS
            predictions = np.full_like(np.atleast_2d(FACET_FACEPOSE_COLUMNS), np.nan)
            empty_facepose = pd.DataFrame(predictions, columns=FACET_FACEPOSE_COLUMNS)
            self._empty_facepose = empty_facepose

        self.info["output_columns"] = (
            FEAT_TIME_COLUMNS
            + FEAT_FACEBOX_COLUMNS
            + openface_2d_landmark_columns
            + auoccur_columns
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
        if (au is None) or (au in ["jaanet", "drml"]):
            auoccur_columns = jaanet_AU_presence
        else:
            auoccur_columns = RF_AU_presence

        self._init_detectors(
            face,
            landmark,
            au,
            emotion,
            facepose,
            auoccur_columns,
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
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"Frame should be a numpy array, not {type(frame)}. If you are passing in an image path try calling .read_image to first load the image data as a numpy array. Then pass the result to this method"
            )
        # check if frame is 4d
        if frame.ndim == 3:
            frame = np.expand_dims(frame, 0)
        assert frame.ndim == 4, "Frame needs to be 4 dimensions (list of images)"
        # height, width, _ = frame.shape
        if "img2pose" in self.info["face_model"]:
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

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"Frame should be a numpy array, not {type(frame)}. If you are passing in an image path try calling .read_image to first load the image data as a numpy array. Then pass the result to this method"
            )
        # check if frame is 4d
        if frame.ndim == 3:
            frame = np.expand_dims(frame, 0)
        assert frame.ndim == 4, "Frame needs to be 4 dimensions (list of images)"

        mean = np.asarray([0.485, 0.456, 0.406])
        std = np.asarray([0.229, 0.224, 0.225])
        self.landmark_detector.eval()
        if self.info["landmark_model"]:
            if self.info["landmark_model"].lower() == "mobilenet":
                out_size = 224
            else:
                out_size = 112

        _, height, width, _ = frame.shape
        landmark_list = []

        concate_arr, len_frames_faces, bbox_list = self._face_preprocesing(
            frame=frame,
            detected_faces=detected_faces,
            mean=mean,
            std=std,
            out_size=out_size,
            height=height,
            width=width,
        )
        # Run through the deep leanring model
        input = torch.from_numpy(concate_arr).float()
        input = torch.autograd.Variable(input)
        if self.info["landmark_model"]:
            if self.info["landmark_model"].lower() == "mobilefacenet":
                landmark = self.landmark_detector(input)[0].cpu().data.numpy()
            else:
                landmark = self.landmark_detector(input).cpu().data.numpy()

        landmark_results = []

        landmark = landmark.reshape(landmark.shape[0], -1, 2)

        for ik in range(landmark.shape[0]):
            landmark2 = bbox_list[ik].reprojectLandmark(landmark[ik, :, :])
            landmark_results.append(landmark2)

        list_concat = []
        new_lens = np.insert(np.cumsum(len_frames_faces), 0, 0)
        for ij in range(len(len_frames_faces)):
            list_concat.append(landmark_results[new_lens[ij] : new_lens[ij + 1]])

        return list_concat

    def _batch_hog(self, frames, detected_faces, landmarks):
        """
        NEW
        Helper function used in batch processing hog features
        frames is a batch of frames
        """

        len_index = [len(aa) for aa in landmarks]
        lenth_cumu = np.cumsum(len_index)
        lenth_cumu2 = np.insert(lenth_cumu, 0, 0)
        new_lands_list = []
        flat_faces = [item for sublist in detected_faces for item in sublist]
        flat_land = [item for sublist in landmarks for item in sublist]
        hogs_arr = None

        for i in range(len(flat_land)):

            frame_assignment = np.where(i < lenth_cumu)[0][0]

            convex_hull, new_lands = self.extract_face(
                frame=frames[frame_assignment],
                detected_faces=[flat_faces[i][0:4]],
                landmarks=flat_land[i],
                size_output=112,
            )
            hogs = self.extract_hog(frame=convex_hull, visualize=False).reshape(1, -1)
            if hogs_arr is None:
                hogs_arr = hogs
            else:
                hogs_arr = np.concatenate([hogs_arr, hogs], 0)

            new_lands_list.append(new_lands)

        new_lands = []
        for i in range(len(lenth_cumu)):
            new_lands.append(new_lands_list[lenth_cumu2[i] : (lenth_cumu2[i + 1])])

        return (hogs_arr, new_lands)

    def _face_preprocesing(
        self, frame, detected_faces, mean, std, out_size, height, width
    ):
        """
        NEW
        Helper function used in batch detecting landmarks
        Let's assume that frame is of shape B x H x W x 3
        """
        lenth_index = [len(ama) for ama in detected_faces]
        lenth_cumu = np.cumsum(lenth_index)

        flat_faces = [
            item for sublist in detected_faces for item in sublist
        ]  # Flatten the faces

        concatenated_face = None
        bbox_list = []
        for k, face in enumerate(flat_faces):
            frame_assignment = np.where(k <= lenth_cumu)[0][0]  # which frame is it?
            x1 = face[0]
            y1 = face[1]
            x2 = face[2]
            y2 = face[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(min([w, h]) * 1.2)
            cx = x1 + w // 2
            cy = y1 + h // 2
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped = frame[
                frame_assignment,
                new_bbox.top : new_bbox.bottom,
                new_bbox.left : new_bbox.right,
            ]
            bbox_list.append(new_bbox)

            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                cropped = cv2.copyMakeBorder(
                    cropped,
                    int(dy),
                    int(edy),
                    int(dx),
                    int(edx),
                    cv2.BORDER_CONSTANT,
                    0,
                )
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
                continue
            test_face = cropped_face.copy()
            test_face = test_face / 255.0
            if self.info["landmark_model"]:
                if self.info["landmark_model"].lower() == "mobilenet":
                    test_face = (test_face - mean) / std
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)

            if concatenated_face is None:
                concatenated_face = test_face
            else:
                concatenated_face = np.concatenate([concatenated_face, test_face], 0)

        return (concatenated_face, lenth_index, bbox_list)

    def extract_face(self, frame, detected_faces, landmarks, size_output=112):
        """Extract a face in a frame with a convex hull of landmarks.

        This function extracts the faces of the frame with convex hulls and masks out the rest.

        Args:
            frame (array): The original image]
            detected_faces (list): face bounding box
            landmarks (list): the landmark information]
            size_output (int, optional): [description]. Defaults to 112.

        Returns:
            resized_face_np: resized face as a numpy array
            new_landmarks: landmarks of aligned face
        """
        detected_faces = np.array(detected_faces)
        landmarks = np.array(landmarks)

        detected_faces = detected_faces.astype(int)

        aligned_img, new_landmarks = align_face_68pts(
            frame, landmarks.flatten(), 2.5, img_size=size_output
        )

        hull = ConvexHull(new_landmarks)
        mask = grid_points_in_poly(
            shape=np.array(aligned_img).shape,
            # for some reason verts need to be flipped
            verts=list(
                zip(
                    new_landmarks[hull.vertices][:, 1],
                    new_landmarks[hull.vertices][:, 0],
                )
            ),
        )
        mask[
            0 : np.min([new_landmarks[0][1], new_landmarks[16][1]]),
            new_landmarks[0][0] : new_landmarks[16][0],
        ] = True
        aligned_img[~mask] = 0
        resized_face_np = aligned_img
        resized_face_np = cv2.cvtColor(resized_face_np, cv2.COLOR_BGR2RGB)

        return (
            resized_face_np,
            new_landmarks,
        )  # , hull, mask, np.array(aligned_img).shape, list(zip(new_landmarks[hull.vertices][:, 1], new_landmarks[hull.vertices][:, 0])), origin_mask

    def extract_hog(
        self,
        frame,
        orientation=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    ):
        """Extract HOG features from a SINGLE frame.

        Args:
            frame (array]): Frame of image]
            orientation (int, optional): Orientation for HOG. Defaults to 8.
            pixels_per_cell (tuple, optional): Pixels per cell for HOG. Defaults to (8,8).
            cells_per_block (tuple, optional): Cells per block for HOG. Defaults to (2,2).
            visualize (bool, optional): Whether to provide the HOG image. Defaults to False.

        Returns:
            hog_output: array of HOG features, and the HOG image if visualize is True.
        """

        hog_output = hog(
            frame,
            orientations=orientation,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=visualize,
            channel_axis=-1,
        )
        if visualize:
            return (hog_output[0], hog_output[1])
        else:
            return hog_output

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
        # Assume that the Raw landmark is given in the format (n_land,2)

        # landmarks = np.transpose(landmarks)
        # if landmarks.shape[-1] == 68:
        #    landmarks = convert68to49(landmarks)
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"Frame should be a numpy array, not {type(frame)}. If you are passing in an image path try calling .read_image to first load the image data as a numpy array. Then pass the result to this method"
            )
        return self.au_model.detect_au(frame, landmarks)

    def _concatenate_batch(self, indexed_length, au_results):
        """
        NEW
        helper function to convert batch AUs to desired list of list
        only useful for our emotion and au prediction results
        Args:
            indexed_length: (list) the list index for number of faces in each frame.
                            if you have 2 faces in each frame and you batch process 4
                            frames, it will be [2,2,2,2]
            au_results: (np.array), immediate result from running our
                        au/emotion models
        Returns:
            list_concat: (list of list). The list which contains the number of faces. for example
            if you process 2 frames and each frame contains 4 faces, it will return:
                [[xxx,xxx,xxx,xxx],[xxx,xxx,xxx,xxx]]
        """
        list_concat = []
        new_lens = np.insert(np.cumsum(indexed_length), 0, 0)
        for ij in range(len(indexed_length)):
            list_concat.append(au_results[new_lens[ij] : new_lens[ij + 1], :])
        return list_concat

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
        if self.info["emotion_model"].lower() == "fer":
            # landmarks = np.transpose(landmarks)
            # if landmarks.shape[-1] == 68:
            #    landmarks = convert68to49(landmarks)
            #    landmarks = landmarks.T
            return self.emotion_model.detect_emo(frame, landmarks)

        elif self.info["emotion_model"].lower() == "resmasknet":
            return self.emotion_model.detect_emo(frame, facebox)

        elif self.info["emotion_model"].lower() in ["svm", "rf"]:
            return self.emotion_model.detect_emo(frame, landmarks)

        else:
            raise ValueError(
                "Cannot recognize input emo model! Please try to re-type emotion model"
            )

    def detect_facepose(self, frame, detected_faces=None, landmarks=None):
        """Detect facepose from image or video frame.

        When used with img2pose, returns *all* detected poses, and facebox and landmarks
        are ignored. Use `detect_face` method in order to obtain bounding boxes
        corresponding to the detected poses returned by this method.

        When used with pnp model, 'facebox' param is ignored, and the passed 2D
        landmarks are used to compute the head pose for the single face associated with
        the passed landmarks.

        Args:
            frame (np.ndarray): list of cv2 images
            detected_faces (list): (num_images, num_faces, 4) faceboxes representing faces in the list of images
            landmarks (np.ndarray): (num_images, num_faces, 68, 2) landmarks for the faces contained in list of images

        Returns:
            np.ndarray: (num_images, num_faces, [pitch, roll, yaw]) - Euler angles (in
            degrees) for each face within in each image


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
        # check if frame is 4d
        if frame.ndim == 3:
            frame = np.expand_dims(frame, 0)
        assert frame.ndim == 4, "Frame needs to be 4 dimensions (list of images)"

        # height, width, _ = frame.shape
        if "img2pose" in self.info["facepose_model"]:
            _, poses = self.facepose_detector(frame)
        else:
            poses = self.facepose_detector(frame, landmarks)

        return poses

    # TODO: probably need to add exceptions. The exception handling is not great yet
    def process_frame(
        self,
        frames,
        input_names,
        counter=0,
        singleframe4error=False,
        skip_frame_rate=1,
        is_video_frame=False,
    ):
        """Function to run face detection, landmark detection, and emotion detection on
        a frame.

        Args:
            frames (np.array): batch of frames, of shape BxHxWxC (read from cv2)
            input_names (list): file names for each frame in the batch
            counter (int, str, default=0): Index used for the prediction results
            dataframe. Tracks the batches
            singleframe4error (bool, default = False): When exception occurs inside a
            batch, instead of nullify the whole batch, process each img in batch
            individually
            is_video_frame (bool): Whether processing is happening over batches of
            otherwise independent IMAGES so len(frames) == len(input_names) or over
            VIDEOFRAMES so len(input_names) == 1 and len(frames) > len(input_names);
            Default False

        Returns:
            feat.data.Fex (dataframe): Prediction results dataframe.
            int: counter - the updated number of counter. Used to track the batch size and image number

        """
        # check if frame is 4d
        if not isinstance(frames, np.ndarray):
            raise TypeError("frames needs to be a 3 or 4d numpy array of image data")
        if frames.ndim == 3:
            frames = np.expand_dims(frames, 0)
        assert frames.ndim == 4, "Frame needs to be 4 dimensions (list of images)"
        if not is_video_frame:
            assert frames.shape[0] == len(
                input_names
            ), "Number of input_names needs to match the number of frames to process"
        out = None
        try:
            detected_faces = self.detect_faces(frame=frames)
            landmarks = self.detect_landmarks(
                frame=frames, detected_faces=detected_faces
            )
            poses = self.detect_facepose(
                frame=frames, detected_faces=detected_faces, landmarks=landmarks
            )
            index_len = [len(ii) for ii in landmarks]

            if self["au_model"].lower() in ["logistic", "svm", "rf"]:
                # landmarks_2 = round_vals(landmarks,3)
                landmarks_2 = landmarks
                hog_arr, new_lands = self._batch_hog(
                    frames=frames, detected_faces=detected_faces, landmarks=landmarks_2
                )
                au_occur = self.detect_aus(frame=hog_arr, landmarks=new_lands)
            else:
                au_occur = self.detect_aus(frame=frames, landmarks=landmarks)

            if self["emotion_model"].lower() in ["svm", "rf"]:
                hog_arr, new_lands = self._batch_hog(
                    frames=frames, detected_faces=detected_faces, landmarks=landmarks
                )
                emo_pred = self.detect_emotions(
                    frame=hog_arr, facebox=None, landmarks=new_lands
                )
            else:
                emo_pred = self.detect_emotions(
                    frame=frames, facebox=detected_faces, landmarks=landmarks
                )

            my_aus = self._concatenate_batch(
                indexed_length=index_len, au_results=au_occur
            )
            my_emo = self._concatenate_batch(
                indexed_length=index_len, au_results=emo_pred
            )

            for i, sessions in enumerate(detected_faces):
                for j, faces in enumerate(sessions):
                    facebox_df = pd.DataFrame(
                        [
                            [
                                faces[0],
                                faces[1],
                                faces[2] - faces[0],
                                faces[3] - faces[1],
                                faces[4],
                            ]
                        ],
                        columns=self["face_detection_columns"],
                        index=[counter + j],
                    )

                    facepose_df = pd.DataFrame(
                        [poses[i][j].flatten(order="F")],
                        columns=self["facepose_model_columns"],
                        index=[counter + j],
                    )

                    landmarks_df = pd.DataFrame(
                        [landmarks[i][j].flatten(order="F")],
                        columns=self["face_landmark_columns"],
                        index=[counter + j],
                    )

                    au_occur_df = pd.DataFrame(
                        my_aus[i][j, :].reshape(1, len(self["au_presence_columns"])),
                        columns=self["au_presence_columns"],
                        index=[counter + j],
                    )

                    emo_pred_df = pd.DataFrame(
                        my_emo[i][j, :].reshape(1, len(FEAT_EMOTION_COLUMNS)),
                        columns=FEAT_EMOTION_COLUMNS,
                        index=[counter + j],
                    )

                    tmp_df = pd.concat(
                        [
                            facebox_df,
                            landmarks_df,
                            au_occur_df,
                            facepose_df,
                            emo_pred_df,
                        ],
                        axis=1,
                    )
                    tmp_df[FEAT_TIME_COLUMNS] = counter
                    tmp_df["input"] = (
                        input_names[0] if is_video_frame else input_names[i]
                    )
                    if out is None:
                        out = tmp_df
                    else:
                        out = pd.concat([out, tmp_df], axis=0)
                    # out[FEAT_TIME_COLUMNS] = counter

                counter += skip_frame_rate
            return out, counter

        except:
            traceback.print_exc()
            self.logger.error("exception occurred in the batch")
            if singleframe4error:
                self.logger.warning(
                    "Trying to process one image at a time in the batch"
                )
                raise FaceDetectionError

            else:
                self.logger.warning(
                    "Since singleframe4error=FALSE, giving up this entire batch result"
                )
                newdf = None
                for cter in range(frames.shape[0]):
                    emotion_df = self._empty_emotion.reindex(index=[counter + cter])
                    facebox_df = self._empty_facebox.reindex(index=[counter + cter])
                    facepose_df = self._empty_facepose.reindex(index=[counter + cter])
                    landmarks_df = self._empty_landmark.reindex(index=[counter + cter])
                    au_occur_df = self._empty_auoccurence.reindex(
                        index=[counter + cter]
                    )

                    out = pd.concat(
                        [
                            facebox_df,
                            landmarks_df,
                            au_occur_df,
                            facepose_df,
                            emotion_df,
                        ],
                        axis=1,
                    )
                    out["input"] = input_names[0] if is_video_frame else input_names[i]
                    out[FEAT_TIME_COLUMNS] = counter + cter
                    if newdf is None:
                        newdf = out
                    else:
                        newdf = pd.concat([newdf, out], axis=0)

                return (newdf, counter + frames.shape[0])

    def detect_video(
        self,
        inputFname,
        batch_size=5,
        outputFname=None,
        return_detection=True,
        skip_frames=1,
        singleframe4error=False,
    ):
        """Detects FEX from a video file.

        Args:
            inputFname (str): Path to video file
            batch_size (int, optional): how many batches of images you want to run at one shot. Larger gives faster speed but is more memory-consuming
            outputFname (str, optional): Path to output file. Defaults to None.
            return_detection (bool, optional): whether to return a Fex object of all
            concatenated detections. To save memory you can process results directly to
            a file by setting this to False and providing an outputFname; Default True
            skip_frames (int, optional): Number of every other frames to skip for speed or if not all frames need to be processed. Defaults to 1.
            singleframe4error (bool, default = False): When set True, when exception
            occurs inside a batch, instead of nullify the whole batch, process each img
            in batch individually

        Returns:
            dataframe: Prediction results dataframe if outputFname is None. Returns True if outputFname is specified.
        """
        self.info["inputFname"] = validate_input(inputFname)
        if not (outputFname or return_detection):
            raise ValueError(
                "If return_detection is False then you must provide an outputFname"
            )

        self.info["outputFname"] = outputFname
        init_df = pd.DataFrame(columns=self["output_columns"])
        if outputFname is not None:
            init_df.to_csv(outputFname, index=False, header=True)

        cap = cv2.VideoCapture(self.info["inputFname"][0])
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = int(np.ceil(length / skip_frames))

        counter = 0
        frame_got = True
        self.logger.info("Processing video.")
        #  single core
        concat_frame = None
        while True:
            frame_got, frame = cap.read()
            if frame_got:
                if counter % skip_frames == 0:
                    # if the
                    if concat_frame is None:
                        concat_frame = np.expand_dims(frame, 0)
                        tmp_counter = counter
                    else:
                        concat_frame = np.concatenate(
                            [concat_frame, np.expand_dims(frame, 0)], 0
                        )
                if (
                    (concat_frame is not None)
                    and (counter != 0)
                    and (concat_frame.shape[0] % batch_size == 0)
                ):  # I think it's probably this error
                    if singleframe4error:
                        try:
                            df, _ = self.process_frame(
                                concat_frame,
                                self.info["inputFname"],
                                counter=tmp_counter,
                                singleframe4error=singleframe4error,
                                skip_frame_rate=skip_frames,
                                is_video_frame=True,
                            )
                        except FaceDetectionError:
                            df = None
                            for id_fr in range(concat_frame.shape[0]):
                                tmp_df, _ = self.process_frame(
                                    concat_frame[id_fr : (id_fr + 1)],
                                    self.info["inputFname"][id_fr : (id_fr + 1)],
                                    counter=tmp_counter,
                                    singleframe4error=False,
                                    skip_frame_rate=skip_frames,
                                    is_video_frame=True,
                                )
                                tmp_counter += 1
                                if df is None:
                                    df = tmp_df
                                else:
                                    df = pd.concat((df, tmp_df), 0)
                    else:
                        df, _ = self.process_frame(
                            concat_frame,
                            self.info["inputFname"],
                            counter=tmp_counter,
                            skip_frame_rate=skip_frames,
                            is_video_frame=True,
                        )

                    if outputFname:
                        df[init_df.columns].to_csv(
                            outputFname, index=False, header=False, mode="a"
                        )
                    if return_detection:
                        init_df = pd.concat([init_df, df[init_df.columns]], axis=0)
                    concat_frame = None
                    tmp_counter = None
                counter = counter + 1
            else:
                # process remaining frames
                if concat_frame is not None:
                    if singleframe4error:
                        try:
                            df, _ = self.process_frame(
                                concat_frame,
                                self.info["inputFname"],
                                counter=tmp_counter,
                                skip_frame_rate=skip_frames,
                                is_video_frame=True,
                            )
                        except FaceDetectionError:
                            df = None
                            for id_fr in range(concat_frame.shape[0]):
                                tmp_df, _ = self.process_frame(
                                    concat_frame[id_fr : (id_fr + 1)],
                                    self.info["inputFname"][id_fr : (id_fr + 1)],
                                    counter=tmp_counter,
                                    singleframe4error=False,
                                    skip_frame_rate=skip_frames,
                                    is_video_frame=True,
                                )
                                tmp_counter += 1
                                if df is None:
                                    df = tmp_df
                                else:
                                    df = pd.concat((df, tmp_df), 0)
                    else:
                        df, _ = self.process_frame(
                            concat_frame,
                            self.info["inputFname"],
                            counter=tmp_counter,
                            skip_frame_rate=skip_frames,
                            is_video_frame=True,
                        )
                    df["input"] = inputFname
                    if outputFname is not None:
                        df[init_df.columns].to_csv(
                            outputFname, index=False, header=False, mode="a"
                        )
                    if return_detection:
                        init_df = pd.concat([init_df, df[init_df.columns]], axis=0)
                break
        cap.release()
        if return_detection:
            return Fex(
                init_df,
                filename=self.info["inputFname"],
                au_columns=self["au_presence_columns"],
                emotion_columns=FEAT_EMOTION_COLUMNS,
                facebox_columns=FEAT_FACEBOX_COLUMNS,
                landmark_columns=openface_2d_landmark_columns,
                facepose_columns=FACET_FACEPOSE_COLUMNS,
                time_columns=FEAT_TIME_COLUMNS,
                detector="Feat",
            )
        # Not returning any detection to save memory
        return True

    def detect_image(
        self,
        inputFname,
        batch_size=5,
        outputFname=None,
        return_detection=True,
        singleframe4error=False,
    ):
        """Detects FEX from an image file.

        Args:
            inputFname (list of str): Path to a list of paths to image files.
            batch_size (int, optional): how many batches of images you want to run at one shot. Larger gives faster speed but is more memory-consuming
            outputFname (str, optional): Path to output file. Defaults to None.
            return_detection (bool, optional): whether to return a Fex object of all
            concatenated detections. To save memory you can process results directly to
            a file by setting this to False and providing an outputFname; Default True
            singleframe4error (bool, default = False): When set True, when exception
            occurs inside a batch, instead of nullify the whole batch, process each img
            in batch individually

        Returns:
            Fex: Prediction results dataframe if outputFname is None. Returns True if outputFname is specified.
        """

        self.info["inputFname"] = validate_input(inputFname)
        if not (outputFname or return_detection):
            raise ValueError(
                "If return_detection is False then you must provide an outputFname"
            )

        init_df = pd.DataFrame(columns=self["output_columns"])
        if outputFname is not None:
            init_df.to_csv(outputFname, index=False, header=True)

        counter = 0
        concat_frame = None
        input_names = []
        while counter < len(self.info["inputFname"]):
            frame = np.expand_dims(cv2.imread(self.info["inputFname"][counter]), 0)
            if concat_frame is None:
                concat_frame = frame
                tmp_counter = counter
            else:
                # NOTE: We need to handle batch counting and divisibility properly which
                # is why we don't use utils.read_pictures or self.read_pictures here
                try:
                    concat_frame = np.concatenate([concat_frame, frame], 0)
                except ValueError as e:
                    raise ValueError(
                        f"Image batching size mis-match error. You set batch_size to{batch_size}, but all of your images don't have the same dimensions. You need to either resize you images prior to using py-feat or you can set batch_size = 1 to prevent any image batching during processing. See these additional error details from numpy: {str(e)}"
                    )
            input_names.append(self.info["inputFname"][counter])
            counter = counter + 1

            if (counter % batch_size == 0) and (concat_frame is not None):
                if singleframe4error:
                    try:
                        df, _ = self.process_frame(
                            concat_frame,
                            input_names,
                            counter=tmp_counter,
                            singleframe4error=singleframe4error,
                        )
                    except FaceDetectionError:
                        df = None
                        for id_fr in range(concat_frame.shape[0]):
                            tmp_df, _ = self.process_frame(
                                concat_frame[id_fr : (id_fr + 1)],
                                input_names[id_fr : (id_fr + 1)],
                                counter=tmp_counter,
                                singleframe4error=False,
                            )
                            tmp_counter += 1
                            if df is None:
                                df = tmp_df
                            else:
                                df = pd.concat((df, tmp_df), 0)
                else:
                    df, _ = self.process_frame(
                        concat_frame, input_names, counter=tmp_counter
                    )

                if outputFname is not None:
                    df[init_df.columns].to_csv(
                        outputFname, index=False, header=False, mode="a"
                    )
                if return_detection:
                    init_df = pd.concat([init_df, df[init_df.columns]], axis=0)

                concat_frame = None
                tmp_counter = None
                input_names = []

        if len(self.info["inputFname"]) % batch_size != 0:
            # process remaining frames
            if concat_frame is not None:
                if singleframe4error:
                    try:
                        df, _ = self.process_frame(
                            concat_frame, input_names, counter=tmp_counter
                        )
                    except FaceDetectionError:
                        df = None
                        for id_fr in range(concat_frame.shape[0]):
                            tmp_df, _ = self.process_frame(
                                concat_frame[id_fr : (id_fr + 1)],
                                input_names[id_fr : (id_fr + 1)],
                                counter=tmp_counter,
                                singleframe4error=False,
                            )
                            tmp_counter += 1
                            if df is None:
                                df = tmp_df
                            else:
                                df = pd.concat((df, tmp_df), 0)
                else:
                    df, _ = self.process_frame(
                        concat_frame, input_names, counter=tmp_counter
                    )

                if outputFname is not None:
                    df[init_df.columns].to_csv(
                        outputFname, index=False, header=False, mode="a"
                    )
                if return_detection:
                    init_df = pd.concat([init_df, df[init_df.columns]], axis=0)

        if return_detection:
            return Fex(
                init_df,
                filename=self.info["inputFname"],
                au_columns=self["au_presence_columns"],
                emotion_columns=FEAT_EMOTION_COLUMNS,
                facebox_columns=FEAT_FACEBOX_COLUMNS,
                landmark_columns=openface_2d_landmark_columns,
                facepose_columns=FACET_FACEPOSE_COLUMNS,
                time_columns=FACET_TIME_COLUMNS,
                detector="Feat",
            )
        # Not returning any detection to save memory
        return True

    def read_images(self, imgname_list):
        """
        Given a string filepath or list of filepath, load the image(s) as a numpy array

        Args:
            imgname_list (list/str): filename or list of filenames

        Returns:
            np.ndarry: 3d or 4d numpy array
        """
        if not isinstance(imgname_list, list):
            imgname_list = [imgname_list]
        return read_pictures(imgname_list)
