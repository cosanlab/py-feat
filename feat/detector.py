from __future__ import division

"""Functions to help detect face, landmarks, emotions, action units from images and videos"""
import traceback  # REMOVE LATER

from collections import deque
from multiprocessing.pool import ThreadPool
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import math
from scipy.spatial import ConvexHull
from skimage.morphology.convex_hull import grid_points_in_poly
from skimage.feature import hog
import cv2
import feat
from feat.data import Fex
from feat.utils import (
    get_resource_path,
    face_rect_to_coords,
    openface_2d_landmark_columns,
    jaanet_AU_presence,
    RF_AU_presence,
    FEAT_EMOTION_MAPPER,
    FEAT_EMOTION_COLUMNS,
    FEAT_FACEBOX_COLUMNS,
    FACET_FACEPOSE_COLUMNS,
    FEAT_TIME_COLUMNS,
    FACET_TIME_COLUMNS,
    BBox,
    convert68to49,
    padding,
    resize_with_padding,
    align_face_68pts,
    FaceDetectionError,
)
from feat.au_detectors.JAANet.JAA_test import JAANet
from feat.au_detectors.DRML.DRML_test import DRMLNet
from feat.au_detectors.StatLearning.SL_test import (
    RandomForestClassifier,
    SVMClassifier,
    LogisticClassifier,
)
from feat.emo_detectors.ferNet.ferNet_test import ferNetModule
from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet
from feat.emo_detectors.StatLearning.EmoSL_test import (
    EmoRandomForestClassifier,
    EmoSVMClassifier,
)
import torch
from feat.face_detectors.FaceBoxes.FaceBoxes_test import FaceBoxes
from feat.face_detectors.MTCNN.MTCNN_test import MTCNN
from feat.face_detectors.Retinaface import Retinaface_test
from feat.landmark_detectors.basenet_test import MobileNet_GDConv
from feat.landmark_detectors.pfld_compressed_test import PFLDInference
from feat.landmark_detectors.mobilefacenet_test import MobileFaceNet
from feat.facepose_detectors.img2pose.img2pose_test import Img2Pose
from feat.facepose_detectors.pnp.pnp_test import PerspectiveNPoint
import json
from torchvision.datasets.utils import download_url
import zipfile


class Detector(object):
    def __init__(
        self,
        face_model="retinaface",
        landmark_model="mobilenet",
        au_model="rf",
        emotion_model="resmasknet",
        facepose_model="pnp",
        n_jobs=1,
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
        self.info = {}
        self.info["n_jobs"] = n_jobs

        if torch.cuda.is_available():
            self.map_location = lambda storage, loc: storage.cuda()
        else:
            self.map_location = "cpu"

        # Handle img2pose mismatch error
        if (
            facepose_model
            and "img2pose" in facepose_model.lower()
            and facepose_model.lower() != face_model.lower()
        ):
            print(
                facepose_model,
                " is both a face detector and pose estimator, and cannot be used with a different "
                "face detector. Setting face detector to use ",
                facepose_model,
                ".",
                sep="",
            )
            face_model = facepose_model

        """ LOAD UP THE MODELS """
        print("Loading Face Detection model: ", face_model)
        # Check if model files have been downloaded. Otherwise download model.
        # get model url.
        with open(os.path.join(get_resource_path(), "model_list.json"), "r") as f:
            model_urls = json.load(f)

        if face_model:
            for url in model_urls["face_detectors"][face_model.lower()]["urls"]:
                download_url(url, get_resource_path())
        if landmark_model:
            for url in model_urls["landmark_detectors"][landmark_model.lower()]["urls"]:
                download_url(url, get_resource_path())
        if au_model:
            for url in model_urls["au_detectors"][au_model.lower()]["urls"]:
                download_url(url, get_resource_path())
                if ".zip" in url:
                    import zipfile

                    with zipfile.ZipFile(
                        os.path.join(get_resource_path(), "JAANetparams.zip"), "r"
                    ) as zip_ref:
                        zip_ref.extractall(os.path.join(get_resource_path()))
                if au_model.lower() in ["logistic", "svm", "rf"]:
                    download_url(
                        model_urls["au_detectors"]["hog-pca"]["urls"][0],
                        get_resource_path(),
                    )
                    download_url(
                        model_urls["au_detectors"]["au_scalar"]["urls"][0],
                        get_resource_path(),
                    )

        if emotion_model:
            for url in model_urls["emotion_detectors"][emotion_model.lower()]["urls"]:
                download_url(url, get_resource_path())
                if emotion_model.lower() in ["svm", "rf"]:
                    download_url(
                        model_urls["emotion_detectors"]["emo_pca"]["urls"][0],
                        get_resource_path(),
                    )
                    download_url(
                        model_urls["emotion_detectors"]["emo_scalar"]["urls"][0],
                        get_resource_path(),
                    )

        if face_model:
            if face_model.lower() == "faceboxes":
                self.face_detector = FaceBoxes()
            elif face_model.lower() == "retinaface":
                self.face_detector = Retinaface_test.Retinaface()
            elif face_model.lower() == "mtcnn":
                self.face_detector = MTCNN()
            elif "img2pose" in face_model.lower():
                # Check if user selected unconstrained or constrained version
                constrained = False  # use by default
                if face_model.lower() == "img2pose-c":
                    constrained = True
                # Used as both face detector and facepose estimator
                self.face_detector = Img2Pose(
                    cpu_mode=self.map_location == "cpu", constrained=constrained
                )
                self.facepose_detector = self.face_detector

        self.info["face_model"] = face_model
        facebox_columns = FEAT_FACEBOX_COLUMNS
        self.info["face_detection_columns"] = facebox_columns
        predictions = np.empty((1, len(facebox_columns)))
        predictions[:] = np.nan
        empty_facebox = pd.DataFrame(predictions, columns=facebox_columns)
        self._empty_facebox = empty_facebox

        print("Loading Face Landmark model: ", landmark_model)
        if landmark_model:
            if landmark_model.lower() == "mobilenet":
                self.landmark_detector = MobileNet_GDConv(136)
                self.landmark_detector = torch.nn.DataParallel(self.landmark_detector)
                checkpoint = torch.load(
                    os.path.join(
                        get_resource_path(),
                        "mobilenet_224_model_best_gdconv_external.pth.tar",
                    ),
                    map_location=self.map_location,
                )
                self.landmark_detector.load_state_dict(checkpoint["state_dict"])

            elif landmark_model.lower() == "pfld":
                self.landmark_detector = PFLDInference()
                checkpoint = torch.load(
                    os.path.join(get_resource_path(), "pfld_model_best.pth.tar"),
                    map_location=self.map_location,
                )
                self.landmark_detector.load_state_dict(checkpoint["state_dict"])

            elif landmark_model.lower() == "mobilefacenet":
                self.landmark_detector = MobileFaceNet([112, 112], 136)
                checkpoint = torch.load(
                    os.path.join(
                        get_resource_path(), "mobilefacenet_model_best.pth.tar"
                    ),
                    map_location=self.map_location,
                )
                self.landmark_detector.load_state_dict(checkpoint["state_dict"])

        self.info["landmark_model"] = landmark_model
        self.info["mapper"] = openface_2d_landmark_columns
        landmark_columns = openface_2d_landmark_columns
        self.info["face_landmark_columns"] = landmark_columns
        predictions = np.empty((1, len(openface_2d_landmark_columns)))
        predictions[:] = np.nan
        empty_landmarks = pd.DataFrame(predictions, columns=landmark_columns)
        self._empty_landmark = empty_landmarks

        print("Loading au model: ", au_model)
        self.info["au_model"] = au_model
        if au_model:
            if au_model.lower() == "jaanet":
                self.au_model = JAANet()
            elif au_model.lower() == "drml":
                self.au_model = DRMLNet()
            elif au_model.lower() == "logistic":
                self.au_model = LogisticClassifier()
            elif au_model.lower() == "svm":
                self.au_model = SVMClassifier()
            elif au_model.lower() == "rf":
                self.au_model = RandomForestClassifier()

        if (au_model is None) or (au_model.lower() in ["jaanet", "drml"]):
            auoccur_columns = jaanet_AU_presence
        else:
            auoccur_columns = RF_AU_presence

        self.info["au_presence_columns"] = auoccur_columns
        predictions = np.empty((1, len(auoccur_columns)))
        predictions[:] = np.nan
        empty_au_occurs = pd.DataFrame(predictions, columns=auoccur_columns)
        self._empty_auoccurence = empty_au_occurs

        print("Loading emotion model: ", emotion_model)
        self.info["emotion_model"] = emotion_model
        if emotion_model:
            if emotion_model.lower() == "fer":
                self.emotion_model = ferNetModule()
            elif emotion_model.lower() == "resmasknet":
                self.emotion_model = ResMaskNet()
            elif emotion_model.lower() == "svm":
                self.emotion_model = EmoSVMClassifier()
            elif emotion_model.lower() == "rf":
                self.emotion_model = EmoRandomForestClassifier()

        self.info["emotion_model_columns"] = FEAT_EMOTION_COLUMNS
        predictions = np.empty((1, len(FEAT_EMOTION_COLUMNS)))
        predictions[:] = np.nan
        empty_emotion = pd.DataFrame(predictions, columns=FEAT_EMOTION_COLUMNS)
        self._empty_emotion = empty_emotion

        predictions = np.empty((1, len(auoccur_columns)))
        predictions[:] = np.nan
        empty_au_occurs = pd.DataFrame(predictions, columns=auoccur_columns)
        self._empty_auoccurence = empty_au_occurs

        print("Loading facepose model: ", facepose_model)
        self.info["facepose_model"] = facepose_model
        if facepose_model:
            if facepose_model.lower() == "pnp":
                self.facepose_detector = PerspectiveNPoint()
            # Note that img2pose case is handled under face_model loading

        self.info["facepose_model_columns"] = FACET_FACEPOSE_COLUMNS
        predictions = np.empty((1, len(FACET_FACEPOSE_COLUMNS)))
        predictions[:] = np.nan
        empty_facepose = pd.DataFrame(predictions, columns=FACET_FACEPOSE_COLUMNS)
        self._empty_facepose = empty_facepose

        self.info["output_columns"] = (
            FEAT_TIME_COLUMNS
            + facebox_columns
            + landmark_columns
            + auoccur_columns
            + FACET_FACEPOSE_COLUMNS
            + FEAT_EMOTION_COLUMNS
            + ["input"]
        )

    def __getitem__(self, i):
        return self.info[i]

    def detect_faces(self, frame):
        """Detect faces from image or video frame

        Args:
            frame (array): image array

        Returns:
            list: face detection results (x, y, x2, y2)

        Examples:
            >>> import cv2
            >>> frame = cv2.imread(imgfile)
            >>> from feat import Detector
            >>> detector = Detector()
            >>> detector.detect_faces(frame)
        """
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
            print("Warning: NO FACE is detected")
        return faces

    def detect_landmarks(self, frame, detected_faces):
        """Detect landmarks from image or video frame

        Args:
            frame (array): image array
            detected_faces (array):

        Returns:
            list: x and y landmark coordinates (1,68,2)

        Examples:
            >>> import cv2
            >>> frame = cv2.imread(imgfile)
            >>> from feat import Detector
            >>> detector = Detector()
            >>> detected_faces = detector.detect_faces(frame)
            >>> detector.detect_landmarks(frame, detected_faces)
        """
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
            multichannel=True,
        )
        if visualize:
            return (hog_output[0], hog_output[1])
        else:
            return hog_output

    def detect_aus(self, frame, landmarks):
        """Detect Action Units from image or video frame

        Args:
            frame (array): image loaded in array format (n, m, 3)
            landmarks (array): 68 landmarks used to localize face.

        Returns:
            array: Action Unit predictions

        Examples:
            >>> import cv2
            >>> frame = cv2.imread(imgfile)
            >>> from feat import Detector
            >>> detector = Detector()
            >>> detector.detect_aus(frame)
        """
        # Assume that the Raw landmark is given in the format (n_land,2)

        # landmarks = np.transpose(landmarks)
        # if landmarks.shape[-1] == 68:
        #    landmarks = convert68to49(landmarks)
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
            >>> import cv2
            >>> frame = cv2.imread(imgfile)
            >>> from feat import Detector
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
            # With img2pose
            >>> import cv2
            >>> frame = [cv2.imread(imgfile)]
            >>> from feat import Detector
            >>> detector = Detector(face_model='imgpose', facepose_model='img2pose')
            >>> detector.detect_facepose([frame]) # one shot computation

            # With PnP
            >>> import cv2
            >>> frame = [cv2.imread(imgfile)]
            >>> from feat import Detector
            >>> detector = Detector(face_model='retinaface', landmark_model='mobilefacenet', facepose_model='pnp')
            >>> faces = detector.detect_faces(frame)
            >>> landmarks = detector.detect_landmarks(detected_faces=faces)
            >>> detector.detect_facepose(frame=frame, landmarks=landmarks) # detect pose for all faces
        """
        # check if frame is 4d
        if frame.ndim == 3:
            frame = np.expand_dims(frame, 0)
        assert frame.ndim == 4, "Frame needs to be 4 dimensions (list of images)"

        # height, width, _ = frame.shape
        if "img2pose" in self.info["face_model"]:
            faces, poses = self.facepose_detector(frame)
        else:
            poses = self.facepose_detector(frame, landmarks)

        return poses

    # TODO: probably need to add exceptions. The exception handling is not great yet
    def process_frame(
        self, frames, counter=0, singleframe4error=False, skip_frame_rate=1
    ):
        """Function to run face detection, landmark detection, and emotion detection on
        a frame.

        Args:
            frames (np.array): batch of frames, of shape BxHxWxC (read from cv2)
            counter (int, str, default=0): Index used for the prediction results
            dataframe. Tracks the batches
            singleframe4error (bool, default = False): When exception occurs inside a
            batch, instead of nullify the whole batch, process each img in batch
            individually

        Returns:
            feat.data.Fex (dataframe): Prediction results dataframe.
            int: counter - the updated number of counter. Used to track the batch size and image number

        """
        # check if frame is 4d
        if frames.ndim == 3:
            frames = np.expand_dims(frames, 0)
        assert frames.ndim == 4, "Frame needs to be 4 dimensions (list of images)"
        out = None
        # TODO Changed here
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
                    if out is None:
                        out = tmp_df
                    else:
                        out = pd.concat([out, tmp_df], axis=0)
                    # out[FEAT_TIME_COLUMNS] = counter

                counter += skip_frame_rate
            return out, counter

        except:
            traceback.print_exc()
            print("exception occurred in the batch")
            if singleframe4error:
                print("Trying to process one image at a time in the batch")
                raise FaceDetectionError

            else:
                print(
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
        skip_frames=1,
        verbose=False,
        singleframe4error=False,
    ):
        """Detects FEX from a video file.

        Args:
            inputFname (str): Path to video file
            outputFname (str, optional): Path to output file. Defaults to None.
            bacth_size (int, optional): how many batches of images you want to run at one shot. Larger gives faster speed but is more memory-consuming
            skip_frames (int, optional): Number of every other frames to skip for speed or if not all frames need to be processed. Defaults to 1.
            singleframe4error (bool, default = False): When set True, when exception
            occurs inside a batch, instead of nullify the whole batch, process each img
            in batch individually

        Returns:
            dataframe: Prediction results dataframe if outputFname is None. Returns True if outputFname is specified.
        """
        self.info["inputFname"] = inputFname
        self.info["outputFname"] = outputFname
        init_df = pd.DataFrame(columns=self["output_columns"])
        if outputFname:
            init_df.to_csv(outputFname, index=False, header=True)

        cap = cv2.VideoCapture(inputFname)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = int(np.ceil(length / skip_frames))

        counter = 0
        frame_got = True
        if verbose:
            print("Processing video.")
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
                                counter=tmp_counter,
                                singleframe4error=singleframe4error,
                                skip_frame_rate=skip_frames,
                            )
                        except FaceDetectionError:
                            df = None
                            for id_fr in range(concat_frame.shape[0]):
                                tmp_df, _ = self.process_frame(
                                    concat_frame[id_fr : (id_fr + 1)],
                                    counter=tmp_counter,
                                    singleframe4error=False,
                                    skip_frame_rate=skip_frames,
                                )
                                tmp_counter += 1
                                if df is None:
                                    df = tmp_df
                                else:
                                    df = pd.concat((df, tmp_df), 0)
                    else:
                        df, _ = self.process_frame(
                            concat_frame,
                            counter=tmp_counter,
                            skip_frame_rate=skip_frames,
                        )

                    df["input"] = inputFname
                    if outputFname:
                        df[init_df.columns].to_csv(
                            outputFname, index=False, header=False, mode="a"
                        )
                    else:
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
                                counter=tmp_counter,
                                skip_frame_rate=skip_frames,
                            )
                        except FaceDetectionError:
                            df = None
                            for id_fr in range(concat_frame.shape[0]):
                                tmp_df, _ = self.process_frame(
                                    concat_frame[id_fr : (id_fr + 1)],
                                    counter=tmp_counter,
                                    singleframe4error=False,
                                    skip_frame_rate=skip_frames,
                                )
                                tmp_counter += 1
                                if df is None:
                                    df = tmp_df
                                else:
                                    df = pd.concat((df, tmp_df), 0)
                    else:
                        df, _ = self.process_frame(
                            concat_frame,
                            counter=tmp_counter,
                            skip_frame_rate=skip_frames,
                        )
                    df["input"] = inputFname
                    if outputFname:
                        df[init_df.columns].to_csv(
                            outputFname, index=False, header=False, mode="a"
                        )
                    else:
                        init_df = pd.concat([init_df, df[init_df.columns]], axis=0)
                break
        cap.release()
        if outputFname:
            return True
        else:
            return Fex(
                init_df,
                filename=inputFname,
                au_columns=self["au_presence_columns"],
                emotion_columns=FEAT_EMOTION_COLUMNS,
                facebox_columns=FEAT_FACEBOX_COLUMNS,
                landmark_columns=openface_2d_landmark_columns,
                facepose_columns=FACET_FACEPOSE_COLUMNS,
                time_columns=FEAT_TIME_COLUMNS,
                detector="Feat",
            )

    def detect_image(
        self,
        inputFname,
        batch_size=5,
        outputFname=None,
        verbose=False,
        singleframe4error=False,
    ):
        """Detects FEX from an image file.

        Args:
            inputFname (list of str): Path to a list of paths to image files.
            bacth_size (int, optional): how many batches of images you want to run at one shot. Larger gives faster speed but is more memory-consuming
            outputFname (str, optional): Path to output file. Defaults to None.
            singleframe4error (bool, default = False): When set True, when exception
            occurs inside a batch, instead of nullify the whole batch, process each img
            in batch individually

        Returns:
            Fex: Prediction results dataframe if outputFname is None. Returns True if outputFname is specified.
        """
        assert (
            type(inputFname) == str or type(inputFname) == list
        ), "inputFname must be a string path to image or list of image paths"
        if type(inputFname) == str:
            inputFname = [inputFname]
        for inputF in inputFname:
            if not os.path.exists(inputF):
                raise FileNotFoundError(f"File {inputF} not found.")
        self.info["inputFname"] = inputFname

        init_df = pd.DataFrame(columns=self["output_columns"])
        if outputFname:
            init_df.to_csv(outputFname, index=False, header=True)

        counter = 0
        concat_frame = None
        input_names = []
        while counter < len(inputFname):
            # if counter % skip_frames == 0:
            frame = np.expand_dims(cv2.imread(inputFname[counter]), 0)
            if concat_frame is None:
                concat_frame = frame
                tmp_counter = counter
            else:
                concat_frame = np.concatenate([concat_frame, frame], 0)
            input_names.append(inputFname[counter])
            counter = counter + 1

            if (counter % batch_size == 0) and (concat_frame is not None):
                if singleframe4error:
                    try:
                        df, _ = self.process_frame(
                            concat_frame,
                            counter=tmp_counter,
                            singleframe4error=singleframe4error,
                        )
                    except FaceDetectionError:
                        df = None
                        for id_fr in range(concat_frame.shape[0]):
                            tmp_df, _ = self.process_frame(
                                concat_frame[id_fr : (id_fr + 1)],
                                counter=tmp_counter,
                                singleframe4error=False,
                            )
                            tmp_counter += 1
                            if df is None:
                                df = tmp_df
                            else:
                                df = pd.concat((df, tmp_df), 0)
                else:
                    df, _ = self.process_frame(concat_frame, counter=tmp_counter)

                df["input"] = input_names
                if outputFname:
                    df[init_df.columns].to_csv(
                        outputFname, index=False, header=False, mode="a"
                    )
                else:
                    init_df = pd.concat([init_df, df[init_df.columns]], axis=0)

                concat_frame = None
                tmp_counter = None
                input_names = []

        if len(inputFname) % batch_size != 0:
            # process remaining frames
            if concat_frame is not None:
                if singleframe4error:
                    try:
                        df, _ = self.process_frame(concat_frame, counter=tmp_counter)
                    except FaceDetectionError:
                        df = None
                        for id_fr in range(concat_frame.shape[0]):
                            tmp_df, _ = self.process_frame(
                                concat_frame[id_fr : (id_fr + 1)],
                                counter=tmp_counter,
                                singleframe4error=False,
                            )
                            tmp_counter += 1
                            if df is None:
                                df = tmp_df
                            else:
                                df = pd.concat((df, tmp_df), 0)
                else:
                    df, _ = self.process_frame(concat_frame, counter=tmp_counter)

                # Handle pandas assignment issue where we have a single file name, but
                # our dataframe contains multiple faces (i.e. multiple rows). So we need
                # to broadcast the *contents* of input_names since it's length doesn't
                # match the number of rows
                if df.shape[0] > 1 and len(input_names) == 1:
                    input_names = input_names[0]
                df["input"] = input_names
                if outputFname:
                    df[init_df.columns].to_csv(
                        outputFname, index=False, header=False, mode="a"
                    )
                else:
                    init_df = pd.concat([init_df, df[init_df.columns]], axis=0)

        if outputFname:
            return True
        else:
            return Fex(
                init_df,
                filename=inputFname,
                au_columns=self["au_presence_columns"],
                emotion_columns=FEAT_EMOTION_COLUMNS,
                facebox_columns=FEAT_FACEBOX_COLUMNS,
                landmark_columns=openface_2d_landmark_columns,
                facepose_columns=FACET_FACEPOSE_COLUMNS,
                time_columns=FACET_TIME_COLUMNS,
                detector="Feat",
            )


if __name__ == "__main__":
    my_file_path = "/home/tiankang/AU_Dataset/test_case/"
    all_imgs = [my_file_path + "f" + str(i) + ".jpg" for i in range(1, 12)] + [
        "/home/tiankang/AU_Dataset/test_case/" + "f12.png"
    ]
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="svm",
        emotion_model="resmasknet",
    )  # initialize methods. These are the methods I like to use
    # Test the video
    outs2 = detector.detect_video(
        "/home/tiankang/AU_Dataset/test_case/songA.mp4",
        batch_size=512,
        skip_frames=1,
        singleframe4error=True,
    )
    # Test the pictures
    outs = detector.detect_image(
        all_imgs, batch_size=5, outputFname=None, verbose=False, singleframe4error=True
    )
    print("finished")
