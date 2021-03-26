from __future__ import division

"""Functions to help detect face, landmarks, emotions, action units from images and videos"""

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
    FEAT_TIME_COLUMNS,
    FACET_TIME_COLUMNS,
    BBox,
    convert68to49,
    padding,
    resize_with_padding,
    align_face_68pts
)
from feat.au_detectors.JAANet.JAA_test import JAANet
from feat.au_detectors.DRML.DRML_test import DRMLNet
from feat.au_detectors.StatLearning.SL_test import RandomForestClassifier, SVMClassifier, LogisticClassifier
from feat.emo_detectors.ferNet.ferNet_test import ferNetModule
from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet
from feat.emo_detectors.StatLearning.EmoSL_test import EmoRandomForestClassifier, EmoSVMClassifier
import torch
from feat.face_detectors.FaceBoxes.FaceBoxes_test import FaceBoxes
from feat.face_detectors.MTCNN.MTCNN_test import MTCNN
from feat.face_detectors.Retinaface import Retinaface_test
from feat.landmark_detectors.basenet_test import MobileNet_GDConv
from feat.landmark_detectors.pfld_compressed_test import PFLDInference
from feat.landmark_detectors.mobilefacenet_test import MobileFaceNet
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
                face_detection_columns (list): Column names for face detection ouput (x, y, w, h)
                face_landmark_columns (list): Column names for face landmark output (x0, y0, x1, y1, ...)
                emotion_model_columns (list): Column names for emotion model output
                mapper (dict): Class names for emotion model output by index.
                input_shape (dict)

            face_detector: face detector object
            face_landmark: face_landmark object
            emotion_model: emotion_model object

        Examples:
            >> detector = Detector(n_jobs=1)
            >> detector.detect_image("input.jpg")
            >> detector.detect_video("input.mp4")
        """
        self.info = {}
        self.info["n_jobs"] = n_jobs

        if torch.cuda.is_available():
            self.map_location = lambda storage, loc: storage.cuda()
        else:
            self.map_location = "cpu"

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
                    with zipfile.ZipFile(os.path.join(get_resource_path(), "JAANetparams.zip"), 'r') as zip_ref:
                        zip_ref.extractall(os.path.join(get_resource_path()))
                if au_model.lower() in ['logistic', 'svm', 'rf']:
                    download_url(
                        model_urls["au_detectors"]['hog-pca']['urls'][0], get_resource_path())
                    download_url(
                        model_urls["au_detectors"]['au_scalar']['urls'][0], get_resource_path())

        if emotion_model:
            for url in model_urls["emotion_detectors"][emotion_model.lower()]["urls"]:
                download_url(url, get_resource_path())
                if emotion_model.lower() in ['svm', 'rf']:
                    download_url(
                        model_urls["emotion_detectors"]['emo_pca']['urls'][0], get_resource_path())
                    download_url(
                        model_urls["emotion_detectors"]['emo_scalar']['urls'][0], get_resource_path())

        if face_model:
            if face_model.lower() == "faceboxes":
                self.face_detector = FaceBoxes()
            elif face_model.lower() == "retinaface":
                self.face_detector = Retinaface_test.Retinaface()
            elif face_model.lower() == "mtcnn":
                self.face_detector = MTCNN()

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
                self.landmark_detector = torch.nn.DataParallel(
                    self.landmark_detector)
                checkpoint = torch.load(
                    os.path.join(
                        get_resource_path(),
                        "mobilenet_224_model_best_gdconv_external.pth.tar",
                    ),
                    map_location=self.map_location,
                )
                self.landmark_detector.load_state_dict(
                    checkpoint["state_dict"])

            elif landmark_model.lower() == "pfld":
                self.landmark_detector = PFLDInference()
                checkpoint = torch.load(
                    os.path.join(get_resource_path(),
                                 "pfld_model_best.pth.tar"),
                    map_location=self.map_location,
                )
                self.landmark_detector.load_state_dict(
                    checkpoint["state_dict"])

            elif landmark_model.lower() == "mobilefacenet":
                self.landmark_detector = MobileFaceNet([112, 112], 136)
                checkpoint = torch.load(
                    os.path.join(
                        get_resource_path(), "mobilefacenet_model_best.pth.tar"
                    ),
                    map_location=self.map_location,
                )
                self.landmark_detector.load_state_dict(
                    checkpoint["state_dict"])

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
            elif au_model.lower() == 'rf':
                self.au_model = RandomForestClassifier()

        if (au_model is None) or (au_model.lower() in ['jaanet', 'drml']):
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
            elif emotion_model.lower() == 'svm':
                self.emotion_model = EmoSVMClassifier()
            elif emotion_model.lower() == 'rf':
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

        self.info["output_columns"] = (
            FEAT_TIME_COLUMNS
            + facebox_columns
            + landmark_columns
            + auoccur_columns
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
            array: face detection results (x, y, x2, y2)

        Examples: 
            >>> import cv2
            >>> frame = cv2.imread(imgfile)
            >>> from feat import Detector
            >>> detector = Detector()        
            >>> detector.detect_faces(frame)
        """
        height, width, _ = frame.shape
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
        mean = np.asarray([0.485, 0.456, 0.406])
        std = np.asarray([0.229, 0.224, 0.225])
        self.landmark_detector.eval()
        if self.info["landmark_model"]:
            if self.info["landmark_model"].lower() == "mobilenet":
                out_size = 224
            else:
                out_size = 112

        height, width, _ = frame.shape
        landmark_list = []

        for k, face in enumerate(detected_faces):
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
                new_bbox.top: new_bbox.bottom, new_bbox.left: new_bbox.right
            ]
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
            input = torch.from_numpy(test_face).float()
            input = torch.autograd.Variable(input)
            if self.info["landmark_model"]:
                if self.info["landmark_model"].lower() == "mobilefacenet":
                    landmark = self.landmark_detector(
                        input)[0].cpu().data.numpy()
                else:
                    landmark = self.landmark_detector(input).cpu().data.numpy()
            landmark = landmark.reshape(-1, 2)
            landmark = new_bbox.reprojectLandmark(landmark)
            landmark_list.append(landmark)

        return landmark_list

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
        # if (np.any(detected_faces) < 0):
        #     orig_size = np.array(frame).shape
        #     if np.where(detected_faces<0)[0][0]==1:
        #         # extend y
        #         new_size = (orig_size[0], int(orig_size[1] + 2*abs(detected_faces[detected_faces<0][0])))
        #     else:
        #         # extend x
        #         new_size = (int(orig_size[0] + 2*abs(detected_faces[detected_faces<0][0])), orig_size[1])

        #     frame = resize_with_padding(Image.fromarray(frame), new_size)
        #     frame = np.asarray(frame)
        #     detected_faces = np.array(detector.detect_faces(np.array(frame))[0])

        detected_faces = detected_faces.astype(int)

        aligned_img, new_landmarks = align_face_68pts(
            frame, landmarks.flatten(), 2.5, img_size=size_output)

        hull = ConvexHull(new_landmarks)
        mask = grid_points_in_poly(shape=np.array(aligned_img).shape,
                                   # for some reason verts need to be flipped
                                   verts=list(
                                       zip(new_landmarks[hull.vertices][:, 1], new_landmarks[hull.vertices][:, 0]))
                                   )

        mask[0:np.min([new_landmarks[0][1], new_landmarks[16][1]]),
             new_landmarks[0][0]:new_landmarks[16][0]] = True
        aligned_img[~mask] = 0
        resized_face_np = aligned_img
        resized_face_np = cv2.cvtColor(resized_face_np, cv2.COLOR_BGR2RGB)

        return resized_face_np, new_landmarks

    def extract_hog(self, frame, orientation=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False):
        """Extract HOG features from a frame.

        Args:
            frame (array]): Frame of image]
            orientation (int, optional): Orientation for HOG. Defaults to 8.
            pixels_per_cell (tuple, optional): Pixels per cell for HOG. Defaults to (8,8).
            cells_per_block (tuple, optional): Cells per block for HOG. Defaults to (2,2).
            visualize (bool, optional): Whether to provide the HOG image. Defaults to False.

        Returns:
            hog_output: array of HOG features, and the HOG image if visualize is True.
        """

        hog_output = hog(frame, orientations=orientation, pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block, visualize=visualize, multichannel=True)
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

        #landmarks = np.transpose(landmarks)
        #if landmarks.shape[-1] == 68:
        #    landmarks = convert68to49(landmarks)
        return self.au_model.detect_au(frame, landmarks)

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
        if self.info["emotion_model"].lower() == 'fer':
            landmarks = np.transpose(landmarks)
            if landmarks.shape[-1] == 68:
                landmarks = convert68to49(landmarks)
                landmarks = landmarks.T
            return self.emotion_model.detect_emo(frame, landmarks)

        elif self.info["emotion_model"].lower() == 'resmasknet':
            return self.emotion_model.detect_emo(frame, facebox)

        elif self.info["emotion_model"].lower() in ['svm', 'rf']:
            return self.emotion_model.detect_emo(frame, landmarks)

        else:
            raise ValueError(
                'Cannot recognize input emo model! Please try to re-type emotion model')

    def process_frame(self, frame, counter=0):
        """Helper function to run face detection, landmark detection, and emotion detection on a frame.

        Args:
            frame (np.array): Numpy array of image, ideally loaded through Pillow.Image
            counter (int, str, default=0): Index used for the prediction results dataframe.

        Returns:
            df (dataframe): Prediction results dataframe.

        Example:
            >>> from pil import Image
            >>> frame = Image.open("input.jpg")
            >>> detector = Detector()
            >>> detector.process_frame(np.array(frame))
        """
        try:
            # detect faces
            detected_faces = self.detect_faces(frame=frame)
            out = None
            for i, faces in enumerate(detected_faces):
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
                    index=[counter + i],
                )
                # detect landmarks
                landmarks = self.detect_landmarks(
                    frame=frame, detected_faces=[faces[0:4]]
                )
                landmarks_df = pd.DataFrame(
                    [landmarks[0].flatten(order="F")],
                    columns=self["face_landmark_columns"],
                    index=[counter + i],
                )
                # detect AUs
                if self["au_model"].lower() in ['logistic', 'svm', 'rf']:
                    convex_hull, new_lands = self.extract_face(
                        frame=frame, detected_faces=[faces[0:4]], landmarks=landmarks, size_output=112)
                    hogs = self.extract_hog(frame=convex_hull, visualize=False)
                    au_occur = self.detect_aus(frame=hogs, landmarks=new_lands)
                else:
                    au_occur = self.detect_aus(
                        frame=frame, landmarks=landmarks)

                au_occur_df = pd.DataFrame(
                    au_occur, columns=self["au_presence_columns"], index=[
                        counter + i]
                )
                # detect emotions
                if self["emotion_model"].lower() in ['svm', 'rf']:
                    emo_pred = self.detect_emotions(
                        frame=hogs, facebox=None, landmarks=new_lands)
                else:
                    emo_pred = self.detect_emotions(
                        frame=frame, facebox=[faces], landmarks=landmarks[0])

                emo_pred_df = pd.DataFrame(
                    emo_pred, columns=FEAT_EMOTION_COLUMNS, index=[counter + i]
                )
                tmp_df = pd.concat(
                    [facebox_df, landmarks_df, au_occur_df, emo_pred_df], axis=1
                )
                if out is None:
                    out = tmp_df
                else:
                    out = pd.concat([out, tmp_df], axis=0)
            out[FEAT_TIME_COLUMNS] = counter
            return out
        except:
            print("exception occurred")
            emotion_df = self._empty_emotion.reindex(index=[counter])
            facebox_df = self._empty_facebox.reindex(index=[counter])
            landmarks_df = self._empty_landmark.reindex(index=[counter])
            au_occur_df = self._empty_auoccurence.reindex(index=[counter])

            out = pd.concat([facebox_df, landmarks_df,
                             au_occur_df, emotion_df], axis=1)
            out[FEAT_TIME_COLUMNS] = counter
            return out

    def detect_video(self, inputFname, outputFname=None, skip_frames=1, verbose=False):
        """Detects FEX from a video file.

        Args:
            inputFname (str): Path to video file
            outputFname (str, optional): Path to output file. Defaults to None.
            skip_frames (int, optional): Number of every other frames to skip for speed or if not all frames need to be processed. Defaults to 1.
            
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

        # Determine whether to use multiprocessing.
        n_jobs = self["n_jobs"]
        if n_jobs == -1:
            thread_num = cv2.getNumberOfCPUs()  # get available cpus
        else:
            thread_num = n_jobs
        if verbose:
            print(f"Using {thread_num} cpus")
        pool = ThreadPool(processes=thread_num)
        pending_task = deque()
        counter = 0
        processed_frames = 0
        frame_got = True
        detected_faces = []
        if verbose:
            print("Processing video.")
        #  single core
        while True:
            frame_got, frame = cap.read()
            if counter % skip_frames == 0:
                df = self.process_frame(frame, counter=counter)
                df["input"] = inputFname
                if outputFname:
                    df[init_df.columns].to_csv(
                        outputFname, index=False, header=False, mode="a"
                    )
                else:
                    init_df = pd.concat([init_df, df[init_df.columns]], axis=0)
            counter = counter + 1
            if not frame_got:
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
                time_columns=FACET_TIME_COLUMNS,
                detector="Feat",
            )

    def detect_image(self, inputFname, outputFname=None, verbose=False):
        """Detects FEX from an image file.

        Args:
            inputFname (str, or list of str): Path to image file or a list of paths to image files.
            outputFname (str, optional): Path to output file. Defaults to None.

        Rseturns:
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

        for inputF in inputFname:
            if verbose:
                print(f"processing {inputF}")
            frame = cv2.imread(inputF)
            df = self.process_frame(frame)
            df["input"] = inputF
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
                au_columns=self['au_presence_columns'],
                emotion_columns=FEAT_EMOTION_COLUMNS,
                facebox_columns=FEAT_FACEBOX_COLUMNS,
                landmark_columns=openface_2d_landmark_columns,
                time_columns=FACET_TIME_COLUMNS,
                detector="Feat",
            )
