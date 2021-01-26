# %%
from __future__ import division

"""Functions to help detect face, landmarks, emotions, action units from images and videos"""

from collections import deque
from multiprocessing.pool import ThreadPool
import tensorflow as tf
from tensorflow.python.keras import optimizers, models
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2

from feat.data import Fex
from feat.utils import get_resource_path, face_rect_to_coords, openface_2d_landmark_columns, FEAT_EMOTION_MAPPER, FEAT_EMOTION_COLUMNS, FEAT_FACEBOX_COLUMNS, FACET_TIME_COLUMNS, BBox, convert68to49
from feat.models.JAA_test import JAANet
# from data import Fex
# from utils import get_resource_path, face_rect_to_coords, openface_2d_landmark_columns, FEAT_EMOTION_MAPPER, FEAT_EMOTION_COLUMNS, FEAT_FACEBOX_COLUMNS, FACET_TIME_COLUMNS, BBox, convert68to49
# from models.JAA_test import JAANet
# from models.mtcnn_Net import MTCNN

import torch
from feat.face_detectors.FaceBoxes import FaceBoxes
from feat.face_detectors.Retinaface import Retinaface
from feat.face_detectors.MTCNN import MTCNN

from feat.landmark_detectors.basenet import MobileNet_GDConv
from feat.landmark_detectors.pfld_compressed import PFLDInference
from feat.landmark_detectors.mobilefacenet import MobileFaceNet


class Detector(object):
    def __init__(self, face_model='FaceBoxes', landmark_model='MobileNet', au_occur_model='jaanet', emotion_model='fer', n_jobs=1):
        """Detector class to detect FEX from images or videos.

        Detector is a class used to detect faces, facial landmarks, emotions, and action units from images and videos.

        Args:
            n_jobs (int, default=1): Number of processes to use for extraction. 

        Attributes:
            info (dict):
                n_jobs (int): Number of jobs to be used in parallel.
                face_detection_model (str, default=haarcascade_frontalface_alt.xml): Path to face detection model.
                face_detection_columns (list): Column names for face detection ouput (x, y, w, h)
                face_landmark_model (str, default=lbfmodel.yaml): Path to landmark model.
                face_landmark_columns (list): Column names for face landmark output (x0, y0, x1, y1, ...)
                emotion_model (str, default=fer_aug_model.h5): Path to emotion detection model.
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
        self.info['n_jobs'] = n_jobs

        if torch.cuda.is_available():
            self.map_location = lambda storage, loc: storage.cuda()
        else:
            self.map_location = 'cpu'

        """ LOAD UP THE MODELS """
        print("Loading Face Detection model: ", face_model)
        self.info['Face_Model'] = face_model
        if face_model == "FaceBoxes":
            self.face_detector = FaceBoxes()
        elif face_model == "RetinaFace":
            self.face_detector = Retinaface.Retinaface()
        elif face_model == 'MTCNN':
            self.face_detector = MTCNN()

        # if face_model == 'haarcascade':
        #     self.detect_face = True
        #     face_detection_model_path = cv.data.haarcascades + \
        #         "haarcascade_frontalface_alt.xml"
        #     if not os.path.exists(face_detection_model_path):
        #         print("Face detection model not found. Check haarcascade_frontalface_alt.xml exists in your opencv installation (cv.data).")
        #     face_cascade = cv.CascadeClassifier(face_detection_model_path)
        #     face_detection_columns = FEAT_FACEBOX_COLUMNS
        #     facebox_empty = np.empty((1, 4))
        #     facebox_empty[:] = np.nan
        #     empty_facebox = pd.DataFrame(
        #         facebox_empty, columns=face_detection_columns)
        #     self.info["face_detection_model"] = face_detection_model_path
        #     self.info["face_detection_columns"] = face_detection_columns
        #     self.face_detector = face_cascade
        #     self._empty_facebox = empty_facebox
        # elif face_model == 'mtcnn':
        #     self.face_detector = MTCNN()
        #self.detect_face = (face_model is not None)

        print("Loading Face Landmark model: ", landmark_model)
        self.info['Landmark_Model'] = landmark_model
        if landmark_model == 'MobileNet':
            self.landmark_detector = MobileNet_GDConv(136)
            self.landmark_detector = torch.nn.DataParallel(
                self.landmark_detector)
            # download model from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing
            checkpoint = torch.load(
                'landmark_detectors/weights/mobilenet_224_model_best_gdconv_external.pth.tar', map_location=self.map_location)
            print('Use MobileNet as backbone')
            self.landmark_detector.load_state_dict(checkpoint['state_dict'])

        elif landmark_model == 'PFLD':
            self.landmark_detector = PFLDInference()
            # download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
            checkpoint = torch.load(
                'landmark_detectors/weights/pfld_model_best.pth.tar', map_location=self.map_location)
            print('Use PFLD as backbone')
            self.landmark_detector.load_state_dict(checkpoint['state_dict'])
            # download from https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing
        elif landmark_model == 'MobileFaceNet':
            self.landmark_detector = MobileFaceNet([112, 112], 136)
            checkpoint = torch.load(
                'landmark_detectors/weights/mobilefacenet_model_best.pth.tar', map_location=self.map_location)
            print('Use MobileFaceNet as backbone')
            self.landmark_detector.load_state_dict(checkpoint['state_dict'])

        print("Loading au occurence model: ", au_occur_model)
        self.info['AU_Occur_Model'] = au_occur_model
        if au_occur_model == 'jaanet':
            self.au_model = JAANet()
        # if landmark_model == 'lbf':
        #     face_landmark = cv.face.createFacemarkLBF()
        #     face_landmark_model_path = os.path.join(
        #         get_resource_path(), 'lbfmodel.yaml')
        #     if not os.path.exists(face_landmark_model_path):
        #         print("Face landmark model not found. Please run download_models.py.")
        #     face_landmark.loadModel(face_landmark_model_path)
        #     face_landmark_columns = np.array(
        #         [(f'x_{i}', f'y_{i}') for i in range(68)]).reshape(1, 136)[0].tolist()
        #     landmark_empty = np.empty((1, 136))
        #     landmark_empty[:] = np.nan
        #     empty_landmark = pd.DataFrame(
        #         landmark_empty, columns=face_landmark_columns)
        #     self.info["face_landmark_model"] = face_landmark_model_path
        #     self.info['face_landmark_columns'] = face_landmark_columns
        #     self.face_landmark = face_landmark
        #     self._empty_landmark = empty_landmark
        # elif landmark_model == 'another thing':
            # TODO: Implement the initialization of another landmark detection model
        #self.detect_landmarks = (landmark_model is not None)
        #self.detect_au_occurence = (au_occur_model is not None)

        print("Loading emotion model: ", emotion_model)
        if emotion_model == 'fer':
            emotion_model = 'fer_aug_model.h5'
            emotion_model_path = os.path.join(
                get_resource_path(), 'fer_aug_model.h5')
            if not os.path.exists(emotion_model_path):
                print(
                    "Emotion prediction model not found. Please run download_models.py.")
            model = models.load_model(emotion_model_path)  # Load model to use.
            # model input shape.
            (_, img_w, img_h, img_c) = model.layers[0].input_shape
            self.info["input_shape"] = {
                "img_w": img_w, "img_h": img_h, "img_c": img_c}
            self.info["emotion_model"] = emotion_model_path
            self.info["mapper"] = FEAT_EMOTION_MAPPER
            self.emotion_model = model
            emotion_columns = [key for key in self.info["mapper"].values()]
            self.info['emotion_model_columns'] = emotion_columns

            # create empty df for predictions
            predictions = np.empty((1, len(self.info["mapper"])))
            predictions[:] = np.nan
            empty_emotion = pd.DataFrame(
                predictions, columns=self.info["mapper"].values())
            self._empty_emotion = empty_emotion
        #self.detect_emotion = (emotion_model is not None)

#        frame_columns = ["frame"]
#        self.info["output_columns"] = frame_columns + emotion_columns + \
#            face_detection_columns + face_landmark_columns

    def __getitem__(self, i):
        return self.info[i]

    def face_detect(self, frame):
        # suppose frame=cv2.imread(imgname)
        height, width, _ = frame.shape
        faces = self.face_detector(frame)

        if len(faces) == 0:
            print("Warning: NO FACE is detected")

        return (faces)

    def landmark_detect(self, frame, detected_faces):
        mean = np.asarray([0.485, 0.456, 0.406])
        std = np.asarray([0.229, 0.224, 0.225])
        self.landmark_detector.eval()
        if self.info['Landmark_Model'] == 'MobileNet':
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
            size = int(min([w, h])*1.2)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
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
            cropped = frame[new_bbox.top:new_bbox.bottom,
                            new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(
                    edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
                continue
            test_face = cropped_face.copy()
            test_face = test_face/255.0
            if self.info['Landmark_Model'] == 'MobileNet':
                test_face = (test_face-mean)/std
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)
            input = torch.from_numpy(test_face).float()
            input = torch.autograd.Variable(input)
            if self.info['Landmark_Model'] == 'MobileFaceNet':
                landmark = self.landmark_detector(input)[0].cpu().data.numpy()
            else:
                landmark = self.landmark_detector(input).cpu().data.numpy()
            landmark = landmark.reshape(-1, 2)
            landmark = new_bbox.reprojectLandmark(landmark)
            landmark_list.append(landmark)

        return landmark_list

    def au_occur_detect(self, frame, landmarks):
        # Assume that the Raw landmark is given in the format (n_land,2)
        landmarks = np.transpose(landmarks)
        if landmarks.shape[-1] == 68:
            landmarks = convert68to49(landmarks)
        print(landmarks)
        print(landmarks.shape)
        return (self.au_model.detect_au(frame, landmarks))

    # def process_frame(self, frame, raw_landmarks = None, counter=0):
    #     """Helper function to run face detection, landmark detection, and emotion detection on a frame.

    #     Args:
    #         frame (np.array): Numpy array of image, ideally loaded through Pillow.Image
    #         counter (int, str, default=0): Index used for the prediction results dataframe.

    #     Returns:
    #         df (dataframe): Prediction results dataframe.

    #     Example:
    #         >> from pil import Image
    #         >> frame = Image.open("input.jpg")
    #         >> detector = Detector()
    #         >> detector.process_frame(np.array(frame))
    #     """
    #     # How you would use MTCNN in this case:
    #     # my_model.detect(img = im01,landmarks=False)[0] will return a bounding box array of shape (1,4)

    #     try:
    #         # change image to grayscale
    #         grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #         # find faces
    #         detected_faces = self.face_detector.detectMultiScale(grayscale_image)
    #         # detect landmarks
    #         ok, landmarks = self.face_landmark.fit(grayscale_image, detected_faces)
    #         landmarks_df = pd.DataFrame(landmarks[0][0].reshape(1, 136), columns = self["face_landmark_columns"], index=[counter])
    #         # Use Tiankang's colde to align the faces to the center

    #         # crop just the face area
    #         if len(detected_faces)>0:
    #             facebox_df = pd.DataFrame([detected_faces[0]], columns = self["face_detection_columns"], index=[counter])
    #             grayscale_cropped_face = Image.fromarray(grayscale_image).crop(face_rect_to_coords(detected_faces[0]))
    #             # resize face to newsize 48 x 48
    #             # print("resizeface", grayscale_cropped_face.shape, img_w, img_h, img_c)
    #             grayscale_cropped_resized_face = grayscale_cropped_face.resize((self['input_shape']["img_w"], self['input_shape']["img_h"]))
    #             # reshape to put in model
    #             grayscale_cropped_resized_reshaped_face = np.array(grayscale_cropped_resized_face).reshape(1, self['input_shape']["img_w"], self['input_shape']["img_h"], self['input_shape']["img_c"])
    #             # normalize
    #             normalize_grayscale_cropped_resized_reshaped_face = grayscale_cropped_resized_reshaped_face/255.
    #             # make tensor
    #             tensor_img = tf.convert_to_tensor(normalize_grayscale_cropped_resized_reshaped_face)
    #             # make predictions
    #             predictions = self.emotion_model.predict(tensor_img)
    #             emotion_df = pd.DataFrame(predictions, columns = self["mapper"].values(), index=[counter])

    #             #=======================AU prediction==================================
    #             if raw_landmarks is not None:
    #                 au_landmarks = raw_landmarks
    #             else:
    #                 au_landmarks = convert68to49(landmarks[0][0]).flatten()

    #             au_df = pd.DataFrame(self.au_model.detect_au(frame,au_landmarks), columns = ["1","2","4","6","7","10","12","14","15","17","23","24"], index=[counter])

    #             return pd.concat([emotion_df, facebox_df, landmarks_df, au_df], axis=1)
    #     except:
    #         emotion_df = self._empty_emotion.reindex(index=[counter])
    #         facebox_df = self._empty_facebox.reindex(index=[counter])
    #         landmarks_df = self._empty_landmark.reindex(index=[counter])
    #         au_df = self._empty_auoccurence.reindex(index=[counter])

    #         return pd.concat([emotion_df, facebox_df, landmarks_df, au_df], axis=1)

    def detect_video(self, inputFname, outputFname=None, skip_frames=1):
        """Detects FEX from a video file.

        Args:
            inputFname (str): Path to video file
            outputFname (str, optional): Path to output file. Defaults to None.
            skip_frames (int, optional): Number of every other frames to skip for speed or if not all frames need to be processed. Defaults to 1.

        Returns:
            dataframe: Prediction results dataframe if outputFname is None. Returns True if outputFname is specified.
        """
        self.info['inputFname'] = inputFname
        self.info['outputFname'] = outputFname
        init_df = pd.DataFrame(columns=self["output_columns"])
        if outputFname:
            init_df.to_csv(outputFname, index=False, header=True)

        cap = cv.VideoCapture(inputFname)

        # Determine whether to use multiprocessing.
        n_jobs = self['n_jobs']
        if n_jobs == -1:
            thread_num = cv.getNumberOfCPUs()  # get available cpus
        else:
            thread_num = n_jobs
        pool = ThreadPool(processes=thread_num)
        pending_task = deque()
        counter = 0
        processed_frames = 0
        frame_got = True
        detected_faces = []
        print("Processing video.")
        while True:
            # Consume the queue.
            while len(pending_task) > 0 and pending_task[0].ready():
                df = pending_task.popleft().get()
                # Save to output file.
                if outputFname:
                    df.to_csv(outputFname, index=True, header=False, mode='a')
                else:
                    init_df = pd.concat([init_df, df], axis=0)
                processed_frames = processed_frames + 1

            if not frame_got:
                break

            # Populate the queue.
            if len(pending_task) < thread_num:
                frame_got, frame = cap.read()
                # Process at every seconds.
                if counter % skip_frames == 0:
                    if frame_got:
                        task = pool.apply_async(
                            self.process_frame, (frame.copy(), counter))
                        pending_task.append(task)
                counter = counter + 1
        cap.release()
        if outputFname:
            return True
        else:
            return init_df

    def detect_image(self, inputFname, outputFname=None):
        """Detects FEX from a video file.

        Args:
            inputFname (str, or list of str): Path to image file or a list of paths to image files.
            outputFname (str, optional): Path to output file. Defaults to None.

        Rseturns:
            Fex: Prediction results dataframe if outputFname is None. Returns True if outputFname is specified.
        """
        assert type(inputFname) == str or type(
            inputFname) == list, "inputFname must be a string path to image or list of image paths"
        if type(inputFname) == str:
            inputFname = [inputFname]
        for inputF in inputFname:
            if not os.path.exists(inputF):
                raise FileNotFoundError(f"File {inputF} not found.")
        self.info['inputFname'] = inputFname

        init_df = pd.DataFrame(columns=self["output_columns"])
        if outputFname:
            init_df.to_csv(outputFname, index=False, header=True)

        for inputF in inputFname:
            frame = Image.open(inputF)
            df = self.process_frame(np.array(frame))

            if outputFname:
                df.to_csv(outputFname, index=True, header=False, mode='a')
            else:
                init_df = pd.concat([init_df, df], axis=0)

        if outputFname:
            return True
        else:
            return Fex(init_df, filename=inputFname, au_columns=None, emotion_columns=FEAT_EMOTION_COLUMNS, facebox_columns=FEAT_FACEBOX_COLUMNS, landmark_columns=openface_2d_landmark_columns, time_columns=FACET_TIME_COLUMNS, detector="Feat")


# %%
# Test case:

# A01 = Detector(face_model='RetinaFace',emotion_model=None, landmark_model = "MobileFaceNet")
# test_img = cv2.imread("F:/test_case/0010.jpg")
# bboxes = A01.face_detect(test_img)
# x,y,w,h,_ = bboxes[0]
# x = int(x)
# y = int(y)
# w = int(w)
# h = int(h)
# cv2.rectangle(test_img,(x,y),(w,h),(0,255,0),2)
# cv2.putText(test_img,'Moth Detected',(w+10,h),0,0.3,(0,255,0))
# cv2.imshow("Show",test_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
#b_box_ = BBox(bboxes[0])
# kruska0 = A01.landmark_detect(test_img,bboxes)
# A01.landmark_detect()
# imm00 = drawLandmark(test_img,b_box_,kruska0)
# cv2.imshow('image',imm00)
# cv2.waitKey(0)

# test_img = cv2.imread("F:/test_case/0010.jpg")
# bboxes = A01.face_detect(test_img)
# lands = A01.landmark_detect(test_img,bboxes)
# aus = A01.au_occur_detect(test_img,lands)
