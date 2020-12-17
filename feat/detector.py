from __future__ import division

"""Functions to help detect face, landmarks, emotions, action units from images and videos"""

from collections import deque
from multiprocessing.pool import ThreadPool
import tensorflow as tf
from tensorflow.python.keras import optimizers, models
import os
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
import cv2 as cv
from feat.utils import get_resource_path, face_rect_to_coords

class Detector(object):
    def __init__(self, n_jobs=1):
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
        """ LOAD UP THE MODELS """
        print("Loading Face Detection model.")
        face_detection_model_path = cv.data.haarcascades + "haarcascade_frontalface_alt.xml"
        if not os.path.exists(face_detection_model_path):
            print("Face detection model not found. Check haarcascade_frontalface_alt.xml exists in your opencv installation (cv.data).")
        face_cascade = cv.CascadeClassifier(face_detection_model_path)
        face_detection_columns = ["facebox_x", "facebox_y", "facebox_w", "facebox_h"]
        facebox_empty = np.empty((1,4))
        facebox_empty[:] = np.nan
        empty_facebox = pd.DataFrame(facebox_empty, columns = face_detection_columns)
        self.info["face_detection_model"] = face_detection_model_path
        self.info["face_detection_columns"] = face_detection_columns
        self.face_detector = face_cascade
        self._empty_facebox = empty_facebox

        print("Loading Face Landmark model.")
        face_landmark = cv.face.createFacemarkLBF()
        face_landmark_model_path = os.path.join(get_resource_path(), 'lbfmodel.yaml')
        if not os.path.exists(face_landmark_model_path):
            print("Face landmark model not found. Please run download_models.py.")
        face_landmark.loadModel(face_landmark_model_path)
        face_landmark_columns = np.array([(f'x_{i}',f'y_{i}') for i in range(68)]).reshape(1,136)[0].tolist()
        landmark_empty = np.empty((1,136))
        landmark_empty[:] = np.nan
        empty_landmark = pd.DataFrame(landmark_empty, columns = face_landmark_columns)
        self.info["face_landmark_model"] = face_landmark_model_path
        self.info['face_landmark_columns'] = face_landmark_columns
        self.face_landmark = face_landmark
        self._empty_landmark = empty_landmark

        print("Loading FEX DCNN emotion model.")
        emotion_model = 'fer_aug_model.h5'
        emotion_model_path = os.path.join(get_resource_path(),'fer_aug_model.h5')
        if not os.path.exists(emotion_model_path):
            print("Emotion prediction model not found. Please run download_models.py.")
        model = models.load_model(emotion_model_path) # Load model to use.
        (_, img_w, img_h, img_c) = model.layers[0].input_shape # model input shape.
        self.info["input_shape"] = {"img_w": img_w, "img_h": img_h, "img_c": img_c}
        self.info["emotion_model"] = emotion_model_path
        self.info["mapper"] = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
        self.emotion_model = model
        emotion_columns = [key for key in self.info["mapper"].values()]
        self.info['emotion_model_columns'] = emotion_columns

        # create empty df for predictions 
        predictions = np.empty((1, len(self.info["mapper"])))
        predictions[:] = np.nan
        empty_emotion = pd.DataFrame(predictions, columns = self.info["mapper"].values())
        self._empty_emotion = empty_emotion

        frame_columns = ["frame"]
        self.info["output_columns"] = frame_columns + emotion_columns + face_detection_columns + face_landmark_columns

    def __getitem__(self, i):
        return self.info[i]

    def process_frame(self, frame, counter=0):
        """Helper function to run face detection, landmark detection, and emotion detection on a frame. 

        Args:
            frame (np.array): Numpy array of image, ideally loaded through Pillow.Image
            counter (int, str, default=0): Index used for the prediction results dataframe.

        Returns:
            df (dataframe): Prediction results dataframe.

        Example:
            >> from pil import Image
            >> frame = Image.open("input.jpg")
            >> detector = Detector()
            >> detector.process_frame(np.array(frame))
        """
        try:
            # change image to grayscale
            grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # find faces
            detected_faces = self.face_detector.detectMultiScale(grayscale_image)
            # detect landmarks
            ok, landmarks = self.face_landmark.fit(grayscale_image, detected_faces)
            landmarks_df = pd.DataFrame(landmarks[0][0].reshape(1, 136), columns = self["face_landmark_columns"], index=[counter])
            # Use Tiankang's colde to align the faces to the center 

            # crop just the face area
            if len(detected_faces)>0:
                facebox_df = pd.DataFrame([detected_faces[0]], columns = self["face_detection_columns"], index=[counter])
                grayscale_cropped_face = Image.fromarray(grayscale_image).crop(face_rect_to_coords(detected_faces[0]))
                # resize face to newsize 48 x 48
                # print("resizeface", grayscale_cropped_face.shape, img_w, img_h, img_c) 
                grayscale_cropped_resized_face = grayscale_cropped_face.resize((self['input_shape']["img_w"], self['input_shape']["img_h"]))
                # reshape to put in model
                grayscale_cropped_resized_reshaped_face = np.array(grayscale_cropped_resized_face).reshape(1, self['input_shape']["img_w"], self['input_shape']["img_h"], self['input_shape']["img_c"])
                # normalize
                normalize_grayscale_cropped_resized_reshaped_face = grayscale_cropped_resized_reshaped_face/255.
                # make tensor
                tensor_img = tf.convert_to_tensor(normalize_grayscale_cropped_resized_reshaped_face)
                # make predictions
                predictions = self.emotion_model.predict(tensor_img)
                emotion_df = pd.DataFrame(predictions, columns = self["mapper"].values(), index=[counter])
                return pd.concat([emotion_df, facebox_df, landmarks_df], axis=1)
        except:
            emotion_df = self._empty_emotion.reindex(index=[counter])
            facebox_df = self._empty_facebox.reindex(index=[counter])
            landmarks_df = self._empty_landmark.reindex(index=[counter])
            return pd.concat([emotion_df, facebox_df, landmarks_df], axis=1)

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
        if n_jobs==-1:
            thread_num = cv.getNumberOfCPUs() # get available cpus
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
                if counter%skip_frames == 0:
                    if frame_got:
                        task = pool.apply_async(self.process_frame, (frame.copy(), counter))
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

        Returns:
            dataframe: Prediction results dataframe if outputFname is None. Returns True if outputFname is specified.
        """
        assert type(inputFname)==str or type(inputFname)==list, "inputFname must be a string path to image or list of image paths"
        if type(inputFname)==str:
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
            return init_df