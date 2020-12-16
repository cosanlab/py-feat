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

    """
    Detector is a class used to detect faces, facial landmarks, emotions, and action units from images and videos.
    """

    def __init__(self):
        self.info = {}
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
        self.info["emotion_model"] = emotion_model_path
        self.info["input_shape"] = {"img_w": img_w, "img_h": img_h, "img_c": img_c}
        self.info["mapper"] = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
        self.emotion_model = model
        emotion_columns = [key for key in self.info["mapper"].values()]

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
        """
        Takes a frame from OpenCV and prepares it as a tensor to be predicted by model. 

        frame: Image input. 
        counter: Frame number. Defaults to 0
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

    def detect_video(self, inputFname, outputFname=None, skip_frames=1, n_jobs=1):
        """
        Inputs

        Outputs
            df: dataframe with columns 
        """
        self.info['inputFname'] = inputFname
        self.info['outputFname'] = outputFname 
        init_df = pd.DataFrame(columns=self["output_columns"])
        if outputFname:
            init_df.to_csv(outputFname, index=False, header=True)

        cap = cv.VideoCapture(inputFname)

        # Determine whether to use multiprocessing.
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
                print(df)
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
        """
        Inputs

        Outputs
            If outputFname is specified, saves results to file and returns True.
            If outputFname is not specified, returns the results in a dataframe.
        """
        self.info['inputFname'] = inputFname
        frame = Image.open(inputFname)
        df = self.process_frame(np.array(frame))
        if outputFname:
            df.to_csv(outputFname, index=False, header=True)
            return True
        else:
            return df