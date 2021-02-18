# %%
from __future__ import division

"""Functions to help detect face, landmarks, emotions, action units from images and videos"""

from collections import deque
from multiprocessing.pool import ThreadPool
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import feat
from feat.data import Fex
from feat.utils import get_resource_path, face_rect_to_coords, openface_2d_landmark_columns, jaanet_AU_presence, FEAT_EMOTION_MAPPER, FEAT_EMOTION_COLUMNS, FEAT_FACEBOX_COLUMNS, FEAT_TIME_COLUMNS, FACET_TIME_COLUMNS, BBox, convert68to49
from feat.au_detectors.JAANet.JAA_test import JAANet
from feat.au_detectors.DRML.DRML_test import DRMLNet
from feat.emo_detectors.ferNet.ferNet_test import ferNetModule
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
        
        if face_model:
            if face_model.lower() == "faceboxes":
                self.face_detector = FaceBoxes()
            elif face_model.lower() == "retinaface":
                self.face_detector = Retinaface.Retinaface()
            elif face_model.lower() == 'mtcnn':
                self.face_detector = MTCNN()

        self.info['Face_Model'] = face_model
        #self.info["mapper"] = FEAT_FACEBOX_COLUMNS
        facebox_columns = FEAT_FACEBOX_COLUMNS
        self.info['face_detection_columns'] = facebox_columns
        predictions = np.empty((1, len(facebox_columns)))
        predictions[:] = np.nan
        empty_facebox = pd.DataFrame(predictions, columns=facebox_columns)
        self._empty_facebox = empty_facebox


        print("Loading Face Landmark model: ", landmark_model)
        #self.info['Landmark_Model'] = landmark_model
        if landmark_model:
            if landmark_model.lower() == 'mobilenet':
                self.landmark_detector = MobileNet_GDConv(136)
                self.landmark_detector = torch.nn.DataParallel(
                    self.landmark_detector)
                # download model from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing
                #CHANGEME
                checkpoint = torch.load(os.path.join(feat.__path__[0], 'landmark_detectors/weights/mobilenet_224_model_best_gdconv_external.pth.tar'), map_location=self.map_location)
                print('Use MobileNet as backbone')
                self.landmark_detector.load_state_dict(checkpoint['state_dict'])

            elif landmark_model.lower() == 'pfld':
                self.landmark_detector = PFLDInference()
                # download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
                checkpoint = torch.load(os.path.join(feat.__path__[0],'landmark_detectors/weights/pfld_model_best.pth.tar'), map_location=self.map_location)
                print('Use PFLD as backbone')
                self.landmark_detector.load_state_dict(checkpoint['state_dict'])
                # download from https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing
            elif landmark_model.lower() == 'mobilefacenet':
                self.landmark_detector = MobileFaceNet([112, 112], 136)
                checkpoint = torch.load(os.path.join(feat.__path__[0],'landmark_detectors/weights/mobilefacenet_model_best.pth.tar'), map_location=self.map_location)
                print('Use MobileFaceNet as backbone')
                self.landmark_detector.load_state_dict(checkpoint['state_dict'])

        self.info['landmark_model'] = landmark_model
        self.info["mapper"] = openface_2d_landmark_columns
        landmark_columns = openface_2d_landmark_columns
        self.info['face_landmark_columns'] = landmark_columns
        predictions = np.empty((1, len(openface_2d_landmark_columns)))
        predictions[:] = np.nan
        empty_landmarks = pd.DataFrame(predictions, columns=landmark_columns)
        self._empty_landmark = empty_landmarks


        print("Loading au occurence model: ", au_occur_model)
        self.info['AU_Occur_Model'] = au_occur_model
        if au_occur_model:
            if au_occur_model.lower() == 'jaanet':
                self.au_model = JAANet()
            elif au_occur_model.lower() == 'drml':
                self.au_model = DRMLNet()

        self.info['auoccur_model'] = au_occur_model
        #self.info["mapper"] = jaanet_AU_presence
        auoccur_columns = jaanet_AU_presence
        self.info['au_presence_columns'] = auoccur_columns
        predictions = np.empty((1, len(auoccur_columns)))
        predictions[:] = np.nan
        empty_au_occurs = pd.DataFrame(predictions, columns=auoccur_columns)
        self._empty_auoccurence = empty_au_occurs



        print("Loading emotion model: ", emotion_model)
        self.info['Emo_Model'] = emotion_model
        if emotion_model:
            if emotion_model.lower() == 'fer':
                self.emo_model = ferNetModule()
        
        self.info['emotion_model_columns'] = FEAT_EMOTION_COLUMNS
        predictions = np.empty((1, len(FEAT_EMOTION_COLUMNS)))
        predictions[:] = np.nan
        empty_emotion = pd.DataFrame(predictions, columns=FEAT_EMOTION_COLUMNS)
        self._empty_emotion = empty_emotion

        #self.info['auoccur_model'] = au_occur_model
        #self.info["mapper"] = jaanet_AU_presence
        auoccur_columns = jaanet_AU_presence
        self.info['au_presence_columns'] = auoccur_columns
        predictions = np.empty((1, len(auoccur_columns)))
        predictions[:] = np.nan
        empty_au_occurs = pd.DataFrame(predictions, columns=auoccur_columns)
        self._empty_auoccurence = empty_au_occurs

        frame_columns = ["frame"]

        self.info["output_columns"] = frame_columns  + \
           facebox_columns + landmark_columns + auoccur_columns + FEAT_EMOTION_COLUMNS

    def __getitem__(self, i):
        return self.info[i]

    def face_detect(self, frame):
        # suppose frame=cv2.imread(imgname)
        height, width, _ = frame.shape
        faces = self.face_detector(frame)

        if len(faces) == 0:
            print("Warning: NO FACE is detected")
        return faces

    def landmark_detect(self, frame, detected_faces):
        mean = np.asarray([0.485, 0.456, 0.406])
        std = np.asarray([0.229, 0.224, 0.225])
        self.landmark_detector.eval()
        if self.info['landmark_model']:
            if self.info['landmark_model'].lower() == 'mobilenet':
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
            if self.info['landmark_model']:
                if self.info['landmark_model'].lower() == 'mobilenet':
                    test_face = (test_face-mean)/std
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)
            input = torch.from_numpy(test_face).float()
            input = torch.autograd.Variable(input)
            if self.info['landmark_model']:
                if self.info['landmark_model'].lower() == 'mobilefacenet':
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
        return self.au_model.detect_au(frame, landmarks)

    def emo_detect(self, frame, facebox):

        return self.emo_model.detect_emo(frame,facebox)

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
        # How you would use MTCNN in this case:
        # my_model.detect(img = im01,landmarks=False)[0] will return a bounding box array of shape (1,4)

        try:
            # detect faces
            detected_faces = self.face_detect(frame=frame)
            out = None
            for i, faces in enumerate(detected_faces):
                facebox_df = pd.DataFrame([[faces[0], faces[1], faces[2] - faces[0], faces[3] - faces[1]]], columns = self["face_detection_columns"], index=[counter+i])
                # detect landmarks
                landmarks = self.landmark_detect(frame=frame, detected_faces=[faces[0:4]])
                landmarks_df = pd.DataFrame([landmarks[0].flatten(order="F")], columns = self["face_landmark_columns"], index=[counter+i])
                # detect AUs
                au_occur = self.au_occur_detect(frame=frame, landmarks=landmarks)
                au_occur_df = pd.DataFrame(au_occur, columns = self["au_presence_columns"], index = [counter+i])
                # detect emotions
                emo_pred = self.emo_detect(frame=frame, facebox=[faces])
                emo_pred_df = pd.DataFrame(emo_pred, columns = FEAT_EMOTION_COLUMNS, index=[counter+i])
                tmp_df = pd.concat([facebox_df, landmarks_df, au_occur_df, emo_pred_df], axis=1)
                if out is None:
                    out = tmp_df
                else:
                    out = pd.concat([out,tmp_df],axis=0)
            out[FEAT_TIME_COLUMNS] = counter
            return out
        
        except:
            print("exception occurred")
            emotion_df = self._empty_emotion.reindex(index=[counter])
            facebox_df = self._empty_facebox.reindex(index=[counter])
            landmarks_df = self._empty_landmark.reindex(index=[counter])
            au_occur_df = self._empty_auoccurence.reindex(index=[counter])
    
            out = pd.concat([facebox_df, landmarks_df, au_occur_df, emotion_df], axis=1)
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
        self.info['inputFname'] = inputFname
        self.info['outputFname'] = outputFname 
        init_df = pd.DataFrame(columns=self["output_columns"])
        if outputFname:
            init_df.to_csv(outputFname, index=False, header=True)

        cap = cv2.VideoCapture(inputFname)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = int(np.ceil(length / skip_frames))

        # Determine whether to use multiprocessing.
        n_jobs = self['n_jobs']
        if n_jobs==-1:
            thread_num = cv2.getNumberOfCPUs() # get available cpus
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
            if counter%skip_frames == 0:
                df = self.process_frame(frame, counter=counter)
                init_df = pd.concat([init_df, df], axis=0)
            counter = counter + 1
            if not frame_got:
                break
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
            print(f"processing {inputF}")
            frame = cv2.imread(inputF)
            df = self.process_frame(frame)
            df['input'] = inputF
            # df = self.process_frame(np.array(frame))
            if outputFname:
                df.to_csv(outputFname, index=True, header=False, mode='a')
            else:
                init_df = pd.concat([init_df, df], axis=0)

        if outputFname:
            return True
        else:
            return Fex(init_df, filename=inputFname, au_columns=jaanet_AU_presence, emotion_columns=FEAT_EMOTION_COLUMNS, facebox_columns=FEAT_FACEBOX_COLUMNS, landmark_columns=openface_2d_landmark_columns, time_columns=FACET_TIME_COLUMNS, detector="Feat")


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

if __name__ == '__main__':
    A01 = Detector(face_model='RetinaFace',emotion_model='fer', landmark_model="MobileFaceNet", au_occur_model='jaanet')
    test_img = cv2.imread("F:/test_case/0010.jpg")
    ress = A01.process_frame(frame=test_img)
# %%
