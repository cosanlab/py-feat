#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""

from feat.detector import Detector
from feat.data import Fex
from feat.utils import get_resource_path
from feat.tests.utils import get_test_data_path
import pandas as pd
import feat
import os
import wget
import cv2
import numpy as np
# def test_models():
#     print("Downloading FEX emotion model.")
#     fex_emotion_model = "https://github.com/cosanlab/feat/releases/download/v0.1/fer_aug_model.h5"
#     wget.download(fex_emotion_model, get_resource_path())

#     if os.path.exists(os.path.join(get_resource_path(), "fer_aug_model.h5")):
#         print("\nFEX emotion model downloaded successfully.\n")
#     else:
#         print("Something went wrong. Model not found in directory.")

#     print("Downloading landmark detection model.")
#     lbfmodel = "https://github.com/cosanlab/feat/releases/download/v0.1/lbfmodel.yaml"
#     wget.download(lbfmodel, get_resource_path())

#     if os.path.exists(os.path.join(get_resource_path(), "lbfmodel.yaml")):
#         print("\nLandmark detection model downloaded successfully.\n")
#     else:
#         print("Something went wrong. Model not found in directory.")

#     emotion_model = "fer_aug_model.h5"
#     emotion_model_path = os.path.join(get_resource_path(), emotion_model)
#     print("PATH TO EMOTION MODEL",emotion_model_path)
#     assert os.path.exists(emotion_model_path)==True

#     landmark_model = "lbfmodel.yaml"
#     landmark_model_path = os.path.join(get_resource_path(), landmark_model)
#     assert os.path.exists(landmark_model_path)==True


def test_detector():

    detector = Detector(n_jobs=1)
    assert detector['n_jobs'] == 1
    assert type(detector) == Detector

    # Face Detector Test Case:
    detector01 = Detector(face_model='FaceBoxes', landmark_model=None,
                          au_occur_model=None, emotion_model=None, n_jobs=1)
    inputFname = os.path.join(get_test_data_path(), "sampler0000.jpg")
    img01 = cv2.imread(inputFname)
    h, w, _ = img01.shape
    out = detector01.face_detect(img01)
    bbox_left = out[0][0]
    bbox_right = out[0][1]
    bbox_top = out[0][2]
    bbox_bottom = out[0][3]
    assert len(out[0]) == 5
    assert bbox_left > 0 and bbox_right > 0 and bbox_top > 0 and bbox_bottom > 0 and \
        bbox_left < bbox_right and bbox_top < bbox_bottom and bbox_left < w and \
        bbox_right < w and bbox_top < h and bbox_bottom < h

    detector02 = Detector(face_model='RetinaFace', landmark_model=None,
                          au_occur_model=None, emotion_model=None, n_jobs=1)
    out = detector02.face_detect(img01)
    bbox_left = out[0][0]
    bbox_right = out[0][1]
    bbox_top = out[0][2]
    bbox_bottom = out[0][3]
    assert len(out[0]) == 5
    assert bbox_left > 0 and bbox_right > 0 and bbox_top > 0 and bbox_bottom > 0 and \
        bbox_left < bbox_right and bbox_top < bbox_bottom and bbox_left < w and \
        bbox_right < w and bbox_top < h and bbox_bottom < h

    detector03 = Detector(face_model='MTCNN', landmark_model=None,
                          au_occur_model=None, emotion_model=None, n_jobs=1)
    out = detector03.face_detect(img01)
    bbox_left = out[0][0]
    bbox_right = out[0][1]
    bbox_top = out[0][2]
    bbox_bottom = out[0][3]
    assert len(out[0]) == 5
    assert bbox_left > 0 and bbox_right > 0 and bbox_top > 0 and bbox_bottom > 0 and \
        bbox_left < bbox_right and bbox_top < bbox_bottom and bbox_left < w and \
        bbox_right < w and bbox_top < h and bbox_bottom < h

    # Landmark Detector Test Case:
    inputFname = os.path.join(get_test_data_path(), "sampler0000.jpg")
    img01 = cv2.imread(inputFname)
    h, w, _ = img01.shape
    detector01 = Detector(face_model='RetinaFace',
                          emotion_model=None, landmark_model="MobileFaceNet")
    bboxes = detector01.face_detect(img01)
    landmarks = detector01.landmark_detect(img01, bboxes)
    assert landmarks[0].shape == (68, 2)
    assert np.any(landmarks[0][:, 0] > 0) and np.any(landmarks[0][:, 0] < w) and \
        np.any(landmarks[0][:, 1] > 0) and np.any(landmarks[0][:, 1] < h)

    detector02 = Detector(face_model='RetinaFace',
                          emotion_model=None, landmark_model="MobileNet")
    bboxes = detector02.face_detect(img01)
    landmarks = detector02.landmark_detect(img01, bboxes)
    assert landmarks[0].shape == (68, 2)
    assert np.any(landmarks[0][:, 0] > 0) and np.any(landmarks[0][:, 0] < w) and \
        np.any(landmarks[0][:, 1] > 0) and np.any(landmarks[0][:, 1] < h)

    detector03 = Detector(face_model='RetinaFace',
                          emotion_model=None, landmark_model="PFLD")
    bboxes = detector03.face_detect(img01)
    landmarks = detector03.landmark_detect(img01, bboxes)
    assert landmarks[0].shape == (68, 2)
    assert np.any(landmarks[0][:, 0] > 0) and np.any(landmarks[0][:, 0] < w) and \
        np.any(landmarks[0][:, 1] > 0) and np.any(landmarks[0][:, 1] < h)

    # AU Detection Case:
    inputFname = os.path.join(get_test_data_path(), "sampler0000.jpg")
    img01 = cv2.imread(inputFname)
    detector1 = Detector(face_model='RetinaFace',emotion_model=None, landmark_model = "MobileFaceNet", au_occur_model="jaanet")
    bboxes = detector1.face_detect(img01)
    lands = detector1.landmark_detect(img01,bboxes)
    aus = detector1.au_occur_detect(img01,lands)
    assert aus.shape[-1] == 12

    # Test detect image
    inputFname = os.path.join(get_test_data_path(), "input.jpg")
    out = detector.detect_image(inputFname=inputFname)
    assert type(out) == Fex
    assert len(out) == 1
    #assert out.happiness.values[0] > 0

    outputFname = os.path.join(get_test_data_path(), "output.csv")
    out = detector.detect_image(inputFname=inputFname, outputFname=outputFname)
    assert out
    assert os.path.exists(outputFname)
    out = pd.read_csv(outputFname)
    #assert out.happiness.values[0] > 0

    # Test detect video
    inputFname = os.path.join(get_test_data_path(), "input.mp4")
    out = detector.detect_video(inputFname=inputFname,skip_frames=20)
    assert len(out) == 4

    # outputFname = os.path.join(get_test_data_path(), "output.csv")
    # out = detector.detect_video(inputFname=inputFname, outputFname=outputFname)
    # assert out
    # assert os.path.exists(outputFname)
    # out = pd.read_csv(outputFname)
    # assert out.happiness.values.max() > 0

    # Test processing everything:
    detector04 = Detector(face_model='RetinaFace', emotion_model='fer', landmark_model="PFLD", au_occur_model='jaanet')
    inputFname = os.path.join(get_test_data_path(), "sampler0000.jpg")
    img01 = cv2.imread(inputFname)
    files = detector04.process_frame(img01,0)


test_detector()