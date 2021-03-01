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

inputFname = os.path.join(get_test_data_path(), "sampler0000.jpg")
img01 = cv2.imread(inputFname)
h, w, _ = img01.shape


def test_detector():
    detector = Detector(n_jobs=1)
    assert detector["n_jobs"] == 1
    assert type(detector) == Detector


def test_faceboxes():
    # Face Detector Test Case:
    detector01 = Detector(
        face_model="FaceBoxes",
        landmark_model=None,
        au_occur_model=None,
        emotion_model=None,
        n_jobs=1,
    )
    out = detector01.face_detect(img01)
    bbox_left = out[0][0]
    assert bbox_left != None
    bbox_right = out[0][1]
    bbox_top = out[0][2]
    bbox_bottom = out[0][3]
    assert len(out[0]) == 5
    assert (
        bbox_left > 0
        and bbox_right > 0
        and bbox_top > 0
        and bbox_bottom > 0
        and bbox_left < bbox_right
        and bbox_top < bbox_bottom
        and bbox_left < w
        and bbox_right < w
        and bbox_top < h
        and bbox_bottom < h
    )


def test_retinaface():
    detector02 = Detector(
        face_model="RetinaFace",
        landmark_model=None,
        au_occur_model=None,
        emotion_model=None,
        n_jobs=1,
    )
    out = detector02.face_detect(img01)
    bbox_left = out[0][0]
    assert bbox_left != None
    bbox_right = out[0][1]
    bbox_top = out[0][2]
    bbox_bottom = out[0][3]
    assert len(out[0]) == 5
    assert (
        bbox_left > 0
        and bbox_right > 0
        and bbox_top > 0
        and bbox_bottom > 0
        and bbox_left < bbox_right
        and bbox_top < bbox_bottom
        and bbox_left < w
        and bbox_right < w
        and bbox_top < h
        and bbox_bottom < h
    )


def test_mtcnn():
    detector03 = Detector(
        face_model="MTCNN",
        landmark_model=None,
        au_occur_model=None,
        emotion_model=None,
        n_jobs=1,
    )
    out = detector03.face_detect(img01)
    bbox_left = out[0][0]
    assert bbox_left != None
    bbox_right = out[0][1]
    bbox_top = out[0][2]
    bbox_bottom = out[0][3]
    assert len(out[0]) == 5
    assert (
        bbox_left > 0
        and bbox_right > 0
        and bbox_top > 0
        and bbox_bottom > 0
        and bbox_left < bbox_right
        and bbox_top < bbox_bottom
        and bbox_left < w
        and bbox_right < w
        and bbox_top < h
        and bbox_bottom < h
    )


def test_mobilefacenet():
    # Landmark Detector Test Case:
    detector01 = Detector(
        face_model="RetinaFace", emotion_model=None, landmark_model="MobileFaceNet"
    )
    bboxes = detector01.face_detect(img01)
    landmarks = detector01.landmark_detect(img01, bboxes)
    assert landmarks[0].shape == (68, 2)
    assert (
        np.any(landmarks[0][:, 0] > 0)
        and np.any(landmarks[0][:, 0] < w)
        and np.any(landmarks[0][:, 1] > 0)
        and np.any(landmarks[0][:, 1] < h)
    )


def test_mobilenet():
    detector02 = Detector(
        face_model="RetinaFace", emotion_model=None, landmark_model="MobileNet"
    )
    bboxes = detector02.face_detect(img01)
    landmarks = detector02.landmark_detect(img01, bboxes)
    assert landmarks[0].shape == (68, 2)
    assert (
        np.any(landmarks[0][:, 0] > 0)
        and np.any(landmarks[0][:, 0] < w)
        and np.any(landmarks[0][:, 1] > 0)
        and np.any(landmarks[0][:, 1] < h)
    )


def test_pfld():
    detector03 = Detector(
        face_model="RetinaFace", emotion_model=None, landmark_model="PFLD"
    )
    bboxes = detector03.face_detect(img01)
    landmarks = detector03.landmark_detect(img01, bboxes)
    assert landmarks[0].shape == (68, 2)
    assert (
        np.any(landmarks[0][:, 0] > 0)
        and np.any(landmarks[0][:, 0] < w)
        and np.any(landmarks[0][:, 1] > 0)
        and np.any(landmarks[0][:, 1] < h)
    )


def test_jaanet():
    # AU Detection Case:
    detector1 = Detector(
        face_model="RetinaFace",
        emotion_model=None,
        landmark_model="MobileFaceNet",
        au_occur_model="jaanet",
    )
    bboxes = detector1.face_detect(img01)
    lands = detector1.landmark_detect(img01, bboxes)
    aus = detector1.au_occur_detect(img01, lands)
    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 12


def test_drml():
    # AU Detection Case2:
    inputFname = os.path.join(get_test_data_path(), "sampler0000.jpg")
    img01 = cv2.imread(inputFname)
    detector1 = Detector(
        face_model="RetinaFace",
        emotion_model=None,
        landmark_model="MobileFaceNet",
        au_occur_model="drml",
    )
    bboxes = detector1.face_detect(img01)
    lands = detector1.landmark_detect(img01, bboxes)
    aus = detector1.au_occur_detect(img01, lands)
    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 12


def test_resmasknet():
    inputFname = os.path.join(get_test_data_path(), "sampler0000.jpg")
    detector1 = Detector(emotion_model="resmasknet")
    out = detector1.detect_image(inputFname)
    assert out.emotions()["neutral"].values > 0.5


def test_detect_image():
    # Test detect image
    detector = Detector(n_jobs=1)
    inputFname = os.path.join(get_test_data_path(), "input.jpg")
    out = detector.detect_image(inputFname=inputFname)
    assert type(out) == Fex
    assert len(out) == 1
    assert out.happiness.values[0] > 0

    outputFname = os.path.join(get_test_data_path(), "output.csv")
    out = detector.detect_image(inputFname=inputFname, outputFname=outputFname)
    assert out
    assert os.path.exists(outputFname)
    out = pd.read_csv(outputFname)
    assert out.happiness.values[0] > 0


def test_multiface():
    inputFname2 = os.path.join(
        get_test_data_path(), "tim-mossholder-hOF1bWoet_Q-unsplash.jpg"
    )
    img02 = cv2.imread(inputFname2)
    detector = Detector(
        face_model="RetinaFace",
        emotion_model="fer",
        landmark_model="PFLD",
        au_occur_model="jaanet",
    )
    files = detector.process_frame(img02, 0)
    assert files.shape[0] == 5


def test_detect_video():
    # Test detect video
    detector = Detector(n_jobs=1)
    inputFname = os.path.join(get_test_data_path(), "input.mp4")
    out = detector.detect_video(inputFname=inputFname, skip_frames=60)
    assert len(out) == 2


def test_detect_video_parallel():
    # Test detect video
    detector = Detector(n_jobs=2)
    inputFname = os.path.join(get_test_data_path(), "input.mp4")
    out = detector.detect_video(inputFname=inputFname, skip_frames=20, verbose=True)
    assert len(out) == 4

    outputFname = os.path.join(get_test_data_path(), "output.csv")
    out = detector.detect_video(
        inputFname=inputFname, outputFname=outputFname, skip_frames=10
    )
    assert out
    assert os.path.exists(outputFname)
    out = pd.read_csv(outputFname)
    assert out.happiness.values.max() > 0


def test_simultaneous():
    # Test processing everything:
    detector04 = Detector(
        face_model="RetinaFace",
        emotion_model="fer",
        landmark_model="PFLD",
        au_occur_model="jaanet",
    )
    files = detector04.process_frame(img01, 0)
