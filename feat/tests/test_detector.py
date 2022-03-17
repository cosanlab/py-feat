#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""

from feat.detector import Detector
from feat.data import Fex
from feat.utils import get_resource_path, read_pictures
from feat.tests.utils import get_test_data_path
import pandas as pd
import feat
import os
import cv2
import numpy as np
import pytest

inputFname = os.path.join(get_test_data_path(), "input.jpg")
img01 = read_pictures([inputFname])
_, h, w, _ = img01.shape


# TODO: Since most of these are parameterization tests, refactor using pytest fixtures
def test_detector():
    detector = Detector(n_jobs=1)
    assert detector["n_jobs"] == 1
    assert type(detector) == Detector


def test_faceboxes():
    # Face Detector Test Case:
    detector01 = Detector(
        face_model="FaceBoxes",
        landmark_model=None,
        au_model=None,
        emotion_model=None,
        facepose_model=None,
        n_jobs=1,
    )
    out = detector01.detect_faces(img01)[0]
    bbox_x = out[0][0]
    assert bbox_x != None
    bbox_width = out[0][1]
    bbox_y = out[0][2]
    bbox_height = out[0][3]
    assert len(out[0]) == 5
    assert 180 < bbox_x < 200


def test_retinaface():
    detector02 = Detector(
        face_model="RetinaFace",
        landmark_model=None,
        au_model=None,
        emotion_model=None,
        facepose_model=None,
        n_jobs=1,
    )
    out = detector02.detect_faces(img01)[0]
    bbox_x = out[0][0]
    assert bbox_x != None
    bbox_width = out[0][1]
    bbox_y = out[0][2]
    bbox_height = out[0][3]
    assert len(out[0]) == 5
    assert 180 < bbox_x < 200


def test_mtcnn():
    detector03 = Detector(
        face_model="MTCNN",
        landmark_model=None,
        au_model=None,
        emotion_model=None,
        facepose_model=None,
        n_jobs=1,
    )
    out = detector03.detect_faces(img01)[0]
    bbox_x = out[0][0]
    assert bbox_x != None
    bbox_width = out[0][1]
    bbox_y = out[0][2]
    bbox_height = out[0][3]
    assert len(out[0]) == 5
    assert bbox_x > 180 and bbox_x < 200


def test_img2pose():
    # Test that both face detection and facepose estimation work
    detector = Detector(face_model="img2pose", facepose_model="img2pose")
    # Face detection
    faces = detector.detect_faces(img01)[0]
    bbox_x = faces[0][0]
    assert bbox_x is not None
    assert len(faces[0]) == 5
    assert 180 < bbox_x < 200

    # Pose estimation
    poses = detector.detect_facepose(img01)[0]
    pose_to_test = poses[0][0]  # first image and first face
    pitch, roll, yaw = pose_to_test.reshape(-1)
    assert -10 < pitch < 10
    assert -5 < roll < 5
    assert -10 < yaw < 10


def test_img2pose_mismatch():
    # Check that `detector` properly handles case where user selects img2pose as face pose estimator
    # but selects a different face detector. Detector should tell user they must use img2pose as both face detector and
    # pose estimator, and force face model to be `img2pose`.
    detector = Detector(face_model="RetinaFace", facepose_model="img2pose-c")
    assert detector.info["face_model"] == "img2pose-c"


def test_mobilefacenet():
    # Landmark Detector Test Case:
    detector01 = Detector(
        face_model="RetinaFace", emotion_model=None, landmark_model="MobileFaceNet"
    )
    bboxes = detector01.detect_faces(img01)
    landmarks = detector01.detect_landmarks(img01, bboxes)[0]
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
    bboxes = detector02.detect_faces(img01)[0]
    landmarks = detector02.detect_landmarks(img01, [bboxes])[0]
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
    bboxes = detector03.detect_faces(img01)[0]
    landmarks = detector03.detect_landmarks(img01, [bboxes])[0]
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
        au_model="jaanet",
    )

    detected_faces = detector1.detect_faces(img01)
    landmarks = detector1.detect_landmarks(img01, detected_faces)

    aus = detector1.detect_aus(img01, landmarks)
    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 12


def test_logistic():
    # AU Detection Case:
    detector1 = Detector(
        face_model="RetinaFace",
        emotion_model=None,
        landmark_model="MobileFaceNet",
        au_model="logistic",
    )

    detected_faces = detector1.detect_faces(img01)
    landmarks = detector1.detect_landmarks(img01, detected_faces)
    hogs, new_lands = detector1._batch_hog(
        frames=img01, detected_faces=detected_faces, landmarks=landmarks
    )

    aus = detector1.detect_aus(frame=hogs, landmarks=new_lands)

    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 20


def test_svm():
    # AU Detection Case:
    detector1 = Detector(
        face_model="RetinaFace",
        emotion_model=None,
        landmark_model="MobileFaceNet",
        au_model="svm",
    )
    detected_faces = detector1.detect_faces(img01)
    landmarks = detector1.detect_landmarks(img01, detected_faces)
    hogs, new_lands = detector1._batch_hog(
        frames=img01, detected_faces=detected_faces, landmarks=landmarks
    )
    aus = detector1.detect_aus(frame=hogs, landmarks=new_lands)

    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 20


def test_rf():
    # AU Detection Case:
    detector1 = Detector(
        face_model="RetinaFace",
        emotion_model=None,
        landmark_model="MobileFaceNet",
        au_model="RF",
    )
    detected_faces = detector1.detect_faces(img01)
    landmarks = detector1.detect_landmarks(img01, detected_faces)
    hogs, new_lands = detector1._batch_hog(
        frames=img01, detected_faces=detected_faces, landmarks=landmarks
    )

    aus = detector1.detect_aus(frame=hogs, landmarks=new_lands)

    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 20


def test_drml():
    # AU Detection Case2:
    inputFname = os.path.join(get_test_data_path(), "input.jpg")
    img01 = cv2.imread(inputFname)
    detector1 = Detector(
        face_model="RetinaFace",
        emotion_model=None,
        landmark_model="MobileFaceNet",
        au_model="drml",
    )
    bboxes = detector1.detect_faces(img01)[0]
    lands = detector1.detect_landmarks(img01, [bboxes])[0]
    aus = detector1.detect_aus(img01, lands)
    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 12


def test_resmasknet():
    inputFname = os.path.join(get_test_data_path(), "input.jpg")
    detector1 = Detector(emotion_model="resmasknet")
    out = detector1.detect_image(inputFname)
    assert out.emotions()["happiness"].values > 0.5


def test_emotionsvm():
    inputFname = os.path.join(get_test_data_path(), "input.jpg")
    detector1 = Detector(emotion_model="svm")
    out = detector1.detect_image(inputFname)
    assert out.emotions()["happiness"].values > 0.5


def test_emotionrf():
    # Emotion RF models is not good
    inputFname = os.path.join(get_test_data_path(), "input.jpg")
    detector1 = Detector(emotion_model="rf")
    out = detector1.detect_image(inputFname)
    assert out.emotions()["happiness"].values > 0.0


def test_pnp():
    # Test that facepose can be estimated properly using landmarks + pnp algorithm
    detector = Detector(
        face_model="RetinaFace", landmark_model="MobileFaceNet", facepose_model="PnP"
    )
    bboxes = detector.detect_faces(frame=img01)
    lms = detector.detect_landmarks(frame=img01, detected_faces=bboxes)
    poses = detector.detect_facepose(frame=img01, landmarks=lms)
    pose_to_test = poses[0][0]  # first image and first face
    pitch, roll, yaw = pose_to_test.reshape(-1)
    assert -10 < pitch < 10
    assert -5 < roll < 5
    assert -10 < yaw < 10


def test_wrongmodelname():
    with pytest.raises(KeyError):
        detector1 = Detector(emotion_model="badmodelname")


def test_nofile():
    with pytest.raises(FileNotFoundError):
        inputFname = os.path.join(get_test_data_path(), "nosuchfile.jpg")
        detector1 = Detector(emotion_model="svm")
        out = detector1.detect_image(inputFname)


# TODO: Refactor this test with test_multiface
def test_detect_image():

    single_face = os.path.join(get_test_data_path(), "input.jpg")
    multi_face = os.path.join(
        get_test_data_path(), "tim-mossholder-hOF1bWoet_Q-unsplash.jpg"
    )
    outputFname = os.path.join(get_test_data_path(), "output.csv")
    detector = Detector()

    # NUM IMAGES < BATCH SIZE
    # Test single face from single image
    out = detector.detect_image(inputFname=single_face)
    assert type(out) == Fex
    # Default detectors return 173 attributes
    assert out.shape == (1, 173)
    assert out.happiness.values[0] > 0

    # Test writing out detection
    out = detector.detect_image(inputFname=single_face, outputFname=outputFname)
    assert out is not None
    assert os.path.exists(outputFname)
    out = pd.read_csv(outputFname)
    assert out.happiness.values[0] > 0

    # Test single face from multiple images
    out = detector.detect_image(inputFname=[single_face, single_face])
    assert out.shape == (2, 173)

    # Test multiple face detection from a single image
    out = detector.detect_image(inputFname=multi_face)
    assert out.shape == (5, 173)

    # Test multi face from multiple images
    out = detector.detect_image(inputFname=[multi_face, multi_face])
    assert out.shape == (10, 173)

    # Test batch failing on batching mis-matched image sizes
    with pytest.raises(ValueError):
        _ = detector.detect_image(inputFname=[single_face, multi_face])

    # If we don't batch then we effectively process them separately and concat the
    # results. So always make sure  the results are sum(faces_in_image for image in
    # images) not num_images or num_faces
    out = detector.detect_image(inputFname=[single_face, multi_face], batch_size=1)
    assert out.shape == (6, 173)
    # NUM IMAGES >= BATCH SIZE
    out = detector.detect_image(inputFname=[single_face] * 6)
    assert out.shape == (6, 173)


def test_multiface():
    # Test multiple faces
    inputFname2 = os.path.join(
        get_test_data_path(), "tim-mossholder-hOF1bWoet_Q-unsplash.jpg"
    )
    img01 = read_pictures([inputFname])
    _, h, w, _ = img01.shape

    img02 = cv2.imread(inputFname2)
    # @tiankang: seems to be a problem with fer
    detector = Detector(
        face_model="RetinaFace",
        emotion_model="fer",
        landmark_model="PFLD",
        au_model="jaanet",
    )
    files, _ = detector.process_frame(img02, 0)
    assert files.shape[0] == 5


def test_detect_video():
    # Test detect video
    detector = Detector(n_jobs=1)
    inputFname = os.path.join(get_test_data_path(), "input.mp4")
    out = detector.detect_video(inputFname=inputFname, skip_frames=24)
    assert len(out) == 3


def test_detect_video_parallel():
    # Test detect video
    detector = Detector()
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
        emotion_model="resmasknet",
        landmark_model="PFLD",
        au_model="jaanet",
    )
    files = detector04.process_frame(img01, 0)
