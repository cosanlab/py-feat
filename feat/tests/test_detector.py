#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""

from email.policy import default
from feat.detector import Detector
from feat.data import Fex
from feat.utils import read_pictures
from feat.tests.utils import get_test_data_path
import pandas as pd
import os
import cv2
import numpy as np
import pytest

# inputFname = os.path.join(get_test_data_path(), "input.jpg")
# img01 = read_pictures([inputFname])
# _, h, w, _ = img01.shape


def test_empty_init():
    detector = Detector(
        face_model=None,
        emotion_model=None,
        au_model=None,
        facepose_model=None,
        landmark_model=None,
    )
    assert isinstance(detector, Detector)


def test_detect_single_face(default_detector, single_face_img):
    """Test detection of single face from single image. Default detector returns 173 attributes"""
    out = default_detector.detect_image(single_face_img)

    assert type(out) == Fex
    assert out.shape == (1, 173)
    assert out.happiness.values[0] > 0


def test_detect_read_write(default_detector, single_face_img, data_path):
    """Test detection and writing of results to csv"""

    outputFname = os.path.join(data_path, "output.csv")
    _ = default_detector.detect_image(single_face_img, outputFname=outputFname)
    assert os.path.exists(outputFname)
    loaded = pd.read_csv(outputFname)
    assert loaded.shape == (1, 173)
    assert loaded.happiness.values[0] > 0


def test_detect_multi_face(default_detector, multi_face_img):
    """Test detection of multiple faces from single image"""
    out = default_detector.detect_image(multi_face_img)

    assert type(out) == Fex
    assert out.shape == (5, 173)


def test_detect_single_face_multiple_images(default_detector, single_face_img):
    """Test detection of single face from multiple images"""
    out = default_detector.detect_image([single_face_img, single_face_img])
    assert out.shape == (2, 173)


def test_detect_multi_face_multiple_images(default_detector, multi_face_img):
    """Test detection of multiple faces from multiple images"""
    out = default_detector.detect_image([multi_face_img, multi_face_img])
    assert out.shape == (10, 173)


def test_detect_images_with_batching(default_detector, single_face_img):
    """Test if batching works by passing in more images than the default batch size"""

    # Num images > default batch size of 5 so more than one batch to process
    out = default_detector.detect_image([single_face_img] * 6)
    assert out.shape == (6, 173)


def test_detect_mismatch_image_sizes(default_detector, single_face_img, multi_face_img):
    """We don't currently support images of different dimensions in a single batch, but
    do if we don't batch them"""

    # Fail with default batch size of 5
    with pytest.raises(ValueError):
        _ = default_detector.detect_image([multi_face_img, single_face_img])

    # But not if we explicitly don't batch as each image is processed separately and
    # then results are concatenated
    out = default_detector.detect_image(
        inputFname=[single_face_img, multi_face_img], batch_size=1
    )
    assert out.shape == (6, 173)


def test_faceboxes(default_detector, single_face_img):
    """Since .detect_faces expects image data to be loaded already, this also serves as
    a test for .read_images"""
    default_detector.change_model(face_model="FaceBoxes")
    img_data = default_detector.read_images(single_face_img)
    # Returns number of images length list
    out = default_detector.detect_faces(img_data)[0]
    # Returns bounding boxes for each face in image as list
    assert all(e is not None for e in out[0])
    assert len(out[0]) == 5
    bbox_x = out[0][0]
    assert 180 < bbox_x < 200


def test_retinaface(default_detector, single_face_img_data):
    default_detector.change_model(face_model="RetinaFace")
    # This time just use preloaded image data for convenience
    out = default_detector.detect_faces(single_face_img_data)[0]
    assert all(e is not None for e in out[0])
    assert len(out[0]) == 5
    bbox_x = out[0][0]
    assert 180 < bbox_x < 200


def test_mtcnn(default_detector, single_face_img_data):
    default_detector.change_model(face_model="MTCNN")
    out = default_detector.detect_faces(single_face_img_data)[0]
    assert all(e is not None for e in out[0])
    assert len(out[0]) == 5
    bbox_x = out[0][0]
    assert 180 < bbox_x < 200


def test_img2pose(default_detector, single_face_img_data):
    # Test that both face detection and facepose estimation work
    default_detector.change_model(face_model="img2pose", facepose_model="img2pose")

    # Face detection
    faces = default_detector.detect_faces(single_face_img_data)[0]
    assert all(e is not None for e in faces[0])
    assert len(faces[0]) == 5
    bbox_x = faces[0][0]
    assert 180 < bbox_x < 200

    # Pose estimation
    poses = default_detector.detect_facepose(single_face_img_data)[0]
    pose_to_test = poses[0][0]  # first image and first face
    pitch, roll, yaw = pose_to_test.reshape(-1)
    assert -10 < pitch < 10
    assert -5 < roll < 5
    assert -10 < yaw < 10

    # Test using mismatched face and pose model
    with pytest.raises(ValueError):
        default_detector.change_model(face_model="MTCNN", facepost_model="img2pose")


def test_mobilefacenet(default_detector, single_face_img):
    default_detector.change_model(
        face_model="RetinaFace", landmark_model="MobileFaceNet"
    )
    img_data = default_detector.read_images(single_face_img)
    _, h, w, _ = img_data.shape

    bboxes = default_detector.detect_faces(img_data)
    landmarks = default_detector.detect_landmarks(img_data, bboxes)[0]
    assert landmarks[0].shape == (68, 2)
    assert (
        np.any(landmarks[0][:, 0] > 0)
        and np.any(landmarks[0][:, 0] < w)
        and np.any(landmarks[0][:, 1] > 0)
        and np.any(landmarks[0][:, 1] < h)
    )


def test_mobilenet(default_detector, single_face_img):
    default_detector.change_model(face_model="RetinaFace", landmark_model="MobileNet")
    img_data = default_detector.read_images(single_face_img)
    _, h, w, _ = img_data.shape

    bboxes = default_detector.detect_faces(img_data)
    landmarks = default_detector.detect_landmarks(img_data, bboxes)[0]
    assert landmarks[0].shape == (68, 2)
    assert (
        np.any(landmarks[0][:, 0] > 0)
        and np.any(landmarks[0][:, 0] < w)
        and np.any(landmarks[0][:, 1] > 0)
        and np.any(landmarks[0][:, 1] < h)
    )


def test_pfld(default_detector, single_face_img):
    default_detector.change_model(face_model="RetinaFace", landmark_model="PFLD")
    img_data = default_detector.read_images(single_face_img)
    _, h, w, _ = img_data.shape

    bboxes = default_detector.detect_faces(img_data)
    landmarks = default_detector.detect_landmarks(img_data, bboxes)[0]
    assert landmarks[0].shape == (68, 2)
    assert (
        np.any(landmarks[0][:, 0] > 0)
        and np.any(landmarks[0][:, 0] < w)
        and np.any(landmarks[0][:, 1] > 0)
        and np.any(landmarks[0][:, 1] < h)
    )


def test_jaanet(default_detector, single_face_img):
    default_detector.change_model(
        face_model="RetinaFace",
        emotion_model=None,
        landmark_model="MobileFaceNet",
        au_model="jaanet",
    )
    img_data = default_detector.read_images(single_face_img)

    detected_faces = default_detector.detect_faces(img_data)
    landmarks = default_detector.detect_landmarks(img_data, detected_faces)
    aus = default_detector.detect_aus(img_data, landmarks)

    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 12


# TODO: Continue updating tests from here
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
    files, _ = detector.process_frame([img02], inputFname2, counter=0)
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
    inputFname = os.path.join(get_test_data_path(), "input.jpg")
    img01 = read_pictures([inputFname])
    detector04 = Detector(
        face_model="RetinaFace",
        emotion_model="resmasknet",
        landmark_model="PFLD",
        au_model="jaanet",
    )
    files = detector04.process_frame([img01], inputFname, counter=0)
