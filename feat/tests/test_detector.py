from feat.detector import Detector
from feat.data import Fex
from feat.utils.io import get_test_data_path
import pandas as pd
import os
import numpy as np
import pytest


@pytest.mark.skip(
    reason="This tests all 900 model detector combinations which takes ~20-30min. Run locally using test-detector-combos branch which will output results to file."
)
def test_detector_combos(
    face_model, landmark_model, au_model, emotion_model, facepose_model, single_face_img
):
    """Builds a grid a of tests using all supported detector combinations defined in
    conftest.py"""

    # Test init and basic detection
    detector = Detector(
        face_model=face_model,
        landmark_model=landmark_model,
        au_model=au_model,
        emotion_model=emotion_model,
        facepose_model=facepose_model,
    )
    out = detector.detect_image(single_face_img)
    assert type(out) == Fex
    assert out.shape[0] == 1


def test_empty_init():

    with pytest.raises(ValueError):
        _ = Detector(
            face_model=None,
            emotion_model=None,
            au_model=None,
            facepose_model=None,
            landmark_model=None,
        )


def test_init_with_wrongmodelname():
    with pytest.raises(ValueError):
        _ = Detector(emotion_model="badmodelname")


def test_nofile(default_detector):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        inputFname = os.path.join(get_test_data_path(), "nosuchfile.jpg")
        _ = default_detector.detect_image(inputFname)


def test_detect_single_face(default_detector, single_face_img):
    """Test detection of single face from single image. Default detector returns 173 attributes"""
    out = default_detector.detect_image(single_face_img)

    assert type(out) == Fex
    assert out.shape == (1, 173)
    assert out.happiness.values[0] > 0


def test_detect_read_write(default_detector, single_face_img, data_path):
    """Test detection and writing of results to csv"""

    fex = default_detector.detect_image(single_face_img)
    assert fex is not None
    assert isinstance(fex, Fex)
    assert fex.shape == (1, 173)
    assert fex.happiness.values[0] > 0


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

    out = default_detector.detect_image([single_face_img] * 6, batch_size=5)
    assert out.shape == (6, 173)


def test_detect_mismatch_image_sizes(default_detector, single_face_img, multi_face_img):
    """Test detection on multiple images of different sizes with and without batching"""

    out = default_detector.detect_image([multi_face_img, single_face_img])
    assert out.shape == (6, 173)

    out = default_detector.detect_image(
        [multi_face_img, single_face_img] * 5, batch_size=5
    )
    assert out.shape == (30, 173)


def test_faceboxes(default_detector, single_face_img_data):

    default_detector.change_model(face_model="FaceBoxes")
    out = default_detector.detect_faces(single_face_img_data)
    assert 180 < out[0][0][0] < 200


def test_retinaface(default_detector, single_face_img_data):

    default_detector.change_model(face_model="RetinaFace")
    out = default_detector.detect_faces(single_face_img_data)
    assert 180 < out[0][0][0] < 200


# FIXME: @tiankang MTCNN's face rect is not the same as faceboxes and retinaface
# the bounding box x coord of the bounding box is > 200
def test_mtcnn(default_detector, single_face_img_data):
    default_detector.change_model(face_model="MTCNN")
    out = default_detector.detect_faces(single_face_img_data)
    assert 180 < out[0][0][0] < 200


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


def test_jaanet(default_detector, single_face_img_data):
    default_detector.change_model(
        face_model="RetinaFace",
        landmark_model="MobileFaceNet",
        au_model="jaanet",
    )

    detected_faces = default_detector.detect_faces(single_face_img_data)
    landmarks = default_detector.detect_landmarks(single_face_img_data, detected_faces)
    aus = default_detector.detect_aus(single_face_img_data, landmarks)

    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 12


def test_logistic(default_detector, single_face_img_data):
    default_detector.change_model(
        face_model="RetinaFace",
        landmark_model="MobileFaceNet",
        au_model="logistic",
    )

    detected_faces = default_detector.detect_faces(single_face_img_data)
    landmarks = default_detector.detect_landmarks(single_face_img_data, detected_faces)
    # TODO: Add explanation of why we need to pass the output of ._batch_hog() to
    # .detect_aus() for this model but not others
    hogs, new_lands = default_detector._batch_hog(
        frames=single_face_img_data, detected_faces=detected_faces, landmarks=landmarks
    )

    aus = default_detector.detect_aus(frame=hogs, landmarks=new_lands)

    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 20


def test_svm(default_detector, single_face_img_data):
    default_detector.change_model(
        face_model="RetinaFace",
        landmark_model="MobileFaceNet",
        au_model="svm",
    )

    detected_faces = default_detector.detect_faces(single_face_img_data)
    landmarks = default_detector.detect_landmarks(single_face_img_data, detected_faces)
    # TODO: Add explanation of why we need to pass the output of ._batch_hog() to
    # .detect_aus() for this model but not others
    hogs, new_lands = default_detector._batch_hog(
        frames=single_face_img_data, detected_faces=detected_faces, landmarks=landmarks
    )

    aus = default_detector.detect_aus(frame=hogs, landmarks=new_lands)

    assert np.sum(np.isnan(aus)) == 0
    assert aus.shape[-1] == 20


def test_resmasknet(default_detector, single_face_img):
    default_detector.change_model(emotion_model="resmasknet")
    out = default_detector.detect_image(single_face_img)
    assert out.emotions["happiness"].values > 0.5


def test_emotionsvm(default_detector, single_face_img):
    default_detector.change_model(emotion_model="svm")
    out = default_detector.detect_image(single_face_img)
    assert out.emotions["happiness"].values > 0.5


def test_pnp(default_detector, single_face_img_data):
    # Test that facepose can be estimated properly using landmarks + pnp algorithm
    default_detector.change_model(
        face_model="RetinaFace", landmark_model="MobileFaceNet", facepose_model="PnP"
    )
    bboxes = default_detector.detect_faces(frame=single_face_img_data)
    lms = default_detector.detect_landmarks(
        frame=single_face_img_data, detected_faces=bboxes
    )
    poses = default_detector.detect_facepose(frame=single_face_img_data, landmarks=lms)
    pose_to_test = poses[0][0]  # first image and first face
    pitch, roll, yaw = pose_to_test.reshape(-1)
    assert -10 < pitch < 10
    assert -5 < roll < 5
    assert -10 < yaw < 10


def test_detect_video(default_detector, single_face_mov):
    out = default_detector.detect_video(single_face_mov, skip_frames=24)
    assert len(out) == 3
    assert out.happiness.values.max() > 0
