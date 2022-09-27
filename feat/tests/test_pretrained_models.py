from feat.detector import Detector
from feat.data import Fex
import pytest
import numpy as np
from torchvision.io import read_image


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


def test_mobilefacenet(default_detector, single_face_img, single_face_img_data):

    _, h, w = read_image(single_face_img).shape

    default_detector.change_model(
        face_model="RetinaFace", landmark_model="MobileFaceNet"
    )
    bboxes = default_detector.detect_faces(single_face_img_data)
    landmarks = default_detector.detect_landmarks(single_face_img_data, bboxes)[0]
    assert landmarks[0].shape == (68, 2)
    assert (
        np.any(landmarks[0][:, 0] > 0)
        and np.any(landmarks[0][:, 0] < w)
        and np.any(landmarks[0][:, 1] > 0)
        and np.any(landmarks[0][:, 1] < h)
    )


def test_mobilenet(default_detector, single_face_img, single_face_img_data):

    _, h, w = read_image(single_face_img).shape

    default_detector.change_model(face_model="RetinaFace", landmark_model="MobileNet")

    bboxes = default_detector.detect_faces(single_face_img_data)
    landmarks = default_detector.detect_landmarks(single_face_img_data, bboxes)[0]
    assert landmarks[0].shape == (68, 2)
    assert (
        np.any(landmarks[0][:, 0] > 0)
        and np.any(landmarks[0][:, 0] < w)
        and np.any(landmarks[0][:, 1] > 0)
        and np.any(landmarks[0][:, 1] < h)
    )


def test_pfld(default_detector, single_face_img, single_face_img_data):

    _, h, w = read_image(single_face_img).shape
    default_detector.change_model(face_model="RetinaFace", landmark_model="PFLD")

    bboxes = default_detector.detect_faces(single_face_img_data)
    landmarks = default_detector.detect_landmarks(single_face_img_data, bboxes)[0]
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

    bboxes = default_detector.detect_faces(single_face_img_data)
    landmarks = default_detector.detect_landmarks(single_face_img_data, bboxes)
    aus = default_detector.detect_aus(single_face_img_data, landmarks)

    assert np.sum(np.isnan(aus)) == 0
    assert aus[0].shape[-1] == 12


def test_logistic(default_detector, single_face_img_data):

    default_detector.change_model(
        face_model="RetinaFace",
        landmark_model="MobileFaceNet",
        au_model="logistic",
    )

    detected_faces = default_detector.detect_faces(single_face_img_data)
    landmarks = default_detector.detect_landmarks(single_face_img_data, detected_faces)
    aus = default_detector.detect_aus(single_face_img_data, landmarks=landmarks)

    assert np.sum(np.isnan(aus)) == 0
    assert aus[0].shape[-1] == 20


def test_svm(default_detector, single_face_img_data):

    default_detector.change_model(
        face_model="RetinaFace",
        landmark_model="MobileFaceNet",
        au_model="svm",
    )

    detected_faces = default_detector.detect_faces(single_face_img_data)
    landmarks = default_detector.detect_landmarks(single_face_img_data, detected_faces)
    aus = default_detector.detect_aus(single_face_img_data, landmarks=landmarks)

    assert np.sum(np.isnan(aus)) == 0
    assert aus[0].shape[-1] == 20


def test_resmasknet(default_detector, single_face_img):
    default_detector.change_model(emotion_model="resmasknet")
    out = default_detector.detect_image(single_face_img)
    assert out.emotions["happiness"].values > 0.5


# FIXME: fails on init because doesn't support device kwarg
def test_emotionsvm(default_detector, single_face_img):
    default_detector.change_model(emotion_model="svm")
    out = default_detector.detect_image(single_face_img)
    assert out.emotions["happiness"].values > 0.5


# FIXME: error in call to .predict where list of landmarks is being caste to float32
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
