"""
This file defines pytest fixtures: reusable bits of code that are shared between
tests. To use them, just add them as an argument to any test function, e.g.

def test_detect_single_face(default_detector, single_face_img):
    default_detector.detect_image(single_face_img)

"""

from pytest import fixture
import os
from feat.detector import Detector
from feat.utils import read_pictures
import numpy as np

# AU constants for plotting
@fixture(scope="module")
def au():
    return np.ones(20)


@fixture(scope="module")
def au2():
    return np.ones(20) * 3


# DETECTOR COMBINATIONS
@fixture(
    scope="module",
    params=["retinaface", "faceboxes", "mtcnn", "img2pose", "img2pose-c"],
)
def face_model(request):
    """Supported face detectors"""
    return request.param


@fixture(
    scope="module",
    params=["mobilenet", "mobilefacenet", "pfld"],
)
def landmark_model(request):
    """Supported landmark detectors"""
    return request.param


@fixture(
    scope="module",
    params=["rf", "svm", "logistic", "jaanet", "drml"],
)
def au_model(request):
    """Supported au detectors"""
    return request.param


@fixture(
    scope="module",
    params=["resmasknet", "rf", "svm", "fer"],
)
def emotion_model(request):
    """Supported emotion detectors"""
    return request.param


@fixture(
    scope="module",
    params=["pnp", "img2pose", "img2pose-c"],
)
def facepose_model(request):
    """Supported pose detectors"""
    return request.param


# IMAGE AND VIDEO SAMPLES
@fixture(scope="module")
def data_path():
    return os.path.join(os.path.dirname(__file__), "data")


@fixture(scope="module")
def default_detector():
    """This detector instance is shared across all test in the same file"""
    return Detector()


@fixture(scope="module")
def single_face_img(data_path):
    return os.path.join(data_path, "single_face.jpg")


@fixture(scope="module")
def single_face_img_data(single_face_img):
    """The actual numpy array of img data"""
    return read_pictures([single_face_img])


@fixture(scope="module")
def single_face_mov(data_path):
    return os.path.join(data_path, "single_face.mp4")


@fixture(scope="module")
def multi_face_img(data_path):
    return os.path.join(data_path, "multi_face.jpg")
