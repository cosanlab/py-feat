"""
This file defines pytest fixtures: reusable bits of code that are shared between
tests. To use them, just add them as an argument to any test function, e.g.

def test_detect_single_face(default_detector, single_face_img):
    default_detector.detect_image(single_face_img)

"""

from pytest import fixture
import os
from feat.detector import Detector
import numpy as np
from torchvision.io import read_image
import pandas as pd
from feat.data import Fex

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
    params=["svm", "xgb"],
)
def au_model(request):
    """Supported au detectors"""
    return request.param


@fixture(
    scope="module",
    params=["resmasknet", "svm"],
)
def emotion_model(request):
    """Supported emotion detectors"""
    return request.param


@fixture(
    scope="module",
    params=["img2pose", "img2pose-c"],
)
def facepose_model(request):
    """Supported pose detectors"""
    return request.param


# IMAGE AND VIDEO SAMPLES
@fixture(scope="module")
def data_path():
    return os.path.join(os.path.dirname(__file__), "data")


@fixture()
def default_detector():
    """This detector instance is shared across all test in the same file"""
    return Detector()


@fixture(scope="module")
def no_face_img(data_path):
    return os.path.join(data_path, "free-mountain-vector-01.jpg")


@fixture(scope="module")
def single_face_img(data_path):
    return os.path.join(data_path, "single_face.jpg")


@fixture(scope="module")
def single_face_img_data(single_face_img):
    """The actual numpy array of img data"""
    return read_image(single_face_img)


@fixture(scope="module")
def single_face_mov(data_path):
    return os.path.join(data_path, "single_face.mp4")


@fixture(scope="module")
def multi_face_img(data_path):
    return os.path.join(data_path, "multi_face.jpg")


@fixture(scope="module")
def multiple_images_for_batch_testing(data_path):
    from glob import glob

    return list(glob(os.path.join(data_path, "*-ph.jpg")))


@fixture(scope="module")
def imotions_data(data_path):
    FACET_EMOTION_COLUMNS = [
        "Joy",
        "Anger",
        "Surprise",
        "Fear",
        "Contempt",
        "Disgust",
        "Sadness",
        "Confusion",
        "Frustration",
        "Neutral",
        "Positive",
        "Negative",
    ]
    FACET_FACEBOX_COLUMNS = [
        "FaceRectX",
        "FaceRectY",
        "FaceRectWidth",
        "FaceRectHeight",
    ]
    FACET_TIME_COLUMNS = ["Timestamp", "MediaTime", "FrameNo", "FrameTime"]
    FACET_FACEPOSE_COLUMNS = ["Pitch", "Roll", "Yaw"]
    FACET_DESIGN_COLUMNS = ["StimulusName", "SlideType", "EventSource", "Annotation"]

    filename = os.path.join(data_path, "iMotions_Test_v6.txt")
    d = pd.read_csv(filename, skiprows=5, sep="\t")
    cols2drop = [col for col in d.columns if "Intensity" in col]
    d = d.drop(columns=cols2drop)
    d.columns = [col.replace(" Evidence", "") for col in d.columns]
    d.columns = [col.replace(" Degrees", "") for col in d.columns]
    d.columns = [col.replace(" ", "") for col in d.columns]
    au_columns = [col for col in d.columns if "AU" in col]
    df = Fex(
        d,
        filename=filename,
        au_columns=au_columns,
        emotion_columns=FACET_EMOTION_COLUMNS,
        facebox_columns=FACET_FACEBOX_COLUMNS,
        facepose_columns=FACET_FACEPOSE_COLUMNS,
        time_columns=FACET_TIME_COLUMNS,
        design_columns=FACET_DESIGN_COLUMNS,
        detector="FACET",
        sampling_freq=None,
    )
    df["input"] = filename
    return df
