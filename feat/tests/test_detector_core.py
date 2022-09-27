from feat.detector import Detector
from feat.data import Fex
from feat.utils.io import get_test_data_path
import os
import pytest


def test_empty_init():
    """Should fail if not models provided"""
    with pytest.raises(ValueError):
        _ = Detector(
            face_model=None,
            emotion_model=None,
            au_model=None,
            facepose_model=None,
            landmark_model=None,
        )


def test_init_with_wrongmodelname():
    """Should fail with unsupported model name"""
    with pytest.raises(ValueError):
        _ = Detector(emotion_model="badmodelname")


def test_nofile(default_detector):
    """Should fail with missing data"""
    with pytest.raises((FileNotFoundError, RuntimeError)):
        inputFname = os.path.join(get_test_data_path(), "nosuchfile.jpg")
        _ = default_detector.detect_image(inputFname)


# Single images
def test_detect_single_img_single_face(default_detector, single_face_img):
    """Test detection of single face from single image. Default detector returns 173 attributes"""
    out = default_detector.detect_image(single_face_img)
    assert type(out) == Fex
    assert out.shape == (1, 173)
    assert out.happiness.values[0] > 0


def test_detect_single_img_multi_face(default_detector, multi_face_img):
    """Test detection of multiple faces from single image"""
    out = default_detector.detect_image(multi_face_img)
    assert type(out) == Fex
    assert out.shape == (5, 173)


# Multiple images
def test_detect_multi_img_single_face(default_detector, single_face_img):
    """Test detection of single face from multiple images"""
    out = default_detector.detect_image([single_face_img, single_face_img])
    assert out.shape == (2, 173)


def test_detect_multi_img_multi_face(default_detector, multi_face_img):
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


# FIXME: This only works for the default detector and produces errors for some other
# models, but we don't have combination testing yet:
# Defaults:
# face_model="retinaface",
# landmark_model="mobilenet",
# au_model="svm",
# emotion_model="resmasknet",
# facepose_model="img2pose",
def test_detect_video(default_detector, single_face_mov):
    """Test detection on video file"""
    out = default_detector.detect_video(single_face_mov, skip_frames=24)
    assert len(out) == 3
    assert out.happiness.values.max() > 0
