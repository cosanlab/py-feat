from feat.detector import Detector
from feat.data import Fex
from feat.utils.io import get_test_data_path
import os
import pytest
import numpy as np


def test_landmark_with_batches(multiple_images_for_batch_testing):
    """
    Make sure that when the same images are passed in with and without batch
    processing, the detected landmarks and poses come out to be the same
    """
    detector = Detector()
    det_result_batch = detector.detect_image(
        input_file_list=multiple_images_for_batch_testing,
        batch_size=5,
    )

    det_result_no_batch = detector.detect_image(
        input_file_list=multiple_images_for_batch_testing,
        batch_size=1,
    )

    assert np.allclose(
        det_result_batch.loc[:, "x_0":"y_67"].to_numpy(),
        det_result_no_batch.loc[:, "x_0":"y_67"].to_numpy(),
    )


def test_detection_and_batching_with_diff_img_sizes(
    single_face_img, multi_face_img, multiple_images_for_batch_testing
):
    """
    Make sure that when the same images are passed in with and without batch
    processing, the detected landmarks and poses come out to be the same
    """
    # Each sublist of images contains different sizes
    all_images = (
        [single_face_img] + [multi_face_img] + multiple_images_for_batch_testing
    )

    detector = Detector()

    # Multiple images with different sizes are ok as long as batch_size == 1
    # Detections will be done in each image's native resolution
    det_output_default = detector.detect_image(input_file_list=all_images, batch_size=1)

    # If batch_size > 1 then output_size must be set otherwise we can't stack to make a
    # tensor
    with pytest.raises(ValueError):
        _ = detector.detect_image(
            input_file_list=all_images,
            batch_size=5,
        )

    # Here we batch by resizing each image to 256xpadding
    batched = detector.detect_image(
        input_file_list=all_images, batch_size=5, output_size=256
    )

    # We can also forcibly resize images even if we don't batch process them
    nonbatched = detector.detect_image(
        input_file_list=all_images, batch_size=1, output_size=256
    )

    # To make sure that resizing doesn't interact unexpectedly with batching, we should
    # check that the detections we get back for the same sized images are the same when
    # processed as a batch or serially. We check each column separately
    bad_cols = []
    for col in batched.columns:
        bcol, nbcol = batched[col].to_numpy(), nonbatched[col].to_numpy()
        try:
            if not np.allclose(bcol, nbcol):
                bad_cols.append(col)
        except TypeError as _:
            if not all(bcol == nbcol):
                bad_cols.append(col)

    if len(bad_cols):
        if len(bad_cols) > 1 or bad_cols[0] != "frame":
            raise AssertionError(
                f"Running the list of images resized to 256 returns different detections when running as a batch vs serially. The columns with different detection include: {bad_cols}\n See batched vs non-batched cols:\n{batched[bad_cols]}\n{nonbatched[bad_cols]}\n"
            )


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
        [multi_face_img, single_face_img] * 5, batch_size=5, output_size=256
    )
    assert out.shape == (30, 173)


def test_detect_video(default_detector, single_face_mov):
    """Test detection on video file"""
    out = default_detector.detect_video(single_face_mov, skip_frames=24)
    assert len(out) == 3
    assert out.happiness.values.max() > 0
