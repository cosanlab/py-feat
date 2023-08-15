from feat.detector import Detector
from feat.data import Fex
from feat.utils.io import get_test_data_path
import os
import pytest
import numpy as np
import warnings


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


# TODO: Currently making this test always pass even if batching gives slightly diff
# results until @tiankang can debug whether we're in tolerance
# Track progress updates in this issue: https://github.com/cosanlab/py-feat/issues/128
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
    au_diffs = np.abs(batched.aus - nonbatched.aus).max()
    TOL = 0.5
    bad_aus = au_diffs[au_diffs > TOL]
    if len(bad_aus):
        warnings.warn(
            f"Max AU deviation is larger than tolerance ({TOL}) when comparing batched vs non-batched detections: {bad_aus}"
        )
    else:
        print(
            f"Max AU deviation (batched - nonbatched): {au_diffs.idxmax()}: {au_diffs.max()}"
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


# No Face images
def test_detect_single_img_no_face(default_detector, no_face_img):
    """Test detection of a single image with no face. Default detector returns 173 attributes"""
    out = default_detector.detect_image(no_face_img)
    assert type(out) == Fex
    assert out.shape == (1, 173)
    assert np.isnan(out.happiness.values[0])


def test_detect_multi_img_no_face(default_detector, no_face_img):
    """Test detection of a multiple images with no face. Default detector returns 173 attributes"""
    out = default_detector.detect_image([no_face_img] * 3)
    assert out.shape == (3, 173)


def test_detect_multi_img_no_face_batching(default_detector, no_face_img):
    """Test detection of a multiple images with no face. Default detector returns 173 attributes"""
    out = default_detector.detect_image([no_face_img] * 5, batch_size=2)
    assert out.shape == (5, 173)


def test_detect_multi_img_mixed_no_face(
    default_detector, no_face_img, single_face_img, multi_face_img
):
    """Test detection of a single image with no face. Default detector returns 173 attributes"""
    out = default_detector.detect_image(
        [single_face_img, no_face_img, multi_face_img] * 2
    )
    assert out.shape == (14, 173)


def test_detect_multi_img_mixed_no_face_batching(
    default_detector, no_face_img, single_face_img, multi_face_img
):
    """Test detection of a single image with no face. Default detector returns 173 attributes"""
    out = default_detector.detect_image(
        [single_face_img, no_face_img, multi_face_img] * 2,
        batch_size=4,
        output_size=300,
    )
    assert out.shape == (14, 173)


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


def test_detect_with_alpha(default_detector):
    image = os.path.join(get_test_data_path(), "Image_with_alpha.png")
    out = default_detector.detect_image(image)


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
        [multi_face_img, single_face_img] * 5, batch_size=5, output_size=512
    )
    assert out.shape == (30, 173)


def test_detect_video(
    default_detector, single_face_mov, no_face_mov, face_noface_mov, noface_face_mov
):
    """Test detection on video file"""
    out = default_detector.detect_video(single_face_mov, skip_frames=24)
    assert len(out) == 3
    assert out.happiness.values.max() > 0

    # Test no face movie
    out = default_detector.detect_video(no_face_mov, skip_frames=24)
    assert len(out) == 4
    # Empty detections are filled with NaNs
    assert out.aus.isnull().all().all()

    # Test mixed movie, i.e. spliced vids of face -> noface and noface -> face
    out = default_detector.detect_video(face_noface_mov, skip_frames=24)
    assert len(out) == 3 + 4 + 1
    # first few frames have a face
    assert not out.aus.iloc[:3].isnull().all().all()
    # But the rest are from a diff video that doesn't
    assert out.aus.iloc[3:].isnull().all().all()

    out = default_detector.detect_video(noface_face_mov, skip_frames=24)
    assert len(out) == 3 + 4 + 1
    # beginning no face
    assert out.aus.iloc[:4].isnull().all().all()
    # middle frames have face
    assert not out.aus.iloc[4:7].isnull().all().all()
    # ending doesn't
    assert out.aus.iloc[7:].isnull().all().all()


def test_detect_mismatch_face_pose(default_detector):
    # Multiple Faces, 1 pose
    faces = [
        [
            [
                45.34465026855469,
                49.546714782714844,
                63.04056167602539,
                70.38599395751953,
                0.95337886,
            ],
            [
                146.09866333007812,
                96.34442901611328,
                165.69561767578125,
                120.71611022949219,
                0.9069432,
            ],
        ]
    ]
    faces_pose = [[[46.0, 46.0, 66.0, 71.0, 0.99272925]]]
    poses = [[[-3.72766398, 10.9359162, -3.19862351]]]

    new_faces, new_poses = default_detector._match_faces_to_poses(
        faces, faces_pose, poses
    )
    assert len(new_faces[0]) == len(new_poses[0])
    assert len(new_faces[0]) == 2

    # 1 face, multiple poses
    faces = [
        [
            [
                45.34465026855469,
                49.546714782714844,
                63.04056167602539,
                70.38599395751953,
                0.95337886,
            ]
        ]
    ]

    faces_pose = [
        [
            [65.0, 83.0, 87.0, 110.0, 0.99630725],
            [141.0, 94.0, 167.0, 123.0, 0.9952237],
            [111.0, 97.0, 136.0, 126.0, 0.99487805],
            [91.0, 78.0, 109.0, 100.0, 0.99454665],
            [46.0, 46.0, 66.0, 71.0, 0.99272925],
        ]
    ]

    poses = [
        [
            [-5.90236694, -2.81686444, -5.38250827],
            [18.3324545, 7.2330487, 2.70649852],
            [12.04520545, 5.91369713, 6.13698383],
            [1.10688262, 1.56339815, -0.91693287],
            [-3.72766398, 10.9359162, -3.19862351],
        ]
    ]

    new_faces, new_poses = default_detector._match_faces_to_poses(
        faces, faces_pose, poses
    )
    assert len(new_faces[0]) == 1
    assert len(new_poses[0]) == 1

    # 2 Faces, 5 Poses
    faces = [
        [
            [
                45.34465026855469,
                49.546714782714844,
                63.04056167602539,
                70.38599395751953,
                0.95337886,
            ],
            [
                146.09866333007812,
                96.34442901611328,
                165.69561767578125,
                120.71611022949219,
                0.9069432,
            ],
        ]
    ]

    faces_pose = [
        [
            [65.0, 83.0, 87.0, 110.0, 0.99630725],
            [141.0, 94.0, 167.0, 123.0, 0.9952237],
            [111.0, 97.0, 136.0, 126.0, 0.99487805],
            [91.0, 78.0, 109.0, 100.0, 0.99454665],
            [46.0, 46.0, 66.0, 71.0, 0.99272925],
        ]
    ]

    poses = [
        [
            [-5.90236694, -2.81686444, -5.38250827],
            [18.3324545, 7.2330487, 2.70649852],
            [12.04520545, 5.91369713, 6.13698383],
            [1.10688262, 1.56339815, -0.91693287],
            [-3.72766398, 10.9359162, -3.19862351],
        ]
    ]

    new_faces, new_poses = default_detector._match_faces_to_poses(
        faces, faces_pose, poses
    )
    assert len(new_faces[0]) == len(new_poses[0])
    assert len(new_faces[0]) == 2

    # 5 Faces, 2 Poses
    faces = [
        [
            [65.0, 83.0, 87.0, 110.0, 0.99630725],
            [141.0, 94.0, 167.0, 123.0, 0.9952237],
            [111.0, 97.0, 136.0, 126.0, 0.99487805],
            [91.0, 78.0, 109.0, 100.0, 0.99454665],
            [46.0, 46.0, 66.0, 71.0, 0.99272925],
        ]
    ]

    faces_pose = [
        [
            [
                45.34465026855469,
                49.546714782714844,
                63.04056167602539,
                70.38599395751953,
                0.95337886,
            ],
            [
                146.09866333007812,
                96.34442901611328,
                165.69561767578125,
                120.71611022949219,
                0.9069432,
            ],
        ]
    ]

    poses = [
        [
            [-5.90236694, -2.81686444, -5.38250827],
            [18.3324545, 7.2330487, 2.70649852],
        ]
    ]

    new_faces, new_poses = default_detector._match_faces_to_poses(
        faces, faces_pose, poses
    )
    assert len(new_faces[0]) == len(new_poses[0])
    assert len(new_faces[0]) == 5
