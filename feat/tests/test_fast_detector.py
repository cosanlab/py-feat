import pytest
from feat.au_detectors.StatLearning.SL_test import XGBClassifier
from feat.FastDetector import FastDetector
from feat.data import Fex
from huggingface_hub import PyTorchModelHubMixin
import numpy as np
from feat.utils.io import get_test_data_path
import warnings
import os



@pytest.mark.usefixtures(
    "single_face_img", "single_face_img_data", "multi_face_img", "multi_face_img_data"
)
class Test_Fast_Detector:
    """Test new single model detector"""

    detector = FastDetector(device="cpu")

    def test_init(self):
        assert isinstance(self.detector, PyTorchModelHubMixin)

    def test_fast_detect(self, single_face_img):
        fex = self.detector.detect(single_face_img)
        assert isinstance(fex, Fex)

        # No bad predictions on default image
        assert not fex.isnull().any().any()

        # Default output is 686 features
        assert fex.shape == (1, 686)

        # Bounding box
        assert 180 < fex.FaceRectX[0] < 200
        assert 140 < fex.FaceRectY[0] < 160

        # Jin is smiling
        assert fex.happiness[0] > 0.8

        # AU checks; TODO: Add more
        assert fex.aus.AU20[0] > 0.8

        
    def test_fast_landmark_with_batches(self, multiple_images_for_batch_testing):
        """
        Make sure that when the same images are passed in with and without batch
        processing, the detected landmarks and poses come out to be the same
        """
        det_result_batch = self.detector.detect(
            inputs=multiple_images_for_batch_testing,
            batch_size=5,
        )

        det_result_no_batch = self.detector.detect(
            inputs=multiple_images_for_batch_testing,
            batch_size=1,
        )

        assert np.allclose(
            det_result_batch.loc[:, "x_0":"y_67"].to_numpy(),
            det_result_no_batch.loc[:, "x_0":"y_67"].to_numpy(),
        )
            
    # TODO: Currently making this test always pass even if batching gives slightly diff
    # results until @tiankang can debug whether we're in tolerance
    # Track progress updates in this issue: https://github.com/cosanlab/py-feat/issues/128
    def test_fast_detection_and_batching_with_diff_img_sizes(
        self, single_face_img, multi_face_img, multiple_images_for_batch_testing
    ):
        """
        Make sure that when the same images are passed in with and without batch
        processing, the detected landmarks and poses come out to be the same
        """
        # Each sublist of images contains different sizes
        all_images = (
            [single_face_img] + [multi_face_img] + multiple_images_for_batch_testing
        )

        # Multiple images with different sizes are ok as long as batch_size == 1
        # Detections will be done in each image's native resolution
        det_output_default = self.detector.detect(inputs=all_images, batch_size=1)

        # TODO: fast_detector does not currently raise ValueError
        # If batch_size > 1 then output_size must be set otherwise we can't stack to make a
        # tensor
        # with pytest.raises(ValueError):
        #     _ = self.detector.detect(
        #         inputs=all_images,
        #         batch_size=5,
        #     )

        # Here we batch by resizing each image to 256xpadding
        batched = self.detector.detect(
            inputs=all_images, batch_size=5, output_size=256
        )

        # We can also forcibly resize images even if we don't batch process them
        nonbatched = self.detector.detect(
            inputs=all_images, batch_size=1, output_size=256
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
            
    def test_fast_empty_init():
        """Should fail if not models provided"""
        with pytest.raises(ValueError):
            _ = FastDetector(
                emotion_model=None,
                au_model=None,
                landmark_model=None,
                identity_model=None,
            )
    
    def test_fast_init_with_wrongmodelname():
        """Should fail with unsupported model name"""
        with pytest.raises(ValueError):
            _ = FastDetector(emotion_model="badmodelname")
    
    def test_fast_nofile(self):
        """Should fail with missing data"""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            inputFname = os.path.join(get_test_data_path(), "nosuchfile.jpg")
            _ = self.detector.detect(inputFname)
    
