import pytest
from feat.au_detectors.StatLearning.SL_test import XGBClassifier
from feat.FastDetector import FastDetector
from feat.data import Fex
from huggingface_hub import PyTorchModelHubMixin
import numpy as np
from feat.utils.io import get_test_data_path



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
            
        
