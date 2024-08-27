import pytest
from feat.au_detectors.StatLearning.SL_test import XGBClassifier
from feat.FastDetector import FastDetector
from feat.data import Fex
from huggingface_hub import PyTorchModelHubMixin


@pytest.mark.usefixtures(
    "single_face_img", "single_face_img_data", "multi_face_img", "multi_face_img_data"
)
class Test_Fast_Detector:
    """Test new single model detector"""

    detector = FastDetector(device="cpu")

    def test_init(self):
        assert isinstance(self.detector, PyTorchModelHubMixin)

    def test_detect_image(self, single_face_img):
        fex = self.detector.detect_image(single_face_img)
        assert isinstance(fex, Fex)

        # No bad predictions on default image
        assert not fex.isnull().any().any()

        # Default output is 686 features
        assert fex.shape == (1, 686)

        # Bounding box
        assert 150 < fex.FaceRectX[0] < 180
        assert 125 < fex.FaceRectY[0] < 140

        # Jin is smiling
        assert fex.happiness[0] > 0.8

        # AU checks; TODO: Add more
        assert fex.aus.AU20[0] > 0.8
