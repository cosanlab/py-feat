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
