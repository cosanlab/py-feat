from feat.detector import Detector
from feat.data import Fex
import pytest
import numpy as np
from torchvision.io import read_image


@pytest.mark.skip(
    reason="This tests ALL model detector combinations which takes ~20-30min. Run locally by commenting this line and using pytest -k 'test_detector_combos'."
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


@pytest.mark.usefixtures("default_detector", "single_face_img_data")
class Test_Face_Models:
    """Test all pretrained face models"""

    def test_retinaface(self, default_detector, single_face_img_data):

        default_detector.change_model(face_model="RetinaFace")
        out = default_detector.detect_faces(single_face_img_data)
        assert 180 < out[0][0][0] < 200

    def test_faceboxes(self, default_detector, single_face_img_data):

        default_detector.change_model(face_model="FaceBoxes")
        out = default_detector.detect_faces(single_face_img_data)
        assert 180 < out[0][0][0] < 200

    def test_mtcnn(self, default_detector, single_face_img_data):

        default_detector.change_model(face_model="MTCNN")
        out = default_detector.detect_faces(single_face_img_data)
        # Mtcnn is a bit less accurate
        assert 180 < out[0][0][0] < 205

    @pytest.mark.skip("TODO")
    def test_img2pose_face(self, default_detector, single_face_img_data):
        pass

    @pytest.mark.skip("TODO")
    def test_img2pose_c_face(self, default_detector, single_face_img_data):
        pass


@pytest.mark.usefixtures("default_detector", "single_face_img", "single_face_img_data")
class Test_Landmark_Models:
    """Test all pretrained face models"""

    def test_mobilenet(self, default_detector, single_face_img, single_face_img_data):

        _, h, w = read_image(single_face_img).shape

        default_detector.change_model(
            face_model="RetinaFace", landmark_model="MobileNet"
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

    def test_mobilefacenet(
        self, default_detector, single_face_img, single_face_img_data
    ):

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

    def test_pfld(self, default_detector, single_face_img, single_face_img_data):

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


@pytest.mark.usefixtures("default_detector", "single_face_img_data")
class Test_AU_Models:
    """Test all pretrained AU models"""

    def test_svm_au(self, default_detector, single_face_img_data):

        default_detector.change_model(
            face_model="RetinaFace",
            landmark_model="MobileFaceNet",
            au_model="svm",
        )

        detected_faces = default_detector.detect_faces(single_face_img_data)
        landmarks = default_detector.detect_landmarks(
            single_face_img_data, detected_faces
        )
        aus = default_detector.detect_aus(single_face_img_data, landmarks=landmarks)

        assert np.sum(np.isnan(aus)) == 0
        assert aus[0].shape[-1] == 20

    def test_xgb_au(self, default_detector, single_face_img_data):

        default_detector.change_model(
            face_model="RetinaFace",
            landmark_model="MobileFaceNet",
            au_model="xgb",
        )

        detected_faces = default_detector.detect_faces(single_face_img_data)
        landmarks = default_detector.detect_landmarks(
            single_face_img_data, detected_faces
        )
        aus = default_detector.detect_aus(single_face_img_data, landmarks=landmarks)

        assert np.sum(np.isnan(aus)) == 0
        assert aus[0].shape[-1] == 20


@pytest.mark.usefixtures("default_detector", "single_face_img")
class Test_Emotion_Models:
    """Test all pretrained emotion models"""

    def test_resmasknet(self, default_detector, single_face_img):
        default_detector.change_model(emotion_model="resmasknet")
        out = default_detector.detect_image(single_face_img)
        assert out.emotions["happiness"].values > 0.5

    def test_svm_emotion(self, default_detector, single_face_img):
        default_detector.change_model(emotion_model="svm")
        out = default_detector.detect_image(single_face_img)
        assert out.emotions["happiness"].values > 0.5


@pytest.mark.usefixtures("default_detector", "single_face_img_data")
class Test_Facepose_Models:
    """Test all pretrained facepose models"""

    @pytest.mark.skip("TODO")
    def test_img2pose_facepose(self, default_detector, single_face_img_data):
        pass

    @pytest.mark.skip("TODO")
    def test_img2pose_c_facepose(self, default_detector, single_face_img_data):
        pass
