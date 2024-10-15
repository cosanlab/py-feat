from feat.detector import Detector
from feat.data import Fex
import pytest
import numpy as np
from torchvision.io import read_image


def is_not_third_sunday():
    from datetime import datetime as dt
    import calendar

    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    year = dt.now().year
    month = dt.now().month
    monthcal = c.monthdatescalendar(year, month)
    third_sunday = [
        day
        for week in monthcal
        for day in week
        if day.weekday() == calendar.SUNDAY and day.month == month
    ][2]
    return not dt.date(dt.now()) == third_sunday


# @pytest.mark.skipif(
#     is_not_third_sunday(),
#     reason="This tests ALL model detector combinations which takes a while, so we only run it once a month on the third sunday. You can run it locally by commenting out this dectorator and using pytest -k 'test_detector_combos'.",
# )
@pytest.mark.skip
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
    assert isinstance(out, Fex)
    assert out.shape[0] == 1


@pytest.mark.skip
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

    def test_img2pose_face(self, default_detector, single_face_img_data):
        default_detector.change_model(face_model="img2pose")
        out = default_detector.detect_faces(single_face_img_data)
        assert 180 < out[0][0][0] < 200

    def test_img2pose_c_face(self, default_detector, single_face_img_data):
        default_detector.change_model(face_model="img2pose-c")
        out = default_detector.detect_faces(single_face_img_data)
        assert 180 < out[0][0][0] < 200


@pytest.mark.skip
@pytest.mark.usefixtures("default_detector", "single_face_img", "single_face_img_data")
class Test_Landmark_Models:
    """Test all pretrained face models"""

    def test_mobilenet(self, default_detector, single_face_img, single_face_img_data):
        _, h, w = read_image(single_face_img).shape

        default_detector.change_model(face_model="RetinaFace", landmark_model="MobileNet")

        bboxes = default_detector.detect_faces(single_face_img_data)
        landmarks = default_detector.detect_landmarks(single_face_img_data, bboxes)[0]
        assert landmarks[0].shape == (68, 2)
        assert (
            np.any(landmarks[0][:, 0] > 0)
            and np.any(landmarks[0][:, 0] < w)
            and np.any(landmarks[0][:, 1] > 0)
            and np.any(landmarks[0][:, 1] < h)
        )

    def test_mobilefacenet(self, default_detector, single_face_img, single_face_img_data):
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
@pytest.mark.usefixtures("default_detector", "single_face_img", "single_face_img_data")
class Test_Facepose_Models:
    """Test all pretrained facepose models"""

    def test_img2pose_facepose(
        self, default_detector, single_face_img, single_face_img_data
    ):
        default_detector.change_model(facepose_model="img2pose")
        poses = default_detector.detect_facepose(single_face_img_data)
        assert np.allclose(poses["poses"], [0.86, -3.80, 6.60], atol=0.5)

        # Test DOF kwarg
        facepose_model_kwargs = {"RETURN_DIM": 6}
        new_detector = Detector(facepose_model_kwargs=facepose_model_kwargs)
        assert new_detector.facepose_detector.RETURN_DIM == 6

        # Run as full detection
        out = new_detector.detect_image(single_face_img)
        assert "X" in out.poses.columns

        # Also run directly
        poses = new_detector.detect_facepose(single_face_img_data)
        assert len(poses["poses"][0][0]) == 6

    def test_img2pose_c_facepose(self, default_detector, single_face_img_data):
        default_detector.change_model(facepose_model="img2pose-c")
        poses = default_detector.detect_facepose(single_face_img_data)
        assert np.allclose(poses["poses"], [0.86, -3.80, 6.60], atol=0.5)


@pytest.mark.skip
@pytest.mark.usefixtures("default_detector", "multi_face_img")
class Test_Identity_Models:
    """Test all pretrained identity models"""

    def test_facenet(self, default_detector, multi_face_img):
        default_detector.change_model(identity_model="facenet")
        out = default_detector.detect_image(multi_face_img)

        # Recompute identities based on embeddings and a new threshold
        out2 = out.compute_identities(threshold=0.2)

        # Identities for each face should change
        assert not out.identities.equals(out2.identities)

        # But embeddings don't as they're simply re-used at the new threshold
        assert out.identity_embeddings.equals(out2.identity_embeddings)

        # Should be equivalent to setting that threshold when first calling detector
        out3 = default_detector.detect_image(multi_face_img, face_identity_threshold=0.2)
        assert out3.identities.equals(out2.identities)
