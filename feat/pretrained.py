from feat.face_detectors.FaceBoxes.FaceBoxes_test import FaceBoxes
from feat.face_detectors.Retinaface.Retinaface_test import Retinaface
from feat.face_detectors.MTCNN.MTCNN_test import MTCNN
from feat.landmark_detectors.basenet_test import MobileNet_GDConv
from feat.landmark_detectors.pfld_compressed_test import PFLDInference
from feat.landmark_detectors.mobilefacenet_test import MobileFaceNet
from feat.facepose_detectors.img2pose.img2pose_test import Img2Pose
from feat.facepose_detectors.pnp.pnp_test import PerspectiveNPoint
from feat.au_detectors.JAANet.JAA_test import JAANet
from feat.au_detectors.DRML.DRML_test import DRMLNet
from feat.au_detectors.StatLearning.SL_test import (
    RandomForestClassifier,
    SVMClassifier,
    LogisticClassifier,
)
from feat.emo_detectors.ferNet.ferNet_test import ferNetModule
from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet
from feat.emo_detectors.StatLearning.EmoSL_test import (
    EmoRandomForestClassifier,
    EmoSVMClassifier,
)
from feat.utils import get_resource_path, download_url
import os, json

# Currently supported pre-trained detectors
PRETRAINED_MODELS = {
    "face_model": [
        {"retinaface": Retinaface},
        {"faceboxes": FaceBoxes},
        {"mtcnn": MTCNN},
        {"img2pose": Img2Pose},
        {"img2pose-c": Img2Pose},
    ],
    "landmark_model": [
        {"mobilenet": MobileNet_GDConv},
        {"mobilefacenet": MobileFaceNet},
        {"pfld": PFLDInference},
    ],
    "au_model": [
        {"rf": RandomForestClassifier},
        {"svm": SVMClassifier},
        {"logistic": LogisticClassifier},
        {"jaanet": JAANet},
        {"drml": DRMLNet},
    ],
    "emotion_model": [
        {"resmasknet": ResMaskNet},
        {"rf": EmoRandomForestClassifier},
        {"svm": EmoSVMClassifier},
        {"fer": ferNetModule},
    ],
    "facepose_model": [
        {"pnp": PerspectiveNPoint},
        {"img2pose": None},
        {"img2pose-c": None},
    ],
}


def get_pretrained_models(
    face_model, landmark_model, au_model, emotion_model, facepose_model, verbose
):
    """Helper function that validates the request model names and downloads them if
    necessary using the URLs in the included JSON file. User by detector init"""

    # Get supported model URLs
    with open(os.path.join(get_resource_path(), "model_list.json"), "r") as f:
        model_urls = json.load(f)

    get_names = lambda s: list(
        map(
            lambda e: list(e.keys())[0],
            PRETRAINED_MODELS[s],
        )
    )

    # Face model
    if face_model is not None:
        face_model = face_model.lower()
        if face_model not in get_names("face_model"):
            raise ValueError(
                f"Requested face_model was {face_model}. Must be one of {PRETRAINED_MODELS['face_model']}"
            )
        for url in model_urls["face_detectors"][face_model]["urls"]:
            download_url(url, get_resource_path(), verbose=verbose)

    # Landmark model
    if landmark_model is not None:
        landmark_model = landmark_model.lower()
        if landmark_model not in get_names("landmark_model"):
            raise ValueError(
                f"Requested landmark_model was {landmark_model}. Must be one of {PRETRAINED_MODELS['landmark_model']}"
            )
        for url in model_urls["landmark_detectors"][landmark_model]["urls"]:
            download_url(url, get_resource_path(), verbose=verbose)

    # AU model
    if au_model is not None:
        au_model = au_model.lower()
        if au_model not in get_names("au_model"):
            raise ValueError(
                f"Requested au_model was {au_model}. Must be one of {PRETRAINED_MODELS['au_model']}"
            )
        for url in model_urls["au_detectors"][au_model]["urls"]:
            download_url(url, get_resource_path(), verbose=verbose)
            if ".zip" in url:
                import zipfile

                with zipfile.ZipFile(
                    os.path.join(get_resource_path(), "JAANetparams.zip"), "r"
                ) as zip_ref:
                    zip_ref.extractall(os.path.join(get_resource_path()))
            if au_model in ["logistic", "svm", "rf"]:
                download_url(
                    model_urls["au_detectors"]["hog-pca"]["urls"][0],
                    get_resource_path(),
                    verbose=verbose,
                )
                download_url(
                    model_urls["au_detectors"]["au_scalar"]["urls"][0],
                    get_resource_path(),
                    verbose=verbose,
                )

    # Emotion model
    if emotion_model is not None:
        emotion_model = emotion_model.lower()
        if emotion_model not in get_names("emotion_model"):
            raise ValueError(
                f"Requested emotion_model was {emotion_model}. Must be one of {PRETRAINED_MODELS['emotion_model']}"
            )
        for url in model_urls["emotion_detectors"][emotion_model]["urls"]:
            download_url(url, get_resource_path(), verbose=verbose)
            if emotion_model in ["svm", "rf"]:
                download_url(
                    model_urls["emotion_detectors"]["emo_pca"]["urls"][0],
                    get_resource_path(),
                    verbose=verbose,
                )
                download_url(
                    model_urls["emotion_detectors"]["emo_scalar"]["urls"][0],
                    get_resource_path(),
                    verbose=verbose,
                )

    # Facepose model
    # Just validate as it's handled by the face_model loading
    if facepose_model is not None:
        facepose_model = facepose_model.lower()
        if facepose_model not in get_names("facepose_model"):
            raise ValueError(
                f"Requested facepose_model was {facepose_model}. Must be one of {PRETRAINED_MODELS['facepose_model']}"
            )
        if "img2pose" in facepose_model and facepose_model != face_model:
            raise ValueError(
                f"{facepose_model} is both a face detector and a pose estimator and cannot be used with a different face detector. Please set face_model to {facepose_model} as well"
            )

    return (
        face_model,
        landmark_model,
        au_model,
        emotion_model,
        facepose_model,
    )


def fetch_model(model_type, model_name):
    """Fetch a pre-trained model class constructor. Used by detector init"""
    if model_name is None:
        return None
    model_type = PRETRAINED_MODELS[model_type]
    matches = list(filter(lambda e: model_name in e.keys(), model_type))[0]
    return list(matches.values())[0]
