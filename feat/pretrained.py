"""
Helper functions specifically for working with included pre-trained models
"""

from feat.face_detectors.FaceBoxes.FaceBoxes_test import FaceBoxes
from feat.face_detectors.Retinaface.Retinaface_test import Retinaface
from feat.face_detectors.MTCNN.MTCNN_test import MTCNN
from feat.landmark_detectors.basenet_test import MobileNet_GDConv
from feat.landmark_detectors.pfld_compressed_test import PFLDInference
from feat.landmark_detectors.mobilefacenet_test import MobileFaceNet
from feat.facepose_detectors.img2pose.img2pose_test import Img2Pose
from feat.au_detectors.StatLearning.SL_test import SVMClassifier, XGBClassifier
from feat.emo_detectors.ResMaskNet.resmasknet_test import ResMaskNet
from feat.emo_detectors.StatLearning.EmoSL_test import (
    EmoSVMClassifier,
)
from feat.utils.io import get_resource_path, download_url
import os
import json

__all__ = ["get_pretrained_models", "fetch_model"]
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
    "au_model": [{"svm": SVMClassifier}, {"xgb": XGBClassifier}],
    "emotion_model": [
        {"resmasknet": ResMaskNet},
        {"svm": EmoSVMClassifier},
    ],
    "facepose_model": [
        {"img2pose": Img2Pose},
        {"img2pose-c": Img2Pose},
    ],
}

# Compatibility support for OpenFace which has diff AU names than feat
AU_LANDMARK_MAP = {
    "OpenFace": [
        "AU01_r",
        "AU02_r",
        "AU04_r",
        "AU05_r",
        "AU06_r",
        "AU07_r",
        "AU09_r",
        "AU10_r",
        "AU12_r",
        "AU14_r",
        "AU15_r",
        "AU17_r",
        "AU20_r",
        "AU23_r",
        "AU25_r",
        "AU26_r",
        "AU45_r",
    ],
    "Feat": [
        "AU01",
        "AU02",
        "AU04",
        "AU05",
        "AU06",
        "AU07",
        "AU09",
        "AU10",
        "AU11",
        "AU12",
        "AU14",
        "AU15",
        "AU17",
        "AU20",
        "AU23",
        "AU24",
        "AU25",
        "AU26",
        "AU28",
        "AU43",
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
    if face_model is None:
        raise ValueError(
            f"face_model must be one of {[list(e.keys())[0] for e in PRETRAINED_MODELS['face_model']]}"
        )
    else:
        face_model = face_model.lower()
        if face_model not in get_names("face_model"):
            raise ValueError(
                f"Requested face_model was {face_model}. Must be one of {[list(e.keys())[0] for e in PRETRAINED_MODELS['face_model']]}"
            )
        for url in model_urls["face_detectors"][face_model]["urls"]:
            download_url(url, get_resource_path(), verbose=verbose)

    # Landmark model
    if landmark_model is None:
        raise ValueError(
            f"landmark_model must be one of {[list(e.keys())[0] for e in PRETRAINED_MODELS['landmark_model']]}"
        )
    else:
        landmark_model = landmark_model.lower()
        if landmark_model not in get_names("landmark_model"):
            raise ValueError(
                f"Requested landmark_model was {landmark_model}. Must be one of {[list(e.keys())[0] for e in PRETRAINED_MODELS['landmark_model']]}"
            )
        for url in model_urls["landmark_detectors"][landmark_model]["urls"]:
            download_url(url, get_resource_path(), verbose=verbose)

    # AU model
    if au_model is None:
        raise ValueError(
            f"au_model must be one of {[list(e.keys())[0] for e in PRETRAINED_MODELS['au_model']]}"
        )
    else:
        au_model = au_model.lower()
        if au_model not in get_names("au_model"):
            raise ValueError(
                f"Requested au_model was {au_model}. Must be one of {[list(e.keys())[0]for e in PRETRAINED_MODELS['au_model']]}"
            )
        for url in model_urls["au_detectors"][au_model]["urls"]:
            download_url(url, get_resource_path(), verbose=verbose)
            if au_model in ["xgb", "svm"]:
                download_url(
                    model_urls["au_detectors"]["hog-pca"]["urls"][0],
                    get_resource_path(),
                    verbose=verbose,
                )
                download_url(
                    model_urls["au_detectors"]["hog-pca"]["urls"][1],
                    get_resource_path(),
                    verbose=verbose,
                )
                download_url(
                    model_urls["au_detectors"]["hog-pca"]["urls"][2],
                    get_resource_path(),
                    verbose=verbose,
                )

    # Emotion model
    if emotion_model is None:
        raise ValueError(
            f"emotion_model must be one of {[list(e.keys())[0] for e in PRETRAINED_MODELS['emotion_model']]}"
        )
    else:
        emotion_model = emotion_model.lower()
        if emotion_model not in get_names("emotion_model"):
            raise ValueError(
                f"Requested emotion_model was {emotion_model}. Must be one of {[list(e.keys())[0] for e in PRETRAINED_MODELS['emotion_model']]}"
            )
        for url in model_urls["emotion_detectors"][emotion_model]["urls"]:
            download_url(url, get_resource_path(), verbose=verbose)
            if emotion_model in ["svm"]:
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
    if facepose_model is None:
        raise ValueError(
            f"facepose_model must be one of {[list(e.keys())[0] for e in PRETRAINED_MODELS['facepose_model']]}"
        )
    else:
        facepose_model = facepose_model.lower()
        if facepose_model not in get_names("facepose_model"):
            raise ValueError(
                f"Requested facepose_model was {facepose_model}. Must be one of {[list(e.keys())[0] for e in PRETRAINED_MODELS['facepose_model']]}"
            )
        for url in model_urls["facepose_detectors"][facepose_model]["urls"]:
            download_url(url, get_resource_path(), verbose=verbose)
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
        raise ValueError(f"{model_type} must be a valid string model name, not None")
    model_type = PRETRAINED_MODELS[model_type]
    matches = list(filter(lambda e: model_name in e.keys(), model_type))[0]
    return list(matches.values())[0]
