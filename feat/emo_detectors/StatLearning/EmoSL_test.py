# Implements different statistical learning algorithms to classify Emotions
# Please see https://www.cl.cam.ac.uk/~mmam3/pub/FG2015.pdf for more details and reasons
# Currently support: SVM (as in the paper), RandomForest (new implementation).
import numpy as np
from feat.utils.io import get_resource_path
import joblib
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle


def load_classifier(cf_path):
    clf = joblib.load(cf_path)
    return clf


def load_classifier_pkl(cf_path):
    clf = pickle.load(open(cf_path, "rb"))
    return clf


class EmoSVMClassifier:
    def __init__(self, **kwargs) -> None:
        self.pca_model = load_classifier(
            os.path.join(get_resource_path(), "emo_hog_pca.joblib")
        )
        self.classifier = load_classifier(
            os.path.join(get_resource_path(), "emoSVM38.joblib")
        )
        self.scaler = load_classifier(
            os.path.join(get_resource_path(), "emo_hog_scalar.joblib")
        )

    def detect_emo(self, frame, landmarks, **kwargs):
        """
        Note that here frame is represented by hogs
        """
        # landmarks = np.array(landmarks)
        # landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])
        # landmarks = landmarks.reshape(-1,landmarks.shape[1]*landmarks.shape[2])
        landmarks = np.concatenate(landmarks)
        landmarks = landmarks.reshape(-1, landmarks.shape[1] * landmarks.shape[2])

        pca_transformed_frame = self.pca_model.transform(
            self.scaler.fit_transform(frame)
        )
        feature_cbd = np.concatenate((pca_transformed_frame, landmarks), 1)
        pred_emo = []
        for keys in self.classifier:
            emo_pred = self.classifier[keys].predict(feature_cbd)
            emo_pred = emo_pred  # probably need to delete this
            pred_emo.append(emo_pred)

        pred_emos = np.array(pred_emo).T
        return pred_emos
