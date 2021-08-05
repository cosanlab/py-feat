# Implements different statistical learning algorithms to classify Emotions
# Please see https://www.cl.cam.ac.uk/~mmam3/pub/FG2015.pdf for more details and reasons
# Currently support: SVM (as in the paper), RandomForest (new implementation).
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from feat.utils import get_resource_path
import joblib
import os

def load_classifier(cf_path):
    clf = joblib.load(cf_path)
    return clf

class EmoRandomForestClassifier():
    def __init__(self) -> None:
        self.pca_model = load_classifier(os.path.join(
            get_resource_path(), "emo_hog_pca.joblib"))
        self.classifier = load_classifier(
            os.path.join(get_resource_path(), "emoRF36.joblib"))
        self.scaler = load_classifier(os.path.join(
            get_resource_path(), "emo_hog_scalar.joblib"))

    def detect_emo(self, frame, landmarks):
        """
        Note that here frame is represented by hogs
        """
        # landmarks = np.array(landmarks)
        # landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])
        # landmarks = landmarks.reshape(-1,landmarks.shape[1]*landmarks.shape[2])
        landmarks = np.concatenate(landmarks)
        landmarks = landmarks.reshape(-1,landmarks.shape[1]*landmarks.shape[2])


        pca_transformed_frame = self.pca_model.transform(
            self.scaler.fit_transform(frame))
        feature_cbd = np.concatenate((pca_transformed_frame, landmarks), 1)
        pred_emo = []
        for keys in self.classifier:
            emo_pred = self.classifier[keys].predict_proba(feature_cbd)
            emo_pred = emo_pred[:, 1]
            pred_emo.append(emo_pred)

        pred_emo = np.array(pred_emo).T
        return pred_emo


class EmoSVMClassifier():
    def __init__(self) -> None:
        self.pca_model = load_classifier(os.path.join(
            get_resource_path(), "emo_hog_pca.joblib"))
        self.classifier = load_classifier(
            os.path.join(get_resource_path(), "emoSVM38.joblib"))
        self.scaler = load_classifier(os.path.join(
            get_resource_path(), "emo_hog_scalar.joblib"))

    def detect_emo(self, frame, landmarks):
        """
        Note that here frame is represented by hogs
        """
        # landmarks = np.array(landmarks)
        # landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])
        # landmarks = landmarks.reshape(-1,landmarks.shape[1]*landmarks.shape[2])
        landmarks = np.concatenate(landmarks)
        landmarks = landmarks.reshape(-1,landmarks.shape[1]*landmarks.shape[2])

        
        pca_transformed_frame = self.pca_model.transform(
            self.scaler.fit_transform(frame))
        feature_cbd = np.concatenate((pca_transformed_frame, landmarks), 1)
        pred_emo = []
        for keys in self.classifier:
            emo_pred = self.classifier[keys].predict(feature_cbd)
            emo_pred = emo_pred  # probably need to delete this
            pred_emo.append(emo_pred)

        pred_emos = np.array(pred_emo).T
        return pred_emos
