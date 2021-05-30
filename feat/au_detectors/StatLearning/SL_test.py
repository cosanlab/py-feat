# Implements different statistical learning algorithms to classify AUs
# Please see https://www.cl.cam.ac.uk/~mmam3/pub/FG2015.pdf for more details and reasons
# Currently support: SVM (as in the paper), RandomForest (new implementation), Logistic Regression
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


class RandomForestClassifier():
    def __init__(self) -> None:
        self.pca_model = load_classifier(os.path.join(
            get_resource_path(), "hog_pca_all_emotio.joblib"))
        self.classifier = load_classifier(
            os.path.join(get_resource_path(), "RF_568.joblib"))
        self.scaler = load_classifier(os.path.join(
            get_resource_path(), "hog_scalar_aus.joblib"))

    def detect_au(self, frame, landmarks):
        """
        Note that here frame is represented by hogs
        """
        #if len(frame.shape) < 2:
        #    frame = frame.reshape(1, -1)
        #if len(landmarks.shape) > 1:
        #    landmarks = landmarks.flatten().reshape(1, -1)
        landmarks = np.array(landmarks)
        landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])
        landmarks = landmarks.reshape(-1,landmarks.shape[1]*landmarks.shape[2])
        pca_transformed_frame = self.pca_model.transform(
            self.scaler.fit_transform(frame))
        feature_cbd = np.concatenate((pca_transformed_frame, landmarks), 1)
        pred_aus = []
        for keys in self.classifier:
            au_pred = self.classifier[keys].predict_proba(feature_cbd)
            au_pred = au_pred[:, 1]
            pred_aus.append(au_pred)
        #print("halt")
        pred_aus = np.array(pred_aus).T
        return pred_aus


class SVMClassifier():
    def __init__(self) -> None:
        self.pca_model = load_classifier(os.path.join(
            get_resource_path(), "hog_pca_all_emotio.joblib"))
        self.classifier = load_classifier(
            os.path.join(get_resource_path(), "svm_568.joblib"))
        self.scaler = load_classifier(os.path.join(
            get_resource_path(), "hog_scalar_aus.joblib"))

    def detect_au(self, frame, landmarks):
        """
        Note that here frame is represented by hogs
        """
        landmarks = np.array(landmarks)
        landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])
        landmarks = landmarks.reshape(-1,landmarks.shape[1]*landmarks.shape[2])

        pca_transformed_frame = self.pca_model.transform(
            self.scaler.fit_transform(frame))
        feature_cbd = np.concatenate((pca_transformed_frame, landmarks), 1)
        pred_aus = []
        for keys in self.classifier:
            au_pred = self.classifier[keys].predict(feature_cbd)
            au_pred = au_pred  # probably need to delete this
            pred_aus.append(au_pred)

        pred_aus = np.array(pred_aus).T
        return pred_aus


class LogisticClassifier():

    def __init__(self) -> None:
        self.pca_model = load_classifier(os.path.join(
            get_resource_path(), "hog_pca_all_emotio.joblib"))
        self.classifier = load_classifier(os.path.join(
            get_resource_path(), "Logistic_520.joblib"))
        self.scaler = load_classifier(os.path.join(
            get_resource_path(), "hog_scalar_aus.joblib"))

    def detect_au(self, frame, landmarks):
        """
        Note that here frame is represented by hogs
        """

        landmarks = np.array(landmarks)
        landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])
        landmarks = landmarks.reshape(-1,landmarks.shape[1]*landmarks.shape[2])

        pca_transformed_frame = self.pca_model.transform(
            self.scaler.fit_transform(frame))
        feature_cbd = np.concatenate((pca_transformed_frame, landmarks), 1)
        pred_aus = []
        for keys in self.classifier:
            au_pred = self.classifier[keys].predict_proba(feature_cbd)
            au_pred = au_pred[:, 1]
            pred_aus.append(au_pred)

        pred_aus = np.array(pred_aus).T
        return pred_aus
