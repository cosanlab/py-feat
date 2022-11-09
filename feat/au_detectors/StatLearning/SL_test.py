# Implements different statistical learning algorithms to classify AUs
# Please see https://www.cl.cam.ac.uk/~mmam3/pub/FG2015.pdf for more details and reasons
# Currently support: SVM (as in the paper), RandomForest (new implementation), Logistic Regression

import numpy as np
from feat.utils.io import get_resource_path
import joblib
import pickle
import os


def load_classifier(cf_path):
    clf = joblib.load(cf_path)
    return clf


def load_classifier_pkl(cf_path):
    clf = pickle.load(open(cf_path, "rb"))
    return clf


class SVMClassifier:
    def __init__(self) -> None:
        self.scaler_upper, self.pca_model_upper = load_classifier_pkl(
            os.path.join(get_resource_path(), "upper_face_pcaSet.pkl")
        )
        self.scaler_lower, self.pca_model_lower = load_classifier_pkl(
            os.path.join(get_resource_path(), "lower_face_pcaSet.pkl")
        )
        self.scaler_full, self.pca_model_full = load_classifier_pkl(
            os.path.join(get_resource_path(), "full_face_pcaSet.pkl")
        )
        self.classifier = load_classifier(
            os.path.join(get_resource_path(), "svm_60_Nov22022.pkl")
        )

    def detect_au(self, frame, landmarks):
        """
        Note that here frame is represented by hogs
        """
        # landmarks = np.array(landmarks)
        # landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])
        # landmarks = landmarks.reshape(-1,landmarks.shape[1]*landmarks.shape[2])
        landmarks = np.concatenate(landmarks)
        landmarks = landmarks.reshape(-1, landmarks.shape[1] * landmarks.shape[2])

        pca_transformed_upper = self.pca_model_upper.transform(
            self.scaler_upper.fit_transform(frame)
        )
        pca_transformed_lower = self.pca_model_lower.transform(
            self.scaler_lower.fit_transform(frame)
        )
        pca_transformed_full = self.pca_model_full.transform(
            self.scaler_full.fit_transform(frame)
        )

        pca_transformed_upper = np.concatenate((pca_transformed_upper, landmarks), 1)
        pca_transformed_lower = np.concatenate((pca_transformed_lower, landmarks), 1)
        pca_transformed_full = np.concatenate((pca_transformed_full, landmarks), 1)
        aus_list = sorted(self.classifier.keys(), key=lambda x: int(x[2::]))

        pred_aus = []
        for keys in aus_list:
            if keys in ["AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU43"]:
                au_pred = self.classifier[keys].predict(pca_transformed_upper)
            elif keys in ["AU10", "AU11", "AU12", "AU15", "AU23", "AU25", "AU28"]:
                au_pred = self.classifier[keys].predict(pca_transformed_lower)
            elif keys in ["AU7", "AU14", "AU17", "AU20", "AU24", "AU26"]:
                au_pred = self.classifier[keys].predict(pca_transformed_full)
            else:
                raise ValueError("unknown AU detected")

            pred_aus.append(au_pred)
        pred_aus = np.array(pred_aus).T
        return pred_aus


class XGBClassifier:
    def __init__(self) -> None:
        self.scaler, self.pca_model = load_classifier_pkl(
            os.path.join(get_resource_path(), "full_face_pcaSet.pkl")
        )
        self.classifier = load_classifier_pkl(
            os.path.join(get_resource_path(), "xgb_60_oct312022.pkl")
        )

    def detect_au(self, frame, landmarks):
        """
        Note that here frame is represented by hogs
        """

        # landmarks = np.array(landmarks)
        landmarks = np.concatenate(landmarks)
        # landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])
        landmarks = landmarks.reshape(-1, landmarks.shape[1] * landmarks.shape[2])

        pca_transformed_frame = self.pca_model.transform(self.scaler.transform(frame))
        feature_cbd = np.concatenate((pca_transformed_frame, landmarks), 1)
        aus_list = sorted(self.classifier.keys(), key=lambda x: int(x[2::]))

        pred_aus = []
        for keys in aus_list:
            au_pred = self.classifier[keys].predict_proba(feature_cbd)[:, 1]
            # au_pred = au_pred[:, 1]
            pred_aus.append(au_pred)

        pred_aus = np.array(pred_aus).T
        return pred_aus
