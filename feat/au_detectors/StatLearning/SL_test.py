# Implements different statistical learning algorithms to classify AUs
# Please see https://www.cl.cam.ac.uk/~mmam3/pub/FG2015.pdf for more details and reasons

import numpy as np
from feat.utils.io import get_resource_path
import joblib
import pickle
import os
import xgboost as xgb


def load_classifier(cf_path):
    clf = joblib.load(cf_path)
    return clf


def load_classifier_pkl(cf_path):
    clf = pickle.load(open(cf_path, "rb"))
    return clf


class SVMClassifier:
    def __init__(self) -> None:
        self.scaler_upper, self.pca_model_upper = load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Upperscalar_June30.pkl")), load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Upperpca_June30.pkl"))
        
        self.scaler_lower, self.pca_model_lower = load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Lowerscalar_June30.pkl")), load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Lowerpca_June30.pkl"))
        
        self.scaler_full, self.pca_model_full = load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Fullscalar_June30.pkl")), load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Fullpca_June30.pkl"))
        
        self.classifier = load_classifier(
            os.path.join(get_resource_path(), "svm_60_July2023.pkl")
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
            self.scaler_upper.transform(frame)
        )
        pca_transformed_lower = self.pca_model_lower.transform(
            self.scaler_lower.transform(frame)
        )
        pca_transformed_full = self.pca_model_full.transform(
            self.scaler_full.transform(frame)
        )

        pca_transformed_upper = np.concatenate((pca_transformed_upper, landmarks), 1)
        pca_transformed_lower = np.concatenate((pca_transformed_lower, landmarks), 1)
        pca_transformed_full = np.concatenate((pca_transformed_full, landmarks), 1)
        aus_list = sorted(self.classifier.keys(), key=lambda x: int(x[2::]))

        pred_aus = []
        for keys in aus_list:
            if keys in ["AU1","AU4","AU6"]:
                au_pred = self.classifier[keys].predict(pca_transformed_upper)
            elif keys in ["AU11", "AU12","AU17"]:
                au_pred = self.classifier[keys].predict(pca_transformed_lower)
            elif keys in ["AU2","AU5","AU7","AU9","AU10","AU14","AU15","AU20","AU23","AU24","AU25","AU26","AU28","AU43"]:
                au_pred = self.classifier[keys].predict(pca_transformed_full)
            else:
                raise ValueError("unknown AU detected")

            pred_aus.append(au_pred)
        pred_aus = np.array(pred_aus).T
        return pred_aus


class XGBClassifier:
    def __init__(self) -> None:
        
        self.scaler_upper, self.pca_model_upper = load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Upperscalar_June30.pkl")), load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Upperpca_June30.pkl"))
        
        self.scaler_lower, self.pca_model_lower = load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Lowerscalar_June30.pkl")), load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Lowerpca_June30.pkl"))
        
        self.scaler_full, self.pca_model_full = load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Fullscalar_June30.pkl")), load_classifier_pkl(os.path.join(get_resource_path(), "all_data_Fullpca_June30.pkl"))
        
        self.au_keys = [
            "AU1",
            "AU2",
            "AU4",
            "AU5",
            "AU6",
            "AU7",
            "AU9",
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
        ]

    def detect_au(self, frame, landmarks):
        """
        Note that here frame is represented by hogs
        """

        # landmarks = np.array(landmarks)
        landmarks = np.concatenate(landmarks)
        # landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])
        landmarks = landmarks.reshape(-1, landmarks.shape[1] * landmarks.shape[2])

        pca_transformed_upper = self.pca_model_upper.transform(
            self.scaler_upper.transform(frame)
        )
        pca_transformed_lower = self.pca_model_lower.transform(
            self.scaler_lower.transform(frame)
        )
        pca_transformed_full = self.pca_model_full.transform(
            self.scaler_full.transform(frame)
        )

        pca_transformed_upper = np.concatenate((pca_transformed_upper, landmarks), 1)
        pca_transformed_lower = np.concatenate((pca_transformed_lower, landmarks), 1)
        pca_transformed_full = np.concatenate((pca_transformed_full, landmarks), 1)

        pred_aus = []
        for keys in self.au_keys:
            
            classifier = xgb.XGBClassifier()
            classifier.load_model(
                os.path.join(get_resource_path(), f"July4_{keys}_XGB.ubj")
            )
        
            if keys in ["AU1","AU2","AU7"]:
                au_pred = classifier.predict_proba(pca_transformed_upper)[:, 1]
            elif keys in ["AU11","AU14","AU17","AU23","AU24","AU26"]:
                au_pred = classifier.predict_proba(pca_transformed_lower)[:, 1]
            elif keys in ["AU4","AU5","AU6","AU9","AU10","AU12","AU15","AU20","AU25","AU28","AU43"]:
                au_pred = classifier.predict_proba(pca_transformed_full)[:, 1]
            else:
                raise ValueError("unknown AU detected")
        
            # au_pred = au_pred[:, 1]
            pred_aus.append(au_pred)

        pred_aus = np.array(pred_aus).T
        return pred_aus
