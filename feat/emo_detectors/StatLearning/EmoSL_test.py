# Implements different statistical learning algorithms to classify Emotions
# Please see https://www.cl.cam.ac.uk/~mmam3/pub/FG2015.pdf for more details and reasons
# Currently support: SVM (as in the paper), RandomForest (new implementation).
import numpy as np
# from feat.utils.io import get_resource_path
# import joblib
# import os
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import pickle
# from feat.pretrained import load_classifier_pkl


class EmoSVMClassifier:
    def __init__(self, **kwargs) -> None:
        self.weights_loaded = False
        
    def load_weights(self, scaler_full=None, pca_model_full=None, classifiers=None):

        self.scaler_full = scaler_full
        self.pca_model_full = pca_model_full
        self.classifiers = classifiers
        self.weights_loaded = True

    def pca_transform(self, frame, scaler, pca_model, landmarks):
        if not self.weights_loaded:
            raise ValueError('Need to load weights before running pca_transform')
        else:
            transformed_frame = pca_model.transform(scaler.transform(frame))
            return np.concatenate((transformed_frame, landmarks), axis=1)      

    def detect_emo(self, frame, landmarks, **kwargs):
        """
        Note that here frame is represented by hogs
        """
        if not self.weights_loaded:
            raise ValueError('Need to load weights before running detect_au')
        else:
            landmarks = np.concatenate(landmarks)
            landmarks = landmarks.reshape(-1, landmarks.shape[1] * landmarks.shape[2])

            pca_transformed_full = self.pca_transform(frame, self.scaler_full, self.pca_model_full, landmarks)
            emo_columns = ["anger", "disgust", "fear", "happ", "sad", "sur", "neutral"]
    
            pred_emo = []
            for keys in emo_columns:
                emo_pred = self.classifiers[keys].predict(pca_transformed_full)
                pred_emo.append(emo_pred)
    
            pred_emos = np.array(pred_emo).T
            return pred_emos
