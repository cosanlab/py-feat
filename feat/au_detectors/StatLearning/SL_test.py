# Implements different statistical learning algorithms to classify AUs
# Please see https://www.cl.cam.ac.uk/~mmam3/pub/FG2015.pdf for more details and reasons

import numpy as np


class SVMClassifier:
    def __init__(self) -> None:
        self.weights_loaded = False
        
    def load_weights(self, scaler_upper=None, pca_model_upper=None, scaler_lower=None, pca_model_lower=None, scaler_full=None, pca_model_full=None, classifiers=None):
        self.scaler_upper = scaler_upper
        self.pca_model_upper = pca_model_upper
        self.scaler_lower = scaler_lower
        self.pca_model_lower = pca_model_lower
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

    def detect_au(self, frame, landmarks):
        """
        Note that here frame is represented by hogs
        """
        if not self.weights_loaded:
            raise ValueError('Need to load weights before running detect_au')
        else:
            landmarks = np.concatenate(landmarks)
            landmarks = landmarks.reshape(-1, landmarks.shape[1] * landmarks.shape[2])
    
            pca_transformed_upper = self.pca_transform(frame, self.scaler_upper, self.pca_model_upper, landmarks)
            pca_transformed_lower = self.pca_transform(frame, self.scaler_lower, self.pca_model_lower, landmarks)
            pca_transformed_full = self.pca_transform(frame, self.scaler_full, self.pca_model_full, landmarks)
            
            aus_list = sorted(self.classifiers.keys(), key=lambda x: int(x[2::]))
    
            pred_aus = []
            for keys in aus_list:
                if keys in ["AU1", "AU4", "AU6"]:
                    au_pred = self.classifiers[keys].predict(pca_transformed_upper)
                elif keys in ["AU11", "AU12", "AU17"]:
                    au_pred = self.classifiers[keys].predict(pca_transformed_lower)
                elif keys in [
                    "AU2",
                    "AU5",
                    "AU7",
                    "AU9",
                    "AU10",
                    "AU14",
                    "AU15",
                    "AU20",
                    "AU23",
                    "AU24",
                    "AU25",
                    "AU26",
                    "AU28",
                    "AU43",
                ]:
                    au_pred = self.classifiers[keys].predict(pca_transformed_full)
                else:
                    raise ValueError("unknown AU detected")
    
                pred_aus.append(au_pred)
            pred_aus = np.array(pred_aus).T
            return pred_aus
class XGBClassifier:
    def __init__(self) -> None:

        self.au_keys = [
                "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU11", "AU12",
                "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43"
            ]
        self.weights_loaded = False
        
    def load_weights(self, scaler_upper=None, pca_model_upper=None, scaler_lower=None, pca_model_lower=None, scaler_full=None, pca_model_full=None, classifiers=None):
        self.scaler_upper = scaler_upper
        self.pca_model_upper = pca_model_upper
        self.scaler_lower = scaler_lower
        self.pca_model_lower = pca_model_lower
        self.scaler_full = scaler_full
        self.pca_model_full = pca_model_full
        self.classifiers = classifiers
        self.weights_loaded = True
        
    def pca_transform(self, frame, scaler, pca_model, landmarks):
        if not self.weights_loaded:
            raise ValueError('Need to load weights before running pca_transform')
        else:
            # NOTES: can directly do math to avoid sklearn overhead
            transformed_frame = pca_model.transform(scaler.transform(frame))
            return np.concatenate((transformed_frame, landmarks), axis=1)

    def detect_au(self, frame, landmarks):
        if not self.weights_loaded:
            raise ValueError('Need to load weights before running detect_au')
        else:
            landmarks = np.concatenate(landmarks)
            landmarks = landmarks.reshape(-1, landmarks.shape[1] * landmarks.shape[2])
            
            pca_transformed_upper = self.pca_transform(frame, self.scaler_upper, self.pca_model_upper, landmarks)
            pca_transformed_lower = self.pca_transform(frame, self.scaler_lower, self.pca_model_lower, landmarks)
            pca_transformed_full = self.pca_transform(frame, self.scaler_full, self.pca_model_full, landmarks)
    
            pred_aus = []
            for key in self.au_keys:
                classifier = self.classifiers[key]
    
                if key in ["AU1", "AU2", "AU7"]:
                    au_pred = classifier.predict_proba(pca_transformed_upper)[:, 1]
                elif key in ["AU11", "AU14", "AU17", "AU23", "AU24", "AU26"]:
                    au_pred = classifier.predict_proba(pca_transformed_lower)[:, 1]
                else:
                    au_pred = classifier.predict_proba(pca_transformed_full)[:, 1]
    
                pred_aus.append(au_pred)
    
            return np.array(pred_aus).T
    
      