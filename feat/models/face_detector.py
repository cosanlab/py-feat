import cv2 as cv

class face_detector(Object):
    def __init__(self, model_type):
        if model_type == 'haarcascade':
            face_detection_model_path = cv.data.haarcascades + \
                "haarcascade_frontalface_alt.xml"
            if not os.path.exists(face_detection_model_path):
                print("Face detection model not found. Check haarcascade_frontalface_alt.xml exists in your opencv installation (cv.data).")
            face_cascade = cv.CascadeClassifier(face_detection_model_path)
            face_detection_columns = FEAT_FACEBOX_COLUMNS
            facebox_empty = np.empty((1, 4))
            facebox_empty[:] = np.nan
            empty_facebox = pd.DataFrame(
                facebox_empty, columns=face_detection_columns)
            self.info["face_detection_model"] = face_detection_model_path
            self.info["face_detection_columns"] = face_detection_columns
            self.face_detector = face_cascade
            self._empty_facebox = empty_facebox
        elif face_model == 'mtcnn':
        #TODO: Implement initializaton of mtcnn below  
        

#print("Loading Face Detection model: ", face_cascade)
if face_model == 'haarcascade':
    self.detect_face = True
    face_detection_model_path = cv.data.haarcascades + \
        "haarcascade_frontalface_alt.xml"
    if not os.path.exists(face_detection_model_path):
        print("Face detection model not found. Check haarcascade_frontalface_alt.xml exists in your opencv installation (cv.data).")
    face_cascade = cv.CascadeClassifier(face_detection_model_path)
    face_detection_columns = FEAT_FACEBOX_COLUMNS
    facebox_empty = np.empty((1, 4))
    facebox_empty[:] = np.nan
    empty_facebox = pd.DataFrame(
        facebox_empty, columns=face_detection_columns)
    self.info["face_detection_model"] = face_detection_model_path
    self.info["face_detection_columns"] = face_detection_columns
    self.face_detector = face_cascade
    self._empty_facebox = empty_facebox
   