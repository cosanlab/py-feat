import os
import cv2
import numpy as np
from feat.utils import get_resource_path
from feat.facepose_detectors.utils import convert_to_euler
THREED_FACE_MODEL = os.path.join(get_resource_path(), "reference_3d_68_points_trans.npy")


class PerspectiveNPointModel:
    """ Class that leverages 68 2D facial landmark points to estimate head pose using the Perspective-n-Point
    algorithm.

    Code adapted from https://github.com/yinguobing/head-pose-estimation/ and
    https://github.com/lincolnhard/head-pose-estimation/. Each code base licensed under MIT Licenses, which can be
    found here: https://github.com/yinguobing/head-pose-estimation/blob/master/LICENSE and here:
    https://github.com/lincolnhard/head-pose-estimation/blob/master/LICENSE
    """

    def __init__(self):
        """ Initializes the model, with a reference 3D model (xyz coordinates) of a standard face"""
        # self.model_points = get_full_model_points(os.path.join(get_resource_path(), "3d_face_model.txt"))
        self.model_points = np.load(THREED_FACE_MODEL, allow_pickle=True)

    def predict(self, img, landmarks):
        """ Determines headpose using passed 68 2D landmarks

        Args:
            img (np.ndarray) : The cv2 image from which the landmarks were produced
            landmarks (np.ndarray) : The landmarks to use to produce the headpose estimate

        Returns:
            np.ndarray: Euler angles ([pitch, roll, yaw])
        """
        # Obtain camera intrinsics to solve PnP algorithm. These intrinsics represent defaults - users may modify this
        # code to pass their own camera matrix and distortion coefficients if they happen to have calibrated their
        # camera: https://learnopencv.com/camera-calibration-using-opencv/
        h, w = img.shape[:2]
        camera_matrix = np.array([[w + h, 0, w // 2],
                                  [0, w + h, h // 2],
                                  [0, 0, 1]], dtype='float32')
        dist_coeffs = np.zeros((4, 1), dtype='float32')  # Assuming no lens distortion

        # Solve PnP using all 68 points:
        landmarks = landmarks.astype('float32')
        _, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, landmarks, camera_matrix, dist_coeffs,
                                                              flags=cv2.SOLVEPNP_EPNP)

        # Convert to Euler Angles
        euler_angles = convert_to_euler(np.squeeze(rotation_vector))

        # PnP may give values outside the range of (-90, 90), and sometimes misinterprets a face as facing
        # AWAY from the camera (since 2D landmarks do not convey whether face is facing towards or away from camera)
        # Thus, we adjust below to ensure the face is interpreted as front-facing
        euler_angles[euler_angles > 90] -= 180
        euler_angles[euler_angles < -90] += 180
        return euler_angles
