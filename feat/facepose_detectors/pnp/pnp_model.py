import os
import cv2
import numpy as np
from feat.utils import get_resource_path


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
        self.model_points = get_full_model_points(os.path.join(get_resource_path(), "3d_face_model.txt"))

    def predict(self, img, landmarks):
        """ Determines headpose using passed 68 2D landmarks

        Args:
            img (np.ndarray) : The cv2 image from which the landmarks were produced
            landmarks (np.ndarray) : The landmarks to use to produce the headpose estimate

        Returns:
            np.ndarray: Euler angles ([pitch, roll, yaw])
        """
        height, width = img.shape[:2]

        # Obtain camera intrinsics to solve PnP algorithm
        focal_length = width
        center = np.float32([width / 2, height / 2])
        camera_matrix = np.float32([[focal_length, 0.0, center[0]],
                                    [0.0, focal_length, center[1]],
                                    [0.0, 0.0, 1.0]])
        dist_coeffs = np.zeros((4, 1), dtype="float32")  # Assuming no lens distortion

        # Solve PnP
        landmarks = landmarks.astype('float32')

        # Use all 68 points:
        model_points = self.model_points

        # Use only a selection of points:
        # pts_of_interest = [35, 44, 29, 47, 53]  # 5-point landmarks
        # model_points = self.model_points[pts_of_interest]
        # landmarks = landmarks[pts_of_interest]

        _, rotation_vector, translation_vector = cv2.solvePnP(model_points, landmarks, camera_matrix,
                                                              dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Convert to Euler Angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = euler_angle
        euler_angles = np.array([pitch, roll, -yaw]).reshape(1, -1)

        # PnP may give values outside the range of (-90, 90), and sometimes misinterprets a face as facing
        # AWAY from the camera (since 2D landmarks do not convey whether face is facing towards or away from camera)
        # Thus, we adjust below to ensure the face is interpreted as front-facing
        euler_angles[euler_angles > 90] -= 180
        euler_angles[euler_angles < -90] += 180
        return euler_angles


def get_full_model_points(filename):
    """ Gets the model face coordinates necessary to compute the picture-in-point algorithm, which compares the 2D
    face landmarks identified in an image to the 3D model face coordinates in order to determine the pitch, roll, yaw
    of the head.

    Args:
        filename: Text file containing 3d coordinates for a model face

    Returns:
        np.ndarray: (68, 3) 3D landmarks on model face
    """
    raw_value = []
    with open(filename) as file:
        for line in file:
            raw_value.append(line)
    model_points = np.array(raw_value, dtype=np.float32)
    model_points = np.reshape(model_points, (3, -1)).T
    model_points[:, 2] *= -1
    model_points = model_points.astype('float32')
    return model_points
