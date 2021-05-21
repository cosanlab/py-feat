from .pnp_model import PerspectiveNPointModel


class PerspectiveNPoint:

    def __init__(self):
        self.model = PerspectiveNPointModel()

    def __call__(self, img, landmarks):
        """ Determines headpose using passed 68 2D landmarks
        Args:
            img (np.ndarray) : The cv2 image from which the landmarks were produced
            landmarks (np.ndarray) : The landmarks to use to produce the headpose estimate

        Returns:
            np.ndarray: Euler angles ([pitch, roll, yaw])
        """
        return self.model.predict(img, landmarks)
