from .pnp_model import PerspectiveNPointModel


class PerspectiveNPoint:
    def __init__(self):
        self.model = PerspectiveNPointModel()

    def __call__(self, frames, landmarks):
        """Determines headpose using passed 68 2D landmarks
        Args:
            frames (np.ndarray) : A list of cv2 images from which the landmarks were produced
            landmarks (np.ndarray) : The landmarks used to produce headpose estimates

        Returns:
            np.ndarray: (num_images, num_faces, [pitch, roll, yaw]) - Euler angles (in degrees) for each face within in
                        each image
        """
        all_poses = []
        for image, image_landmarks in zip(frames, landmarks):
            poses_in_this_img = []
            for face_landmarks in image_landmarks:
                poses_in_this_img.append(self.model.predict(image, face_landmarks))
            all_poses.append(poses_in_this_img)

        return all_poses
