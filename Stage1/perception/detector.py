import cv2
import numpy as np


class Detector:
    """
    Class for 1d image feature detection and description.
    """

    def detect_and_compute(self, image):
        """
        Detects keypoints and computes their descriptors.
        :param image: input image - np.array of shape (1, w, 3)
        :return: two-element tuple (keypoints, descriptors), where the first element is a list of keypoints of class
        KeyPoint and the second is a list of descriptors.
        """

        keypoints = list()
        descriptors = list()

        for i in range(image.shape[1] - 1):
            px0 = image[:, i, :]
            px1 = image[:, i + 1, :]
            if not np.array_equal(px0, px1):
                # Keypoint is a point between two pixels with different colors
                keypoints.append(cv2.KeyPoint(i + 0.5, 0, 2))
                # Descriptor is a concatenation of the left and right pixel arrays
                descriptors.append(np.concatenate([px0.flatten(), px1.flatten()], axis=0))

        return keypoints, descriptors
