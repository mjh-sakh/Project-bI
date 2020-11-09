import unittest
import numpy as np

from perception.detector import Detector


class TestDetector(unittest.TestCase):
    """ Tests for Detector class """

    def test_detect_and_compute(self):
        """
        Test for detect_and_compute method.
        :return:
        """
        image = np.zeros((1, 6, 3), dtype=np.uint8)
        for i in range(3):
            image[:, i, :] = np.array([100, 100, 100])
        for i in range(3, 6):
            image[:, i, :] = np.array([200, 200, 200])

        detector = Detector()
        keypoints, descriptors = detector.detect_and_compute(image)

        self.assertEqual(len(keypoints), len(descriptors))
        self.assertEqual(len(keypoints), 1)
        self.assertEqual(keypoints[0].pt[0], 2.5)
        self.assertEqual(keypoints[0].pt[1], 0)
        np.testing.assert_array_equal(descriptors[0], np.array([100] * 3 + [200] * 3))


if __name__ == '__main__':
    unittest.main()
