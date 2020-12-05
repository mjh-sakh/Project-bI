import unittest
import numpy as np
from time import sleep

from perception.detector import Detector
from environment.environment2d import Env2D


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


class TestEnvironment(unittest.TestCase):
    """
    Tests for Env2D
    """

    def setUp(self) -> None:
        # 20x20 px, outer pixel frame is pure green, next is black, rest is white
        self.image_test_map_path = "test_map.bmp"
        self.text_test_map_path = "test_map.csv"

    def tearDown(self) -> None:
        pass

    def test_Env2d_init_with_image(self):
        env = Env2D(self.image_test_map_path)

        self.assertEqual(env.plan_file_path, self.image_test_map_path)
        self.assertEqual(env.plan_type, "image")
        self.assertEqual(env.plan.shape, (20, 20, 3))  # size and rgb

    def test_Env2d_init_with_text(self):
        env = Env2D(self.text_test_map_path)

        self.assertEqual(env.plan_file_path, self.text_test_map_path)
        self.assertEqual(env.plan_type, "text")
        self.assertEqual(env.plan.shape, (4, 2, 2))  # 4 lines, 2 points, 2 coordinates

        self.assertEqual(len(env.viewer.geoms), 5)  # 4 edges + 1 drone

        env.drone_transform.set_translation(250, 250)
        # env.plan_trasnform.set_translation(200, 200)
        env.plan_trasnform.set_rotation(0.1)
        env.viewer.render(return_rgb_array=False)
        sleep(2)


if __name__ == '__main__':
    unittest.main()
