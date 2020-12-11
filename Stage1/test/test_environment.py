import unittest
import numpy as np
from time import sleep

from environment.environment2d import Env2D


class TestEnvironmentInit(unittest.TestCase):
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
        self.assertEqual(env.plan.shape, (3, 2, 2))  # 3 lines, 2 points, 2 coordinates
        self.assertEqual(len(env.viewer.geoms), 5)  # 3 edges + background object + 1 drone

    def test_Env2d_place_drone_for_text_plan(self):
        env = Env2D(self.text_test_map_path)

        for i in range(20):
            env.reset()
            x, y, theta, _ = env.state
            self.assertTrue(y > x)  # map is top left part of square divided by half diagonally

            # env.viewer.render(return_rgb_array=False)
            # sleep(.5)

class TestEnvironmentRender(unittest.TestCase):
    """
    Tests for Env2D rendering
    """

    def setUp(self) -> None:
        # 20x20 px, outer pixel frame is pure green, next is black, rest is white
        self.image_test_map_path = "test_map.bmp"
        self.text_test_map_path = r"../environment/demo_map_4.csv"

    def tearDown(self) -> None:
        pass

    def test_Env2D_render_text_plan(self):
        env = Env2D(self.text_test_map_path, screen_height=600, screen_width=800)

        #visual test
        for _ in range(10):
            env.reset()
            env.render()
            sleep(.1)

        # visual test for scaling
        for i in range(10):
            env.rescale_view(1 - i/50)
            env.reset()
            env.render()
            sleep(.1)


if __name__ == '__main__':
    unittest.main()
