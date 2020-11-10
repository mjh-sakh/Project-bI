import numpy as np
import cv2
import os
import gym
import pyglet
from gym import spaces
from gym.envs.classic_control import rendering


class ImageObject:
    def __init__(self, bites):
        self.bites = bites

    def read(self):
        return self.bites


class NPImage(rendering.Geom):
    def __init__(self, fname, array_img, width, height):
        rendering.Geom.__init__(self)
        self.extension = os.path.splitext(fname)[1].lower()
        self.set_color(1.0, 1.0, 1.0)
        self.width = width
        self.height = height
        file = self.conv_array_to_bytes(array_img)
        img = pyglet.image.load(fname, file=file)
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(0, 0, width=self.width, height=self.height)

    def conv_array_to_bytes(self, array: np.array):
        success, encoded_image = cv2.imencode(self.extension, array)
        return ImageObject(encoded_image.tobytes())


class Env1d(gym.Env):
    """
    Env1d will be an interface for other areas to operate in environment.
    """

    def __init__(self, plan_file_path: str):
        self.plan_file_path = plan_file_path
        self.plan = self.load_plan(plan_file_path)
        self.tau = 0.2  # seconds between state update

        """
        Actions space:
            0: speed, px/sec
            1: turn, radians 
        """
        self.action_space = spaces.Box(low=np.array([0, -2 * np.pi]), high=np.array([100, 2 * np.pi]))

        """
        Observation space = State:
            0: location x, px
            1: location y, px
            2: direction theta, radians from x axis ccw
            3: speed, px/s
        """
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]),
                                            high=np.array([self.plan.shape[0], self.plan.shape[1], 2 * np.pi, 100]))

        self.drone_coordinates = None
        self.state = None
        self.viewer = None

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        speed, turn = action
        x, y, theta, _ = self.state
        theta = (theta + turn) % (2 * np.pi)

        x += speed * np.cos(theta) * self.tau
        y += speed * np.sin(theta) * self.tau

        # TODO: collision check comes here

        self.state = [x, y, theta, speed]

        done = False  # never ending story :)
        reward = 0  # and fruitless one :(

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.drone_coordinates = self.place_drone()
        self.state = [self.drone_coordinates[0], self.drone_coordinates[1], np.random.rand() * 2 * np.pi, 0]
        return np.array(self.state)

    def render(self, mode='human', plan_background=None):
        screen_width = self.plan.shape[0]
        screen_height = self.plan.shape[1]

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            plan = NPImage(self.plan_file_path, self.plan, screen_width, screen_height)
            self.viewer.add_geom(plan)
            drone = rendering.FilledPolygon([(-20, -5), (-20, 5), (0, 0)])
            self.dronetrans = rendering.Transform()
            drone.add_attr(self.dronetrans)
            self.viewer.add_geom(drone)

        if plan_background is not None:
            new_plan = NPImage(self.plan_file_path, plan_background, screen_width, screen_height)
            self.viewer.geoms[0] = new_plan
        else:
            old_plan = NPImage(self.plan_file_path, self.plan, screen_width, screen_height)
            self.viewer.geoms[0] = old_plan

        if self.state is None:
            return None

        x, y, theta, _ = self.state
        self.dronetrans.set_translation(y, screen_width - x)
        self.dronetrans.set_rotation(theta - np.pi / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    @staticmethod
    def load_plan(file: str) -> np.array:
        """
        Read plan *file* in 'jpg' format and returns plan array.
        """
        assert os.path.isfile(file), f"No file '{file}' exist to read. Please check file name and path."
        return cv2.imread(file)

    @staticmethod
    def save_plan(plan: np.array, file: str):
        """
        Saves plan to file.
        """
        assert False, "Function is not implemented"

    @staticmethod
    def show_plan(plan: np.array, window_name="Plan") -> int:
        """
        Shows plan in window until any button is hit.
        Returns key code which was pressed when window closed.
        """
        cv2.imshow(window_name, plan)
        key_pressed = cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        return key_pressed

    def show_layers(self, *layers, **kwargs):
        """
        Takes *layers* (list of np.arrays) and shows them together until any button is hit.
        Takes *layers* and shows them together until any button is hit.
        *layers* is list of tuples (img, placement_coordinates, anchor):
            img is np.array with shape (x, y, 3)
            placement_coordinates is where img should be inserted on top of base
            anchor is optional, by default top left corner of img is placed at placement_coordinates,
                so default value of anchor is [0, 0], provide different if some offset is needed
                typical use is center point of img

        Returns key code which was pressed when window closed.
        """

        def overlay_pic(pic, placement_coordinates, pic_anchor=[0, 0]):
            pic_anchor = np.array(pic_anchor)
            placement_coordinates = np.array(placement_coordinates)
            top_left = placement_coordinates - pic_anchor
            bottom_right = top_left + np.array(pic.shape[:2])
            blend[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = pic

        blend = self.plan.copy()
        for layer in layers:
            if len(layer) == 3:
                img, placement_coordinates, anchor = layer
            else:
                img, placement_coordinates = layer
                anchor = [0, 0]

            overlay_pic(img, placement_coordinates, anchor)

        if 'window_name' in kwargs:
            key_pressed = self.show_plan(blend, kwargs['window_name'])
        else:
            key_pressed = self.show_plan(blend)

        return key_pressed

    def place_drone(self) -> (int, int):
        """
        Randomly places agent on map/plan at valid point.
        Returns coordinates.
        """
        pixel = [0, 0, 0]
        while sum(pixel) < 255 * 3:
            x = np.random.randint(0, self.plan.shape[0])
            y = np.random.randint(0, self.plan.shape[1])
            pixel = self.plan[x, y]
        return x, y
