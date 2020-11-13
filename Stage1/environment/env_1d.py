import numpy as np
import cv2
import os
import gym
import pyglet
from gym import spaces
from gym.envs.classic_control import rendering
from perception.cameras import is_wall


class ImageObject:
    """
    Image Object that holds image information and provides it on read method.
    It's required to trick pyglet.image.load.
    """
    def __init__(self, _bytes):
        self.bytes = _bytes

    def read(self):
        return self.bytes


class NPImage(rendering.Geom):
    """
    Alternative Image class for OpenAI Gym.
    It creates image from provided numpy array.
    """
    def __init__(self, fname, array_img, width, height):
        rendering.Geom.__init__(self)
        self.extension = os.path.splitext(fname)[1].lower()
        self.set_color(1.0, 1.0, 1.0)
        self.width = width
        self.height = height
        file = self.convert_array_to_bytes(array_img)
        img = pyglet.image.load(fname, file=file)
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(0, 0, width=self.width, height=self.height)

    def convert_array_to_bytes(self, array: np.array):
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
            1: rotation speed, radians/sec 
        """
        self.action_space = spaces.Box(low=np.array([-50, -1]), high=np.array([100, 1]))

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

        speed, turn_speed = action
        x, y, theta, _ = self.state
        theta = (theta + turn_speed * self.tau) % (2 * np.pi)

        x += speed * np.cos(theta) * self.tau
        y += speed * np.sin(theta) * self.tau

        if not is_wall(self.plan[int(x), int(y)]):
            self.state = [x, y, theta, speed]
        else:
            self.state[3] = 0

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


    @staticmethod
    def draw_plan_and_frames(plan, frame_prev, frame_curr, matches, frame_width=15):
        """
        Creates an image with plan, current and previous frames and matching keypoints.
        :param plan: environment plan
        :param frame_prev: previous frame
        :param frame_curr: current frame
        :param matches: matched keypoints between frames
        :param frame_width: width of frame stripe at the bottom of plan image in px
        :return: resulting image
        """

        scale = plan.shape[1] / frame_prev.image.shape[1]

        scan_prev_resized = cv2.resize(frame_prev.image, None, fx=scale, fy=frame_width, interpolation=cv2.INTER_NEAREST)
        scan_curr_resized = cv2.resize(frame_curr.image, None, fx=scale, fy=frame_width, interpolation=cv2.INTER_NEAREST)

        kp1_resized = [cv2.KeyPoint((kp.pt[0] + 0.5) * scale, 0.5 * frame_width, kp.size) for kp in frame_prev.keypoints]
        kp2_resized = [cv2.KeyPoint((kp.pt[0] + 0.5) * scale, 0.5 * frame_width, kp.size) for kp in frame_curr.keypoints]

        for kp in kp1_resized:
            cv2.circle(scan_prev_resized, tuple(np.round(kp.pt).astype(np.int)), 5, (255, 255, 255), 2)
        for kp in kp2_resized:
            cv2.circle(scan_curr_resized, tuple(np.round(kp.pt).astype(np.int)), 5, (255, 255, 255), 2)

        scans = np.vstack([scan_prev_resized, scan_curr_resized])

        for match in matches:
            p1 = np.array(kp1_resized[match.queryIdx].pt)
            p2 = np.array(kp2_resized[match.trainIdx].pt)
            p1_coord = tuple(np.round(p1).astype(np.int))
            p2_coord = tuple(np.round(p2 + [0, scan_prev_resized.shape[0]]).astype(np.int))
            cv2.line(scans, p1_coord, p2_coord, (255, 255, 255), 2)

        output = np.vstack([plan, scans])
        return output
