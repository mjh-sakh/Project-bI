import numpy as np
import cv2
import os
import gym
import pyglet
from gym import spaces
from gym.envs.classic_control import rendering
from perception.sensors import is_wall


class MockImageObject:
    """
    Image Object that holds image information and provides it on read method.
    It's required to trick pyglet.image.load.
    """

    def __init__(self, _bytes):
        self.bytes = _bytes

    def read(self):
        return self.bytes


class ImageAsArray(rendering.Geom):
    """
    Alternative Image class for OpenAI Gym.
    It creates image from provided numpy array.
    """

    def __init__(self, fname, array_img):
        rendering.Geom.__init__(self)
        self.extension = os.path.splitext(fname)[1].lower()
        self.set_color(1.0, 1.0, 1.0)
        self.width, self.height = array_img.shape[:2]
        file = self.convert_array_to_bytes(array_img)
        img = pyglet.image.load(fname, file=file)
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(0, 0)

    def convert_array_to_bytes(self, array: np.array):
        success, encoded_image = cv2.imencode(self.extension, array)
        return MockImageObject(encoded_image.tobytes())


class Env2D(gym.Env):
    """
    Env2D is a container for drone navigation on given plan.
    Implemented feature:
        step based on provided actions [speed, rotation speed]
        render
        basic collision detection (drone will not drive into wall)
    """

    def __init__(self, plan_file_path: str, **kwargs):
        self.plan_file_path = plan_file_path
        self.kwargs = kwargs
        self.plan_type = None  # image or text
        self.plan = self.load_plan()
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
        self.screen_width = kwargs.get("screen_width", 500)
        self.screen_height = kwargs.get("screen_height", 500)
        self.set_up_window()

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

    def set_up_window(self):
        """
        Fills self.viewer with required minimum objects:
        - pyglet.window via gym Viewer as a container
        - adds drone geometry into Viewer container

        Note: order is important
        :return:
        """
        self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

        if self.plan_type == "image":
            plan = ImageAsArray(self.plan_file_path, self.plan)
            self.viewer.add_geom(plan)
        elif self.plan_type == "text":
            assert "Not implemented"
        else:
            assert "Rendering of this type of plan is not implemented."

        drone = rendering.FilledPolygon([(-20, -5), (-20, 5), (0, 0)])
        self.drone_transform = rendering.Transform()
        drone.add_attr(self.drone_transform)
        self.viewer.add_geom(drone)

    def render(self, mode='human', plan_background=None):
        if self.state is None:
            assert "Environment should be reset before first render."

        if plan_background is not None:
            new_plan = ImageAsArray(self.plan_file_path, plan_background)
            self.viewer.geoms[0] = new_plan
        else:
            old_plan = ImageAsArray(self.plan_file_path, self.plan)
            self.viewer.geoms[0] = old_plan

        x, y, theta, _ = self.state
        self.drone_transform.set_translation(y, self.screen_width - x)  # TODO: requires proper transformation
        self.drone_transform.set_rotation(theta - np.pi / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def load_plan(self) -> np.array:
        """
        Checks plans extension and initiates right load process:
        - from image (jpg, bmp)
        - from text (txt, csv)
        Read plan *file* in 'jpg' format and returns plan array.
        """
        assert os.path.isfile(
            self.plan_file_path), f"No file '{self.plan_file_path}' exist to read. Please check file name and path."
        plan_extension = os.path.splitext(self.plan_file_path)[1].lower()
        if plan_extension in {".jpg", ".bmp"}:
            self.plan_type = "image"
            return self.load_plan_from_image()
        if plan_extension in {".txt", ".csv"}:
            self.plan_type = "text"
            return self.load_plan_from_text()

        assert f"Unsupported plan file extension - {plan_extension}"

    def load_plan_from_image(self) -> np.array:
        """
        Read plan file in 'jpg' format and returns plan array.
        :return: plan array
        """
        return cv2.imread(self.plan_file_path)

    def load_plan_from_text(self) -> np.array:
        """
        Read plan file in 'jpg' format and returns plan array of pairs of vertices.
        :return:
        """
        assert "Not implemented"

    def save_plan(self, plan: np.array, file: str):
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

    def place_drone(self) -> (int, int):
        """
        Randomly places agent on map/plan at valid point.
        Returns coordinates.
        """

        if self.plan_type == "image":
            while True:
                x = np.random.randint(0, self.plan.shape[0])
                y = np.random.randint(0, self.plan.shape[1])
                pixel = self.plan[x, y]
                if sum(pixel) == 255 * 3:
                    return x, y
        else:
            assert f"Drone placement is not implemented for '{self.plan_type}' plan type"

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

        scan_prev_resized = cv2.resize(frame_prev.image, None, fx=scale, fy=frame_width,
                                       interpolation=cv2.INTER_NEAREST)
        scan_curr_resized = cv2.resize(frame_curr.image, None, fx=scale, fy=frame_width,
                                       interpolation=cv2.INTER_NEAREST)

        kp1_resized = [cv2.KeyPoint((kp.pt[0] + 0.5) * scale, 0.5 * frame_width, kp.size) for kp in
                       frame_prev.keypoints]
        kp2_resized = [cv2.KeyPoint((kp.pt[0] + 0.5) * scale, 0.5 * frame_width, kp.size) for kp in
                       frame_curr.keypoints]

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

        output = plan.copy()
        output[- frame_width * 2:, :, :] = scans
        return output
