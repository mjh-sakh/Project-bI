import numpy as np
import cv2
import os
import gym
import pyglet
import csv
from gym import spaces
from gym.envs.classic_control import rendering
from perception.sensors import is_wall
from common import geometry


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
        self.plan = self._load_plan()
        if self.plan_type == "image":
            self.plan_shape = self.plan.shape[:2]
        elif self.plan_type == "text":
            self.plan_shape = max(self.plan[:, 0, 0]), max(self.plan[:, 0, 1])

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
        self.screen_width = kwargs.get("screen_width", min(self.plan_shape[0], 800))
        self.screen_height = kwargs.get("screen_height", min(self.plan_shape[1], 600))
        self.view_scale = 1
        self._set_up_window()

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        speed, turn_speed = action
        x, y, theta, _ = self.state
        theta = (theta + turn_speed * self.tau) % (2 * np.pi)

        x += speed * np.cos(theta) * self.tau
        y += speed * np.sin(theta) * self.tau

        if not self._check_collision(x, y):
            self.state = [x, y, theta, speed]
        else:
            self.state[3] = 0

        done = False  # never ending story :)
        reward = 0  # and fruitless one :(

        return np.array(self.state), reward, done, {}

    def _check_collision(self, x, y):
        """
        checks that
        :param x: current x coordinate of drone
        :param y: current y coordinate of drone
        :return: True if about to hit wall, False if not
        """
        if self.plan_type == "image":
            return is_wall(self.plan[int(x), int(y)])
        elif self.plan_type == "text":
            return not geometry.check_point_in_polygon((x, y), self.plan[:, 0, :])  # mind not

        assert f"Collision check for '{self.plan_type}' type of plan is not implemented."

    def reset(self):
        self.drone_coordinates = self._place_drone()
        self.state = [self.drone_coordinates[0], self.drone_coordinates[1], np.random.rand() * 2 * np.pi, 0]
        return np.array(self.state)

    def _set_up_window(self) -> None:
        """
        Fills self.viewer with required minimum objects:
        - pyglet.window via gym Viewer as a container
        - adds drone geometry into Viewer container

        Note: order is important
        Current order is following:
        - plan (one ImageAsArray or several Line classes)
        - background [-2]
        - drone [-1]
        """
        self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
        self.rescale_view(self.view_scale)

        self.plan_transform = rendering.Transform()
        if self.plan_type == "image":
            plan = ImageAsArray(self.plan_file_path, self.plan.swapaxes(0, 1)[::-1, :, :])
            plan.add_attr(self.plan_transform)
            self.viewer.add_geom(plan)
        elif self.plan_type == "text":
            for edge in self._generate_edges_from_vertices(self.plan):
                edge.add_attr(self.plan_transform)
                self.viewer.add_geom(edge)
        else:
            assert "Rendering of this type of plan is not implemented."

        # adding element right on top of plan so it can be shown on top of plan and hide it
        background = rendering.Point()
        background.add_attr(self.plan_transform)
        background.set_color(255, 255, 255)
        self.viewer.add_geom(background)

        drone = rendering.FilledPolygon([(-20, -5), (-20, 5), (0, 0)])
        self.drone_transform = rendering.Transform()
        drone.add_attr(self.drone_transform)
        self.viewer.add_geom(drone)

    def _generate_edges_from_vertices(self, vertices: np.array, linewidth=2, color=(0, 0, 0)):
        """
        Takes list of vertices of shape (number of vertices, 2, 2).
        :param vertices: np.array of shape (len, 2, 2)
        :return: list of OpenAI Gym rendering Line
        """
        edges = []
        for vertices_pair in vertices:
            edge = rendering.Line(vertices_pair[0], vertices_pair[1])
            edge.attrs[-1] = rendering.LineWidth(linewidth)  # Line class adds LineWidth attr at init, overriding it
            edge.set_color(*color)
            edges.append(edge)

        return edges

    def render(self, mode='human', plan_background=None):
        assert self.state, "Environment should be reset before first render."

        if plan_background is not None:
            background = ImageAsArray(self.plan_file_path, plan_background.swapaxes(0, 1)[::-1, :, :])
            background.add_attr(self.plan_transform)
            self.viewer.geoms[-2] = background
        else:
            background = rendering.Point()
            background.add_attr(self.plan_transform)
            background.set_color(255, 255, 255)
            self.viewer.geoms[-2] = background

        x, y, theta, _ = self.state
        self.drone_transform.set_translation(x, y)
        self.drone_transform.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def rescale_view(self, zoom_factor:float) -> None:
        """
        Zooms in/out rendered view based
        :param zoom_factor:
        :return:
        """
        self.view_scale = zoom_factor
        self.viewer.transform.set_scale(self.view_scale, self.view_scale)
        self.viewer.transform.set_translation((self.screen_width - self.plan_shape[0] * self.view_scale)/2,
                                                   (self.screen_height - self.plan_shape[1] * self.view_scale)/2)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _load_plan(self) -> np.array:
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
            return self._load_plan_from_image()
        if plan_extension in {".txt", ".csv"}:
            self.plan_type = "text"
            return self._load_plan_from_text()

        assert f"Unsupported plan file extension - {plan_extension}"

    def _load_plan_from_image(self) -> np.array:
        """
        Read plan file in 'jpg' format and returns plan array.
        :return: plan array
        """
        return cv2.imread(self.plan_file_path)

    def _load_plan_from_text(self) -> np.array:
        """
        Read plan file in 'jpg' format and returns plan array of pairs of vertices.
        :return:
        """

        plan = []
        with open(self.plan_file_path, 'r', encoding="UTF-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            for line in csv_reader:
                points = [int(x) for x in line]
                assert len(points) == 4, f"Wrong plan file format, should have 4 points per row, got {len(points)}"
                plan.append([points[:2], points[-2:]])

        return np.array(plan)

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

    def _place_drone(self) -> (int, int):
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
        elif self.plan_type == "text":
            while True:
                x = np.random.randint(0, self.plan_shape[0])
                y = np.random.randint(0, self.plan_shape[1])
                if geometry.check_point_in_polygon((x, y), self.plan[:, 0, :]):
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
