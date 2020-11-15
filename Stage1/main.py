from environment.environment2d import Env2D
from perception.detector import Detector
from perception.frame import Frame
from planning.movement import wall_to_wall_movement
from perception.sensors import camera_1d
from perception.sensors import is_wall

import numpy as np
import cv2


def main():
    env2d = Env2D(r'environment/demo_map_2.bmp')
    env2d.reset()
    env2d.state[2] = np.pi / 2

    detector = Detector()
    matcher = cv2.BFMatcher.create(normType=cv2.NORM_L2, crossCheck=True)

    scan_size = 20
    frame_prev = Frame(np.zeros((1, scan_size, 3), dtype=np.uint8), list(), list())

    actions = [0, .05]

    # only one turn
    while True:
        plan_with_scan = np.copy(env2d.plan)

        observation, _, _, _ = env2d.step(actions)
        x, y, theta, _ = observation

        scan = camera_1d(plan_with_scan, (x, y), theta, np.pi / 6, scan_size, is_wall)
        scan = np.expand_dims(scan, axis=0)

        kp, des = detector.detect_and_compute(scan)
        frame_curr = Frame(scan, kp, des)

        matches = list()
        if frame_curr.descriptors and frame_prev.descriptors:
            matches = matcher.match(np.array(frame_prev.descriptors), np.array(frame_curr.descriptors))

        image_output = env2d.draw_plan_and_frames(plan_with_scan, frame_prev, frame_curr, matches)
        frame_prev = frame_curr
        env2d.render(plan_background=image_output)

        if theta > 2 * np.pi - 0.1:
            break

    # movement to the wall

    wall_to_wall_movement(env2d)
    env2d.close()


if __name__ == "__main__":
    main()
