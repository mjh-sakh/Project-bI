import numpy as np

from perception.cameras import *
from environment.env_1d import *
from perception.detector import Detector
from perception.frame import Frame


def main():
    plan = Env1d.load_plan(r'environment/demo_map_2.bmp')

    detector = Detector()
    matcher = cv2.BFMatcher.create(normType=cv2.NORM_L2, crossCheck=True)

    scan_size = 20
    frame_prev = Frame(np.zeros((1, scan_size, 3), dtype=np.uint8), list(), list())

    count = 0
    while True:
        phi = count % 360
        count += 1

        plan_with_scan = np.copy(plan)

        scan = camera_1d(plan_with_scan, (230, 230), np.deg2rad(phi), np.pi / 6, 20, is_wall)
        scan = np.expand_dims(scan, axis=0)

        kp, des = detector.detect_and_compute(scan)
        frame_curr = Frame(scan, kp, des)

        matches = list()
        if frame_curr.descriptors and frame_prev.descriptors:
            matches = matcher.match(np.array(frame_prev.descriptors), np.array(frame_curr.descriptors))

        image_output = draw_pland_and_frames(plan_with_scan, frame_prev, frame_curr, matches)

demo_map = Env1d(r'environment/demo_map_1.jpg')

for _ in range(20):
    scan = camera_1d(demo_map.plan, demo_map.drone_coordinates, np.random.rand() * 2 * np.pi, np.pi / 6, 20, is_wall)
    demo_map.drone_coordinates = demo_map.place_drone()

demo_map.show_plan(demo_map.plan)


if __name__ == "__main__":
    main()
