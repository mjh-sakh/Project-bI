from environment.env_1d import *
from perception.detector import Detector
from perception.frame import Frame
from planning.movement import *
import time


def main():
    env2d = Env1d(r'environment/demo_map_2.bmp')
    env2d.reset()
    env2d.state[2] = np.pi / 2

    detector = Detector()
    matcher = cv2.BFMatcher.create(normType=cv2.NORM_L2, crossCheck=True)

    scan_size = 20
    frame_prev = Frame(np.zeros((1, scan_size, 3), dtype=np.uint8), list(), list())

    actions = [0, .05]

    #only one turn
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

    #movement to the wall

    wall_to_wall_movement(env2d)
    env2d.close()


def example_work_with_env():
    """
    Just an example of how to work with OpenGym environment.
    """
    env1d = Env2D(r'environment/demo_map_2.bmp')
    env1d.reset()
    actions = [0, 0]
    plan_with_scan = None

    for _ in range(100):

        env1d.render(plan_background=plan_with_scan)
        time.sleep(.1)

        observation, reward, done, info = env1d.step(actions)
        x, y, theta, speed = observation

        print(x, y)
        if not is_wall(env1d.plan[int(x), int(y)]):
            plan_with_scan = env1d.plan.copy()
            scan = camera_1d(plan_with_scan, (x, y), theta, np.pi / 6, 20, is_wall)
        else:
            plan_with_scan = None
        actions = demo_drive(actions)
        # actions = env1d.action_space.sample()  # random selection of speed and turn

    env1d.close()


def demo_drive(actions):
    """
    slowly turns and accelerates drone to show it's movement on the map
    """
    speed = actions[0] + 1
    turn = .1
    return [speed, turn]


if __name__ == "__main__":
    # example_work_with_env()
    main()
