from perception.cameras import *
from environment.env_1d import *
from perception.detector import Detector
from perception.frame import Frame
import time


def move_to_wall(env1d):
    env1d.reset()
    plan_with_scan = None
    env1d.render(plan_background=plan_with_scan)
    print(env1d.state)
    _, _, curr_dir, _ = env1d.state

    turns = list(map(abs, [i * np.pi / 2 - curr_dir for i in range(3)]))
    min_turn = turns.index(min(turns))
    turn = min_turn*np.pi/2 - curr_dir
    print(curr_dir, min_turn*np.pi/2,  turn)
    actions = [0, turn]
    observation, reward, done, info = env1d.step(actions)
    x, y, theta, speed = observation

    while not is_wall(env1d.plan[int(x), int(y)]):
        env1d.render(plan_background=plan_with_scan)
        actions = [10, 0]
        observation, reward, done, info = env1d.step(actions)
        x, y, theta, speed = observation
        plan_with_scan = env1d.plan.copy()
        scan = camera_1d(plan_with_scan, (x, y), theta, np.pi / 6, 20, is_wall)
        time.sleep(.1)

    env1d.close()
