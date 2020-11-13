from perception.cameras import *
from environment.env_1d import *
import time


def move_to_wall(env1d):
    """
    moves towards the wall in direction with the minimal turn
    :param env1d:
    :return:
    """

    plan_with_scan = None
    env1d.render(plan_background=plan_with_scan)
    time.sleep(1)

    min_turn_to_axis(env1d)

    actions = [10, 0]
    observation, reward, done, info = env1d.step(actions)
    x, y, theta, speed = observation

    while speed:
        env1d.render(plan_background=plan_with_scan)
        observation, reward, done, info = env1d.step(actions)
        x, y, theta, speed = observation
        plan_with_scan = env1d.plan.copy()
        scan = camera_1d(plan_with_scan, (x, y), theta, np.pi / 6, 20, is_wall)
        time.sleep(.1)


def min_turn_to_axis(env1d):
    """
    makes a minimal turn to the axis
    :param env1d:
    :return:
    """

    plan_with_scan = None
    _, _, theta, _ = env1d.state
    turns = list(map(abs, [i * np.pi / 2 - theta for i in range(4)]))
    min_turn = turns.index(min(turns))
    turn = int((min_turn * np.pi / 2 - theta) > 0)

    while abs(theta - min_turn * np.pi / 2) > 0.05:
        env1d.render(plan_background=plan_with_scan)
        actions = [0, 0.1 * (2 * turn - 1)]
        observation, reward, done, info = env1d.step(actions)
        x, y, theta, speed = observation
