from perception.sensors import camera_1d
from perception.sensors import is_wall
import time

import numpy as np


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
    _, _, theta, _ = env1d.state
    turns = list(map(abs, [i * np.pi / 2 - theta for i in range(4)]))
    min_turn = turns.index(min(turns))
    direction = min_turn * np.pi / 2
    turn_to_direction(env1d, direction)


def turn_to_direction(env1d, direction):
    """
    make turn to direction
    :param env1d:
    :param direction:
    :return:
    """
    plan_with_scan = None
    _, _, theta, _ = env1d.state
    clockwise = theta + np.pi > direction
    actions = [0, 0.1 * (2 * clockwise - 1)]
    while abs(theta - direction) > 0.05:
        env1d.render(plan_background=plan_with_scan)
        observation, reward, done, info = env1d.step(actions)
        x, y, theta, speed = observation


def wall_to_wall_movement(env1d):
    """
    random route from wall to wall
    :param env1d:
    :return:
    """
    plan_with_scan = None
    env1d.render(plan_background=plan_with_scan)
    time.sleep(1)
    actions = [10, 0]
    while True:
        env1d.render(plan_background=plan_with_scan)
        observation, reward, done, info = env1d.step(actions)
        x, y, theta, speed = observation
        plan_with_scan = env1d.plan.copy()
        scan = camera_1d(plan_with_scan, (x, y), theta, np.pi / 6, 20, is_wall)
        time.sleep(.1)
        if not speed:
            direction = np.random.rand() * 2 * np.pi
            turn_to_direction(env1d, direction)
