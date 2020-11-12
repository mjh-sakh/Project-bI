from perception.cameras import *
from environment.env_1d import *
import time


def move_to_wall(env1d):
    '''
    puts drone on a random point, counts the min turn, moves towards the wall in this direction 
    :param env1d: 
    :return: 
    '''''

    env1d.reset()
    plan_with_scan = None
    env1d.render(plan_background=plan_with_scan)
    _, _, curr_dir, _ = env1d.state

    turns = list(map(abs, [i * np.pi / 2 - curr_dir for i in range(3)]))
    min_turn = turns.index(min(turns))
    turn = min_turn*np.pi/2 - curr_dir
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
