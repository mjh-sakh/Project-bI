from perception.cameras import *
from environment.env_1d import *

plan = load_plan(r'environment/demo_map_1.jpg')

for _ in range(20):
    scan = camera_1d(plan, place_drone(plan), np.random.rand() * 2 * np.pi, np.pi / 6, 20, is_wall)

show_plan(plan)

# print(scan)