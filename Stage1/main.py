from perception.cameras import *
from environment.env_1d import *

demo_map = Env1d(r'environment/demo_map_1.jpg')

for _ in range(20):
    scan = camera_1d(demo_map.plan, demo_map.drone_coordinates, np.random.rand() * 2 * np.pi, np.pi / 6, 20, is_wall)
    demo_map.drone_coordinates = demo_map.place_drone(demo_map.plan)

demo_map.show_plan(demo_map.plan)

# print(scan)