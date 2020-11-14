from environment.env_1d import *
from functools import reduce


def is_wall(pixel: np.array):
    """
    checks if pixel on map is wall or not
    """
    ray_color = np.array([90, 90, 0])
    if sum(pixel) > 250 * 3 or np.array_equal(pixel, ray_color):
        return False

    # if sum(pixel) > 250 * 3:
    #     return False

    return True


def cast_ray(plan: np.array, center: (float, float), angle: float, condition, display=False) -> (int, int):
    """
    Casts a ray on *plan* from observer *center* point in direction defined by *angle*.
    Returns coordinates of first pixel that matches provided *condition*.
    """
    x_start, y_start = center
    max_distance = np.sqrt(plan.shape[0]**2 + plan.shape[1]**2)
    x_end = x_start + max_distance * np.cos(angle)
    y_end = y_start + max_distance * np.sin(angle)
    ray = bresenham(int(x_start), int(y_start), int(x_end), int(y_end))
    for point in ray:
        x, y = point
        if condition(plan[x, y]):
            break
        if display:
            plan[x, y] = [90, 90, 0]
    return x, y


def camera_1d(plan: np.array, center: (float, float), direction: float, view_angle: float, resolution: int, condition):
    """
    Represents 1d camera which scans *plan* from *center* point in given *direction*. 
    Camera has *view_angle* in radians and *resolution*. 
    Wall or obscure object is defined by *condition*. 
    Returns scan, which is numpy array with shape (resolution, 1).
    """
    scan = []
    for i in range(resolution):
        # display = True if any([i == 0, i == int(resolution/2), i == resolution - 1]) else False
        display = True
        ray_i_angle = direction - view_angle * (i / resolution - 0.5)
        wall_pixel_coordinates = cast_ray(plan, center, ray_i_angle, condition, display)
        wall_pixel = plan[wall_pixel_coordinates]
        scan.append(wall_pixel)
    return np.array(scan)


def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    Source: https://github.com/encukou/bresenham
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy
