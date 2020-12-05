import numpy as np

def check_point_in_polygon(point, vertices) -> bool:
    """
    Checks if point is inside Polygon.
    "Left and horizontal bottom edges are in, right and horizontal top edges are out"

    References
    ----------
    .. [1] O'Rourke (1998), "Computational Geometry in C",
           Second Edition, Cambridge Unversity Press, Chapter 7

    :param point: tuple
    :param vertices: list of tuples
    :return: True if inside, False if outside
    """

    _vertices = np.array(vertices).astype(np.float)
    _vertices -= np.array(point)
    crossings = 0

    for i in range(len(vertices)):
        x0, y0 = _vertices[i]
        x1, y1 = _vertices[i-1]
        # check if edge straddles the x-axis
        if ((y0 > 0) != (y1 > 0)):
            # check if it crosses the ray to the right
            if ((x0 * y1 - x1 * y0) / (y1 - y0)) > 0:
                crossings += 1

    return True if (crossings & 1) else False
