class KeyPoint:
    """
    Class for 1d image keypoint.
    """

    def __init__(self, x):
        """
        Constructor.
        :param x: x-coordinate of keypoint.
        """
        self.__x = x

    @property
    def x(self):
        """
        Get or set x value
        :return: x value
        """
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = x
