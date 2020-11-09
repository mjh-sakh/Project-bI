class Frame:
    """
    Class representing image frame with the corresponding features.
    """

    def __init__(self, image, keypoints, descriptors):
        """
        Frame constructor.
        :param image: frame image
        :param keypoints: frame keypoints
        :param descriptors: frame keypoints descriptors
        """
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
