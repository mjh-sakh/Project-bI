import numpy as np
import cv2 as cv
import os


def load_plan(file: str) -> np.array:
    """
    Read plan *file* in 'jpg' format and returns plan array.
    """
    assert os.path.isfile(file), f"No file '{file}' exist to read. Please check file name and path."
    return cv.imread(file)


def save_plan(plan: np.array, file: str):
    """
    Saves plan to file.
    """
    assert False, "Function is not implemented"


def show_plan(plan: np.array, window_name="Plan") -> int:
    """
    Shows plan in window until any button is hit.
    Returns key code which was pressed when window closed.
    """
    cv.imshow(window_name, plan)
    key_pressed = cv.waitKey(0)
    cv.destroyWindow(window_name)
    return key_pressed


def show_layers(base: np.array, *layers, **kwargs):
    """
    Takes *layers* (list of np.arrays) and shows them together until any button is hit.
    Takes *layers* and shows them together until any button is hit.
    *layers* is list of tuples (img, placement_coordinates, anchor):
        img is np.array with shape (x, y, 3)
        placement_coordiantes is where img should be inserted on top of base
        anchor is optional, by default top left corner of img is placed at placement_coordinates,
            so default value of anchor is [0, 0], provide different if some offset is needed
            typical use is center point of img

    Returns key code which was pressed when window closed.
    """

    def overlay_pic(pic, placement_coordinates, pic_anchor=[0, 0]):
        pic_anchor = np.array(pic_anchor)
        placement_coordinates = np.array(placement_coordinates)
        top_left = placement_coordinates - pic_anchor
        bottom_right = top_left + np.array(pic.shape[:2])
        blend[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = pic

    blend = base.copy()
    for layer in layers:
        if len(layer) == 3:
            img, placement_coordiantes, anchor = layer
        else:
            img, placement_coordiantes = layer
            anchor = [0, 0]

        overlay_pic(img, placement_coordiantes, anchor)

    if 'window_name' in kwargs:
        key_pressed = show_plan(blend, kwargs['window_name'])
    else:
        key_pressed = show_plan(blend)

    return key_pressed


def place_drone(plan) -> (int, int):
    """
    Randomly places agent on map/plan at valid point.
    Returns coordinates.
    """
    pixel = [0, 0, 0]
    while sum(pixel) < 255 * 3:
        x = np.random.randint(0, plan.shape[0])
        y = np.random.randint(0, plan.shape[1])
        pixel = plan[x, y]
    return x, y
