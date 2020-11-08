import numpy as np
import cv2
import os


def load_plan(file: str) -> np.array:
    """
    Read plan *file* in 'jpg' format and returns plan array.
    """
    assert os.path.isfile(file), f"No file '{file}' exist to read. Please check file name and path."
    return cv2.imread(file)


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
    cv2.imshow(window_name, plan)
    key_pressed = cv2.waitKey(0)
    cv2.destroyWindow(window_name)
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

def draw_pland_and_frames(plan, frame_prev, frame_curr, matches):
    """
    Creates an image with plan, current and previous frames and matching keypoints.
    :param plan: environment plan
    :param frame_prev: previous frame
    :param frame_curr: current frame
    :param matches: matched keypoints between frames
    :return: resulting image
    """

    scale = plan.shape[1] / frame_prev.image.shape[1]

    scan_prev_resized = cv2.resize(frame_prev.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    scan_curr_resized = cv2.resize(frame_curr.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    kp1_resized = [cv2.KeyPoint((kp.pt[0] + 0.5) * scale, 0.5 * scale, kp.size) for kp in frame_prev.keypoints]
    kp2_resized = [cv2.KeyPoint((kp.pt[0] + 0.5) * scale, 0.5 * scale, kp.size) for kp in frame_curr.keypoints]

    for kp in kp1_resized:
        cv2.circle(scan_prev_resized, tuple(np.round(np.array(kp.pt)).astype(np.int)), 5, (255, 255, 255), 2)
    for kp in kp2_resized:
        cv2.circle(scan_curr_resized, tuple(np.round(np.array(kp.pt)).astype(np.int)), 5, (255, 255, 255), 2)

    scans = np.vstack([scan_prev_resized, scan_curr_resized])

    for match in matches:
        p1 = kp1_resized[match.queryIdx].pt
        p2 = kp2_resized[match.trainIdx].pt
        p1_coord = tuple(np.round(np.array(p1)).astype(np.int))
        p2_coord = tuple(np.round(np.array(p2) + [0, scan_prev_resized.shape[0]]).astype(np.int))
        cv2.line(scans, p1_coord, p2_coord, (255, 255, 255), 2)

    output = np.vstack([plan, scans])
    return output
