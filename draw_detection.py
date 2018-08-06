import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_labeled_bounding_boxes(img, labeled_frame, num_objects):
    """
    Starting from labeled regions, draw enclosing rectangles in the original color frame.
    """
    # Iterate through all detected cars
    for car_number in range(1, num_objects+1):

        # Find pixels with each car_number label value
        rows, cols = np.where(labeled_frame == car_number)

        # Find minimum enclosing rectangle
        x_min, y_min = np.min(cols), np.min(rows)
        x_max, y_max = np.max(cols), np.max(rows)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=6)

    return img


def compute_heatmap_from_detections(frame, hot_windows, threshold=5):
    """
    Compute heatmaps from windows classified as positive, in order to filter false positives.
    """
    h, w, c = frame.shape

    heatmap = np.zeros(shape=(h, w), dtype=np.uint8)

    for bbox in hot_windows:
        # for each bounding box, add heat to the corresponding rectangle in the image
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        heatmap[y_min:y_max, x_min:x_max] += 1  # add heat

    # apply threshold + morphological closure to remove noise
    _, heatmap_thresh = cv2.threshold(heatmap, threshold, 255, type=cv2.THRESH_BINARY)
    heatmap_thresh = cv2.morphologyEx(heatmap_thresh, op=cv2.MORPH_CLOSE,
                                      kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)

    return heatmap, heatmap_thresh


def draw_boxes(img, bbox_list, color=(0, 0, 255), thick=5):
    """
    Draw all bounding boxes in `bbox_list` onto a given image.
    :param img: input image
    :param bbox_list: list of bounding boxes
    :param color: color used for drawing boxes
    :param thick: thickness of the box line
    :return: a new image with the bounding boxes drawn
    """
    # Make a copy of the image
    img_copy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bbox_list:
        tl_corner = tuple(bbox[0])
        br_corner = tuple(bbox[1])
        cv2.rectangle(img_copy, tl_corner, br_corner, color, thick)
    return img_copy


if __name__=='__main__':
    pass