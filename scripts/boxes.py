import numpy as np
from scripts.masks import get_iou as masks_get_iou
from scripts.utils import enumerate_over_pairs, remove_indices


def rescale_box(box, x_scale, y_scale):
    min_x, min_y, max_x, max_y = box
    return (min_x * x_scale, min_y * y_scale, max_x * x_scale, max_y * y_scale)


def get_area(b):
    """Get area of box (x1,y1,x2,y2)."""
    return (b[3] - b[1]) * (b[2] - b[0])


def get_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = min(max(0, ax2 - bx1), max(0, bx2 - ax1))
    iy = min(max(0, ay2 - by1), max(0, by2 - ay1))
    return ix * iy


def get_iou(a, b):
    """Compute iou between two boxes. Box is a tuple (x1,y1,x2,y2). Returns tuple (IOU,U,I)."""
    A = get_area(a)
    B = get_area(b)
    I = get_intersection_area(a, b)
    U = A + B - I
    return I / U, U, I


def get_pairwise_iou(boxes1, boxes2):
    iou = np.zeros((len(boxes1), len(boxes2)))
    for n, a in enumerate(boxes1):
        for m, b in enumerate(boxes2):
            iou[n, m] = get_iou(a, b)[0]
    return iou


def smallest_possible_mask_shape_for_boxes(boxes):
    """Get the shape of smallest possible 2D mask that contains all boxes."""
    return (max([i[2] for i in boxes] + [0]),
            max([i[3] for i in boxes] + [0]))


def boxes_to_mask(boxes, shape=None):
    """Turn set of boxes [(x1,y1,x2,y2),...] to binary mask of given shape.
    If shape is None, smallest possible shape is used."""
    if shape is None:
        shape = smallest_possible_mask_shape_for_boxes(boxes)
    mask = np.zeros(shape, dtype=np.bool)
    for (min_x, min_y, max_x, max_y) in boxes:
        mask[min_y:max_y, min_x:max_x] = True
    return mask


def get_iou_between_box_sets(boxes1, boxes2):
    """IOU between two sets of boxes. Every set of boxes is first turned into binary mask.
    Returns: IOU,Union area, Intersection area"""
    shape = smallest_possible_mask_shape_for_boxes(
        list(boxes1) + list(boxes2))
    mask1 = boxes_to_mask(boxes1, shape)
    mask2 = boxes_to_mask(boxes2, shape)
    return masks_get_iou(mask1, mask2)


def are_inside(a, b):
    """Check if two boxes are one inside another."""
    small, big = sorted([a, b], key=get_area)
    small_x1, small_y1, small_x2, small_y2 = small
    big_x1, big_y1, big_x2, big_y2 = big
    return (small_x1 >= big_x1) and (small_y1 >= big_y1) and (small_x2 <= big_x2) and (small_y2 <= big_y2)


def join_boxes(boxes, t=0.5):
    """Join together boxes with iou>t or which are one inside another.
    Is recursive, i.e. joined boxes can be also joined with other boxes.
    The order of joining matters, so we sort from the biggest to the smallest.
    Sometimes a bit wonky (e.g. if there is a chain of overlapping boxes, only part of them will be joined together and the chain will be split)."""
    result = [i for i in boxes]
    while True:
        result.sort(key=lambda x: -get_area(x))
        for n, m, a, b in enumerate_over_pairs(result, result):
            iou, _, _ = get_iou(a, b)
            if iou > t or are_inside(a, b):
                break  # if we found boxes to join, join them and start again
        else:
            break  # if we didn't find anything to join, break the while loop
        joined_box = (min(a[0], b[0]), min(a[1], b[1]),
                      max(a[2], b[2]), max(a[3], b[3]))
        remove_indices(result, [m, n])
        result.append(joined_box)
    return result
