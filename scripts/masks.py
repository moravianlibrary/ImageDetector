import numpy as np
from skimage import measure


def get_iou(a, b):
    """Get IOU between two binary masks with the same shape. Returns tuple (IOU,U,I)."""
    A = np.count_nonzero(a)
    B = np.count_nonzero(b)
    I = np.count_nonzero(np.logical_and(a, b))
    U = A + B - I
    if not U:
        return 1, U, I
    return I / U, U, I


def mask_to_box(mask):
    """Convert mask to a bounding box. Assumes there is only one box."""
    nonzero = np.nonzero(mask)
    min_x = min(nonzero[1])
    max_x = max(nonzero[1]) + 1
    min_y = min(nonzero[0])
    max_y = max(nonzero[0]) + 1
    return(min_x, min_y, max_x, max_y)


def mask_to_boxes(mask):
    """Convert mask to boxes. Assumes there are multiple non-overlapping boxes.
    If boxes overlap, draws box around every overlapping group of boxes.
    """
    regions, num_regions = measure.label(mask, return_num=True)
    boxes = []
    for r in range(1, num_regions + 1):
        nonzero = np.nonzero(regions == r)
        min_x = min(nonzero[1])
        max_x = max(nonzero[1]) + 1
        min_y = min(nonzero[0])
        max_y = max(nonzero[0]) + 1
        box = (min_x, min_y, max_x, max_y)
        boxes.append(box)
    return boxes


def clean_margins(mask, margin_ratio=0.125):
    """Turns to False everything inside the margins that doesn't have True value next to itself in direction away from edge.
    Corners are not handled properly, but it's good enough.

    !CHANGES MASK IN PLACE!

    Input: 
    mask: boolean array
    margin_ratio: ratio of margin width compared to width and height of the mask

    Examples:

    before cleaning:

    _____XXXXXXXX___XXX__  < - top edge of mask
    _______XXX___________
    _______XXXXX___XX____  < - last line of margin
    ____XXXXXXXX___XXXX__

    after cleaning:

    _______XXX___________  < - top edge of mask
    _______XXX___________
    _______XXXXX___XX____  < - last line of margin
    ____XXXXXXXX___XXXX__


    """
    assert margin_ratio < 0.5
    h, w = mask.shape
    mx = int(margin_ratio * w)
    my = int(margin_ratio * h)
    #top and bottom
    for i in range(my + 1):
        y = my - i
        mask[y, :] = np.where(
            mask[y + 1, :], mask[y, :], False)
        mask[-y, :] = np.where(mask[-y - 1, :],
                               mask[-y, :], False)
    #left and right
    for i in range(mx + 1):
        x = mx - i
        mask[:, x] = np.where(
            mask[:, x + 1], mask[:, x], False)
        mask[:, -x] = np.where(mask[:, -x - 1],
                               mask[:, -x], False)
