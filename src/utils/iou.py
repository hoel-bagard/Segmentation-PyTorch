def get_iou(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    TODO: Add a named tuple BBox  (to make things more readable)

    Args:
        bbox1: Tuple with the bbox's coordinates: (left, top, width, height)
        bbox2: Tuple with the bbox's coordinates: (left, top, width, height)

    Returns:
        The IoU (a float in [0, 1])
    """
    # Coordinates of the interesection bbox
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[0], bbox2[0])
    x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox1[2])
    y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox1[3])

    # No overlapp
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute the areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]

    # Compute the IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou
