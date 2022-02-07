import numpy as np


def get_iou_bboxes(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> float:
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
    y_top = max(bbox1[1], bbox2[1])
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


def get_iou_masks(mask_label: np.ndarray, mask_pred: np.ndarray, color: tuple[int, int, int]) -> float:
    """Get the IoU of two masks for the given color."""
    assert mask_label.shape == mask_pred.shape, (f"Masks must have the same shape, but got {mask_label.shape} "
                                                 f"and {mask_pred.shape}")
    # Transform the masks into booleans (color / not color)
    labels_bool = (mask_label == np.asarray(color)).all(-1)
    pred_bool = (mask_pred == np.asarray(color)).all(-1)
    intersection = np.sum(np.logical_and(labels_bool, pred_bool))
    union = np.sum(labels_bool) + np.sum(pred_bool) - intersection
    return intersection / union
