from pathlib import Path

import cv2
import numpy as np

from config.model_config import ModelConfig


def load_dice(data_path: Path, limit: int = None, load_data: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads VOC labels for image segmentation.
    Args:
        data_path: Path to the dataset folder
        limit (int, optional): If given then the number of elements for each class in the dataset
                            will be capped to this number
        load_data: If true then this function returns the images instead of their paths
    Return:
        numpy array containing the paths/images and the associated label
    """
    data, labels = [], []

    for i, image_path in enumerate(data_path.glob("*.jpg")):
        segmentation_map_path = Path(str(image_path.stem) + "_segDotsTopOnly.png")
        if load_data:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if ModelConfig.IMAGE_SIZES:
                img = cv2.resize(img, ModelConfig.IMAGE_SIZES, interpolation=cv2.INTER_AREA)
            data.append(img)

            seg_map = cv2.imread(segmentation_map_path)
            seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB)
            if ModelConfig.IMAGE_SIZES:
                seg_map = cv2.resize(seg_map, ModelConfig.IMAGE_SIZES, interpolation=cv2.INTER_AREA)
            labels.append(seg_map)
        else:
            data.append(image_path)
            labels.append(segmentation_map_path)

        if i >= limit:
            break

    return np.asarray(data), np.asarray(labels)
