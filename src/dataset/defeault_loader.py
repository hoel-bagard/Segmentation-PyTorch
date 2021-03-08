from pathlib import Path
from typing import (
    Callable,
    Union,
    Optional
)

import cv2
import numpy as np

from config.data_config import DataConfig
from src.torch_utils.utils.misc import clean_print


def default_loader(data_path: Path, get_mask_path_fn: Callable[[Path], Path],
                   limit: int = None, load_data: bool = False,
                   data_preprocessing_fn: Optional[Callable[[Path], np.ndarray]] = None,
                   labels_preprocessing_fn: Optional[Callable[[Path], np.ndarray]] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads image and masks for image segmentation.
    Args:
        data_path: Path to the dataset folder
        get_mask_path: Function that returns the mask's path corresponding to a given image path
        limit (int, optional): If given then the number of elements for each class in the dataset
                            will be capped to this number
        load_data: If true then this function returns the images instead of their paths using the preprocessing_fns
        data_preprocessing_fn: Function used to load data from their paths.
        labels_preprocessing_fn: Function used to load labels from their paths.
    Return:
        numpy array containing the paths/images and the associated label
    """
    data, labels = [], []

    exts = [".jpg", ".png"]
    file_list = list([p for p in data_path.rglob('*') if p.suffix in exts and "mask" not in str(p)])
    nb_imgs = len(file_list)
    for i, img_path in enumerate(file_list, start=1):
        clean_print(f"Processing image {img_path.name}    ({i}/{nb_imgs})", end="\r")

        segmentation_map_path = get_mask_path_fn(img_path)
        if load_data:
            data.append(data_preprocessing_fn(img_path))
            labels.append(labels_preprocessing_fn(segmentation_map_path))
        else:
            data.append(img_path)
            labels.append(segmentation_map_path)

        if limit and i >= limit:
            break

    return np.asarray(data), np.asarray(labels)


def default_load_data(data: Union[Path, list[Path]], size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """
    Function that loads image(s) from path(s)
    Args:
        data: either an image path or a batch of image paths, and return the loaded image(s)
        size: size to which resize the images
    Returns:
        Image or batch of image
    """
    if isinstance(data, Path):
        img = cv2.imread(str(data))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize is in the data loading function because it needs to be done
        # before transforming the segmentation map into a one hot.
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        return img
    else:
        imgs = []
        for image_path in data:
            imgs.append(default_load_data(image_path))
        return np.asarray(imgs)


def default_load_labels(label_paths: Union[Path, list[Path]], size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """
    Function that loads image(s) from path(s)
    Args:
        data: either an image path or a batch of image paths, and return the loaded image(s)
        size: size to which resize the images
    Returns:
        Segmentation mask or batch of segmentation masks
    """
    if isinstance(label_paths, Path):
        img = cv2.imread(str(label_paths))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        # Transform the mask into a one hot mask
        width, height, _ = img.shape
        one_hot_mask = np.zeros((width, height, DataConfig.OUTPUT_CLASSES))
        for key in range(len(DataConfig.COLOR_MAP)):
            one_hot_mask[:, :, key][(img == DataConfig.COLOR_MAP[key]).all(axis=-1)] = 1

        # Assert to check that each pixel of the segmentation mask has a class. Not used for performance.
        # assert np.sum(one_hot_mask) == width * height, f"At least one pixel has no class in image {str(label_paths)}"

        return one_hot_mask
    else:
        img_masks = np.asarray([default_load_labels(image_path) for image_path in label_paths])
        return img_masks
