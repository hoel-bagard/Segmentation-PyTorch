from pathlib import Path
from typing import (
    Callable,
    Optional
)

import cv2
import numpy as np

from config.data_config import get_data_config
from src.torch_utils.utils.misc import clean_print


def default_loader(data_path: Path,
                   get_mask_path_fn: Callable[[Path], Path],
                   limit: int = None,
                   load_data: bool = False,
                   data_preprocessing_fn: Optional[Callable[[Path], np.ndarray]] = None,
                   labels_preprocessing_fn: Optional[Callable[[Path], np.ndarray]] = None,
                   verbose: bool = True
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Loads image and masks for image segmentation.

    This function assumes that the masks' paths contain either "mask" or "seg" (and that the main image does not).

    Args:
        data_path (Path): Path to the dataset folder
        get_mask_path_fn (callable): Function that returns the mask's path corresponding to a given image path
        limit (int, optional): If given then the number of elements for each class in the dataset
                            will be capped to this number
        load_data (bool): If true then this function returns the images already loaded instead of their paths.
                          The images are loaded using the preprocessing functions (they must be provided)
        data_preprocessing_fn (callable, optional): Function used to load data from their paths.
        labels_preprocessing_fn (callable, optional): Function used to load labels from their paths.
        verbose (bool): Verbose mode, print loading progress.

    Return:
        numpy arrays containing the paths/images and the associated label
    """
    data: list[np.ndarray | Path] = []
    labels: list[np.ndarray | Path] = []

    exts = [".jpg", ".png", ".bmp"]
    file_list = list([p for p in data_path.rglob('*') if p.suffix in exts
                      and "seg" not in str(p) and "mask" not in str(p)])
    nb_imgs = len(file_list)
    for i, img_path in enumerate(file_list, start=1):
        if verbose:
            clean_print(f"Processing image {img_path.name}    ({i}/{nb_imgs})", end="\r" if i != nb_imgs else "\n")

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


def default_load_data(data: Path | list[Path]) -> np.ndarray:
    """Function that loads image(s) from path(s).

    Args:
        data (Path, list[Path]): Either an image path or a batch of image paths, and return the loaded image(s)

    Returns:
        Image or batch of images in RGB format.
    """
    if isinstance(data, Path):
        img = cv2.imread(str(data))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
    else:
        imgs = []
        for image_path in data:
            imgs.append(default_load_data(image_path))
        return np.asarray(imgs)


def default_load_labels(label_paths: Path | list[Path]) -> np.ndarray:
    """Function that loads segmentation mask(s) from path(s).

    Args:
        data (Path, list[Path]): Either a mask path or a batch of mask paths, and return the loaded mask(s)

    Returns:
        Segmentation mask or batch of segmentation masks.
    """
    if isinstance(label_paths, Path):
        mask = cv2.imread(str(label_paths))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        height, width, _ = mask.shape

        # Transform the mask into a one hot mask
        data_config = get_data_config()
        one_hot_mask = np.zeros((height, width, data_config.OUTPUT_CLASSES))
        for key in range(len(data_config.IDX_TO_COLOR)):
            one_hot_mask[:, :, key][(mask == data_config.IDX_TO_COLOR[key]).all(axis=-1)] = 1

        # Assert to check that each pixel of the segmentation mask has a class. Not used for performance reasons.
        # assert np.sum(one_hot_mask) == np.prod(ModelConfig.IMAGE_SIZES), (f"At least one pixel has no class"
        #                                                                   f"in image {str(label_paths)}")

        return one_hot_mask
    else:
        img_masks = np.asarray([default_load_labels(image_path) for image_path in label_paths])
        return img_masks
