from pathlib import Path
from itertools import product
from typing import (
    Callable,
    Union,
    Optional
)

import cv2
import numpy as np

from config.model_config import ModelConfig
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


def default_load_data(data: Union[Path, list[Path]], crop: bool = False,
                      top: int = 0, bottom: int = 1, left: int = 0, right: int = 1) -> np.ndarray:
    """
    Function that loads image(s) from path(s)
    Args:
        data: either an image path or a batch of image paths, and return the loaded image(s)
    """
    if isinstance(data, Path):
        img = cv2.imread(str(data))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # TODO: Move resize and crop to the pipeline
        # TODO: use https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html to do the resize on GPU ?
        # (I miss TF2 =(  )
        # # Resize the image in a sample to a given size.
        # if ModelConfig.IMAGE_SIZES:
        #     img = cv2.resize(img, ModelConfig.IMAGE_SIZES, interpolation=cv2.INTER_AREA)
        # # Crop the image
        # if crop:
        #     img[top:-bottom, left:-right]
        return img
    else:
        imgs = []
        for image_path in data:
            imgs.append(default_load_data(image_path))
        return np.asarray(imgs)


def default_load_labels(label_paths: Union[Path, list[Path]], crop: bool = False,
                        top: int = 0, bottom: int = 1, left: int = 0, right: int = 1) -> np.ndarray:
    """
    Function that loads image(s) from path(s)
    Args:
        data: either an image path or a batch of image paths, and return the loaded image(s)
    """
    if isinstance(label_paths, Path):
        img = cv2.imread(str(label_paths))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Transform the mask into a one hot mask
        width, height, _ = img.shape
        one_hot_mask = np.zeros((width, height, DataConfig.OUTPUT_CLASSES))
        for key in range(len(DataConfig.COLOR_MAP)):
            one_hot_mask[:, :, key][(img == DataConfig.COLOR_MAP[key]).all(axis=-1)] = 1

        # TODO: have an assert to check that each "pixel" has a value?

        # TODO: use https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html to do the resize on GPU ?
        # (I miss TF2 =(  )
        # Resize the image in a sample to a given size.
        # if ModelConfig.IMAGE_SIZES:
        #     img = cv2.resize(img, ModelConfig.IMAGE_SIZES, interpolation=cv2.INTER_AREA)
        # Crop the image
        # if crop:
        #     img[top:-bottom, left:-right]
        return one_hot_mask
    else:
        img_masks = np.asarray([default_load_labels(image_path) for image_path in label_paths])
        return img_masks
