from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import numpy.typing as npt

from src.torch_utils.utils.misc import clean_print


def danger_p_loader(data_path: Path,
                    get_mask_path_fn: Callable[[Path], Path],
                    limit: int = None,
                    load_data: bool = False,
                    data_preprocessing_fn: Optional[Callable[[Path], npt.NDArray[np.uint8]]] = None,
                    labels_preprocessing_fn: Optional[Callable[[Path], npt.NDArray[np.uint8]]] = None,
                    verbose: bool = True
                    ) -> tuple[npt.NDArray, npt.NDArray]:
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

    file_list = list([p for p in data_path.rglob("*.jpg") if "disp" not in p.stem])
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


def danger_p_load_data(data: Path | list[Path]) -> np.ndarray:
    """Function that loads image(s) from path(s).

    Args:
        data (Path, list[Path]): Either an image path or a batch of image paths, and return the loaded image(s)

    Returns:
        Image or batch of images in RGB format.
    """
    if isinstance(data, Path):
        # Read the image and the disparity map in grayscale.
        img = cv2.imread(str(data), 0)
        disparity = cv2.imread(str(data.with_stem(data.stem + "_disp")), 0)
        res_img = np.stack((img, disparity), axis=-1)

        return res_img
    else:
        imgs = []
        for image_path in data:
            imgs.append(danger_p_load_data(image_path))
        return np.asarray(imgs)


def danger_p_load_labels(label_paths: Path | list[Path],
                         size: tuple[int, int],
                         idx_to_color: npt.NDArray[np.uint8]) -> np.ndarray:
    """Function that loads segmentation mask(s) from path(s).

    Args:
        data (Path, list[Path]): Either a mask path or a batch of mask paths, and return the loaded mask(s)
        size (tuple[int, int]): The width and height of the network's output.
        idx_to_color (np.ndarray): Array mapping an int (index) to a color.

    Returns:
        Segmentation mask or batch of segmentation masks.
    """

    # Handle missing class with sono ta. Resize to x2

    if isinstance(label_paths, Path):
        mask = cv2.imread(str(label_paths))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST_EXACT)

        # Transform the mask into a one hot mask
        width, height = size
        one_hot_mask = np.zeros((height, width, len(idx_to_color)))
        for key in range(len(idx_to_color)):
            one_hot_mask[:, :, key][(mask == idx_to_color[key]).all(axis=-1)] = 1

        return one_hot_mask
    else:
        img_masks = np.asarray([danger_p_load_labels(image_path, size, idx_to_color) for image_path in label_paths])
        return img_masks


if __name__ == "__main__":
    def _test_load_data():
        from argparse import ArgumentParser
        from src.torch_utils.utils.misc import show_img
        parser = ArgumentParser(description=("Script to test the image loading function. "
                                             "Run with 'python -m src.dataset.danger_p_loader <path>'"))
        parser.add_argument("img_path", type=Path, help="Path to one image.")
        args = parser.parse_args()

        img_path = args.img_path

        img = danger_p_load_data(img_path)
        show_img(img[..., 0], "Grayscale image")
        show_img(img[..., 1], "Disparity map")
    # _test_load_data()

    def _test_load_labels():
        from argparse import ArgumentParser
        from functools import partial

        from config.data_config import get_data_config
        from config.model_config import get_model_config

        parser = ArgumentParser(description=("Script to test the label loading function. "
                                             "Run with 'python -m src.dataset.danger_p_loader <path>'"))
        parser.add_argument("mask_path", type=Path, help="Path to one (8 by 5) mask.")
        args = parser.parse_args()

        mask_path = args.mask_path
        model_config = get_model_config()
        data_config = get_data_config()

        load_labels = partial(danger_p_load_labels,
                              size=model_config.OUTPUT_SIZES,
                              idx_to_color=data_config.IDX_TO_COLOR)

        mask = load_labels(mask_path)
        print(mask.shape)
        # To show the mask, modify the function to return the mask instead of the one hot.
        # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        # from src.torch_utils.utils.misc import show_img
        # show_img(mask)
    _test_load_labels()
