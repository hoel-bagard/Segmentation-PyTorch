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
                         idx_to_color: npt.NDArray[np.uint8],
                         max_danger_lvl: int) -> np.ndarray:
    """Function that loads segmentation mask(s) from path(s).

    Args:
        label_paths (Path, list[Path]): Either a mask path or a batch of mask paths, and return the loaded mask(s)
        size (tuple[int, int]): The width and height of the network's output.
        idx_to_color (np.ndarray): Array mapping an int (index) to a color.
        max_danger_lvl (int): Max danger level that can appear in the danger seg mask.

    Returns:
        Segmentation mask or batch of segmentation masks.
    """
    if isinstance(label_paths, Path):
        height, width = size
        cls_mask = cv2.imread(str(label_paths))
        cls_mask = cv2.cvtColor(cls_mask, cv2.COLOR_BGR2RGB)
        cls_mask = cv2.resize(cls_mask, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)

        # Transform the mask into a one hot mask
        one_hot_cls_mask = np.zeros((height, width, len(idx_to_color)))
        for key in range(len(idx_to_color)):
            # Skip duplicate colors.
            if key > 0 and (idx_to_color[key] == idx_to_color[key-1]).all(axis=0):
                print(idx_to_color[key])
                continue
            one_hot_cls_mask[:, :, key][(cls_mask == idx_to_color[key]).all(axis=-1)] = 1

        # Dirty and hardcoded for the project.
        danger_mask_path = label_paths.with_stem("_".join(label_paths.stem.split("_")[:-1]) + "_danger_mask")
        danger_mask = cv2.imread(str(danger_mask_path), 0)
        danger_mask = cv2.resize(danger_mask, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)

        one_hot_danger_mask = np.zeros((height, width, max_danger_lvl))
        for key in range(max_danger_lvl):
            one_hot_danger_mask[..., key][danger_mask == key] = 1

        # Assume that there are always more classes than danger levels.
        # Pad the danger mask to be able to stack them (makes it easier to handle than a tuple with the existing code)
        one_hot_danger_mask = np.pad(one_hot_danger_mask, ((0, 0), (0, 0), (0, len(idx_to_color)-max_danger_lvl)))
        one_hot_masks = np.stack((one_hot_cls_mask, one_hot_danger_mask), axis=-1)

        return one_hot_masks
    else:
        img_masks = np.asarray([danger_p_load_labels(label_path, size, idx_to_color, max_danger_lvl)
                                for label_path in label_paths])
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
        from src.torch_utils.utils.imgs_misc import show_img

        parser = ArgumentParser(description=("Script to test the label loading function. "
                                             "Run with 'python -m src.dataset.danger_p_loader <path>'"))
        parser.add_argument("mask_path", type=Path, help="Path to one (8 by 5) mask.")
        args = parser.parse_args()

        mask_path = args.mask_path
        model_config = get_model_config()
        data_config = get_data_config()

        load_labels = partial(danger_p_load_labels,
                              size=model_config.OUTPUT_SIZES,
                              idx_to_color=data_config.IDX_TO_COLOR,
                              max_danger_lvl=data_config.MAX_DANGER_LEVEL)

        one_hot_masks = load_labels(mask_path)
        oh_cls_mask, oh_danger_mask = one_hot_masks[..., 0], one_hot_masks[..., 1]

        cls_mask = np.argmax(oh_cls_mask, axis=-1)
        cls_mask_rgb = np.asarray(data_config.IDX_TO_COLOR[cls_mask], dtype=np.uint8)
        cls_mask_bgr = cv2.cvtColor(cls_mask_rgb, cv2.COLOR_RGB2BGR)
        print(f"{oh_cls_mask.shape=}, {cls_mask.shape=}")
        show_img(cls_mask_bgr)

        oh_danger_mask = oh_danger_mask[..., :data_config.MAX_DANGER_LEVEL]  # Remove padding
        danger_mask = np.argmax(oh_danger_mask, axis=-1)
        # Just make it possible to tell the levels from one another.
        danger_mask = (255 * danger_mask / data_config.MAX_DANGER_LEVEL).astype(np.uint8)
        print(f"{oh_danger_mask.shape=}, {danger_mask.shape=}")
        show_img(danger_mask)
    _test_load_labels()
