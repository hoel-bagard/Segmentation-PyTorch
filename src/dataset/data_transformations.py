import random
from typing import Callable

import cv2
import numpy as np
import torch
# from einops import rearrange

NumpyOrTensorType = torch.Tensor | np.ndarray


def compose_transformations(transformations: list[Callable[[NumpyOrTensorType, NumpyOrTensorType],
                                                           tuple[NumpyOrTensorType, NumpyOrTensorType]]]):
    """Returns a function that applies all the given transformations."""
    def compose_transformations_fn(imgs: NumpyOrTensorType, labels: NumpyOrTensorType):
        """Apply transformations on a batch of data."""
        for fn in transformations:
            imgs, labels = fn(imgs, labels)
        return imgs, labels
    return compose_transformations_fn


def horizontal_flip(imgs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Randomly flips the img around the y-axis."""
    for i in range(len(imgs)):
        if random.random() > 0.5:
            imgs[i] = cv2.flip(imgs[i], 1)
            labels[i] = labels[i, :, ::-1]
    return imgs, labels


def to_tensor():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to_tensor_fn(imgs: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        imgs = imgs.transpose((0, 3, 1, 2))
        # labels = rearrange(labels, "b w h c m -> b m c w h")
        return torch.from_numpy(imgs).float().to(device), torch.from_numpy(labels).to(device)
    return to_tensor_fn


if __name__ == "__main__":
    def _test_fn():
        from argparse import ArgumentParser
        from pathlib import Path
        from config.data_config import get_data_config
        from src.dataset.danger_p_loader import danger_p_load_data, danger_p_load_labels
        from src.torch_utils.utils.imgs_misc import show_img

        parser = ArgumentParser(description=("Script to test the augmentation functions. "
                                             "Run with 'python -m src.dataset.data_transformations <path>'."))
        parser.add_argument("img_path", type=Path, help="Path to the image to use.")
        parser.add_argument("--mask_path", "-m", type=Path, default=None,
                            help="Path to the mask corresponding to the image. Defaults to same name with '_mask.png'.")
        args = parser.parse_args()

        img_path: Path = args.img_path
        mask_path: Path = args.mask_path if args.mask_path else img_path.parent / (img_path.stem + "_mask.png")

        data_config = get_data_config()

        img = danger_p_load_data(img_path)
        one_hot_mask = danger_p_load_labels(mask_path, (10, 16), data_config.IDX_TO_COLOR, data_config.MAX_DANGER_LEVEL)

        rgb_mask = np.asarray(data_config.IDX_TO_COLOR[np.argmax(one_hot_mask[..., 0], axis=-1)], dtype=np.uint8)
        show_img(img[..., 0])
        show_img(rgb_mask)

        flipped_img, flipped_mask = horizontal_flip(np.expand_dims(img, 0), np.expand_dims(one_hot_mask, 0))
        flipped_img, flipped_mask = flipped_img[0], flipped_mask[0]

        flipped_mask = flipped_mask[..., 0]  # Discard the danger mask
        flipped_mask = np.argmax(flipped_mask, axis=-1)
        flipped_mask = np.asarray(data_config.IDX_TO_COLOR[flipped_mask], dtype=np.uint8)

        show_img(img[..., 0])
        show_img(flipped_mask)
    _test_fn()
