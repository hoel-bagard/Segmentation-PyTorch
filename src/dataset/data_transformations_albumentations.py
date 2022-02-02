from pathlib import Path
from typing import Callable

import albumentations
import cv2
import numpy as np

from config.model_config import get_model_config
from src.torch_utils.utils.misc import show_img


def albumentation_wrapper(transform: albumentations.Compose) -> Callable[[np.ndarray, np.ndarray],
                                                                         tuple[np.ndarray, np.ndarray]]:
    """Returns a function that applies the albumentation transforms to a batch."""

    def albumentation_transform_fn(imgs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply transformations on a batch of data."""
        out_sizes = transform(image=imgs[0])["image"].shape[:2]
        out_imgs = np.empty((imgs.shape[0], *out_sizes, 3), dtype=np.float32)
        out_labels = np.empty((imgs.shape[0], *out_sizes, labels.shape[-1]), dtype=np.uint8)
        for i, (img, label) in enumerate(zip(imgs, labels)):
            transformed = transform(image=img, mask=label)
            out_imgs[i] = transformed["image"]
            out_labels[i] = transformed["mask"]
        return out_imgs, out_labels
    return albumentation_transform_fn


if __name__ == "__main__":
    def test_fn():
        from argparse import ArgumentParser
        from config.data_config import get_data_config

        parser = ArgumentParser(description=("Script to test the augmentation pipeline. "
                                             "Run with 'python -m src.dataset.data_transformations_albumentations "
                                             "<path>'."))
        parser.add_argument("img_path", type=Path, help="Path to the image to use.")
        parser.add_argument("--mask_path", "-m", type=Path, default=None,
                            help="Path to the mask corresponding to the image. Defaults to same name with '_mask.png'.")
        parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
        args = parser.parse_args()

        img_path: Path = args.img_path
        mask_path: Path = args.mask_path if args.mask_path else img_path.parent / (img_path.stem + "_mask.png")
        debug: bool = args.debug

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path))
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # This requires having a config and dataset (not ideal), but it's just to test so whatever...
        data_config = get_data_config()
        one_hot_mask = np.zeros((*mask.shape[:2][::1], data_config.OUTPUT_CLASSES))
        for key in range(len(data_config.COLOR_MAP)):
            one_hot_mask[:, :, key][(mask_rgb == data_config.COLOR_MAP[key]).all(axis=-1)] = 1

        model_config = get_model_config()
        img_sizes = model_config.IMAGE_SIZES

        batch_size = 4
        img_batch = np.asarray([img.copy() for _ in range(batch_size)])
        mask_batch = np.asarray([one_hot_mask.copy() for _ in range(batch_size)])

        transform = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.2),
            albumentations.Resize(img_sizes[0], img_sizes[1], interpolation=cv2.INTER_AREA, always_apply=False, p=1)
        ])
        augmentation_pipeline = albumentation_wrapper(transform)

        aug_imgs, aug_one_hot_masks = augmentation_pipeline(img_batch, mask_batch)

        # Prepare the original image / mask so that the can be displayed next to the augmented ones
        img = cv2.resize(img, img_sizes)
        mask = cv2.resize(mask, img_sizes, interpolation=cv2.INTER_NEAREST)
        original = cv2.hconcat([img, mask])

        for i in range(batch_size):
            # One hot to bgr
            aug_mask = np.argmax(aug_one_hot_masks[i], axis=-1)
            aug_mask_rgb = np.asarray(data_config.COLOR_MAP[aug_mask], dtype=np.uint8)
            aug_mask_bgr = cv2.cvtColor(aug_mask_rgb, cv2.COLOR_RGB2BGR)

            if debug:
                print(f"{aug_one_hot_masks[i].shape=}")
                print(f"{set(aug_one_hot_masks[i].flat)=}")
                print(f"{aug_mask.shape=}")
                print(f"{set(aug_mask.flat)=}")
            assert set(mask.flat) == set(aug_mask_bgr.flat), ("Impurities detected!    "
                                                              f"({set(mask.flat)} vs {set(aug_mask_bgr.flat)})")

            aug_result = cv2.hconcat([aug_imgs[i], aug_mask_bgr])
            display_img = cv2.vconcat([original, aug_result])
            show_img(display_img)
    test_fn()
