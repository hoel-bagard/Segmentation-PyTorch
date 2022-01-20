from pathlib import Path

import albumentations
import cv2
import numpy as np

from config.model_config import get_model_config
from src.torch_utils.utils.misc import show_img


def albumentation_wrapper(transform):
    """Returns a function that applies the albumentation transforms to a batch."""
    model_config = get_model_config()
    img_sizes = model_config.IMAGE_SIZES

    def albumentation_transform_fn(imgs: np.ndarray, labels: np.ndarray):
        """Apply transformations on a batch of data."""
        out_imgs = np.empty((imgs.shape[0], *img_sizes, 3), dtype=np.uint8)
        out_labels = np.empty((imgs.shape[0], *img_sizes), dtype=np.uint8)
        for i, (img, label) in enumerate(zip(imgs, labels)):
            transformed = transform(image=img, mask=label)
            out_imgs[i] = transformed["image"]
            out_labels[i] = transformed["mask"]
        return out_imgs, out_labels
    return albumentation_transform_fn


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description=("Script to test the augmentation pipeline."
                                         "Run with 'python -m src.dataset.data_transformations_albumentation <path>'."))
    parser.add_argument("img_path", type=Path, help="Path to the image to use.")
    parser.add_argument("--mask_path", "-m", type=Path, default=None,
                        help="Path to the mask corresponding to the image. Defaults to same name with '_mask.png'.")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    args = parser.parse_args()

    img_path: Path = args.img_path
    mask_path: Path = args.mask_path if args.mask_path else img_path.parent / (img_path.stem + "_mask.png")

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    model_config = get_model_config()
    img_sizes = model_config.IMAGE_SIZES

    batch_size = 4
    img_batch = np.asarray([img.copy() for _ in range(batch_size)])
    mask_batch = np.asarray([mask.copy() for _ in range(batch_size)])

    transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.2),
        albumentations.Resize(img_sizes[0], img_sizes[1], interpolation=cv2.INTER_AREA, always_apply=False, p=1)
    ])
    augmentation_pipeline = albumentation_wrapper(transform)

    aug_imgs, aug_masks = augmentation_pipeline(img_batch, mask_batch)

    # Prepare the original image / mask so that the can be displayed next to the augmented ones
    img = cv2.resize(img, img_sizes)
    mask = cv2.resize(mask, img_sizes, interpolation=cv2.INTER_NEAREST)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    original = cv2.hconcat([img, mask_bgr])

    assert set(mask.flat) == set(aug_masks[0].flat), "Impurities detected!"

    for i in range(batch_size):
        aug_masks_bgr = cv2.cvtColor(aug_masks[i], cv2.COLOR_GRAY2BGR)
        aug_result = cv2.hconcat([aug_imgs[i], aug_masks_bgr])
        display_img = cv2.vconcat([original, aug_result])
        show_img(display_img)
