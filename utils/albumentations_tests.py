import shutil
from argparse import ArgumentParser
from pathlib import Path

import albumentations
import cv2
import numpy as np


def show_img(img: np.ndarray, window_name: str = "Image"):
    """Displays an image until the user presses the "q" key.

    Args:
        img: The image that is to be displayed.
        window_name (str): The name of the window in which the image will be displayed.
    """
    while True:
        # Make the image full screen if it's above a given size (assume the screen isn't too small^^)
        if any(img.shape[:2] > np.asarray([1080, 1440])):
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def denormalize(img: np.ndarray, mean: tuple[float, float, float], std: tuple[float, float, float]) -> np.ndarray:
    """Undo the normalization process on an image.

    Args:
        img (np.ndarray): The normalized image.
        mean (tuple): The mean values that were used to normalize the image.
        std (tuple): The std values that were used to normalize the image.

    Returns:
        The denormalized image.
    """
    std = np.asarray(std)
    mean = np.asarray(mean)
    img = img * (255*std) + 255*mean
    return img.astype(np.uint8)


def main():
    parser = ArgumentParser(description="Script to do data augmentation tests with albumentations.")
    parser.add_argument("data_path", type=Path, help="Path to the dataset.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    sizes = (512, 512)
    p_value = 1

    # Data augmentation done on cpu.
    pipeline = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        # albumentations.RandomRotate90(p=0.2),
        # albumentations.CLAHE(),
        albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p_value),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=p_value),
        albumentations.ShiftScaleRotate(scale_limit=0.05, rotate_limit=10, shift_limit=0.06, p=p_value,
                                        border_mode=cv2.BORDER_CONSTANT, value=0),
        # albumentations.GridDistortion(p=0.5),
        albumentations.Normalize(mean=(0.041, 0.129, 0.03), std=(0.054, 0.104, 0.046), max_pixel_value=255.0, p=1.0),
        albumentations.Resize(*sizes, interpolation=cv2.INTER_LINEAR)
    ])

    # This assumes that the masks' paths contain either "mask" or "seg" (and that the main image does not).
    exts = [".jpg", ".png", ".tiff"]
    img_path_list = list([p for p in data_path.rglob('*') if p.suffix in exts
                          and "seg" not in str(p) and "mask" not in str(p)])

    nb_imgs = len(img_path_list)
    for i, img_path in enumerate(img_path_list, start=1):
        msg = f"Processing image {img_path.name}    ({i}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = pipeline(image=img)
        aug_img = transformed["image"]

        show_img(img, "Original image")
        show_img(aug_img, "Augmented normalized image")
        denormalized_img = denormalize(aug_img, mean=(0.041, 0.129, 0.03), std=(0.054, 0.104, 0.046))
        show_img(denormalized_img, "Augmented denormalized image")


if __name__ == "__main__":
    main()
