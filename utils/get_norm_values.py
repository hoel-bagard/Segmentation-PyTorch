import shutil
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = ArgumentParser(description="Script to get the mean and std of a dataset.")
    parser.add_argument("data_path", type=Path, help="Path to the dataset.")
    args = parser.parse_args()

    data_path: Path = args.data_path

    # This assumes that the masks' paths contain either "mask" or "seg" (and that the main image does not).
    exts = [".jpg", ".png", ".tiff"]
    img_path_list = list([p for p in data_path.rglob('*') if p.suffix in exts
                          and "seg" not in str(p) and "mask" not in str(p)])

    mean, std = np.zeros(3), np.zeros(3)
    nb_imgs = len(img_path_list)
    for i, img_path in enumerate(img_path_list, start=1):
        msg = f"Processing image {img_path.name}    ({i}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        img = cv2.imread(str(img_path))
        mean[0] += np.mean(img[..., 2])
        mean[1] += np.mean(img[..., 1])
        mean[2] += np.mean(img[..., 0])

        std[0] += np.std(img[..., 2])
        std[1] += np.std(img[..., 1])
        std[2] += np.std(img[..., 0])

    mean /= nb_imgs
    std /= nb_imgs
    np.set_printoptions(precision=3)
    print(f"Mean: {mean}, std: {std}    (RGB format)")
    print("Divided by 255:")
    print(f"Mean: {mean/255}, std: {std/255}    (RGB format)")


if __name__ == "__main__":
    main()
