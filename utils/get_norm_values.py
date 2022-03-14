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

    # Get the grayscale images (masks are in .png)
    img_path_list = [p for p in data_path.rglob("*.jpg") if "disp" not in p.stem]
    nb_imgs = len(img_path_list)
    res_dict = {"img_mean": 0., "img_std": 0., "disp_mean": 0., "disp_std": 0.}

    for i, img_path in enumerate(img_path_list, start=1):
        msg = f"Processing image {img_path.name}    ({i}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        img = cv2.imread(str(img_path), 0)
        res_dict["img_mean"] += np.mean(img)
        res_dict["img_std"] += np.std(img)

        disp = cv2.imread(str(img_path.with_stem(img_path.stem + "_disp")), 0)
        res_dict["disp_mean"] += np.mean(disp)
        res_dict["disp_std"] += np.std(disp)

    print("\n")
    for value_name, value in res_dict.items():
        print(f"{value_name}: {value / nb_imgs:.3f}   (Divided by 255: {value / (255*nb_imgs):.3f})")


if __name__ == "__main__":
    main()
