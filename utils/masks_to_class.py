from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Pool
import os
import shutil

import cv2
import numpy as np


def sort_worker(args: tuple[Path, Path, Path]):
    """
    Worker in charge of sorting an image
    Args:
        img_path: Path to the image to sort
        good_output_path: Path to where the images with an empty mask should be copied
        bad_output_path: Path to where the images with a non-empty mask should be copied
    Return:
        0 it the image was a bad one, 1 if it was a good one
    """
    img_path, good_output_path, bad_output_path = args
    img_mask_name = "_".join(str(img_path.name).split("_")[:-1]) + "_mask_" + str(img_path.name).split("_")[-1]
    img_mask_path = img_path.parent / img_mask_name

    assert img_mask_path.exists(), f"\nMask for image {img_path} is missing"

    img_mask = cv2.imread(str(img_mask_path))
    height, width, _ = img_mask.shape

    # Check if there is a red pixel somewhere on the mask
    if any([np.array_equal(img_mask[i][j], [0, 0, 255]) for i in range(width) for j in range(height)]):
        shutil.copy(img_mask_path, bad_output_path / img_mask_name)
        shutil.copy(img_path, bad_output_path / img_path.name)
        return 0
    else:
        shutil.copy(img_mask_path, good_output_path / img_mask_name)
        shutil.copy(img_path, good_output_path / img_path.name)
        return 1


def main():
    parser = ArgumentParser("Sorts image tiles into good and bad samples. Expects masks to be in red")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("output_path", type=Path, help="Output path")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path
    good_output_path = output_path / "good"
    bad_output_path = output_path / "bad"

    good_output_path.mkdir(parents=True, exist_ok=True)
    bad_output_path.mkdir(parents=True, exist_ok=True)

    # Get a list of all the images
    exts = [".jpg", ".png"]
    file_list = list([p for p in data_path.rglob('*') if p.suffix in exts and "mask" not in str(p)])
    nb_imgs = len(file_list)

    mp_args = list([(img_path, good_output_path, bad_output_path) for img_path in file_list])
    results = []  # Use to count the number of good / bad samples
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for result in pool.imap(sort_worker, mp_args, chunksize=10):
            results.append(result)
            msg = f"Processing status: ({len(results)}/{nb_imgs})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

    nb_good_samples: int = sum(results)
    nb_bad_samples: int = len(results) - nb_good_samples

    print("\nFinished processing dataset")
    print(f"Found {nb_good_samples} good samples and {nb_bad_samples} bad samples")


if __name__ == "__main__":
    main()
