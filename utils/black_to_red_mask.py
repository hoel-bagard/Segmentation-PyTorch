from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Pool
import os
import shutil

import cv2
import numpy as np


def convert_worker(args: tuple[Path, Path, int]):
    """
    Worker in charge of thresholding and converting an image
    Args:
        img_path: Path to the image to convert
        output_path: Folder to where the new mask will be saved
    Return:
        output_file_path: Path of the saved image.
    """
    img_path, output_path, threshold = args
    output_file_path = output_path / img_path.relative_to(output_path.parent)

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape
    mask_img = np.zeros((height, width, 4))

    for i in range(height):
        for j in range(width):
            if img[i, j, 3] > threshold:
                mask_img[i, j] = [0, 0, 255, 255]

    cv2.imwrite(str(output_file_path), mask_img)
    return output_file_path


def main():
    # Because writing python is faster than learning Gimp.
    parser = ArgumentParser("Converts a 'black' mask to red after thresholding it."
                            "Saves the data in a converted_masks folder, in the dataset's folder")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = data_path / "converted_masks"
    output_path.mkdir(parents=True, exist_ok=True)

    threshold: int = 200  # Alpha threshold

    # Get a list of all the images
    exts = [".jpg", ".png"]
    file_list = list([p for p in data_path.rglob('*') if p.suffix in exts and "mask" in str(p)])
    nb_imgs = len(file_list)

    mp_args = list([(img_path, output_path, threshold) for img_path in file_list])
    nb_images_processed = 0  # Use to count the number of good / bad samples
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for result in pool.imap(convert_worker, mp_args, chunksize=10):
            nb_images_processed += 1
            msg = f"Processing status: ({nb_images_processed}/{nb_imgs})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

    print(f"\nFinished processing dataset. Converted {nb_images_processed} images, and saved them in {output_path}")


if __name__ == "__main__":
    main()
