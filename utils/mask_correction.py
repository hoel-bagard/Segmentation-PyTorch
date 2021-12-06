import json
import os
import shutil
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def worker(args: tuple[Path, Path, np.ndarray]):
    """Worker in charge of converting an image to valid colors only.

    Args:
        args: Tuple containing:
            img_path (Path): Path to the image to convert
            output_dir (Path): Path to the output folder
            colors (np.ndarray): List of the valid colors

    Return:
        None
    """
    file_path: Path
    output_dir: Path
    colors: np.ndarray
    file_path, output_dir, colors = args

    img = cv2.imread(str(file_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Colors are in RGB

    height, width, _ = img.shape

    # Passes a kernel on the image.
    # If a pixel does not have a valid color, then it takes the value of the most represented color within the kernel.
    kernel_size = 5//2    # //2 here for convenience
    for i in range(height):
        for j in range(width):
            if not (img[i, j] == colors).all(1).any():
                sub_img = img[max(0, i-kernel_size):i+kernel_size, max(0, j-kernel_size):j+kernel_size]
                unique, counts = np.unique(np.reshape(sub_img, (-1, 3)), return_counts=True, axis=0)
                most_neighboring = unique[counts.argmax()]
                img[i, j] = most_neighboring

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    output_path: Path = output_dir / file_path.with_suffix(".png").name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

    return None


def main():
    parser = ArgumentParser("Remove 'mixed' colors to have only those corresponding to a class.")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("class_path", type=Path,
                        help=("Path to a json file with the valid colors and classes."
                              "File should be a list of {'name': 'class_name', 'color': [R, G, B]'}."))
    parser.add_argument("--output_path", "--o", default=None, type=Path,
                        help="Output path, defaults to 'data_path/../out_imgs'")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path if args.output_path else data_path.parent / "out_imgs"
    class_path: Path = args.class_path

    colors = []  # List of valid colors
    with open(class_path) as json_file:
        data = json.load(json_file)
        for _key, entry in enumerate(data):
            colors.append(entry["color"])
    colors = np.asarray(colors)

    exts = [".jpg", ".png", ".bmp"]
    file_list = list([p for p in data_path.rglob('*') if (p.suffix in exts
                                                          and ("_seg" in str(p.name) or "_mask" in str(p.name)))])
    nb_imgs = len(file_list)
    print(f"Found {nb_imgs} images to process, starting to convert them.")

    imgs_processed = 0
    mp_args = list([(img_path, output_path, colors) for img_path in file_list])
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for _ in pool.imap(worker, mp_args, chunksize=10):
            imgs_processed += 1
            msg = f"Processing status: ({imgs_processed}/{nb_imgs})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

    print("\nFinished processing dataset")


if __name__ == "__main__":
    main()
