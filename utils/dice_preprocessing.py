from argparse import ArgumentParser
from multiprocessing import Pool
import os
from pathlib import Path
import shutil

import cv2


def worker(args: tuple[Path]):
    """
    Worker in charge of converting an image to true black/white
    Args:
        img_path: Path to the image to convert
    Return:
        None
    """
    file_path = args[0]
    img = cv2.imread(str(file_path))
    height, width, _ = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j, 0] < 150:
                img[i, j] = [0, 0, 0]
            else:
                img[i, j] = [255, 255, 255]

    # Do not use jpg for masks since it is a lossy format.
    # cv2.imwrite(str(file_path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(str(file_path.with_suffix(".png")), img)
    file_path.unlink()

    return None


def main():
    parser = ArgumentParser("Turns black-ish into true black and white-ish into true white")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()

    data_path: Path = args.data_path

    exts = [".jpg", ".png"]
    file_list = list([p for p in data_path.rglob('*') if p.suffix in exts and "_seg" in str(p)])
    nb_imgs = len(file_list)
    print(f"Found {nb_imgs} images to process, starting to convert them to true black/white")

    imgs_processed = 0
    mp_args = list([(img_path,) for img_path in file_list])
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for _ in pool.imap(worker, mp_args, chunksize=10):
            imgs_processed += 1
            msg = f"Processing status: ({imgs_processed}/{nb_imgs})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

    print("\nFinished processing dataset")


if __name__ == "__main__":
    main()
