from argparse import ArgumentParser
from pathlib import Path
from shutil import get_terminal_size
from typing import Optional

import cv2
import numpy as np


def stitch_images(data_path: Path, output_dir: Path, tile_size: int = 128, limit: Optional[int] = None):
    """Stitch together the tiles of each image."""
    csv_paths = list(data_path.rglob("*.csv"))
    nb_imgs = len(csv_paths)
    for img_idx, csv_path in enumerate(csv_paths, start=1):
        dir_path = csv_path.parent  # Name of the image.
        msg = f"Processing image {dir_path.name} ({img_idx}/{nb_imgs})"
        print(msg + ' ' * (get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        with open(csv_path, encoding="shift-jis") as f:
            annotations = f.readlines()
        # In the csv, the first column contains the tile's coordinates (first 2 digits for Y, last 2 for X)
        # Those coordinates are also used as the tile's image name.
        tile_names = [line.strip().split(',')[0] for line in annotations]
        # CSV coordinates start from 1, here we make then start at 0.
        coordinates = [(int(name[:2])-1, int(name[2:])-1) for name in tile_names]
        # Read all the images in grayscale format.
        imgs = [cv2.imread(str(dir_path / "pic" / (tile_name + ".bmp")), 0) for tile_name in tile_names]

        # 5 and 8 are hard coded.
        # They are different from max(coord_y) and max(coord_x) because there is some overlapp between tiles.
        img = np.zeros((5*tile_size, 8*tile_size))
        # Fuse the tiles into one image.
        for i, (coord_y, coord_x) in enumerate(coordinates):
            if coord_y % 2 == 0:
                coord_y //= 2
                img[coord_y*tile_size:coord_y*tile_size + tile_size,
                    coord_x*tile_size:coord_x*tile_size + tile_size] = imgs[i]

        rel_path = dir_path.parent.relative_to(data_path)
        output_path = output_dir / rel_path / (dir_path.name + ".jpg")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path), img)

        if limit and img_idx == limit:
            break
    print(f"\nFinished stitching the images. Result saved in {output_dir}")


def stitch_disparities(data_path: Path,
                       output_dir: Path,
                       tile_size: int = 128,
                       save_as_np: bool = False,
                       limit: Optional[int] = None):
    """Stitch together the tiles of each disparity map."""
    csv_paths = list(data_path.rglob("*.csv"))
    nb_imgs = len(csv_paths)
    for img_idx, csv_path in enumerate(csv_paths):
        dir_path = csv_path.parent  # Name of the image.
        msg = f"Processing disparity map for image {dir_path.name} ({img_idx+1}/{nb_imgs})"
        print(msg + ' ' * (get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        # See function above for explainations.
        with open(csv_path, encoding="shift-jis") as f:
            annotations = f.readlines()
        tile_names = [line.strip().split(',')[0] for line in annotations]
        coordinates = [(int(name[:2])-1, int(name[2:])-1) for name in tile_names]
        # Read the disparity maps.
        disps = [np.fromfile(dir_path / "pic" / (tile_name + ".bin"), '<f4') for tile_name in tile_names]
        disps = [disp.reshape(tile_size, tile_size) for disp in disps]  # Go from 1D array to 2D
        disps = [np.flipud(disp) for disp in disps]  # Tiles need to be flipped upside down for some reason.

        disp = np.zeros((5*tile_size, 8*tile_size))
        for i, (coord_y, coord_x) in enumerate(coordinates):
            if coord_y % 2 == 0:
                coord_y //= 2
                disp[coord_y*tile_size:coord_y*tile_size + tile_size,
                     coord_x*tile_size:coord_x*tile_size + tile_size] = disps[i]

        rel_path = dir_path.parent.relative_to(data_path)
        if save_as_np:
            output_path = output_dir / rel_path / (dir_path.name + "_disp.npy")
            output_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(output_path, disp)
        else:
            output_path = output_dir / rel_path / (dir_path.name + "_disp.jpg")
            output_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(output_path), disp)

        if limit and img_idx == limit:
            break
    print(f"\nFinished stitching the disparity maps. Result saved in {output_dir}")


def main():
    parser = ArgumentParser(description="Stitch together tiles of images and disparity maps.")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("output_dir", type=Path, help="Output path")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Break after N samples.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_dir: Path = args.output_dir
    limit: int = args.limit

    stitch_images(data_path, output_dir, limit=limit)
    stitch_disparities(data_path, output_dir, limit=limit)


if __name__ == "__main__":
    main()
