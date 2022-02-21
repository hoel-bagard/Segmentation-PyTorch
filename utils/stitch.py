from argparse import ArgumentParser
from pathlib import Path
from shutil import get_terminal_size

import cv2
import numpy as np


def stitch_images(data_path: Path, output_dir: Path, tile_size: int = 128):
    """Stitch together the tiles of each image."""
    csv_paths = list(data_path.rglob("*.csv"))
    nb_imgs = len(csv_paths)
    for i, csv_path in enumerate(csv_paths):
        dir_path = csv_path.parent  # Name of the image.
        msg = f"Processing image {dir_path.name} ({i+1}/{nb_imgs})"
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
        for j, (coord_y, coord_x) in enumerate(coordinates):
            if coord_y % 2 == 0:
                coord_y //= 2
                img[coord_y*tile_size:coord_y*tile_size + tile_size,
                    coord_x*tile_size:coord_x*tile_size + tile_size] = imgs[j]

        rel_path = dir_path.parent.relative_to(data_path)
        output_path = output_dir / rel_path / (dir_path.name + ".png")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path), img)
    print(f"\nFinished stitching the images. Result saved in {output_dir}")


def stitch_disparities(data_path: Path, output_dir: Path, tile_size: int = 128):
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

        disp = np.zeros((5*tile_size, 8*tile_size))
        for i, (coord_y, coord_x) in enumerate(coordinates):
            if coord_y % 2 == 0:
                coord_y //= 2
                disp[coord_y*tile_size:coord_y*tile_size + tile_size,
                     coord_x*tile_size:coord_x*tile_size + tile_size] = disps[i].reshape(tile_size, tile_size)

        rel_path = dir_path.parent.relative_to(data_path)
        output_path = output_dir / rel_path / (dir_path.name + "_disp.npy")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(output_path, disp)
    print(f"\nFinished stitching the disparity maps. Result saved in {output_dir}")


def main():
    parser = ArgumentParser(description="Stitch together tiles of images and disparity maps.")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("output_dir", type=Path, help="Output path")
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir

    stitch_images(data_path, output_dir)
    stitch_disparities(data_path, output_dir)


if __name__ == "__main__":
    main()
