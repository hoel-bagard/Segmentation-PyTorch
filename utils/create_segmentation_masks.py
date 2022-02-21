from argparse import ArgumentParser
from pathlib import Path
from shutil import get_terminal_size

import cv2
import numpy as np

from config.data_config import get_data_config


def create_masks(data_path: Path, output_dir: Path, tile_size: int = 128):
    """Create the segmentation masks from the csvs."""
    data_config = get_data_config()

    csv_paths = list(data_path.rglob("*.csv"))
    nb_imgs = len(csv_paths)
    for i, csv_path in enumerate(csv_paths):
        dir_path = csv_path.parent  # Name of the image.
        msg = f"Processing image {dir_path.name} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        with open(csv_path, encoding="shift-jis") as f:
            annotations = f.readlines()
        # In the csv, the first column contains the tile's coordinates (first 2 digits for Y, last 2 for X)
        # (Those coordinates are also used as the tile's image name.)
        tile_names = [line.strip().split(',')[0] for line in annotations]
        # CSV coordinates start from 1, here we make then start at 0.
        coordinates = [(int(name[:2])-1, int(name[2:])-1) for name in tile_names]
        classes = [line.strip().split(',')[1] for line in annotations]

        # 5 and 8 are hard coded.
        # They are different from max(coord_y) and max(coord_x) because there is some overlapp between tiles.
        mask = np.zeros((5*tile_size, 8*tile_size, 3))
        # Fuse the tiles into one image.
        for cls, (coord_y, coord_x) in zip(classes, coordinates):
            cls = cls if cls in data_config.NAME_TO_COLOR.keys() else "安全"
            if coord_y % 2 == 0:
                coord_y //= 2
                mask[coord_y*tile_size:coord_y*tile_size + tile_size,
                     coord_x*tile_size:coord_x*tile_size + tile_size] = data_config.NAME_TO_COLOR[cls]

        rel_path = dir_path.parent.relative_to(data_path)
        output_path = output_dir / rel_path / (dir_path.name + "_mask.png")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path), mask)
    print(f"\nFinished creating the masks. Result saved in {output_dir}")


def main():
    parser = ArgumentParser(description=("Create segmentation masks from the csvs."
                                         " Call with python -m utils/create_segmentation_masks"))
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("output_dir", type=Path, help="Output path")
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir

    create_masks(data_path, output_dir)


if __name__ == "__main__":
    main()
