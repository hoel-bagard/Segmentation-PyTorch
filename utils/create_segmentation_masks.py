from argparse import ArgumentParser
from pathlib import Path
from shutil import get_terminal_size
from typing import Optional

import cv2
import numpy as np

from config.data_config import get_data_config


def create_masks(data_path: Path, output_dir: Path, limit: Optional[int] = None, generate_full: bool = False):
    """Create the classification and danger segmentation masks from the csvs."""
    data_config = get_data_config()
    tile_size: int = 128

    csv_paths = list(data_path.rglob("*.csv"))
    nb_imgs = len(csv_paths)
    for i, csv_path in enumerate(csv_paths, start=1):
        dir_path = csv_path.parent  # Name of the image.
        msg = f"Processing image {dir_path.name} ({i}/{nb_imgs})"
        print(msg + ' ' * (get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        with open(csv_path, encoding="shift-jis") as f:
            annotations = f.readlines()
        # In the csv, the first column contains the tile's coordinates (first 2 digits for Y, last 2 for X)
        # (Those coordinates are also used as the tile's image name.)
        tile_names = [line.strip().split(',')[0] for line in annotations]
        # CSV coordinates start from 1, here we make then start at 0.
        coordinates = [(int(name[:2])-1, int(name[2:])-1) for name in tile_names]
        classes = [line.strip().split(',')[1] for line in annotations]
        danger_lvls = [int(line.strip().split(',')[2]) for line in annotations]

        # 5 and 8 are hard coded.
        # They are different from max(coord_y) and max(coord_x) because there is some overlapp between tiles.
        mask = np.zeros((5, 8, 3), dtype=np.uint8)  # Actual mask used as label.
        danger_mask = np.zeros((5, 8), dtype=np.uint8)
        if generate_full:
            # Masks with the same size as the images. Used only for visualization.
            mask_full = np.zeros((5*tile_size, 8*tile_size, 3), dtype=np.uint8)
            danger_mask_full = np.zeros((5*tile_size, 8*tile_size), dtype=np.uint8)
        # Fuse the tiles into one image.
        for cls, danger_lvl, (coord_y, coord_x) in zip(classes, danger_lvls, coordinates):
            cls = cls if cls in data_config.NAME_TO_COLOR.keys() else "その他"
            if coord_y % 2 == 0:
                coord_y //= 2
                mask[coord_y, coord_x] = data_config.NAME_TO_COLOR[cls]
                danger_mask[coord_y, coord_x] = danger_lvl
                if generate_full:
                    x_start, y_start = coord_x*tile_size, coord_y*tile_size  # Start points of the tiles (full masks)
                    mask_full[y_start:y_start + tile_size,
                              x_start:x_start + tile_size] = data_config.NAME_TO_COLOR[cls]
                    danger_mask_full[y_start:y_start + tile_size,
                                     x_start:x_start + tile_size] = int(255 * danger_lvl / data_config.MAX_DANGER_LEVEL)

        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

        rel_path = dir_path.parent.relative_to(data_path)
        output_path = output_dir / rel_path / (dir_path.name + "_mask.png")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path), mask)
        cv2.imwrite(str(output_path.with_stem(dir_path.name + "_danger_mask")), danger_mask)
        if generate_full:
            mask_full = cv2.cvtColor(mask_full, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path.with_stem(dir_path.name + "_full_mask")), mask_full)
            cv2.imwrite(str(output_path.with_stem(dir_path.name + "_danger_full_mask")), danger_mask_full)

        if limit and i == limit:
            break
    print(f"\nFinished creating the masks. Result saved in {output_dir}")


def main():
    parser = ArgumentParser(description=("Create segmentation masks from the csvs."
                                         " Call with python -m utils/create_segmentation_masks"))
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("output_dir", type=Path, help="Output path")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Break after N samples.")
    parser.add_argument("--full", "-f", action="store_true", help="Also generate full size masks for visualization.")
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir
    limit: int = args.limit
    generate_full: bool = args.full

    create_masks(data_path, output_dir, limit=limit, generate_full=generate_full)


if __name__ == "__main__":
    main()
