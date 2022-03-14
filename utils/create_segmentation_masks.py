import itertools
from argparse import ArgumentParser
from pathlib import Path
from shutil import get_terminal_size
from typing import Optional

import cv2
import numpy as np

from config.data_config import get_data_config


def create_masks(data_path: Path,
                 output_dir: Path,
                 limit: Optional[int] = None,
                 generate_full: bool = False,
                 fallback_class: str = "その他"):
    """Create the classification and danger segmentation masks from the csvs.

    Args:
        data_path: Path to the dataset folder.
        output_dir: Path to where the created masks should be saved.
        limit: Break after N samples.
        generate_full: Also generate full size masks for visualization.
        fallback_class: If a class in the csv is not in the data config, then it is set to that class.
                        (the fallback_class needs to be set in the data config)

    The original data for the masks is classified tiles. But those tiles have been created from two overlapping images.
    (Note: Last column is missing for the second image (I think, then again no doc))
    To get a mask resolution of 16x10 instead of 8x5, we need to fuse the masks of the two images somehow:
    Instructions on how to assemble the labels:
        If the overlapping area have the same labels, keep it.
        If they overlap with "安全", Priotize the object.
        If two objects overlap, let define a rule: based on their general sizes, put some priorities.
            1. ポール (pole)
            2. 人間
            3. その他
            4. 段差
            5. 自転車
            6. 自動車
            7. 安全
    """
    data_config = get_data_config()
    name_to_color = data_config.NAME_TO_COLOR
    tile_size: int = 128
    danger_lvl_map = {1: 0, 3: 1, 5: 2}  # I was told to change the values to those.

    csv_paths = list(data_path.rglob("*.csv"))
    nb_imgs = len(csv_paths)
    for i, csv_path in enumerate(csv_paths, start=1):
        dir_path = csv_path.parent  # Name of the image.
        msg = f"Processing image {dir_path.name} ({i}/{nb_imgs})"
        print(msg + ' ' * (get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        with open(csv_path, encoding="shift-jis") as f:
            annotations = f.readlines()
        # In the csv, the first column contains the tile's coordinates
        # First 2 digits for Y (row index), last 2 for X (column index).
        # (Those coordinates are also used as the tile's image name.)
        tile_names = [line.strip().split(',')[0] for line in annotations]
        # CSV coordinates start from 1, here we make them start at 0.
        coordinates = [(int(name[:2])-1, int(name[2:])-1) for name in tile_names]
        classes = [line.strip().split(',')[1] for line in annotations]
        danger_lvls = [int(line.strip().split(',')[2]) for line in annotations]

        # Mask size for one image is 5x8. But we upsample it to 10x16 by merging the masks.
        mask1, mask2 = np.zeros((10, 16, 3), dtype=np.uint8), np.full((10, 16, 3), name_to_color["安全"], dtype=np.uint8)
        danger_mask1, danger_mask2 = np.zeros((10, 16), dtype=np.uint8), np.zeros((10, 16), dtype=np.uint8)
        for cls, danger_lvl, (y, x) in zip(classes, danger_lvls, coordinates):
            danger_lvl = danger_lvl_map[danger_lvl]
            cls = cls if cls in data_config.NAME_TO_COLOR.keys() else fallback_class
            x *= 2  # Move x by increments of 2 (to go from 8 to 16 columns)

            # The two images can be differentiated by their y index.
            if y % 2 == 0:
                mask1[y:y+2, x:x+2] = data_config.NAME_TO_COLOR[cls]
                danger_mask1[y:y+2, x:x+2] = danger_lvl
            else:
                # Note: I've no idea if anything done bellow is correct, I just tried stuff...
                if y == 0:  # Skip top row.
                    continue
                y -= 3  # Move the labels two rows up.
                # Skip first half of the first column
                if x == 0:
                    mask2[y:y+2, x+1] = data_config.NAME_TO_COLOR[cls]
                    danger_mask2[y:y+2, x+1] = danger_lvl
                # Skip last half of the last column
                elif x == 2*(7-1):  # 7 is max x index in the csv for the second image.
                    mask2[y:y+2, x] = data_config.NAME_TO_COLOR[cls]
                    danger_mask2[y:y+2, x] = danger_lvl
                else:
                    mask2[y:y+2, x:x+2] = data_config.NAME_TO_COLOR[cls]
                    danger_mask2[y:y+2, x:x+2] = danger_lvl

        # First process the danger_mask
        merged_danger_mask = np.maximum(danger_mask1, danger_mask2)

        # Then merge the two classification masks following the rules given in the docstring.
        # Written in a verbose way (starting from an array full of zeros) for easier readability.
        color_to_priority = {tuple(color): i for i, color in enumerate(data_config.IDX_TO_COLOR)}
        merged_mask = np.zeros((10, 16, 3), dtype=np.uint8)
        for y, x in itertools.product(range(10), range(16)):
            # If the masks have the same value, just keep it
            if (mask1[y, x] == mask2[y, x]).all():
                merged_mask[y, x] = mask1[y, x]

            # Priotize any object over "安全"  (included in the rule bellow, but...)
            elif (mask1[y, x] == name_to_color["安全"]).all() and (mask2[y, x] != name_to_color["安全"]).all():
                merged_mask[y, x] = mask2[y, x]
            elif (mask1[y, x] != name_to_color["安全"]).all() and (mask2[y, x] == name_to_color["安全"]).all():
                merged_mask[y, x] = mask1[y, x]

            # If two objects overlap, keep the one with the highest priority (as defined by the config order).
            # Highest priority means lower index, not necessarily super intuitive but good enough.
            elif color_to_priority[tuple(mask1[y, x])] < color_to_priority[tuple(mask2[y, x])]:
                merged_mask[y, x] = mask1[y, x]
            else:
                merged_mask[y, x] = mask2[y, x]

        mask = cv2.cvtColor(merged_mask, cv2.COLOR_RGB2BGR)

        rel_path = dir_path.parent.relative_to(data_path)
        output_path = output_dir / rel_path / (dir_path.name + "_mask.png")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path), mask)
        cv2.imwrite(str(output_path.with_stem(dir_path.name + "_danger_mask")), danger_mask1)

        if generate_full:
            # Masks with the same size as the images. Used only for visualization.
            img_size = (8*tile_size, 5*tile_size)
            mask_full = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST_EXACT)
            cv2.imwrite(str(output_path.with_stem(dir_path.name + "_full_mask")), mask_full)

            # Uncomment bellow for debugging.
            # mask = cv2.cvtColor(mask1, cv2.COLOR_RGB2BGR)
            # mask_full = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST_EXACT)
            # cv2.imwrite(str(output_path.with_stem(dir_path.name + "_full_mask1")), mask_full)

            # mask = cv2.cvtColor(mask2, cv2.COLOR_RGB2BGR)
            # mask_full = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST_EXACT)
            # cv2.imwrite(str(output_path.with_stem(dir_path.name + "_full_mask2")), mask_full)

            # For the danger mask, make the difference between danger levels visible.
            danger_mask = (255. * merged_danger_mask / data_config.MAX_DANGER_LEVEL).astype(np.uint8)
            danger_mask_full = cv2.resize(danger_mask, img_size, interpolation=cv2.INTER_NEAREST_EXACT)
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
