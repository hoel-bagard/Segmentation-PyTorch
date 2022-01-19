import argparse
from pathlib import Path
from shutil import get_terminal_size

import cv2


def main():
    parser = argparse.ArgumentParser("Cuts images and corresponding masks into small tiles")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("output_path", type=Path, help="Output path")
    parser.add_argument("--tile_size", "-ts", nargs=2, default=[256, 256], type=int, help="Size of the tiles (w, h)")
    parser.add_argument("--stride", "-s", nargs=2, default=[100, 100], type=int, help="Strides (w, h)")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path

    output_path.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height = args.tile_size
    stride_width, stride_height = args.stride

    exts = [".jpg", ".png"]
    img_path_list = list([p for p in data_path.rglob('*') if p.suffix in exts and "mask" not in str(p)])
    nb_imgs = len(img_path_list)
    for i, img_path in enumerate(img_path_list):
        msg = f"Processing image {img_path.name} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        img_mask_path = Path(str(img_path.with_suffix('')) + "_mask.png")
        assert img_mask_path.exists(), f"\nMask for image {img_path} is missing"

        img = cv2.imread(str(img_path))
        height, width, _ = img.shape
        img_mask = cv2.imread(str(img_mask_path))
        assert img_mask.shape[0] == height and img_mask.shape[1] == width, (
            f"\nShape of the image and the mask do not match for image {img_path}")

        tile_index = 0
        for x in range(0, width-tile_width, stride_width):
            for y in range(0, height-tile_height, stride_height):
                tile = img[y:y+tile_height, x:x+tile_width]
                tile_mask = img_mask[y:y+tile_height, x:x+tile_width]

                rel_path = img_path.relative_to(data_path)
                new_tile_name = img_path.stem + '_' + str(tile_index).zfill(5) + img_path.suffix
                tile_path = output_path / rel_path.parent / new_tile_name
                tile_path.parent.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(tile_path), tile)

                new_tile_mask_name = (''.join(img_mask_path.stem.split('_')[:-1])
                                      + '_' + str(tile_index).zfill(5) + "_mask.png")
                tile_mask_path = output_path / rel_path.parent / new_tile_mask_name
                cv2.imwrite(str(tile_mask_path), tile_mask)

                tile_index += 1

    print("\nFinished tiling dataset")


if __name__ == "__main__":
    main()
