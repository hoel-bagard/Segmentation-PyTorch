import argparse
from pathlib import Path
import shutil
from random import shuffle


def main():
    parser = argparse.ArgumentParser("Validation/Train splitting")
    parser.add_argument("data_path", type=Path, help="Path to the train dataset")
    parser.add_argument("--split_ratio", "--s", type=float, default=0.8, help="Fraction of the dataset used for train")
    args = parser.parse_args()

    data_path = args.data_path
    val_path: Path = data_path.parent / "Validation"
    val_path.mkdir(parents=True, exist_ok=True)

    exts = [".jpg", ".png"]
    img_path_list = list([p for p in data_path.rglob('*') if p.suffix in exts and "mask" not in str(p)])
    nb_imgs = len(img_path_list)
    shuffle(img_path_list)
    for i, img_path in enumerate(img_path_list):
        msg = f"Processing image {img_path.name} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        img_mask_name = "_".join(str(img_path.name).split("_")[:-1]) + "_mask_" + str(img_path.name).split("_")[-1]
        img_mask_path = img_path.parent / img_mask_name

        assert img_mask_path.exists(), f"\nMask for image {img_path} is missing"

        if i >= args.split_ratio*nb_imgs:
            dest_img_path = (val_path / img_path.relative_to(data_path)).parent
            dest_mask_path = (val_path / img_mask_path.relative_to(data_path)).parent
            dest_img_path.mkdir(parents=True, exist_ok=True)
            shutil.move(img_path, dest_img_path)
            shutil.move(img_mask_path, dest_mask_path)

    print("\nFinished splitting dataset")


if __name__ == "__main__":
    main()
