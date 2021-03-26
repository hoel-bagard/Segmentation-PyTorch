import argparse
from pathlib import Path
import shutil
from random import shuffle


def get_files_dice(img_path: Path) -> list[Path]:
    return [Path(str(img_path.stem) + "_segDotsTopOnly.jpg"),
            Path(str(img_path.stem) + "_segDots.jpg"),
            Path(str(img_path.stem) + "_segDie.jpg"),
            Path(str(img_path.stem) + "_label.json")]


def get_mask_path_tape(img_p: Path) -> list[Path]:
    return [img_p.parent / ("_".join(str(img_p.name).split("_")[:-1]) + "_mask_" + str(img_p.name).split("_")[-1])]


def main():
    parser = argparse.ArgumentParser("Validation/Train splitting")
    parser.add_argument("data_path", type=Path, help="Path to the train dataset")
    parser.add_argument("--split_ratio", "--s", type=float, default=0.8, help="Fraction of the dataset used for train")
    parser.add_argument("--max_in_val", "--m", type=int, default=None,
                        help="If given, then the validation dataset will contain at most N elements."
                             "Override / takes precedent over the split_ratio if needed.")
    parser.add_argument("--dataset_name", "--n", type=str, default="dice",
                        help="Name of the dataset, used to know how to get the masks' paths")
    args = parser.parse_args()

    data_path = args.data_path
    val_path: Path = data_path.parent / "Validation"
    val_path.mkdir(parents=True, exist_ok=True)

    exts = [".jpg", ".png"]

    get_path_mask_fn = get_files_dice if args.dataset_name == "dice" else get_mask_path_tape
    # This assumes that the masks' paths contain either "mask" or "seg" (and that the main image does not).
    img_path_list = list([p for p in data_path.rglob('*') if p.suffix in exts
                          and "seg" not in str(p) and "mask" not in str(p)])

    nb_imgs: int = len(img_path_list)
    nb_moved_samples: int = 0
    shuffle(img_path_list)
    for i, img_path in enumerate(img_path_list):
        msg = f"Processing image {img_path.name} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        img_mask_names = get_path_mask_fn(img_path)
        img_mask_paths = [img_path.parent / img_mask_name for img_mask_name in img_mask_names]

        assert img_mask_paths[0].exists(), f"\nMask {img_mask_paths[0]} for image {img_path} is missing"

        if i >= args.split_ratio*nb_imgs:
            if args.max_in_val and nb_moved_samples >= args.max_in_val:
                break

            nb_moved_samples += 1
            dest_img_path = (val_path / img_path.relative_to(data_path)).parent
            dest_mask_paths = [(val_path / img_mask_path.relative_to(data_path)).parent
                               for img_mask_path in img_mask_paths]

            dest_img_path.mkdir(parents=True, exist_ok=True)
            shutil.move(img_path, dest_img_path)
            for img_mask_path, dest_mask_path in zip(img_mask_paths, dest_mask_paths):
                shutil.move(img_mask_path, dest_mask_path)

    print(f"\nFinished splitting dataset, moved {nb_moved_samples} from Train to Validation")


if __name__ == "__main__":
    main()
