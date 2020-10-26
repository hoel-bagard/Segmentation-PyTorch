import argparse
import os
import glob
import shutil
from random import shuffle


def main():
    parser = argparse.ArgumentParser("Validation/Train splitting")
    parser.add_argument('data_path', help='Path to the train dataset')
    args = parser.parse_args()

    val_path = os.path.join(os.path.dirname(args.data_path.strip("/")), "Validation")
    os.makedirs(val_path, exist_ok=True)

    img_index: int = 0
    file_list = glob.glob(os.path.join(args.data_path, "**", "*.jpg"), recursive=True)
    nb_imgs = len(file_list)
    shuffle(file_list)  # Too lazy to split each class individually
    for i, file_path in enumerate(file_list):
        print(f"Processing image {os.path.basename(file_path)} ({i+1}/{nb_imgs})", end='\r')
        if i >= 0.9*nb_imgs:
            # dogs-vs-cats (I think)
            new_nb = str(img_index).zfill(5)
            new_name = f"{new_nb}.jpg"
            shutil.move(file_path, os.path.join(val_path, new_name))
            img_index += 1
    print("\nFinished splitting dataset")


if __name__ == "__main__":
    main()
