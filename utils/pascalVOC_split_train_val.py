import argparse
import glob
import os
import shutil
import xml.etree.ElementTree as ET  # noqa: N817
from random import shuffle


def main():
    parser = argparse.ArgumentParser("Validation/Train splitting")
    parser.add_argument('data_path', help='Path to the train dataset')
    args = parser.parse_args()

    val_path = os.path.join(os.path.dirname(args.data_path.strip("/")), "Validation")
    os.makedirs(val_path, exist_ok=True)

    file_list = glob.glob(os.path.join(args.data_path, "**", "*.xml"), recursive=True)
    nb_imgs = len(file_list)
    shuffle(file_list)
    for i, label_path in enumerate(file_list):
        print(f"Processing {os.path.basename(label_path)} ({i+1}/{nb_imgs})", end='\r')
        root: ET.Element = ET.parse(label_path).getroot()
        image_path: str = root.find("path").text

        image_subpath = os.path.join(*image_path.split(os.path.sep)[-2:])

        # It seems like glob does not work with Japanese characters
        f = []
        for dirpath, _subdirs, files in os.walk(os.path.join(args.data_path, "images")):
            f.extend(os.path.join(dirpath, x) for x in files)

        for filename in f:
            # Should use the full subpath to avoid duplicates  (Seems like there is an issue with Japanese characters)
            if image_subpath.split(os.path.sep)[-1] in filename:
                image_path = filename

        # This looks really dirty, make it better if possible
        label_subpath, image_subpath = [], []
        found = False
        for path_part in image_path.split(os.path.sep):
            if path_part == "images":
                found = True
            if found:
                image_subpath.append(path_part)
        found = False
        for path_part in label_path.split(os.path.sep):
            if path_part == "annotations":
                found = True
            if found:
                label_subpath.append(path_part)

        if i >= 0.9*nb_imgs:
            new_label_path = os.path.join(val_path, *label_subpath[:-1])
            new_image_path = os.path.join(val_path, *image_subpath[:-1])
            os.makedirs(new_label_path, exist_ok=True)
            os.makedirs(new_image_path, exist_ok=True)
            shutil.move(label_path, new_label_path)
            shutil.move(image_path, new_image_path)

    print("\nFinished splitting dataset")


if __name__ == "__main__":
    main()
