import argparse
import os
import glob
import xml.etree.ElementTree as ET
from typing import Tuple, Dict

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser("Display data for a segmentation project with PascalVOC labels")
    parser.add_argument("data_path", type=str, help="Path to data folder with classes.names in parent folder")
    parser.add_argument("--resize", nargs=2, default=[1080, 720], type=int, help="Resizes the images to given size")
    parser.add_argument("--superpose", action="store_true", help="Displays segmentation on top of orginal image")
    args = parser.parse_args()

    data_path = args.data_path
    resize = args.resize  # [::-1]  # Python convention vs usual convention
    # Build a map between id and names
    label_map = {}
    with open(os.path.join(data_path, "..", "classes.names")) as table_file:
        for key, line in enumerate(table_file):
            label = line.strip()
            label_map[key] = label

    # Display images one by one
    for label_path in glob.glob(os.path.join(data_path, "**", "*.xml"), recursive=True):
        root: ET.Element = ET.parse(label_path).getroot()
        image_path: str = root.find("path").text

        image_subpath = os.path.join(*image_path.split(os.path.sep)[-2:])

        # It seems like glob does not work with Japanese characters
        f = []
        for dirpath, subdirs, files in os.walk(os.path.join(data_path, "images")):
            f.extend(os.path.join(dirpath, x) for x in files)

        for filename in f:
            # Should use the full subpath to avoid duplicates  (Seems like there is an issue with Japanese characters)
            if image_subpath.split(os.path.sep)[-1] in filename:
                image_path = filename

        # Load data directly if everything should be in RAM
        resized_img, seg_map = prepare_data(image_path, label_path, label_map, resize)

        if args.superpose:
            displayed_img = cv2.addWeighted(resized_img, 1, seg_map, 0.5, 0)
        else:
            displayed_img = seg_map

        displayed_img = cv2.cvtColor(displayed_img, cv2.COLOR_BGR2RGB)
        while True:
            if any([size > 1080 for size in resize]):
                cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Image", displayed_img)
            if cv2.waitKey(10) == ord("q"):
                break


def prepare_data(image_path: str, label_path: str, label_map: Dict,
                 new_sizes: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes in image and label paths, returns ready to use data.
    Args:
        image_path: Path to the image
        label_path: Path to the corresponding xml file (pascal VOC label)
        new_sizes: Size to which the images will be resized
    Return:
        Image and associated segmentation map
    """
    # Load and resize image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape
    resized_img = cv2.resize(img, tuple(new_sizes), interpolation=cv2.INTER_AREA)

    # Read label file and make the segmentation map
    img_labels = parse_voc2007_annotation(label_path, label_map)
    resized_img_labels = resize_labels(img_labels, height, width, new_sizes)
    seg_map = draw_segmentation_map(resized_img_labels, new_sizes)
    return resized_img, seg_map


def parse_voc2007_annotation(xml_path: str, label_map: Dict) -> np.ndarray:
    root: ET.Element = ET.parse(xml_path).getroot()
    objects: ET.Element = root.findall("object")

    labels = []
    for item in objects:
        difficult = int(item.find("difficult").text)
        if difficult:
            continue
        labels.append([])

        cls = next(key for key, value in label_map.items() if value == item.find("name").text)

        labels[-1].append(cls)
        bbox = np.asarray([(int(item.find("bndbox").find("xmin").text)),
                           (int(item.find("bndbox").find("ymin").text)),
                           (int(item.find("bndbox").find("xmax").text)),
                           (int(item.find("bndbox").find("ymax").text))], dtype=np.int32)
        labels[-1].append(bbox)

    return np.asarray(labels)


def draw_segmentation_map(labels: np.ndarray, shape: Tuple[int, int]):
    # Image should be RGB
    seg_map = np.full((*shape[::-1], 3), [255, 255, 255])

    for label in labels:
        xmin, ymin, xmax, ymax = label[1]
        seg_map[ymin:ymax, xmin:xmax] = [0, 0, 0]

    return seg_map.astype(np.uint8)


def resize_labels(img_labels: np.ndarray, org_height: int, org_width: int, new_sizes: Tuple[int, int]):
    """
    Resizes labels so that they match the resized image
    Args:
        img_labels: labels for one image, array of  [class, bbox]
        org_width, org_height: dimensions of the image before it got resized
        new_sizes: Size to which the images will be resized
    """
    resized_labels = []
    for cls, bbox in img_labels:
        new_x_min = int(bbox[0] * new_sizes[0] / org_width)
        new_y_min = int(bbox[1] * new_sizes[1] / org_height)
        new_x_max = int(bbox[2] * new_sizes[0] / org_width)
        new_y_max = int(bbox[3] * new_sizes[1] / org_height)

        new_bbox = np.asarray([new_x_min, new_y_min, new_x_max, new_y_max], dtype=np.int32)
        resized_labels.append(np.asarray([cls, new_bbox]))

    return np.asarray(resized_labels)


if __name__ == "__main__":
    main()
