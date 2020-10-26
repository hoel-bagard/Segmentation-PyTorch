import os
import glob
import xml.etree.ElementTree as ET
from typing import Tuple, Dict

import cv2
import numpy as np

from config.model_config import ModelConfig


def load_voc_seg(data_path: str, label_map: Dict,
                 limit: int = None, load_data: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads VOC labels for image segmentation.
    Args:
        data_path: Path to the VOC2007 folder (included)
        label_map: Dictionnary mapping int to class name
        limit (int, optional): If given then the number of elements for each class in the dataset
                            will be capped to this number
        load_data: If true then this function returns the videos instead of their paths
    Return:
        numpy array containing the paths/images and the associated label
    """
    data = []
    for i, label_path in enumerate(glob.glob(os.path.join(data_path, "**", "*.xml"), recursive=True)):
        root: ET.Element = ET.parse(label_path).getroot()
        image_path: str = root.find("path").text

        image_subpath = os.path.join(*image_path.split(os.path.sep)[-2:])

        # It seems like glob does not work with Japanese characters
        f = []
        for dirpath, subdirs, files in os.walk(os.path.join(data_path, "images")):
            f.extend(os.path.join(dirpath, x) for x in files)

        for filename in f:
            # TODO: use the full subpath to avoid duplicates  (Seems like there is an issue with Japanese characters)
            if image_subpath.split(os.path.sep)[-1] in filename:
                image_path = filename

        # Load data directly if everything should be in RAM
        if load_data:
            resized_img, seg_map = prepare_data(image_path, label_path, label_map)
            data.append([resized_img, seg_map])
        else:
            data.append([image_path, label_path])

        if limit and i == limit-1:
            break

    data = np.asarray(data, dtype=object)

    return data


def prepare_data(image_path: str, label_path: str, label_map: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes in image and label paths, returns ready to use data.
    Args:
        image_path: Path to the image
        label_path: Path to the corresponding xml file (pascal VOC label)
    Return:
        Image and associated segmentation map
    """
    # TODO: Put the resize in a nn.Module / Transform.
    # Load and resize image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape
    resized_img = cv2.resize(img, ModelConfig.IMAGE_SIZES, interpolation=cv2.INTER_AREA)

    # Read label file and make the segmentation map
    img_labels = parse_voc2007_annotation(label_path, label_map)
    resized_img_labels = resize_labels(img_labels, height, width)
    seg_map = draw_segmentation_map(resized_img_labels, ModelConfig.IMAGE_SIZES)
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

        try:
            cls = next(key for key, value in label_map.items() if value == item.find("name").text)
        except StopIteration:
            print(f"Class {item.find('name').text} is not in the label map:")
            print(label_map)
            exit()

        labels[-1].append(cls)
        bbox = np.asarray([(int(item.find("bndbox").find("xmin").text)),
                           (int(item.find("bndbox").find("ymin").text)),
                           (int(item.find("bndbox").find("xmax").text)),
                           (int(item.find("bndbox").find("ymax").text))], dtype=np.int32)
        labels[-1].append(bbox)

    return np.asarray(labels, dtype=object)


def draw_segmentation_map(labels: np.ndarray, shape: Tuple[int, int]):
    seg_map = np.full((*shape[::-1], 1), 1)

    for label in labels:
        xmin, ymin, xmax, ymax = label[1]
        seg_map[ymin:ymax, xmin:xmax] = 0

    return seg_map.astype(np.uint8)


def resize_labels(img_labels: np.ndarray, org_height: int, org_width: int):
    """
    Resizes labels so that they match the resized image
    Args:
        img_labels: labels for one image, array of  [class, bbox]
        org_width, org_height: dimensions of the image before it got resized
    """
    resized_labels = []
    for cls, bbox in img_labels:
        new_x_min = int(bbox[0] * ModelConfig.IMAGE_SIZES[0] / org_width)
        new_y_min = int(bbox[1] * ModelConfig.IMAGE_SIZES[1] / org_height)
        new_x_max = int(bbox[2] * ModelConfig.IMAGE_SIZES[0] / org_width)
        new_y_max = int(bbox[3] * ModelConfig.IMAGE_SIZES[1] / org_height)

        new_bbox = np.asarray([new_x_min, new_y_min, new_x_max, new_y_max], dtype=np.int32)
        resized_labels.append(np.asarray([cls, new_bbox], dtype=object))

    return np.asarray(resized_labels, dtype=object)
