import os
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
        limit (int, optional): If given then the number of elements for each class in the dataset
                            will be capped to this number
        load_videos: If true then this function returns the videos instead of their paths
    Return:
        numpy array containing the paths/images and the associated label
    """
    # TODO: project specific
    if mode == "Train":
        data_names = open(os.path.join(data_path, "ImageSets/Main/train.txt"), "r")
    elif mode == "Validation":
        data_names = open(os.path.join(data_path, "ImageSets/Main/val.txt"), "r")

    imgs = []
    labels = []
    for name in data_names:
        # TODO: project specific
        image_path = os.path.join(data_path, "JPEGImages", name.strip()+".jpg")
        label_path = os.path.join(data_path, "Annotations", name.strip()+".xml")

        # Load data directly if everything should be in RAM
        if load_data:
            resized_img, seg_map = prepare_data(image_path, label_path)
            imgs.append(resized_img)
            labels.append(seg_map)
        else:
            imgs.append(image_path)
            labels.append(label_path)

    labels = np.asarray(labels)
    imgs = np.asarray(imgs)

    return imgs, labels


def prepare_data(image_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
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
    height, width, _ = img.shape
    resized_img = cv2.resize(img, (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Read label file and make the segmentation map
    img_labels = parse_voc2007_annotation(label_path)
    resized_img_labels = resize_labels(img_labels, height, width)
    seg_map = draw_segmentation_map(resized_img_labels, (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE))
    return resized_img, seg_map


def parse_voc2007_annotation(xml_path: str) -> np.ndarray:
    root: ET.Element = ET.parse(xml_path).getroot()
    objects: ET.Element = root.findall("object")

    labels = []
    for item in objects:
        difficult = int(item.find("difficult").text)
        if difficult:
            continue
        labels.append([])

        cls = CLASS_TO_INT_DICT[item.find("name").text]
        labels[-1].append(cls)
        bbox = np.asarray([(int(item.find("bndbox").find("xmin").text)),
                           (int(item.find("bndbox").find("ymin").text)),
                           (int(item.find("bndbox").find("xmax").text)),
                           (int(item.find("bndbox").find("ymax").text))], dtype=np.int32)
        labels[-1].append(bbox)

    return np.asarray(labels)


def draw_segmentation_map(labels: np.ndarray, shape: Tuple[int, int]):
    seg_map = np.full(shape, [255, 255, 255])

    for label in labels:
        xmin, ymin, xmax, ymax = label[1]
        seg_map[ymin:ymax, xmin:xmax] = [0, 0, 0]

    return seg_map


def resize_labels(img_labels: np.ndarray, org_height: int, org_width: int):
    """
    Resizes labels so that they match the resized image
    Args:
        img_labels: labels for one image, array of  [class, bbox]
        org_width, org_height: dimensions of the image before it got resized
    """
    resized_labels = []
    for cls, bbox in img_labels:
        new_x_min = int(bbox[0] * ModelConfig.IMG_SIZE / org_width)
        new_y_min = int(bbox[1] * ModelConfig.IMG_SIZE / org_height)
        new_x_max = int(bbox[2] * ModelConfig.IMG_SIZE / org_width)
        new_y_max = int(bbox[3] * ModelConfig.IMG_SIZE / org_height)

        new_bbox = np.asarray([new_x_min, new_y_min, new_x_max, new_y_max], dtype=np.int32)
        resized_labels.append(np.asarray([cls, *new_bbox]))

    return np.asarray(resized_labels)
