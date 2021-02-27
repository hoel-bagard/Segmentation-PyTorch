import random
from typing import Callable

import cv2
import numpy as np
import torch


# TODO: Use functools instead ?
# https://docs.python.org/3/howto/functional.html#the-functools-module
# https://docs.python.org/3/library/functools.html#functools.reduce
def compose_transformations(functions: list[Callable[[np.ndarray, np.ndarray], [np.ndarray, np.ndarray]]]):
    """ Returns a function that applies all the given functions"""
    def compose_transformations_fn(img: np.ndarray, label: np.ndarray):
        for fn in functions:
            img, label = fn(img, label)
        return img, label
    return compose_transformations_fn


def crop(self, top: int = 0, bottom: int = 1, left: int = 0, right: int = 1):
    """ Returns a function that crops an image """
    def crop_fn(img, label):
        return img[top:-bottom, left:-right], label[top:-bottom, left:-right]
    return crop_fn


def random_crop(reduction_factor: int = 0.9):
    """ Randomly crops image """
    def random_crop_fn(img, label):
        h = random.randint(0, int(img.shape[0]*(1-reduction_factor))-1)
        w = random.randint(0, int(img.shape[1]*(1-reduction_factor))-1)
        cropped_img = img[h:h+int(img.shape[0]*reduction_factor), w:w+int(img.shape[1]*reduction_factor)]
        cropped_label = label[h:h+int(label.shape[0]*reduction_factor), w:w+int(label.shape[1]*reduction_factor)]
        return cropped_img, cropped_label



class Resize(object):
    """ Resize the image in a sample to a given size. """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img = cv2.resize(img, (self.width, self.height))
        label = cv2.resize(label, (self.width, self.height))
        return {'img': img, 'label': label}


class Normalize(object):
    """ Normalize the image so that its values are in [0, 1] """

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        return {'img': img/255.0, 'label': label}


class VerticalFlip(object):
    """ Randomly flips the img around the x-axis """

    def __call__(self, sample):
        img, label = sample["img"], sample["label"]
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            label = cv2.flip(label, 0)
        return {"img": img, "label": label}


class HorizontalFlip(object):
    """ Randomly flips the img around the y-axis """

    def __call__(self, sample):
        img, label = sample["img"], sample["label"]
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)

        return {"img": img, "label": label}


class Rotate180(object):
    """ Randomly rotate the image by 180 degrees """

    def __call__(self, sample):
        img, label = sample["img"], sample["label"]
        if random.random() > 0.5:
            img = cv2.rotate(img, cv2.ROTATE_180)
            label = cv2.rotate(label, cv2.ROTATE_180)
        return {"img": img, "label": label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, label = sample['img'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        img = img.transpose((2, 0, 1))
        label = np.expand_dims(label, axis=-1)  # Because opencv removes the channel dimension for greyscale imgs
        label = label.transpose((2, 0, 1))
        return {'img': torch.from_numpy(img),
                'label': torch.from_numpy(label)}


class Noise(object):
    """ Add random noise to the image """

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        noise_offset = (torch.rand(img.shape)-0.5)*0.05
        noise_scale = (torch.rand(img.shape) * 0.2) + 0.9

        img = img * noise_scale + noise_offset
        img = torch.clamp(img, 0, 1)

        return {'img': img, 'label': label}
