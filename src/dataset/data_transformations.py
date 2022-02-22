import random
from typing import Callable

import cv2
import numpy as np
import torch

NumpyOrTensorType = torch.Tensor | np.ndarray


def compose_transformations(transformations: list[Callable[[NumpyOrTensorType, NumpyOrTensorType],
                                                           tuple[NumpyOrTensorType, NumpyOrTensorType]]]):
    """Returns a function that applies all the given transformations."""
    def compose_transformations_fn(imgs: NumpyOrTensorType, labels: NumpyOrTensorType):
        """Apply transformations on a batch of data."""
        for fn in transformations:
            imgs, labels = fn(imgs, labels)
        return imgs, labels
    return compose_transformations_fn


def crop(top: int, bottom: int, left: int, right: int):
    def crop_fn(imgs: np.ndarray, labels: np.ndarray):
        imgs = imgs[:, top:-bottom, left:-right]
        labels = labels[:, top:-bottom, left:-right]
        return imgs, labels
    return crop_fn


def random_crop(reduction_factor: float = 0.9):
    """Randomly crops image."""
    def random_crop_fn(imgs: np.ndarray, labels: np.ndarray):
        """Randomly crops a batch of data (the "same" patch is taken across all images)."""
        h = random.randint(0, int(imgs.shape[1]*(1-reduction_factor))-1)
        w = random.randint(0, int(imgs.shape[2]*(1-reduction_factor))-1)
        cropped_imgs = imgs[:, h:h+int(imgs.shape[1]*reduction_factor), w:w+int(imgs.shape[2]*reduction_factor)]
        cropped_labels = labels[:, h:h+int(labels.shape[1]*reduction_factor), w:w+int(labels.shape[2]*reduction_factor)]
        return cropped_imgs, cropped_labels
    return random_crop_fn


def vertical_flip(imgs: np.ndarray, labels: np.ndarray):
    """Randomly flips the img around the x-axis."""
    for i in range(len(imgs)):
        if random.random() > 0.5:
            imgs[i] = cv2.flip(imgs[i], 0)
            labels[i] = labels[i, ::-1]
    return imgs, labels


def horizontal_flip(imgs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Randomly flips the img around the y-axis."""
    for i in range(len(imgs)):
        if random.random() > 0.5:
            imgs[i] = cv2.flip(imgs[i], 1)
            labels[i] = labels[i, :, ::-1]
    return imgs, labels


def rotate180(imgs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Randomly rotate the image by 180 degrees."""
    for i in range(len(imgs)):
        if random.random() > 0.5:
            imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
            labels[i] = np.rot90(labels[i], k=2)
    return imgs, labels


def to_tensor():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to_tensor_fn(imgs: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        imgs = imgs.transpose((0, 3, 1, 2))
        labels = labels.transpose((0, 3, 1, 2))
        return torch.from_numpy(imgs).float().to(device), torch.from_numpy(labels).to(device)
    return to_tensor_fn


def normalize(labels_too: bool = False):
    """Normalize a batch of images so that its values are in [0, 1]."""
    def normalize_fn(imgs: torch.Tensor, labels: torch.Tensor):
        return imgs/255.0, labels/255.0 if labels_too else labels
    return normalize_fn


def noise():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def noise_fn(imgs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add random noise to the image."""
        noise_offset = (torch.rand(imgs.shape, device=device)-0.5)*0.05
        noise_scale = (torch.rand(imgs.shape, device=device) * 0.2) + 0.9

        imgs = imgs * noise_scale + noise_offset
        imgs = torch.clamp(imgs, 0, 1)

        return imgs, labels
    return noise_fn
