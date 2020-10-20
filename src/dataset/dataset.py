import torch
import numpy as np

from src.dataset.dataset_utils import (
    load_voc_seg,
    prepare_data
)
from config.data_config import DataConfig


class Dataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(self, data_path: str, transform=None, limit: int = None, load_data: bool = True):
        """
        Args:
            data_path:
                Path to the root folder of the dataset.
                This folder is expected to contain subfolders with the xml labels and the pictures.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): If given then the number of elements for each class in the dataset
                                   will be capped to this number
            load_data: If True then all the data is loaded into ram
        """
        self.transform = transform
        self.load_data = load_data

        self.labels = load_voc_seg(data_path, DataConfig.LABEL_MAP, limit=limit, load_data=self.load_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        if self.load_data:
            img, label = self.labels[i].astype(np.uint8)
        else:
            img, label = prepare_data(self.labels[i, 0], self.labels[i, 1])

        sample = {"img": img, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
