import itertools
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from einops import rearrange

from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.metrics import Metrics
from src.torch_utils.utils.misc import clean_print


class SegmentationMetrics(Metrics):
    """Class computing usefull metrics for segmentation like tasks."""
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: BatchGenerator,
                 val_dataloader: BatchGenerator,
                 idx_to_name: dict[int, str],
                 max_danger: int,
                 max_batches: Optional[int] = 10):
        """Initialize the instance.

        Args:
            model (nn.Module): The PyTorch model being trained
            train_dataloader (BatchGenerator): DataLoader containing train data
            val_dataloader (BatchGenerator): DataLoader containing validation data
            idx_to_name  (dict): Dictionary linking class index to class name
            max_danger: Max danger level
            max_batches (int): If not None, then the metrics will be computed using at most this number of batches
        """
        super().__init__(model, train_dataloader, val_dataloader, max_batches)

        self.label_map = idx_to_name
        self.nb_output_classes = len(idx_to_name)
        self.max_danger_lvl = max_danger

        self.cls_cm: npt.NDArray[np.float64]
        self.danger_cm: npt.NDArray[np.float64]

    def compute_confusion_matrices(self, mode: str = "Train"):
        """Computes the confusion matrices. This function has to be called before using the get functions.

        Args:
            mode (str): Either "Train" or "Validation"
        """
        self.cls_cm = np.zeros((self.nb_output_classes, self.nb_output_classes))
        self.danger_cm = np.zeros((self.max_danger_lvl, self.max_danger_lvl))

        dataloader = self.train_dataloader if mode == "Train" else self.val_dataloader
        for step, (imgs, labels) in enumerate(dataloader, start=1):
            cls_oh_preds, danger_oh_preds = self.model(imgs)

            cls_masks_labels: np.ndarray = torch.argmax(labels[..., 0], dim=-1).cpu().detach().numpy()
            danger_masks_labels: np.ndarray = torch.argmax(labels[..., 1], dim=-1).cpu().detach().numpy()

            # Note: for this project I haven't switched the channels for the labels (4D tensor, annoying...).
            cls_oh_preds = rearrange(cls_oh_preds, "b c w h -> b w h c")
            cls_masks_preds: np.ndarray = torch.argmax(cls_oh_preds, dim=-1).cpu().detach().numpy()
            danger_oh_preds = rearrange(danger_oh_preds, "b c w h -> b w h c")
            danger_masks_preds: np.ndarray = torch.argmax(danger_oh_preds, dim=-1).cpu().detach().numpy()

            for (label_pixel, pred_pixel) in zip(cls_masks_labels.flatten(), cls_masks_preds.flatten()):
                self.cls_cm[label_pixel, pred_pixel] += 1

            for (label_pixel, pred_pixel) in zip(danger_masks_labels.flatten(), danger_masks_preds.flatten()):
                self.danger_cm[label_pixel, pred_pixel] += 1

            if self.max_batches and step >= self.max_batches:
                break
        dataloader.reset_epoch()  # Reset the epoch to not cause issues for other functions

    def get_avg_acc(self) -> tuple[float, float]:
        """Uses the confusion matrix to return the average accuracy of the model.

        Returns:
            Average classification and danger accuracies
        """
        cls_acc = np.sum([self.cls_cm[i, i] for i in range(len(self.cls_cm))]) / np.sum(self.cls_cm)
        danger_acc = np.sum([self.danger_cm[i, i] for i in range(len(self.danger_cm))]) / np.sum(self.danger_cm)
        return cls_acc, danger_acc

    def get_class_accuracy(self) -> tuple[list[float], list[float]]:
        """Uses the confusion matrix to return the average accuracy of the model.

        Returns:
            Arrays containing the accuracy for each class, for the classification and danger tasks
        """
        per_class_acc = [self.cls_cm[i, i] / max(1, np.sum(self.cls_cm[i])) for i in range(len(self.cls_cm))]
        per_lvl_acc = [self.danger_cm[i, i] / max(1, np.sum(self.danger_cm[i])) for i in range(len(self.danger_cm))]
        return per_class_acc, per_lvl_acc

    def get_class_iou(self) -> tuple[list[float], list[float]]:
        """Uses the confusion matrices to return the iou for each class.

        Returns:
            list: List of the IOU for each class, for each task.
        """
        intersections = [self.cls_cm[i, i] for i in range(len(self.cls_cm))]
        unions = [np.sum(self.cls_cm[i, :]) + np.sum(self.cls_cm[:, i]) - self.cls_cm[i, i]
                  for i in range(self.nb_output_classes)]
        per_class_iou_cls = [intersections[i] / max(1, unions[i]) for i in range(self.nb_output_classes)]

        intersections = [self.danger_cm[i, i] for i in range(len(self.danger_cm))]
        unions = [np.sum(self.danger_cm[i, :]) + np.sum(self.danger_cm[:, i]) - self.danger_cm[i, i]
                  for i in range(self.max_danger_lvl)]
        per_class_iou_danger = [intersections[i] / max(1, unions[i]) for i in range(self.max_danger_lvl)]
        return per_class_iou_cls, per_class_iou_danger

    @staticmethod
    def get_confusion_matrix(confusion_matrix: npt.NDArray[np.float64], class_names: list[str]) -> np.ndarray:
        """Returns an image containing the plotted confusion matrix.

        Taken from: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12

        Returns:
            np.ndarray: Image of the confusion matrix.
        """
        # Normalize the confusion matrix.
        cm = np.around(confusion_matrix.astype("float") / np.maximum(1, confusion_matrix.sum(axis=1)[:, np.newaxis]),
                       decimals=2)

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, fontname="TakaoGothic")
        plt.yticks(tick_marks, class_names, fontname="TakaoGothic")

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel("True label", labelpad=-5)
        plt.xlabel("Predicted label")
        fig.canvas.draw()

        # Convert matplotlib plot to normal image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)  # Close figure explicitly to avoid memory leak

        return img

    def get_metrics(self, mode: str = "Train", **kwargs) -> dict[str, dict[str, Any]]:
        """See base class."""
        metrics: dict[str, dict] = {"scalars": {}, "imgs": {}}

        clean_print("Computing confusion matrices", end="\r")
        self.compute_confusion_matrices(mode=mode)

        clean_print("Computing average accuracy", end="\r")
        cls_acc, danger_acc = self.get_avg_acc()
        metrics["scalars"]["Average Accuracy/Object Classification"] = cls_acc
        metrics["scalars"]["Average Accuracy/Danger Level"] = danger_acc

        clean_print("Computing per class accuracy", end="\r")
        per_class_acc, per_lvl_acc = self.get_class_accuracy()
        for key, acc in enumerate(per_class_acc):
            metrics["scalars"][f"Per Class Accuracy/{self.label_map[key]}"] = acc
        for i, acc in enumerate(per_lvl_acc):
            metrics["scalars"][f"Per Danger Level Accuracy/{i}"] = acc

        clean_print("Computing per class IOU", end="\r")
        per_class_iou_cls, per_class_iou_danger = self.get_class_iou()
        for key, iou in enumerate(per_class_iou_cls):
            metrics["scalars"][f"Per Class IOU/{self.label_map[key]}"] = iou
        for key, iou in enumerate(per_class_iou_danger):
            metrics["scalars"][f"Per Danger Level IOU/{key}"] = iou

        clean_print("Creating confusion matrix image", end="\r")
        cls_confusion_matrix = self.get_confusion_matrix(self.cls_cm, list(self.label_map.values()))
        danger_confusion_matrix = self.get_confusion_matrix(self.danger_cm,
                                                            list([str(i) for i in range(self.max_danger_lvl)]))
        metrics["imgs"]["Cls Confusion Matrix"] = cls_confusion_matrix
        metrics["imgs"]["Danger Confusion Matrix"] = danger_confusion_matrix

        return metrics
