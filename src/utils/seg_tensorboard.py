from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from einops import rearrange

from config.data_config import get_data_config
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.misc import clean_print
from src.torch_utils.utils.tensorboard_template import TensorBoard
from src.utils.draw_seg import draw_segmentation
from src.utils.grasping_metrics import GraspingMetrics


class SegmentationTensorBoard(TensorBoard):
    """Class with TensorBoard functions for segmentation.

    Args:
        model (nn.Module): Pytorch model whose performance are to be recorded
        tb_dir (Path): Path to where the tensorboard files will be saved
        metrics (Metrics, optional): Instance of the Metrics class, used to compute classification metrics
        denormalize_imgs_fn (Callable): Function to destandardize a batch of images.
        write_graph (bool): If True, add the network graph to the TensorBoard
        max_outputs (int): Maximal number of images kept and displayed in TensorBoard (per function call)
    """
    def __init__(self,
                 model: nn.Module,
                 tb_dir: Path,
                 metrics: GraspingMetrics,
                 denormalize_imgs_fn: Callable,
                 train_dataloader: BatchGenerator,
                 val_dataloader: BatchGenerator,
                 write_graph: bool = True,
                 max_outputs: int = 4):
        super().__init__(model, tb_dir, train_dataloader, val_dataloader, metrics, write_graph)
        self.max_outputs = max_outputs
        self.denormalize_imgs_fn = denormalize_imgs_fn
        self.data_config = get_data_config()

    def write_images(self,
                     epoch: int,
                     mode: str = "Train"):
        """Writes images with predictions written on them to TensorBoard.

        Args:
            epoch (int): Current epoch
            dataloader (BatchGenerator): The images will be sampled from this dataset
            draw_fn (callable): Function that takes in the tensor images, labels and predictions
                                and draws on the images before returning them.
            mode (str): Either "Train" or "Validation"
            preprocess_fn (callable, optional): Function called before inference.
                                                Gets data and labels as input, expects them as outputs
            postprocess_fn (callable, optional): Function called after inference.
                                                 Gets data and predictions as input, expects them as outputs
        """
        clean_print("Writing images", end="\r")
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        dataloader = self.train_dataloader if mode == "Train" else self.val_dataloader

        batch = dataloader.next_batch()
        dataloader.reset_epoch()  # Reset the epoch to not cause issues for other functions

        imgs, labels = batch[0][:self.max_outputs], batch[1][:self.max_outputs]

        # Get some predictions
        preds = self.model(imgs)

        imgs = rearrange(imgs, "b c w h -> b w h c").cpu().detach().numpy()
        imgs_batch: npt.NDArray[np.uint8] = self.denormalize_img_fn(imgs)

        one_hot_masks_preds = rearrange(preds, "b c w h -> b w h c")
        masks_preds: np.ndarray = torch.argmax(one_hot_masks_preds, dim=-1).cpu().detach().numpy()
        one_hot_masks_labels = rearrange(labels, "b c w h -> b w h c")
        masks_labels: np.ndarray = torch.argmax(one_hot_masks_labels, dim=-1).cpu().detach().numpy()

        out_imgs = draw_segmentation(imgs_batch,
                                     masks_labels,
                                     masks_preds,
                                     color_map=self.data_config.IDX_TO_COLOR)

        # Add them to TensorBoard
        for image_index, img in enumerate(out_imgs):
            tb_writer.add_image(f"{mode}/segmentation_output_{image_index}", img, global_step=epoch, dataformats="HWC")
