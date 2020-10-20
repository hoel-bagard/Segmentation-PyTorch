import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config.data_config import DataConfig
from config.model_config import ModelConfig


class TensorBoard():
    def __init__(self, model: nn.Module, max_outputs: int = 4):
        """
        Args:
            model: Model whose performance are to be recorded
            max_outputs: Number of images kept and dislpayed in TensorBoard
        """
        super(TensorBoard, self).__init__()
        self.max_outputs = max_outputs
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_tb_writer = SummaryWriter(os.path.join(DataConfig.TB_DIR, "Train"))
        self.val_tb_writer = SummaryWriter(os.path.join(DataConfig.TB_DIR, "Validation"))
        self.train_tb_writer.add_graph(model, (torch.empty(1, 3, *ModelConfig.IMAGE_SIZES, device=self.device), ))
        self.train_tb_writer.flush()

    def write_images(self, epoch: int, dataloader: torch.utils.data.DataLoader, mode: str = "Train"):
        """
        Writes images with predictions written on them to TensorBoard
        Args:
            epoch: Current epoch
            dataloader: The images will be sampled from this dataset
            mode: Either "Train" or "Validation"
        """
        print("Writing images" + ' ' * (os.get_terminal_size()[0] - len("Writing images")), end="\r", flush=True)
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        batch = next(iter(dataloader))
        in_imgs, labels = batch["img"][:self.max_outputs].float(), batch["label"][:self.max_outputs]
        out_imgs = self.model(in_imgs.to(self.device))
        for image_index, out_img in enumerate(out_imgs):
            out_img = np.transpose(out_img, (2, 0, 1))  # HWC -> CHW
            tb_writer.add_image(f"{mode}/prediction_{image_index}", out_img, global_step=epoch)

    def write_loss(self, epoch: int, loss: float, mode: str = "Train"):
        """
        Writes loss metric in TensorBoard
        Args:
            epoch: Current epoch
            loss: Epoch loss that will be added to the TensorBoard
            mode: Either "Train" or "Validation"
        """
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        tb_writer.add_scalar("Loss", loss, epoch)
        self.train_tb_writer.flush()

    def write_lr(self, epoch: int, lr: float):
        """
        Writes learning rate in the TensorBoard
        Args:
            epoch: Current epoch
            lr: Learning rate for the given epoch
        """
        self.train_tb_writer.add_scalar("Learning Rate", lr, epoch)
