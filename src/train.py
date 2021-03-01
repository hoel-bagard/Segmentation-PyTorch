import time
import os
from typing import (
    Callable,
    Optional
)

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.loss import CE_Loss
from src.torch_utils.utils.trainer import Trainer
from src.torch_utils.utils.tensorboard import TensorBoard
from src.torch_utils.utils.metrics import Metrics
from src.torch_utils.utils.batch_generator import BatchGenerator


def train(model: nn.Module, train_dataloader: BatchGenerator, val_dataloader: BatchGenerator,
          aug_pipeline: Optional[Callable[[np.ndarray, np.ndarray], tuple[Tensor, Tensor]]] = None):
    """
    Creates model corresponding to the given name.
    Args:
        model: Model to train
        train_dataloader: BatchGenerator of training data
        val_dataloader: BatchGenerator of validation data
        aug_pipeline: Function that takes in data and labels and do some data augmentation on them
    """
    loss_fn = CE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ModelConfig.LR, weight_decay=ModelConfig.REG_FACTOR)
    trainer = Trainer(model, loss_fn, optimizer, train_dataloader, val_dataloader, aug_pipeline=aug_pipeline)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=ModelConfig.LR_DECAY)

    if DataConfig.USE_TB:
        metrics = Metrics(model, loss_fn, train_dataloader, val_dataloader,
                          DataConfig.LABEL_MAP, max_batches=None)
        tensorboard = TensorBoard(model, metrics, DataConfig.LABEL_MAP, DataConfig.TB_DIR,
                                  ModelConfig.IMAGE_SIZES)

    best_loss = 1000
    last_checkpoint_epoch = 0
    train_start_time = time.time()

    try:
        for epoch in range(ModelConfig.MAX_EPOCHS):
            epoch_start_time = time.perf_counter()
            print(f"\nEpoch {epoch}/{ModelConfig.MAX_EPOCHS}")

            epoch_loss = trainer.train_epoch()
            if DataConfig.USE_TB:
                tensorboard.write_loss(epoch, epoch_loss)
                tensorboard.write_lr(epoch, scheduler.get_last_lr()[0])

            if (epoch_loss < best_loss and DataConfig.USE_CHECKPOINT and epoch >= DataConfig.RECORD_START
                    and (epoch - last_checkpoint_epoch) >= DataConfig.CHECKPT_SAVE_FREQ):
                save_path = os.path.join(DataConfig.CHECKPOINT_DIR, f"train_{epoch}.pt")
                print(f"\nLoss improved from {best_loss:.5e} to {epoch_loss:.5e},"
                      f"saving model to {save_path}", end='\r')
                best_loss, last_checkpoint_epoch = epoch_loss, epoch
                torch.save(model.state_dict(), save_path)

            print(f"\nEpoch loss: {epoch_loss:.5e}  -  Took {time.perf_counter() - epoch_start_time:.5f}s")

            # Validation and other metrics
            if epoch % DataConfig.VAL_FREQ == 0 and epoch >= DataConfig.RECORD_START:
                with torch.no_grad():
                    validation_start_time = time.perf_counter()
                    epoch_loss = trainer.val_epoch()

                    if DataConfig.USE_TB:
                        print("\nStarting to compute TensorBoard metrics", end="\r", flush=True)
                        tensorboard.write_loss(epoch, epoch_loss, mode="Validation")

                        # Write exemple of predictions
                        tensorboard.write_images(epoch, train_dataloader)
                        tensorboard.write_images(epoch, val_dataloader, mode="Validation")



                        # TODO: Validation loop used for videos
                        # print("\nStarting to compute TensorBoard metrics", end="\r", flush=True)
                        # # TODO: Uncomment line bellow and see if it works properly
                        # # tensorboard.write_weights_grad(epoch)
                        # tensorboard.write_loss(epoch, epoch_loss, mode="Validation")

                        # # Metrics for the Train dataset
                        # tensorboard.write_images(epoch, train_dataloader, input_is_video=True,
                        #                          preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                        # if epoch % (3*DataConfig.VAL_FREQ) == 0:
                        #     tensorboard.write_videos(epoch, train_dataloader,
                        #                              preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                        # train_acc = tensorboard.write_metrics(epoch, write_defect_acc=True)

                        # # Metrics for the Validation dataset
                        # tensorboard.write_images(epoch, val_dataloader, mode="Validation", input_is_video=True,
                        #                          preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                        # if epoch % (3*DataConfig.VAL_FREQ) == 0:
                        #     tensorboard.write_videos(epoch, val_dataloader, mode="Validation",
                        #                              preprocess_fn=preprocess_fn, postprocess_fn=postprocess_fn)
                        # val_acc = tensorboard.write_metrics(epoch, mode="Validation", write_defect_acc=True)

                        # print(f"Train accuracy: {train_acc:.3f}  -  Validation accuracy: {val_acc:.3f}",
                        #       end='\r', flush=True)

                    print(f"\nValidation loss: {epoch_loss:.5e}  -"
                          f"  Took {time.perf_counter() - validation_start_time:.5f}s", flush=True)
            scheduler.step()
    except KeyboardInterrupt:
        print("\n")

    train_stop_time = time.time()
    tensorboard.close_writers()
    memory_peak, gpu_memory = resource_usage()
    print("Finished Training"
          f"\n\tTraining time : {train_stop_time - train_start_time:.03f}s"
          f"\n\tRAM peak : {memory_peak // 1024} MB\n\tVRAM usage : {gpu_memory}")
