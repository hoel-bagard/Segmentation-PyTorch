import time
import os
from subprocess import CalledProcessError

import torch
import torch.nn as nn


from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.loss import MSE_Loss
from src.torch_utils.utils.trainer import Trainer
from src.torch_utils.utils.tensorboard import TensorBoard
from src.torch_utils.utils.classification_metrics import ClassificationMetrics
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.ressource_usage import resource_usage


def train(model: nn.Module, train_dataloader: BatchGenerator, val_dataloader: BatchGenerator):
    """
    Trains and validate the given model using the datasets.
    Args:
        model: Model to train
        train_dataloader: BatchGenerator of training data
        val_dataloader: BatchGenerator of validation data
    """
    loss_fn = MSE_Loss(negative_loss_factor=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=ModelConfig.LR, weight_decay=ModelConfig.REG_FACTOR)
    trainer = Trainer(model, loss_fn, optimizer, train_dataloader, val_dataloader)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=ModelConfig.LR_DECAY)

    if DataConfig.USE_TB:
        metrics = ClassificationMetrics(model, train_dataloader, val_dataloader,
                                        DataConfig.LABEL_MAP, max_batches=10, segmentation=True)
        tensorboard = TensorBoard(model, DataConfig.TB_DIR, ModelConfig.IMAGE_SIZES, metrics, DataConfig.LABEL_MAP,
                                  segmentation=True, color_map=DataConfig.COLOR_MAP)

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
                        tensorboard.write_weights_grad(epoch)
                        tensorboard.write_loss(epoch, epoch_loss, mode="Validation")

                        # Metrics for the Train dataset
                        tensorboard.write_segmentation(epoch, train_dataloader)
                        tensorboard.write_metrics(epoch)
                        train_acc = metrics.get_avg_acc()

                        # Metrics for the Validation dataset
                        tensorboard.write_segmentation(epoch, val_dataloader, mode="Validation")
                        tensorboard.write_metrics(epoch, mode="Validation")
                        val_acc = metrics.get_avg_acc()

                        print(f"Train accuracy: {train_acc:.3f}  -  Validation accuracy: {val_acc:.3f}", end='\r')

                    print(f"\nValidation loss: {epoch_loss:.5e}  -  Took {time.time() - validation_start_time:.5f}s")
            scheduler.step()
    except KeyboardInterrupt:
        print("\n")

    train_stop_time = time.time()
    print("Finished Training"
          f"\n\tTraining time : {train_stop_time - train_start_time:.03f}s")
    tensorboard.close_writers()
    try:
        memory_peak, gpu_memory = resource_usage()
        print(f"\n\tRAM peak : {memory_peak // 1024} MB\n\tVRAM usage : {gpu_memory}")
    except CalledProcessError:
        pass
