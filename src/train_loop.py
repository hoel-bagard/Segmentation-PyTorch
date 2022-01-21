import os
import time
from subprocess import CalledProcessError

import torch
import torch.nn as nn


from config.data_config import get_data_config
from config.model_config import get_model_config
from src.loss import MSE_Loss
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.classification_metrics import ClassificationMetrics
from src.torch_utils.utils.ressource_usage import resource_usage
from src.torch_utils.utils.tensorboard import TensorBoard
from src.torch_utils.utils.trainer import Trainer


def train(model: nn.Module, train_dataloader: BatchGenerator, val_dataloader: BatchGenerator):
    """Trains and validate the given model using the datasets.

    Args:
        model: Model to train
        train_dataloader: BatchGenerator of training data
        val_dataloader: BatchGenerator of validation data
    """
    data_config = get_data_config()
    model_config = get_model_config()

    loss_fn = MSE_Loss(negative_loss_factor=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.LR, weight_decay=model_config.REG_FACTOR)
    trainer = Trainer(model, loss_fn, optimizer, train_dataloader, val_dataloader)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=model_config.LR_DECAY)

    if data_config.USE_TB:
        metrics = ClassificationMetrics(model, train_dataloader, val_dataloader,
                                        data_config.LABEL_MAP, max_batches=10, segmentation=True)
        tensorboard = TensorBoard(model, data_config.TB_DIR, model_config.IMAGE_SIZES, metrics, data_config.LABEL_MAP,
                                  color_map=data_config.COLOR_MAP)

    best_loss = 1000
    last_checkpoint_epoch = 0
    train_start_time = time.time()

    try:
        for epoch in range(model_config.MAX_EPOCHS):
            epoch_start_time = time.perf_counter()
            print(f"\nEpoch {epoch}/{model_config.MAX_EPOCHS}")

            epoch_loss = trainer.train_epoch()
            if data_config.USE_TB:
                tensorboard.write_loss(epoch, epoch_loss)
                tensorboard.write_lr(epoch, scheduler.get_last_lr()[0])

            if (epoch_loss < best_loss and data_config.USE_CHECKPOINT and epoch >= data_config.RECORD_START
                    and (epoch - last_checkpoint_epoch) >= data_config.CHECKPT_SAVE_FREQ):
                save_path = os.path.join(data_config.CHECKPOINT_DIR, f"train_{epoch}.pt")
                print(f"\nLoss improved from {best_loss:.5e} to {epoch_loss:.5e},"
                      f"saving model to {save_path}", end='\r')
                best_loss, last_checkpoint_epoch = epoch_loss, epoch
                torch.save(model.state_dict(), save_path)

            print(f"\nEpoch loss: {epoch_loss:.5e}  -  Took {time.perf_counter() - epoch_start_time:.5f}s")

            # Validation and other metrics
            if epoch % data_config.VAL_FREQ == 0 and epoch >= data_config.RECORD_START:
                with torch.no_grad():
                    validation_start_time = time.perf_counter()
                    epoch_loss = trainer.val_epoch()

                    if data_config.USE_TB:
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
