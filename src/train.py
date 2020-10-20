import time
import os

import torch
import torch.nn as nn

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.loss import CE_Loss
from src.utils.trainer import Trainer
from src.utils.tensorboard import TensorBoard


def train(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader):
    loss_fn = CE_Loss()
    trainer = Trainer(model, loss_fn, train_dataloader, val_dataloader)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer, gamma=ModelConfig.LR_DECAY)

    if DataConfig.USE_TB:
        tensorboard = TensorBoard(model)

    best_loss = 1000
    last_checkpoint_epoch = 0

    for epoch in range(ModelConfig.MAX_EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch}/{ModelConfig.MAX_EPOCHS}")

        epoch_loss = trainer.train_epoch()
        if DataConfig.USE_TB:
            tensorboard.write_loss(epoch, epoch_loss)
            tensorboard.write_lr(epoch, scheduler.get_last_lr()[0])

        if (epoch_loss < best_loss and DataConfig.USE_CHECKPOINT and
                epoch >= DataConfig.RECORD_START and (epoch - last_checkpoint_epoch) >= DataConfig.CHECKPT_SAVE_FREQ):
            save_path = os.path.join(DataConfig.CHECKPOINT_DIR, f"train_{epoch}.pt")
            print(f"\nLoss improved from {best_loss:.5e} to {epoch_loss:.5e}, saving model to {save_path}", end='\r')
            best_loss, last_checkpoint_epoch = epoch_loss, epoch
            torch.save(model.state_dict(), save_path)

        print(f"\nEpoch loss: {epoch_loss:.5e}, Learning rate: {scheduler.get_last_lr()[0]:.3e}"
              + f"  -  Took {time.time() - epoch_start_time:.5f}s")

        # Validation and other metrics
        if epoch % DataConfig.VAL_FREQ == 0 and epoch >= DataConfig.RECORD_START:
            validation_start_time = time.time()
            epoch_loss = trainer.val_epoch()

            if DataConfig.USE_TB:
                print("\nStarting to compute TensorBoard metrics", end="\r", flush=True)
                tensorboard.write_loss(epoch, epoch_loss, mode="Validation")

                # Write exemple of predictions
                tensorboard.write_images(epoch, train_dataloader)
                tensorboard.write_images(epoch, val_dataloader, mode="Validation")

            print(f"\nValidation loss: {epoch_loss:.5e}  -  Took {time.time() - validation_start_time:.5f}s", flush=1)
        scheduler.step()

    print("Finished Training")
