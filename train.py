import argparse
import sys
import time
import traceback
from functools import partial
from subprocess import CalledProcessError

import albumentations
import cv2
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import src.dataset.data_transformations as transforms
from config.data_config import get_data_config
from config.model_config import get_model_config
from src.dataset.data_transformations_albumentations import albumentation_wrapper
from src.dataset.dataset_specific_fn import default_get_mask_path as get_mask_path
from src.dataset.default_loader import default_loader as loader_fn
from src.dataset.default_loader import load_data_fn as load_data_fn
from src.dataset.default_loader import load_labels_fn as load_labels_fn
from src.losses import DiceBCELoss
from src.networks.build_network import build_model
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.classification_metrics import ClassificationMetrics
from src.torch_utils.utils.draw import denormalize_np
from src.torch_utils.utils.logger import create_logger
from src.torch_utils.utils.misc import get_dataclass_as_dict
from src.torch_utils.utils.prepare_folders import prepare_folders
from src.torch_utils.utils.ressource_usage import resource_usage
from src.torch_utils.utils.tensorboard import TensorBoard
from src.torch_utils.utils.torch_summary import summary
from src.torch_utils.utils.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Segmentation training")
    parser.add_argument("--limit", default=None, type=int, help="Limits the number of apparition of each class")
    parser.add_argument("--load_data", action="store_true", help="Loads all the videos into RAM")
    parser.add_argument("--name", type=str, default="Train",
                        help="Use it to know what a train is when using ps. Also name of the logger.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    name: str = args.name
    verbose_level: str = args.verbose_level

    data_config = get_data_config()
    model_config = get_model_config()

    prepare_folders(data_config.TB_DIR if data_config.USE_TB else None,
                    data_config.CHECKPOINTS_DIR if data_config.USE_CHECKPOINTS else None,
                    repo_name="Segmentation-PyTorch")
    log_dir = data_config.CHECKPOINTS_DIR / "print_logs" if data_config.USE_CHECKPOINTS else None
    logger = create_logger(name, log_dir=log_dir, verbose_level=verbose_level)
    logger.info("Finished preparing tensorboard and checkpoints folders.")

    torch.backends.cudnn.benchmark = True   # Makes training quite a bit faster

    train_data, train_labels = loader_fn(data_config.DATA_PATH / "Train",
                                         get_mask_path_fn=get_mask_path,
                                         limit=args.limit,
                                         load_data=args.load_data,
                                         data_preprocessing_fn=load_data_fn if args.load_data else None,
                                         labels_preprocessing_fn=load_labels_fn if args.load_data else None)
    logger.info("Train data loaded")

    val_data, val_labels = loader_fn(data_config.DATA_PATH / "Validation",
                                     get_mask_path_fn=get_mask_path,
                                     limit=args.limit,
                                     load_data=args.load_data,
                                     data_preprocessing_fn=load_data_fn if args.load_data else None,
                                     labels_preprocessing_fn=load_labels_fn if args.load_data else None)
    logger.info("Validation data loaded")

    # Data augmentation done on cpu.
    augmentation_pipeline = albumentation_wrapper(albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        # albumentations.RandomRotate90(p=0.2),
        # albumentations.CLAHE(),
        albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(scale_limit=0.05, rotate_limit=10, shift_limit=0.06, p=0.5,
                                        border_mode=cv2.BORDER_CONSTANT,  # cv2.BORDER_REFLECT_101
                                        value=0, mask_value=[1]+[0]*(data_config.OUTPUT_CLASSES-1)),
        # albumentations.GridDistortion(p=0.5),
    ]))

    common_pipeline = albumentation_wrapper(albumentations.Compose([
        albumentations.Normalize(mean=model_config.MEAN, std=model_config.STD, max_pixel_value=255.0, p=1.0),
        albumentations.Resize(*model_config.IMAGE_SIZES, interpolation=cv2.INTER_LINEAR)
    ]))
    train_pipeline = transforms.compose_transformations((augmentation_pipeline, common_pipeline))

    with BatchGenerator(train_data,
                        train_labels,
                        model_config.BATCH_SIZE,
                        nb_workers=data_config.NB_WORKERS,
                        data_preprocessing_fn=load_data_fn if not args.load_data else None,
                        labels_preprocessing_fn=load_labels_fn if not args.load_data else None,
                        cpu_pipeline=train_pipeline,
                        gpu_pipeline=transforms.to_tensor(),
                        shuffle=True) as train_dataloader, \
        BatchGenerator(val_data,
                       val_labels,
                       model_config.BATCH_SIZE,
                       nb_workers=data_config.NB_WORKERS,
                       data_preprocessing_fn=load_data_fn if not args.load_data else None,
                       labels_preprocessing_fn=load_labels_fn if not args.load_data else None,
                       cpu_pipeline=common_pipeline,
                       gpu_pipeline=transforms.to_tensor(),
                       shuffle=False) as val_dataloader:

        print(f"\nLoaded {len(train_dataloader)} train data and",
              f"{len(val_dataloader)} validation data", flush=True)

        print("Building model. . .", end="\r")
        model = build_model(model_config.MODEL, data_config.OUTPUT_CLASSES, **get_dataclass_as_dict(model_config))

        logger.info(f"{'-'*24} Starting train {'-'*24}")
        logger.info("From command : " + ' '.join(sys.argv))
        logger.info(f"Input shape: {train_dataloader.data_shape}")
        logger.info("")
        logger.info("Using model:")
        for line in summary(model, train_dataloader.data_shape):
            logger.info(line)
        logger.info("")

        loss_fn = DiceBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.LR, weight_decay=model_config.WEIGHT_DECAY)
        trainer = Trainer(model, loss_fn, optimizer, train_dataloader, val_dataloader)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=model_config.LR_DECAY)
        scheduler = CosineAnnealingLR(optimizer, model_config.MAX_EPOCHS, eta_min=5e-6)
        # TODO: Try this https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/cosine_lr.py

        if data_config.USE_TB:
            metrics = ClassificationMetrics(model, train_dataloader, val_dataloader,
                                            data_config.LABEL_MAP, max_batches=10, segmentation=True)
            tensorboard = TensorBoard(model, data_config.TB_DIR, model_config.IMAGE_SIZES, metrics,
                                      data_config.LABEL_MAP, color_map=data_config.COLOR_MAP,
                                      denormalize_img_fn=partial(denormalize_np,
                                                                 mean=model_config.MEAN,
                                                                 std=model_config.STD))

        best_loss = 1000
        last_checkpoint_epoch = 0
        train_start_time = time.time()
        try:
            for epoch in range(model_config.MAX_EPOCHS):
                epoch_start_time = time.perf_counter()
                print()  # logger doesn't handle \n super well
                logger.info(f"Epoch {epoch}/{model_config.MAX_EPOCHS}")

                epoch_loss = trainer.train_epoch()
                if data_config.USE_TB:
                    tensorboard.write_loss(epoch, epoch_loss)
                    tensorboard.write_lr(epoch, scheduler.get_last_lr()[0])

                if (epoch_loss < best_loss and data_config.USE_CHECKPOINTS and epoch >= data_config.RECORD_START
                        and (epoch - last_checkpoint_epoch) >= data_config.CHECKPT_SAVE_FREQ):
                    save_path = data_config.CHECKPOINTS_DIR / f"train_{epoch}.pt"
                    logger.info(f"\nLoss improved from {best_loss:.5e} to {epoch_loss:.5e},"
                                f"saving model to {save_path}")
                    best_loss, last_checkpoint_epoch = epoch_loss, epoch
                    torch.save(model.state_dict(), save_path)

                logger.info(f"Epoch loss: {epoch_loss:.5e}  -  Took {time.perf_counter() - epoch_start_time:.5f}s")

                # Validation and other metrics
                if epoch % data_config.VAL_FREQ == 0 and epoch >= data_config.RECORD_START:
                    # if data_config.USE_TB:
                    #     tensorboard.write_weights_grad(epoch)
                    with torch.no_grad():
                        validation_start_time = time.perf_counter()
                        epoch_loss = trainer.val_epoch()

                        if data_config.USE_TB:
                            print("Starting to compute TensorBoard metrics", end="\r", flush=True)
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

                            logger.info(f"Train accuracy: {train_acc:.3f}  -  Validation accuracy: {val_acc:.3f}")

                        logger.info(f"Validation loss: {epoch_loss:.5e}  -  "
                                    f"Took {time.perf_counter() - validation_start_time:.5f}s")
                scheduler.step()
        except KeyboardInterrupt:
            print("\n")
        except Exception as error:
            logger.error(''.join(traceback.format_exception(*sys.exc_info())))
            raise error

    if data_config.USE_TB:
        tensorboard.close_writers()

    train_stop_time = time.time()
    end_msg = f"Finished Training\n\tTraining time : {train_stop_time - train_start_time:.03f}s"
    try:
        memory_peak, gpu_memory = resource_usage()
        end_msg += f"\n\tRAM peak : {memory_peak // 1024} MB\n\tVRAM usage : {gpu_memory}"
    except CalledProcessError:
        pass
    logger.info(end_msg)


if __name__ == "__main__":
    main()
