import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path
from shutil import copy, rmtree

import albumentations
import cv2
import torch

import src.dataset.data_transformations as transforms
from config.data_config import get_data_config
from config.model_config import get_model_config
from src.dataset.data_transformations_albumentations import albumentation_wrapper
from src.dataset.dataset_specific_fn import default_get_mask_path as get_mask_path
from src.dataset.default_loader import (
    default_load_data,
    default_load_labels,
    default_loader
)
from src.networks.build_network import build_model
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.logger import create_logger
from src.torch_utils.utils.misc import clean_print, get_dataclass_as_dict
from src.torch_utils.utils.torch_summary import summary
from src.train_loop import train


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

    if data_config.USE_CHECKPOINT:
        log_dir = Path("logs") / date.today().strftime("%Y-%m-%d")
        logger = create_logger(name, log_dir)
    else:
        logger = create_logger(args.name)

    match verbose_level:
        case "debug":
            logger.setLevel(logging.DEBUG)
        case "info":
            logger.setLevel(logging.INFO)
        case "error":
            logger.setLevel(logging.ERROR)

    if not data_config.KEEP_TB:
        while data_config.TB_DIR.exists():
            rmtree(data_config.TB_DIR, ignore_errors=True)
            time.sleep(0.5)
    data_config.TB_DIR.mkdir(parents=True, exist_ok=False)

    if data_config.USE_CHECKPOINT:
        if not data_config.KEEP_CHECKPOINTS:
            while data_config.CHECKPOINT_DIR.exists():
                rmtree(data_config.CHECKPOINT_DIR, ignore_errors=True)
                time.sleep(0.5)
        try:
            data_config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            logger.info(f"The checkpoint dir {data_config.CHECKPOINT_DIR} already exists")
            return -1

        # Makes a copy of all the code (and config) so that the checkpoints are easy to load and use
        output_folder = data_config.CHECKPOINT_DIR / "Segmentation-PyTorch"
        for filepath in list(Path(".").glob("**/*.py")):
            destination_path = output_folder / filepath
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            copy(filepath, destination_path)
        misc_files = ["README.md", "requirements.txt", "setup.cfg", ".gitignore"]
        for misc_file in misc_files:
            copy(misc_file, output_folder / misc_file)
        logger.info("Finished copying files")

    torch.backends.cudnn.benchmark = True   # Makes training quite a bit faster

    train_data, train_labels = default_loader(data_config.DATA_PATH / "Train",
                                              get_mask_path_fn=get_mask_path,
                                              limit=args.limit,
                                              load_data=args.load_data,
                                              data_preprocessing_fn=default_load_data if args.load_data else None,
                                              labels_preprocessing_fn=default_load_labels if args.load_data else None)
    clean_print("Train data loaded")

    val_data, val_labels = default_loader(data_config.DATA_PATH / "Validation",
                                          get_mask_path_fn=get_mask_path,
                                          limit=args.limit,
                                          load_data=args.load_data,
                                          data_preprocessing_fn=default_load_data if args.load_data else None,
                                          labels_preprocessing_fn=default_load_labels if args.load_data else None)
    clean_print("Validation data loaded")

    # Data augmentation done on cpu.
    augmentation_pipeline = albumentation_wrapper(albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.RandomRotate90(p=0.2),
        # albumentations.CLAHE(),
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        albumentations.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
        albumentations.ShiftScaleRotate(scale_limit=0.05, rotate_limit=10, shift_limit=0.06, p=0.5,
                                        border_mode=cv2.BORDER_CONSTANT, value=0),
        # albumentations.GridDistortion(p=0.5),
    ]))

    common_pipeline = albumentation_wrapper(albumentations.Compose([
        albumentations.Normalize(mean=(0.041, 0.129, 0.03), std=(0.054, 0.104, 0.046), max_pixel_value=255.0, p=1.0),
        albumentations.Resize(*model_config.IMAGE_SIZES, interpolation=cv2.INTER_LINEAR)
    ]))
    train_pipeline = transforms.compose_transformations((augmentation_pipeline, common_pipeline))

    with BatchGenerator(train_data,
                        train_labels,
                        model_config.BATCH_SIZE,
                        nb_workers=data_config.NB_WORKERS,
                        data_preprocessing_fn=default_load_data if not args.load_data else None,
                        labels_preprocessing_fn=default_load_labels if not args.load_data else None,
                        cpu_pipeline=train_pipeline,
                        gpu_pipeline=transforms.to_tensor(),
                        shuffle=True) as train_dataloader, \
        BatchGenerator(val_data,
                       val_labels,
                       model_config.BATCH_SIZE,
                       nb_workers=data_config.NB_WORKERS,
                       data_preprocessing_fn=default_load_data if not args.load_data else None,
                       labels_preprocessing_fn=default_load_labels if not args.load_data else None,
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

        train(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
