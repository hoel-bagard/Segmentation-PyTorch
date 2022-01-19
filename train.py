# from torch.utils.tensorboard import SummaryWriter  # noqa: F401  # Needs to be there to avoid segfaults
import argparse
import time
from pathlib import Path
from shutil import copy, rmtree

import torch
from torchsummary import summary

import src.dataset.data_transformations as transforms
from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.dataset.dataset_specific_fn import default_get_mask_path as get_mask_path
from src.dataset.default_loader import (
    default_load_data,
    default_load_labels,
    default_loader
)
from src.networks.build_network import build_model
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.misc import clean_print
from src.torch_utils.utils.misc import get_config_as_dict
from src.train_loop import train


def main():
    parser = argparse.ArgumentParser("Segmentation project from PascalVOC labels")
    parser.add_argument("--limit", default=None, type=int, help="Limits the number of apparition of each class")
    parser.add_argument("--load_data", action="store_true", help="Loads all the videos into RAM")
    parser.add_argument("--name", type=str, help="Not used in the code. Use it to know what a train is when using ps.")
    args = parser.parse_args()

    if not DataConfig.KEEP_TB:
        while DataConfig.TB_DIR.exists():
            rmtree(DataConfig.TB_DIR, ignore_errors=True)
            time.sleep(0.5)
    DataConfig.TB_DIR.mkdir(parents=True, exist_ok=True)

    if DataConfig.USE_CHECKPOINT:
        if not DataConfig.KEEP_CHECKPOINTS:
            while DataConfig.CHECKPOINT_DIR.exists():
                rmtree(DataConfig.CHECKPOINT_DIR, ignore_errors=True)
                time.sleep(0.5)
        try:
            DataConfig.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            print(f"The checkpoint dir {DataConfig.CHECKPOINT_DIR} already exists")
            return -1

        # Makes a copy of all the code (and config) so that the checkpoints are easy to load and use
        output_folder = DataConfig.CHECKPOINT_DIR / "Segmentation-PyTorch"
        for filepath in list(Path(".").glob("**/*.py")):
            destination_path = output_folder / filepath
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            copy(filepath, destination_path)
        misc_files = ["README.md", "requirements.txt", "setup.cfg", ".gitignore"]
        for misc_file in misc_files:
            copy(misc_file, output_folder / misc_file)
        print("Finished copying files")

    torch.backends.cudnn.benchmark = True   # Makes training quite a bit faster

    # Data augmentation done on cpu.
    augmentation_pipeline = transforms.compose_transformations((
        # transforms.vertical_flip,
        transforms.horizontal_flip,
        # transforms.rotate180,
    ))
    # GPU pipeline used by both validation and train
    base_gpu_pipeline = (
        transforms.to_tensor(),
        transforms.normalize(labels_too=True),
    )
    train_gpu_augmentation_pipeline = transforms.compose_transformations((
        *base_gpu_pipeline,
        transforms.noise()
    ))

    train_data, train_labels = default_loader(DataConfig.DATA_PATH / "Train", get_mask_path_fn=get_mask_path,
                                              limit=args.limit, load_data=args.load_data,
                                              data_preprocessing_fn=default_load_data if args.load_data else None,
                                              labels_preprocessing_fn=default_load_labels if args.load_data else None)
    clean_print("Train data loaded")

    val_data, val_labels = default_loader(DataConfig.DATA_PATH / "Validation", get_mask_path_fn=get_mask_path,
                                          limit=args.limit, load_data=args.load_data,
                                          data_preprocessing_fn=default_load_data if args.load_data else None,
                                          labels_preprocessing_fn=default_load_labels if args.load_data else None)
    clean_print("Validation data loaded")

    with BatchGenerator(train_data, train_labels,
                        ModelConfig.BATCH_SIZE, nb_workers=DataConfig.NB_WORKERS,
                        data_preprocessing_fn=default_load_data if not args.load_data else None,
                        labels_preprocessing_fn=default_load_labels if not args.load_data else None,
                        aug_pipeline=augmentation_pipeline,
                        gpu_augmentation_pipeline=train_gpu_augmentation_pipeline,
                        shuffle=True) as train_dataloader, \
        BatchGenerator(val_data, val_labels, ModelConfig.BATCH_SIZE, nb_workers=DataConfig.NB_WORKERS,
                       data_preprocessing_fn=default_load_data if not args.load_data else None,
                       labels_preprocessing_fn=default_load_labels if not args.load_data else None,
                       gpu_augmentation_pipeline=transforms.compose_transformations(base_gpu_pipeline),
                       shuffle=False) as val_dataloader:

        print(f"\nLoaded {len(train_dataloader)} train data and",
              f"{len(val_dataloader)} validation data", flush=True)

        print("Building model. . .", end="\r")
        model = build_model(ModelConfig.MODEL, DataConfig.OUTPUT_CLASSES, **get_config_as_dict(ModelConfig))
        summary(model, train_dataloader.data_shape)

        train(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
