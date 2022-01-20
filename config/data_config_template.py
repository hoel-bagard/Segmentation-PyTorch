import os
from dataclasses import dataclass
from json import load
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class DataConfig:
    # Recording part
    DATA_PATH          = Path("path", "to", "dataset")  # Path to the dataset folder
    USE_CHECKPOINT     = True               # Whether to save checkpoints or not
    CHECKPOINT_DIR     = Path("path", "to", "checkpoint_dir", "AI_Name")  # Path to checkpoint dir
    CHECKPT_SAVE_FREQ  = 10                  # How often to save checkpoints (if they are better than the previous one)
    KEEP_CHECKPOINTS   = False               # Whether to remove the checkpoint dir
    USE_TB             = True                # Whether generate a TensorBoard or not
    TB_DIR             = Path("path", "to", "log_dir", "AI_Name")  # TensorBoard dir
    KEEP_TB            = False               # Whether to remove the TensorBoard dir
    VAL_FREQ           = 10                  # How often to compute accuracy and images (also used for validation freq)
    RECORD_START       = 0                  # Checkpoints and TensorBoard are not recorded before this epoch

    # Dataloading
    NB_WORKERS = int(os.cpu_count() * 0.8)  # Number of workers to use for dataloading

    # Build a map between id and names
    LABEL_MAP = {}   # Maps an int to a class name
    COLOR_MAP = []   # Maps an int to a color (corresponding to a class)
    with open(DATA_PATH / "classes.json") as json_file:
        data = load(json_file)
        for key, entry in enumerate(data):
            LABEL_MAP[key] = entry["name"]
            COLOR_MAP.append(entry["color"])
    COLOR_MAP = np.asarray(COLOR_MAP)
    OUTPUT_CLASSES = len(LABEL_MAP)


def get_data_config() -> DataConfig:
    return DataConfig()
