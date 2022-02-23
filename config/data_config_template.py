import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


class SegClass(NamedTuple):
    """Tuple with a class' name and its associated RGB color."""
    name: str
    color: tuple[int, int, int]


# Notes:
# - 2 classes can be mapped to the same color.
#   In that case, the first to appear will be used when going from RGB to class name.
#   Can be usefull with things like nice_cat, BadCat --> cat
classes = frozenset([SegClass("cat", (0, 255, 0)),
                     SegClass("nice_cat", (0, 255, 0)),
                     SegClass("BadCat", (0, 255, 0)),
                     SegClass("Dog", (34, 113, 179)),
                     SegClass("Human", (144, 70, 132))
                     ])


@dataclass(frozen=True, slots=True)
class DataConfig:
    # Recording part
    DATA_PATH          = Path("path", "to", "dataset")  # Path to the dataset folder

    USE_CHECKPOINTS    = True               # Whether to save checkpoints or not
    CHECKPOINTS_DIR    = Path("path", "to", "checkpoint_dir", "AI_Name")  # Path to checkpoint dir
    CHECKPT_SAVE_FREQ  = 10                  # How often to save checkpoints (if they are better than the previous one)

    USE_TB             = True                # Whether generate a TensorBoard or not
    TB_DIR             = Path("path", "to", "log_dir", "AI_Name")  # TensorBoard dir
    VAL_FREQ           = 10                  # How often to compute accuracy and images (also used for validation freq)
    RECORD_START       = 0                  # Checkpoints and TensorBoard are not recorded before this epoch

    # Dataloading
    NB_WORKERS = int(os.cpu_count() * 0.8)  # Number of workers to use for dataloading

    # Build maps between ids, names and colors
    IDX_TO_COLOR = np.asarray([cls.color for cls in classes])  # Maps an int to a color (corresponding to a class)
    NAME_TO_COLOR = {cls.name: cls.color for i, cls in enumerate(classes)}   # Maps a class name to its color
    LABEL_MAP = {i: cls.name for i, cls in enumerate(classes)}   # Maps an int to a class name
    OUTPUT_CLASSES = len(LABEL_MAP)
    MAX_DANGER_LEVEL: int = 5  # Specific to this project.


def get_data_config() -> DataConfig:
    return DataConfig()
