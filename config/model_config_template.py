from typing import (
    Union,
    Tuple
)

from src.networks.build_network import (
    ModelHelper
)


class ModelConfig:
    # Training parameters
    BATCH_SIZE         = 16            # Batch size
    MAX_EPOCHS         = 2000          # Number of Epochs
    LR                 = 1e-3          # Learning Rate
    LR_DECAY           = 0.998
    REG_FACTOR         = 0.005         # Regularization factor

    # Data processing
    IMAGE_SIZES: Tuple[int, int] = (256, 256)  # All images will be resized to this size

    # Network part
    MODEL = ModelHelper.UDarkNet

    CHANNELS: list[Union[int, Tuple[int, int]]] = [3, 8, 16, 32, 32, 16]
    SIZES: list[Union[int, Tuple[int, int]]]  = [5, 3, 3, 3, 3, 3]   # Kernel sizes
    STRIDES: list[Union[int, Tuple[int, int]]]  = [5, 3, 3, 2, 2, 2]
    PADDINGS: list[Union[int, Tuple[int, int]]]  = [2, 1, 1, 1, 1, 1]
    BLOCKS: list[int] = [1, 2, 2, 1, 1, 1]
