from dataclasses import dataclass, field

from src.networks.build_network import ModelHelper


@dataclass(frozen=True, slots=True)
class ModelConfig:
    # Training parameters
    BATCH_SIZE = 32
    MAX_EPOCHS = 500
    LR = 1e-3
    LR_DECAY = 0.998
    WEIGHT_DECAY = 1e-2   # Weight decay for the optimizer

    # Data processing
    IMAGE_SIZES: tuple[int, int] = field(default_factory=lambda: (512, 512))  # All images will be resized to this size

    # Network part
    MODEL = ModelHelper.UDarkNet

    CHANNELS: list[int] = field(default_factory=lambda: [3, 32, 64, 64, 128, 256])
    SIZES: list[int | tuple[int, int]]  = field(default_factory=lambda: [5, 3, 3, 3, 3, 3])   # Kernel sizes
    STRIDES: list[int | tuple[int, int]]  = field(default_factory=lambda: [5, 2, 2, 2, 2, 2])
    PADDINGS: list[int | tuple[int, int]]  = field(default_factory=lambda: [2, 1, 1, 1, 1, 1])
    BLOCKS: list[int] = field(default_factory=lambda: [1, 2, 2, 1, 1, 1])


def get_model_config() -> ModelConfig:
    return ModelConfig()
