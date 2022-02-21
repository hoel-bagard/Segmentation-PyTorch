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
    IMAGE_SIZES: tuple[int, int] = field(default_factory=lambda: (1024, 640))  # All images will be resized to this size
    OUTPUT_SIZES: tuple[int, int] = field(default_factory=lambda: (16, 10))  # Output size of the network
    # The mean and std used to normalize the dataset.
    MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Network part
    MODEL = ModelHelper.UDarkNet

    # The values bellow are used for the UNet (CHANNELS can be used for the ConvNeXt too if len == 4)
    CHANNELS: list[int] = field(default_factory=lambda: [3, 32, 64, 64, 128, 256])
    SIZES: list[int | tuple[int, int]]  = field(default_factory=lambda: [5, 3, 3, 3, 3, 3])   # Kernel sizes
    STRIDES: list[int | tuple[int, int]]  = field(default_factory=lambda: [5, 2, 2, 2, 2, 2])
    PADDINGS: list[int | tuple[int, int]]  = field(default_factory=lambda: [2, 1, 1, 1, 1, 1])
    BLOCKS: list[int] = field(default_factory=lambda: [1, 2, 2, 1, 1, 1])


def get_model_config() -> ModelConfig:
    return ModelConfig()
