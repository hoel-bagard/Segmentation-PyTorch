from dataclasses import dataclass, field

from src.networks.build_network import ModelHelper


@dataclass(frozen=True, slots=True)
class ModelConfig:
    # Training parameters
    BATCH_SIZE = 32
    MAX_EPOCHS = 500
    START_LR = 1e-3
    END_LR = 5e-6
    WEIGHT_DECAY = 1e-2   # Weight decay for the optimizer

    # Data processing
    # IMAGE_SIZES: tuple[int, int] = field(default_factory=lambda: (640, 1024))  # Images will be resized to this size
    IMAGE_SIZES: tuple[int, int] = field(default_factory=lambda: (320, 512))
    OUTPUT_SIZES: tuple[int, int] = field(default_factory=lambda: (10, 16))  # Output size of the network
    # The mean and std used to normalize the dataset.
    MEAN: tuple[float, float] = field(default_factory=lambda: (0.494, 0.037))
    STD: tuple[float, float] = field(default_factory=lambda: (0.266, 0.074))

    # Network part
    MODEL = ModelHelper.DangerPNet


def get_model_config() -> ModelConfig:
    return ModelConfig()
