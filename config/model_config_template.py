class ModelConfig:
    # Training parameters
    BATCH_SIZE         = 16            # Batch size
    MAX_EPOCHS         = 2000          # Number of Epochs
    LR                 = 1e-3          # Learning Rate
    LR_DECAY           = 0.998
    REG_FACTOR         = 0.005         # Regularization factor

    # Network part
    NETWORK = "WideNet"  # Name of the network to use
    CHANNELS = [3, 8, 16, 32, 32, 16]
    SIZES = [5, 3, 3, 3, 3, 3]   # Kernel sizes
    STRIDES = [5, 3, 3, 2, 2, 2]
    PADDINGS = [2, 1, 1, 1, 1, 1]
    NB_BLOCKS = [1, 2, 2, 1, 1, 1]
    IMAGE_SIZES = (2448, 2048)  # All images will be resized to this size
    OUTPUT_CLASSES = 2   # Do not change this without changing the loss
