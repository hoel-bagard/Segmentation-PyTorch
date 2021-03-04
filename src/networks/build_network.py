from typing import (
    Optional,
)

import torch

from .unet import UDarkNet


class ModelHelper:
    UDarkNet = UDarkNet


def build_model(model_type: type, output_classes: bool, model_path: Optional[str] = None,
                eval_mode: bool = False, **kwargs):
    """
    Creates model corresponding to the given name.
    Args:
        name: Name of the model to create, must be one of the implemented models
        output_classes: Number of classes in the dataset
        model_path: If given, then the weights will be load that checkpoint
        eval: Whether the model will be used for evaluation or not
    Returns:
        model: PyTorch model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    kwargs["output_classes"] = output_classes
    model = model_type(**kwargs)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    if eval:
        model.eval()

    model.to(device)
    return model
