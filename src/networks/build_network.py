from pathlib import Path
from typing import Optional

import torch

from .danger_p_net import DangerPConvNeXt, DangerPNet
from .resnet_18 import DangerResNet
from .resnet_50 import DangerResNet50


class ModelHelper:
    DangerPNet = DangerPNet
    DangerPConvNeXt = DangerPConvNeXt
    DangerResNet18 = DangerResNet
    DangerResNet50 = DangerResNet50


def build_model(model_type: type,
                model_path: Optional[Path] = None,
                eval_mode: bool = False,
                **kwargs):
    """Function that instanciates the given model.

    Args:
        model_type (type): Class of the model to instanciates
        model_path (Path): If given, then the weights will be loaded from that checkpoint
        eval (bool): Whether the model will be used for evaluation or not

    Returns:
        torch.nn.Module: Instantiated PyTorch model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model_type(**kwargs)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    if eval:
        model.eval()

    model.to(device)
    return model
