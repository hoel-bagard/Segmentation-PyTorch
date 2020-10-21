import torch

from .unet import UDarkNet


def build_model(name: str, model_path: str = None, eval: bool = False):
    """
    Creates model corresponding to the given name.
    Args:
        name: Name of the model to create, must be one of the implemented models
        model_path: If given, then the weights will be load that checkpoint
        eval: Whether the model will be used for evaluation or not
    Returns:
        model: PyTorch model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert name in ("UDarkNet")
    if name == "UDarkNet":
        model = UDarkNet()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    if eval:
        model.eval()

    model = model.float()
    model.to(device)
    return model
