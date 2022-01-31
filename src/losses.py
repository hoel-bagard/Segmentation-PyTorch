from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy


class DiceBCELoss(nn.Module):
    """This loss combines Dice loss with the standard binary cross-entropy (BCE) loss.

    This code is taken from:
        This kaggle post: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
        This PyTorch PR thread: https://github.com/pytorch/pytorch/issues/1249

    Combining the two methods allows for some diversity in the loss, while benefitting from the stability of BCE.
    See usage example here: https://www.mdpi.com/2079-9292/11/1/130/htm

    Args:
        smooth (float): Avoids division by zero, similar to an epsilon.
                        Moreover, having a larger smooth value (also known as Laplace smooth, or Additive smooth)
                        can be used to avoid overfitting.
                        The larger the smooth value the closer the following term is to 1 (if everything else is fixed),
                        ((2. * intersection + smooth) /  (iflat.sum() + tflat.sum() + smooth))
                        This decreases the penalty obtained from having 2*intersection different from the sums.
    """
    def __init__(self, smooth: float = 1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1).to(torch.float32)
        targets = targets.contiguous().view(-1).to(torch.float32)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)
        bce = binary_cross_entropy(inputs, targets, reduction="mean")
        dice_bce = bce + dice_loss

        return dice_bce


class CE_Loss(nn.Module):  # noqa: N801
    """PyTorch's cross entropy loss."""
    def __init__(self, weight: Optional[list[int]] = None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.weight = torch.Tensor(weight) if weight else None
        self.loss_fn = nn.CrossEntropyLoss(self.weight)

    def forward(self, input_data: torch.Tensor, input_labels: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(input_data, input_labels.float())
        return loss


class MSE_Loss(nn.Module):  # noqa: N801
    """MSE loss from the past."""
    def __init__(self, negative_loss_factor: int = 50):
        super().__init__()
        self.negative_loss_factor = negative_loss_factor
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, input_data: torch.Tensor, input_labels: torch.Tensor) -> torch.Tensor:
        y_pred = torch.flatten(input_labels, start_dim=1)
        y_true = torch.flatten(input_data, start_dim=1)

        neg_loss = y_true * (-torch.log(y_pred + 1e-8))
        pos_loss = (1 - y_true) * (-torch.log(1 - y_pred + 1e-8))

        # I prefer to avoid using reduce_mean directly because of all the zeros
        neg_loss = torch.sum(neg_loss) / torch.max(torch.tensor(1.0, device=self.device), torch.sum(y_true))
        pos_loss = torch.sum(pos_loss) / torch.max(torch.tensor(1.0, device=self.device), torch.sum(1 - y_true))

        loss = pos_loss + self.negative_loss_factor*neg_loss

        return loss
