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


class DangerPLoss(nn.Module):
    def __init__(self, max_danger_lvl: int):
        super().__init__()
        self.max_danger_lvl = max_danger_lvl
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cls_preds = preds[0]
        danger_preds = preds[1]

        oh_cls_labels = labels[..., 0]
        oh_danger_labels = labels[..., 1]
        oh_danger_labels = oh_danger_labels[..., :self.max_danger_lvl]  # Remove padding

        cls_loss = self.loss_fn(cls_preds, oh_cls_labels)
        danger_loss = self.loss_fn(danger_preds, oh_danger_labels)

        return cls_loss + danger_loss
