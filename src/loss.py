from typing import Optional

import torch
import torch.nn as nn


class CE_Loss(nn.Module):
    """PyTorch's cross entropy loss"""
    def __init__(self, weight: Optional[list[int]] = None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.weight = torch.Tensor(weight) if weight else None
        self.loss_fn = nn.CrossEntropyLoss(self.weight)

    def forward(self, input_data: torch.Tensor, input_labels: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(input_data, input_labels.float())
        return loss


class MSE_Loss(nn.Module):
    """MSE loss from the past"""
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
