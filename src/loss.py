import torch
import torch.nn as nn


class CE_Loss(nn.Module):
    def __init__(self, negative_loss_factor: int = 50):
        super(CE_Loss, self).__init__()
        self.negative_loss_factor = negative_loss_factor

    def forward(self, input_data: torch.Tensor, input_labels: torch.Tensor) -> torch.Tensor:
        y_true = torch.flatten(input_data, start_dim=1)
        y_pred = torch.flatten(input_labels, start_dim=1)

        neg_obj_loss = y_true * (-torch.log(y_pred + 1e-8))
        pos_obj_loss = (1 - y_true) * (-torch.log(1 - y_pred + 1e-8))

        # I prefer to avoid using reduce_mean directly because of all the zeros
        neg_obj_loss = torch.sum(neg_obj_loss) / torch.max(1.0, torch.sum(y_true))
        pos_obj_loss = torch.sum(pos_obj_loss) / torch.max(1.0, torch.sum(1 - y_true))
        obj_loss = pos_obj_loss + self.negative_loss_factor*neg_obj_loss

        return obj_loss
