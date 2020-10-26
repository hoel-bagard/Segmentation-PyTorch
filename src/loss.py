import torch
import torch.nn as nn


class CE_Loss(nn.Module):
    def __init__(self, negative_loss_factor: int = 50):
        super(CE_Loss, self).__init__()
        self.negative_loss_factor = negative_loss_factor
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # nn.BCELoss()  doesn't work (seems like it's full of bugs to me....)
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, input_data: torch.Tensor, input_labels: torch.Tensor) -> torch.Tensor:

        loss = self.mse_loss(input_data, input_labels.float())
        # loss = ((input_data-input_labels)**2).mean()

        # TODO: Try this later (once everything else works)
        # y_pred = torch.flatten(input_labels, start_dim=1)
        # y_true = torch.flatten(input_data, start_dim=1)

        # neg_loss = y_true * (-torch.log(y_pred + 1e-8))
        # pos_loss = (1 - y_true) * (-torch.log(1 - y_pred + 1e-8))

        # # I prefer to avoid using reduce_mean directly because of all the zeros
        # neg_loss = torch.sum(neg_loss) / torch.max(torch.tensor(1.0, device=self.device), torch.sum(y_true))
        # pos_loss = torch.sum(pos_loss) / torch.max(torch.tensor(1.0, device=self.device), torch.sum(1 - y_true))

        # loss = pos_loss + self.negative_loss_factor*neg_loss

        return loss
