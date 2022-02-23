import torch
from torch import nn


class Conv2D(torch.nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(out_filters)
        self.activation = nn.Relu()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x


class DangerPNet(nn.Module):
    def __init__(self, n_classes: int, n_danger_levels: int):
        """Instanciate the network.

        Args:
            n_classes: The number of classes.
            n_danger_levels: The number of danger levels.
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_danger_levels = n_danger_levels

        self.backend = nn.Sequential(
            Conv2D(2, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            Conv2D(32, 64, 3, 2, 2),
            nn.MaxPool2d(2, 2),
            Conv2D(64, 64, 1, 1, 0),
            Conv2D(64, 128, 3, 2, 2),
            nn.MaxPool2d(2, 2),
            Conv2D(128, 128, 1, 1, 0),
            Conv2D(128, 64, 3, 1, 1))
        self.segmentation = nn.Sequential(
            Conv2D(64, 128, 3, 1, 1),  # So bottle neck in the latent ?!
            nn.MaxPool2d(2, 2),
            Conv2D(128, 128, 1, 1, 0),
            Conv2D(128, 64, 3, 1, 1),
            Conv2D(64, 32, 3, 1, 1),
            Conv2D(32, 32, 1, 1, 0),
            nn.Conv2d(32, self.n_classes, 3, 1, 1)
        )
        self.danger = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1, 0, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, self.n_danger_levels, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.backend(x)
        return self.segmentation(latent), self.danger(latent), latent
