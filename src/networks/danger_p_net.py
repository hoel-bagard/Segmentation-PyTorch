from typing import Optional

import torch
from torch import nn

from src.layers.trunc import trunc_normal_
from src.networks.convnext import ConvNeXt


class Conv2D(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(out_filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x


class DangerPNet(nn.Module):
    def __init__(self, output_classes: int, max_danger_level: int, **kwargs):
        """Instanciate the network.

        Args:
            output_classes: The number of classes.
            max_danger_level: The number of danger levels.
        """
        super().__init__()
        self.n_classes = output_classes
        self.n_danger_levels = max_danger_level

        self.backbone = nn.Sequential(
            Conv2D(2, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            Conv2D(32, 64, 3, 2, 2),
            nn.MaxPool2d(2, 2),
            Conv2D(64, 64, 1, 1, 0),
            Conv2D(64, 128, 3, 2, 2),
            nn.MaxPool2d(2, 2),
            Conv2D(128, 128, 1, 1, 0),
            Conv2D(128, 64, 3, 1, 1))

        self.classification_head = nn.Sequential(
            Conv2D(64, 128, 3, 1, 1),  # So bottle neck in the latent ?!
            nn.MaxPool2d(2, 2),
            Conv2D(128, 128, 1, 1, 0),
            Conv2D(128, 64, 3, 1, 1),
            Conv2D(64, 32, 3, 1, 1),
            Conv2D(32, 32, 1, 1, 0),
            nn.Conv2d(32, self.n_classes, 3, 1, 1)
        )
        self.danger_head = nn.Sequential(
            Conv2D(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            Conv2D(128, 128, 1, 1, 0),
            Conv2D(128, 64, 3, 1, 1),
            Conv2D(64, 32, 3, 1, 1),
            Conv2D(32, 32, 1, 1, 0),
            nn.Conv2d(32, self.n_danger_levels, 3, 1, 1)
        )

        def _init_weights(m: nn.Module):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        return self.classification_head(x), self.danger_head(x)


class DangerPConvNeXt(nn.Module):
    def __init__(self, output_classes: int, max_danger_level: int, channels: Optional[list[int]] = None, **kwargs):
        """Network meant for debugging purposes only."""
        super().__init__()
        self.n_classes = output_classes
        self.n_danger_levels = max_danger_level

        channels = channels if channels else [96, 192, 384, 768]
        assert len(channels) == 4, "For this network, the channels must be a list of 4 ints"

        self.backbone = ConvNeXt(2, depths=[3, 3, 9, 3], dims=channels)

        self.classification_head = nn.Sequential(
            Conv2D(channels[-1], 64, 3, 2, 1),
            nn.Conv2d(64, self.n_classes, 3, 1, 1)
        )
        self.danger_head = nn.Sequential(
            Conv2D(channels[-1], 64, 3, 2, 1),
            nn.Conv2d(64, self.n_danger_levels, 3, 1, 1)
        )

        def _init_weights(m: nn.Module):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor):
        conv_outputs = self.backbone(x)

        x = conv_outputs[-1]

        return self.classification_head(x), self.danger_head(x)
