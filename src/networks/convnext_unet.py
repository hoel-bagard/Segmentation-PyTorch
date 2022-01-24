from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn as nn


from src.layers.trunc import trunc_normal_
from src.networks.convnext import ConvNeXt
from src.networks.layers import ConvTranspose, SkipConnection


class UConvNeXt(nn.Module):
    def __init__(self,
                 output_classes: int,
                 channels: Optional[list[int]] = None,
                 layer_init: Callable[[nn.Module], None] = None, **kwargs):
        """'Unet' using a ConvNeXt.

        Args:
            output_classes: Number of output classes
            channels: List with the number of channels for each convolution
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()
        # TODO: torch.nn.functional.gelu
        self.channels = channels if channels else [96, 192, 384, 768]
        assert len(channels) == 4, "For this network, the channels must be a list of 4 ints"

        self.encoder = ConvNeXt(3, depths=[3, 3, 9, 3], dims=channels)

        self.skip_connections = nn.ModuleList([SkipConnection(2*channels[i], channels[i])
                                               for i in range(1, len(channels))])
        self.conv_trans = nn.ModuleList([ConvTranspose(channels[i], channels[i-1], 3)
                                         for i in range(1, len(channels))])
        self.last_conv = nn.Conv2d(in_channels=channels[0], out_channels=output_classes, kernel_size=1, stride=1)

        if layer_init:
            self.apply(layer_init)
        else:
            def _init_weights(m: nn.Module):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    trunc_normal_(m.weight, std=.02)
                    nn.init.constant_(m.bias, 0)
            self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_outputs = self.encoder(x)

        for i in range(len(self.channels)-2, -1, -1):
            x = self.skip_connections[i](conv_outputs[i], x)
            x = self.conv_trans[i](x, output_size=((2*x.shape[-2], 2*x.shape[-1])))

        x = self.last_conv(x)
        x = torch.sigmoid(x)
        return x
