from typing import (
    Optional,
    Callable,
    Union
)

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from src.networks.layers import (
    SkipConnection,
    DarknetBlock,
    ConvTranspose
)
from src.torch_utils.networks.network_utils import layer_init


class UDarkNet(nn.Module):
    def __init__(self, channels: list[Union[int, tuple[int, int, int]]],
                 sizes: list[Union[int, tuple[int, int, int]]],
                 strides: list[Union[int, tuple[int, int, int]]],
                 paddings: list[Union[int, tuple[int, int, int]]],
                 blocks: list[Union[int, tuple[int, int, int]]],
                 output_classes: int,
                 aug_pipeline: Optional[Callable[[np.ndarray, np.ndarray], tuple[Tensor, Tensor]]] = None,
                 layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        Feature extractor
        Args:
            channels: List with the number of channels for each convolution
            sizes: List with the kernel size for each convolution
            strides: List with the stride for each convolution
            padding: List with the padding for each convolution
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()
        self.output_size = 2
        self.channels = channels

        self.conv = nn.ModuleList([DarknetBlock(channels[i-1], channels[i], blocks[i-1])
                                   for i in range(1, len(channels))])
        self.bottle_neck_conv = nn.Conv2d(in_channels=channels[-1], out_channels=channels[-1], kernel_size=3, stride=2)
        self.skip_connections = nn.ModuleList([SkipConnection(2*channels[i], channels[i])
                                               for i in range(1, len(channels))])
        self.conv_trans = nn.ModuleList([ConvTranspose(channels[i], channels[i-1], 3)
                                         for i in range(1, len(channels))])
        self.last_conv = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=1, stride=1)

        if layer_init:
            self.apply(layer_init)

    def forward(self, x):
        conv_outputs = []
        for layer in self.conv:
            x = layer(x)
            conv_outputs.append(x)

        # x = self.bottle_neck_conv(x)

        for i in range(len(self.channels)-2, -1, -1):
            x = self.skip_connections[i](conv_outputs[i], x)
            x = self.conv_trans[i](x, output_size=((2*x.shape[-2], 2*x.shape[-1])))

        x = self.last_conv(x)
        x = torch.sigmoid(x)
        return x
