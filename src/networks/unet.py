from collections import OrderedDict
from typing import (
    Callable,
    Union
)

import torch
import torch.nn as nn

from src.networks.layers import (
    SkipConnection,
    DarknetBlock,
    ConvTranspose
)
from src.torch_utils.networks.network_utils import layer_init


class UDarkNet(nn.Module):
    def __init__(self, channels: list[int],
                 sizes: list[Union[int, tuple[int, int, int]]],
                 strides: list[Union[int, tuple[int, int, int]]],
                 paddings: list[Union[int, tuple[int, int, int]]],
                 blocks: list[int],
                 output_classes: int,
                 layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        Feature extractor
        Args:
            channels: List with the number of channels for each convolution
            sizes: List with the kernel size for each convolution
            strides: List with the stride for each convolution
            paddings: List with the padding for each convolution
            blocks: List with the number of blocks for the darknet blocks
            output_class: Number of output classes
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
        self.last_conv = nn.Conv2d(in_channels=channels[0], out_channels=output_classes, kernel_size=1, stride=1)

        if layer_init:
            self.apply(layer_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def get_weight_and_grads(self) -> OrderedDict:
        weight_grads = OrderedDict()

        for index, block in enumerate(self.conv):
            weight_grads[f"block_{index}_conv"] = block.dark_conv.conv.weight, block.dark_conv.conv.weight.grad
            for j, dark_res_block in enumerate(block.dark_res_blocks):
                weight_grads[f"block_{index}_subblock_{j}_conv1"] = (dark_res_block.darknet_conv1.conv.weight,
                                                                     dark_res_block.darknet_conv1.conv.weight.grad)
                weight_grads[f"block_{index}_subblock_{j}_conv2"] = (dark_res_block.darknet_conv2.conv.weight,
                                                                     dark_res_block.darknet_conv2.conv.weight.grad)

        for index in range(len(self.channels)-2, -1, -1):
            weight_grads[f"deconv_{index}"] = (self.conv_trans[index].conv_transpose.weight,
                                               self.conv_trans[index].conv_transpose.weight.grad)
            weight_grads[f"skip_{index}"] = (self.skip_connections[index].conv.weight,
                                             self.skip_connections[index].conv.weight.grad)

        weight_grads["last_conv"] = self.last_conv.weight, self.last_conv.weight.grad
        return weight_grads
