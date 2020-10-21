import math

import torch
import torch.nn as nn

from src.networks.layers import (
    SkipConnection,
    DarknetBlock,
    ConvTranspose
)
from config.model_config import ModelConfig


class UDarkNet(nn.Module):
    def __init__(self):
        super(UDarkNet, self).__init__()
        self.output_size = 2
        channels = ModelConfig.CHANNELS

        self.cons = [DarknetBlock(channels[i-1], channels[i], ModelConfig.NB_BLOCKS[i-1])
                     for i in range(1, len(channels))]
        self.skip_connections = [SkipConnection(2*channels[i], channels[i]) for i in range(len(channels)-1, 1, -1)]
        self.cons_trans = [ConvTranspose(channels[i], channels[i-1]) for i in range(len(channels)-1, 1, -1)]
        self.last_conv = nn.Conv2d(in_channels=channels[1], out_channels=1, kernel_size=1, stride=1)

        # Initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        conv_outputs = []
        for layer in self.cons:
            x = layer(x)
            conv_outputs.append(x)

        for i in range(len(conv_outputs)):
            x = self.skip_connections[i](conv_outputs[i], x)
            x = self.cons_trans[i](x)

        x = self.last_conv(x)
        x = torch.sigmoid(x)
        return x
