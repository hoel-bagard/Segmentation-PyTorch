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
        """Unet-like network using a ConvNeXt.

        Args:
            output_classes: Number of output classes
            channels: List with the number of channels for each convolution
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()
        # TODO: torch.nn.functional.gelu
        channels = channels if channels else [96, 192, 384, 768]
        assert len(channels) == 4, "For this network, the channels must be a list of 4 ints"

        self.encoder = ConvNeXt(3, depths=[3, 3, 9, 3], dims=channels)

        # Random useless not-a-botlle-neck botlle neck
        self.bottle_neck = nn.Conv2d(in_channels=channels[-1], out_channels=channels[-1], kernel_size=1, stride=1)

        # "Decoder" part
        self.skip_connections = nn.ModuleList([SkipConnection(2*channels[i], channels[i])
                                               for i in range(0, len(channels))])
        channels.insert(0, channels[0]//2)
        self.conv_trans = nn.ModuleList([ConvTranspose(channels[i+1], channels[i], 3)
                                         for i in range(0, len(channels)-1)])

        self.last_conv_trans = ConvTranspose(channels[0], output_classes, 3)
        # self.last_conv = nn.Conv2d(in_channels=channels[0], out_channels=output_classes, kernel_size=1, stride=1)

        if layer_init:
            self.apply(layer_init)
        else:
            def _init_weights(m: nn.Module):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
            self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_outputs = self.encoder(x)

        x = conv_outputs[-1]
        for i in range(len(conv_outputs)-1, -1, -1):
            x = self.skip_connections[i](conv_outputs[i], x)
            x = self.conv_trans[i](x, output_size=((2*x.shape[-2], 2*x.shape[-1])))

        x = self.last_conv_trans(x, output_size=((2*x.shape[-2], 2*x.shape[-1])))
        x = torch.sigmoid(x)
        return x
