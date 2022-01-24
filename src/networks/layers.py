from typing import Callable, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class DarknetConv(torch.nn.Module):
    def __init__(self, in_filters, out_filters, size, stride=1):
        super().__init__()

        # This is a quick fix, need to work on padding in PyTorch
        padding = (size - 1) // 2

        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=size,
                              stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x, 0.1)

        return x


class DarknetResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.darknet_conv1 = DarknetConv(filters, filters//2, 1)
        self.darknet_conv2 = DarknetConv(filters//2, filters, 3)

    def forward(self, inputs):
        x = self.darknet_conv1(inputs)
        x = self.darknet_conv2(x)
        x = torch.add(inputs, x)
        return x


class DarknetBlock(nn.Module):
    def __init__(self, in_filters, out_filters, blocks):
        super().__init__()
        self.dark_conv = DarknetConv(in_filters, out_filters, 3, stride=2)
        self.dark_res_blocks = nn.Sequential(*[DarknetResidualBlock(out_filters) for _ in range(blocks)])

    def forward(self, inputs):
        x = self.dark_conv(inputs)
        for dark_res_block in self.dark_res_blocks:
            x = dark_res_block(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, in_filters: int,
                 out_filters: int,
                 use_bn: bool = True,
                 activation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.activation_fn = activation_fn if activation_fn else partial(F.leaky_relu, negative_slope=0.1)
        self.batch_norm = nn.BatchNorm2d(out_filters) if use_bn else None

    def forward(self, input_op, skip_op):

        # y_padding = skip_op.size()[-2] - input_op.size()[-2]
        # x_padding = skip_op.size()[-1] - input_op.size()[-1]

        # input_op = F.pad(input_op, [x_padding // 2, x_padding - x_padding // 2,
        #                             y_padding // 2, y_padding - y_padding // 2])

        x = torch.cat((input_op, skip_op), dim=-3)
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation_fn(x)
        return x


class ConvTranspose(torch.nn.Module):
    def __init__(self, in_filters: int, out_filters: int,
                 size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 2,
                 padding: int | tuple[int, int] = 1,
                 use_bn: bool = True,
                 activation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_filters, out_filters, kernel_size=size,
                                                 stride=stride, padding=padding, bias=False)
        self.activation_fn = activation_fn if activation_fn else partial(F.leaky_relu, negative_slope=0.1)
        self.batch_norm = nn.BatchNorm2d(out_filters) if use_bn else None

    def forward(self, x, output_size=None):
        x = self.conv_transpose(x, output_size=output_size)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation_fn(x)
        return x
