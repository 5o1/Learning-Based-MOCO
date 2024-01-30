"""One U-net implement for image to image."""

import torch.nn as nn
import torch

from .nn import AdaptedPad,AdaptedCrop, nConv2d, Up, Down, Boot, Output
from .nn.functional import dict_filter


class Unet(nn.Module):
    """One U-net which is for image to image."""

    def __init__(self, in_channels=4, out_channels=4, depth=3, top_channels = 8, dtype = torch.float, padding = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.top_channels = top_channels
        self.dtype = dtype
        self.padding = padding


        self.pad = AdaptedPad(depth)
        self.crop = AdaptedCrop()

        # in
        self._in = Boot(in_channels, top_channels, dtype = dtype)

        # Down convs
        self.down_convs = nn.ModuleList()
        channels = top_channels

        for _ in range(depth):
            self.down_convs.append(
                Down(channels, channels * 2, dtype = dtype)
            )
            channels = channels * 2

        # Up convs
        self.up_convs = nn.ModuleList()

        for _ in range(depth):
            self.up_convs.append(
                Up(channels, channels // 2, dtype = dtype)
            )
            channels = channels // 2

        self._out = Output(channels, out_channels, dtype = dtype)

    def forward(self, x: torch.Tensor):
        # mirror padding
        x = self.pad(x)

        x = self._in(x)
        cropped_x = self.crop(x, self.depth)
        res_x = [cropped_x]

        # down path
        for i, f in enumerate(self.down_convs):
            x = f(x)
            cropped_x = self.crop(x,self.depth - 1 - i)
            res_x.append(cropped_x)

        # up path
        for i, f in enumerate(self.up_convs):
            x = f(x, res_x[-2 - i])

        x = self.pad.crop(x)

        # out
        x = self._out(x)
        return x