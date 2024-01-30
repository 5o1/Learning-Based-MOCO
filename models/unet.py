import torch.nn as nn
import torch

from .nn import AdaptedPad,AdaptedCrop, Up, Down, Boot, Output, CheckNan, CheckInf
from .nn.functional import dict_filter


class Unet(nn.Module):

    def __init__(self, in_channels : int = 4 , out_channels : int = 4 , depth : int = 4, top_channels : int = 64, dtype = torch.float, crop_res : bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.top_channels = top_channels
        self.dtype = dtype
        self.crop_res = crop_res
        self.padding = 0 if crop_res else 'same'

        if self.crop_res:
            self.pad = AdaptedPad(depth)
            self.crop = AdaptedCrop()
        self.checknan = CheckNan()
        self.checkinf = CheckInf()

        self._in = Boot(in_channels, top_channels, dtype = dtype,padding = self.padding) # in
        self.down_convs = nn.ModuleList() # Down convs
        channels = top_channels
        for _ in range(depth):
            self.down_convs.append(Down(channels, channels * 2, dtype = dtype, padding = self.padding))
            channels = channels * 2
        self.up_convs = nn.ModuleList() # Up convs
        for _ in range(depth):
            self.up_convs.append(Up(channels, channels // 2, dtype = dtype, padding = self.padding))
            channels = channels // 2
        self._out = Output(channels, out_channels, dtype = dtype, padding = self.padding)

    def forward(self, x: torch.Tensor):
        # double check
        x = self.checknan(x, 'in')
        x = self.checkinf(x, 'in')
        if self.crop_res:
            x = self.pad(x)
        x = self._in(x)
        if self.crop_res:
            cropped_x = self.crop(x, self.depth)
            res_x = [cropped_x]
        else:
            res_x = [x]
        # down path
        for i, f in enumerate(self.down_convs):
            x = f(x)
            if self.crop_res:
                cropped_x = self.crop(x,self.depth - 1 - i)
                res_x.append(cropped_x)
            else:
                res_x.append(x)
        # up path
        for i, f in enumerate(self.up_convs):
            x = f(x, res_x[-2 - i])
        if self.padding == 0:
            x = self.pad.crop(x)
        # out
        x = self._out(x)
        # double check
        x = self.checknan(x, 'out')
        x = self.checkinf(x, 'out')
        return x
