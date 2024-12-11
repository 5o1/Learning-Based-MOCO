import bottleneck
from sympy import im, re
import torch.nn as nn
import torch
from torchvision import transforms as tf

from . import mynn as mynn

import copy


class nConv2d(nn.Module):
    """ repeated conv -> norm -> act ... -> conv, end with conv."""

    def __init__(self, channels: int, kernel_size : int, depth: int, dtype = torch.float32, padding = 0, norm_layer: nn.Module = None, act_layer : nn.Module = nn.ReLU(), bias = False) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.dtype = dtype
        self.padding = padding
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
        self.act_layer = act_layer if act_layer is not None else nn.Identity()

        self.layers = [nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dtype=dtype, bias=bias)]
        for _ in range(depth - 1):
            self.layers.append(copy.deepcopy(self.norm_layer)) if norm_layer is not None else None
            self.layers.append(copy.deepcopy(self.act_layer))
            self.layers.append(nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dtype=dtype, bias=bias))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return x


class Down(nn.Module):
    """conv -> downsample"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, depth: int = 2, dtype = torch.float32, padding = 0, norm_layer: nn.Module = None, act_layer : nn.Module = nn.ReLU(), downsample = None, bias = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype
        self.padding = padding
        self.norm_layer = norm_layer
        self.conv = nConv2d(channels=in_channels, kernel_size=kernel_size, depth=depth, dtype=dtype, padding=padding, norm_layer=norm_layer, act_layer=act_layer, bias=bias)
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, dtype=dtype, bias=bias) if downsample is None else downsample

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.down(x)
        return x


class CALayer(nn.Module):
    """Channel Attention Layer"""

    def __init__(self, in_channels: int, reduction: int = 16, act_layer: nn.Module = nn.ReLU(), bias = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.act_layer = act_layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=bias),
            copy.deepcopy(act_layer),
            nn.Linear(in_channels // reduction, in_channels, bias=bias),
        )

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.sigmoid(y)
        return x * y


class CABlock(nn.Module):
    """Channel Attention Block
    res: conv -> relu -> conv -> ca
    x: +res(x)"""

    def __init__(self, in_channels: int, reduction: int = 4, act_layer: nn.Module = nn.ReLU(), bias = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.act_layer = act_layer
        self.ca = CALayer(in_channels, reduction, act_layer, bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias),
            copy.deepcopy(act_layer),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, x: torch.Tensor):
        res = self.conv(x)
        res = self.ca(res)
        return x + res

class Up(nn.Module):
    """upsample -> concat -> conv -> reduce -> cablock"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, depth: int = 2, dtype = torch.float32, padding = 0, norm_layer: nn.Module = None, act_layer: nn.Module = nn.ReLU(), upsample = None, bias = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype
        self.padding = padding
        self.norm_layer = norm_layer
        self.conv = nConv2d(channels=out_channels * 2, kernel_size=kernel_size, depth=depth, dtype=dtype, padding=padding, norm_layer=norm_layer, act_layer=act_layer, bias=bias)
        self.reduce = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, dtype=dtype, bias=bias)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, dtype=dtype, bias=bias) if upsample is None else upsample
        self.ca = CABlock(out_channels, bias=bias)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.reduce(x)
        x = self.ca(x)
        return x



    

class AdaptedPad(nn.Module):
    """In order to modify Unet to be able to accept input from images of arbitrary size"""


    def __init__(self, depth : int, kernel_size : int = 2):
        """depth: Unet depth.
        n : nConv2d in one step."""
        super().__init__()
        self.depth = depth
        self.kernel_size = kernel_size

        self.prefix = {}

        for i in range(100):
            right = i
            left = i + 2 * kernel_size
            for _ in range(self.depth):
                left = left * 2  + 2 * kernel_size
                right = right * 2 - 2 * kernel_size
            if right <= 0:
                continue
            self.prefix[right] = left


    def __call__(self, image: torch.Tensor):
        shape = image.shape

        self.shape0 = [shape[-2], shape[-1]]
        shapen = torch.tensor(self.shape0)
        # backward_up
        for _ in range(self.depth):
            shapen = shapen + self.kernel_size * 2
            shapen = shapen // 2 + shapen % torch.tensor(2)

        # forward_up
        for _ in range(self.depth):
            shapen = shapen * 2
            shapen = shapen - self.kernel_size * 2
        shapen = torch.tensor([self.prefix[shapen[0].item()] , self.prefix[shapen[1].item()]])
        # # backward_down
        # for _ in range(self.d):
        #     shapen = shapen + self.n * 2
        #     shapen = shapen * 2

        # # backward_in
        # shapen = shapen + self.n * 2
        
        divshape = shapen - torch.tensor(self.shape0)
        # padding
        pad = tf.Pad([divshape[-1] // 2, divshape[-1] // 2 ,
                      divshape[-2] // 2, divshape[-2] // 2 ],
                      padding_mode='reflect')
        image = pad(image)
        return image
    

    def crop(self, image):
        crop = tf.CenterCrop(self.shape0)
        image = crop(image)
        return image
    

class AdaptedCrop(nn.Module):
    def __init__(self, n : int = 2):
        super().__init__()
        self.n = n


    def __call__(self, image: torch.Tensor, d : int):
        """depth: The number of times it's time to down."""
        if d == 0:
            return image
        shape = image.shape
        shapen = torch.tensor([shape[-2], shape[-1]])
        # down
        for _ in range(d):
            shapen = shapen // 2
            shapen = shapen - 2 * self.n

        # up
        shapen = shapen * 2
        for _ in range(d - 1):
            shapen = shapen - 2 * self.n
            shapen = shapen * 2
        crop = tf.CenterCrop((shapen[0].item(),shapen[1].item()))
        image = crop(image)
        return image

class UNet(nn.Module):

    def __init__(self, in_channels : int = 4 , out_channels : int = 4 , depth : int = 4, top_channels : int = 64, dtype = torch.float32, crop_res : bool = False, norm_layer: nn.Module = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.top_channels = top_channels
        self.dtype = dtype
        self.crop_res = crop_res
        self.padding = 0 if crop_res else 'same'
        self.norm_layer = norm_layer

        if self.crop_res:
            self._pad = AdaptedPad(depth)
            self._crop = AdaptedCrop()
        self._checknan = mynn.CheckNan()
        self._checkinf = mynn.CheckInf()

        self._in = nn.Conv2d(in_channels, top_channels, kernel_size=3, padding=self.padding, dtype=dtype)
        self._down = self.build_encoder()
        self._bottom = self.build_bottleneck()
        self._up = self.build_decoder()
        self._out = nn.Conv2d(top_channels, out_channels, kernel_size=3, padding=self.padding, dtype=dtype)


    def build_encoder(self):
        # self._down = nn.Sequential(*[Down(top_channels * 2 ** i, top_channels * 2 ** (i + 1), dtype = dtype, padding = self.padding, norm_layer = norm_layer) for i in range(depth)])
        encoder = nn.Sequential(
            *[Down(self.top_channels * 2 ** i, self.top_channels * 2 ** (i + 1), dtype = self.dtype, padding = self.padding, norm_layer = self.norm_layer) for i in range(self.depth)]
        )
        return encoder
    
    def build_decoder(self):
        decoder = nn.Sequential(
            *[Up(self.top_channels * 2 ** (self.depth - i), self.top_channels * 2 ** (self.depth - i - 1), dtype = self.dtype, padding = self.padding, norm_layer = self.norm_layer) for i in range(self.depth)]
        )
        return decoder
    
    def build_bottleneck(self):
        bottleneck = nn.Sequential(
            nConv2d(self.top_channels * 2 ** self.depth, 3, 2, self.dtype, self.padding, self.norm_layer),
            CABlock(self.top_channels * 2 ** self.depth)
        )
        return bottleneck

    def forward(self, x: torch.Tensor):
        """[pad] -> in -> down -> bottom -> up -> -> [crop] -> out"""
        # check nan, inf
        x = self._checknan(x, 'in')
        x = self._checkinf(x, 'in')
        # pad
        if self.crop_res:
            x = self._pad(x)
        # in
        x = self._in(x)
        # down path
        if self.crop_res:
            cropped_x = self._crop(x, self.depth)
            res_x = [cropped_x]
        else:
            res_x = [x]
        for i, f in enumerate(self._down):
            x = f(x)
            if self.crop_res:
                cropped_x = self._crop(x,self.depth - 1 - i)
                res_x.append(cropped_x)
            else:
                res_x.append(x)
        # bottom
        x = self._bottom(x)
        # up path
        for i, f in enumerate(self._up):
            x = f(x, res_x[-2 - i])
        if self.padding == 0:
            x = self._pad.crop(x)
        # out
        x = self._out(x)
        # check nan, inf
        x = self._checknan(x, 'out')
        x = self._checkinf(x, 'out')
        return x
