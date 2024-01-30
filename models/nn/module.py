import torch
from torchvision import transforms as tf
from torch import nn
from . import ComplexBatchNorm2d, ComplexReLU
from .functional.common import is_complex_dtype

class nConv2d(nn.Module):
    """The module of every step performing multiple conv layer in left path or right path of U-net."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, n: int = 2, dtype = torch.complex64, padding = 0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.n = n
        self.dtype = dtype
        self.padding = padding

        # The first group of convolutional layers changes the number of channels.
        self.seq = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel, stride=1, padding=padding, dtype = dtype),
                ComplexBatchNorm2d(out_channels) if is_complex_dtype(dtype) else nn.BatchNorm2d(out_channels),
                ComplexReLU() if is_complex_dtype(dtype) else nn.ReLU()
            ])


        # The next convolutional layer.
        for _ in range(n - 1):
            self.seq.extend([
                nn.Conv2d(out_channels, out_channels, kernel, stride=1, padding = padding, dtype = dtype),
                ComplexBatchNorm2d(out_channels) if is_complex_dtype(dtype) else nn.BatchNorm2d(out_channels),
                ComplexReLU() if is_complex_dtype(dtype) else nn.ReLU()
            ])

    def forward(self, x):
        for module in self.seq:
            x = module(x)
        return x
    

class Down(nn.Module):
    """left path / encoder"""

    def __init__(self, in_channels, out_channels, dtype = torch.complex64,padding = 0) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(in_channels = in_channels, out_channels=in_channels,kernel_size=2, stride=2, dtype = dtype) if is_complex_dtype(dtype) else nn.MaxPool2d(kernel_size=2)
        self.conv = nConv2d(in_channels, out_channels, dtype = dtype,padding = padding)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """right path / decoder"""

    def __init__(self, in_channels, out_channels, dtype = torch.complex64, padding = 0) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2, dtype= dtype) # Inverse convolution layer shrinks the number of channels
        self.concat = torch.cat
        self.conv = nConv2d(in_channels, out_channels, dtype = dtype, padding = padding)

    def forward(self, x, res_x):
        x = self.upsample(x)
        x = self.concat([res_x, x], -3)
        x = self.conv(x)

        return x
    

class Boot(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dtype = torch.complex64, padding = 0) -> None:
        super().__init__()
        self.conv = nConv2d(in_channels, out_channels, dtype = dtype, padding = padding)


    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x


class Output(nn.Module):
    """out layer, this is an example of an output image by conv layer."""

    def __init__(self, in_channels: int, out_channels: int, dtype = torch.complex64, padding = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), dtype = dtype, padding = padding)
        # self.bn = ComplexBatchNorm2d(out_channels) if is_complex_dtype(dtype) else nn.BatchNorm2d(out_channels)
        # self.relu = ComplexReLU() if is_complex_dtype(dtype) else nn.ReLU()


    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        # x = self.bn(x)
        # x = self.relu(x)
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
