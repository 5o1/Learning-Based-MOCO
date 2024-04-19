import torch.nn as nn
import torch

from . import Unet


class Exp1_model(nn.Module):

    def __init__(self, in_channels : int = 4 , out_channels : int = 4 , depth : int = 4, top_channels : int = 64, dtype = torch.float, crop_res : bool = True, n_input = 2) -> None:
        super().__init__()
        self.unets = nn.ModuleList()
        for _ in range(n_input):
            self.unets.append(Unet(in_channels = in_channels, out_channels = out_channels, depth = depth, top_channels = top_channels, dtype = dtype, crop_res = crop_res))

        # {conv(same padded) - BN- Relu} * 3 + conv1x1(out_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * n_input, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, out_channels, 1)
        )


    def forward(self, x):
        I = x['x']
        Ii_hat = []
        for unet in self.unets:
            Ii_hat.append(unet(I))
        
        y_hat = torch.cat(Ii_hat, dim = 1)
        y_hat = self.conv1(y_hat)
        return {'y_hat': y_hat, 'Ii_hat': Ii_hat}
