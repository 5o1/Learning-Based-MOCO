import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from zmq import device

from .mynn.transformer import FeedForward, Attention, Transformer


class ViTLine(nn.Module):
 
    def __init__(self, channels, image_size, embed_dim, dropout = 0., depth = 6, heads = 8, dim_head = 64, mlp_dim = 2048):
        super().__init__()
        self.embed_dim = embed_dim

        self.line_embedding = nn.Sequential(
            Rearrange('... c h w -> ... w (c h)'),
            nn.LayerNorm(channels * image_size[-2]),
            nn.Linear(channels * image_size[-2], embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1,image_size[-1], embed_dim))

        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, kspace):
        """
        kspace: (B, C, H, W)
        """
        raw_shape = kspace.shape

        # print(kspace.shape)

        x = self.line_embedding(kspace) # b h (c w)

        # print(1, x.min(), x.max())

        # freq_series = torch.linspace(-1 / raw_shape[-1], 1 / raw_shape[-1], steps=raw_shape[-1], device=x.device)
        # self.pos_embedding = repeat(freq_series, 'w -> 1 w d', d = self.embed_dim)

        # x += self.pos_embedding[:, :]
        # x = self.dropout(x)

        # print(2, x.min(), x.max())

        x = self.transformer(x)

        # print(3, x.min(), x.max())

        x = self.mlp_head(x) # b h nclass

        # print(4, x.min(), x.max())

        x = repeat(x, '... w class -> ... class h w', h=raw_shape[-2])

        # print(5, x.min(), x.max())

        return x