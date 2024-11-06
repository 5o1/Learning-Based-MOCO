import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from .mynn.transformer import FeedForward, Attention, Transformer


class ViTLine(nn.Module):
 
    def __init__(self, channels, image_size, embed_dim, dropout = 0., depth = 6, heads = 8, dim_head = 64, mlp_dim = 2048, nclass = 2):
        super().__init__()

        self.line_embedding = nn.Sequential(
            Rearrange('b c h w -> b h (c w)'),
            nn.LayerNorm(channels * image_size[-1]),
            nn.Linear(channels * image_size[-1], embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1,image_size[-1], embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, nclass),
            nn.Softmax(dim=-1)
        )


    def forward(self, kspace):
        """
        kspace: (B, C, H, W)
        """
        raw_shape = kspace.shape

        x = self.line_embedding(kspace) # b h (c w)

        x += self.pos_embedding[:, :]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.mlp_head(x) # b h nclass

        x = repeat(x, 'b h nclass -> b 1 h w nclass', w=raw_shape[-1])

        return x