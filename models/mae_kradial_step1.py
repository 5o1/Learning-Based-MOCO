from torch import nn
import torch
import math
from typing import List
from einops import rearrange, repeat, pack, unpack

from einops.layers.torch import Rearrange

from . import mynn as mynn
from .mynn import functional as myf

class SinusoidalPositionEncoding1d(nn.Module):
    """Sinusoidal Position Encoding for 1-dimensions"""
    def __init__(self, d_model, pos_scale=1e2):
        super(SinusoidalPositionEncoding1d, self).__init__()
        self.d_model = d_model
        self.pos_scale = pos_scale
        self.conv1d = nn.Conv1d(1, d_model, 1)

        self.register_buffer('_div_term', torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)))

    def forward(self, tensor = None, position = None):
        """tensor:[batch_size, seq_len, d_model]  
        position:[batch_size, seq_len, 1]  
        tensor is None: return positional encoding  
        position is None: return [0-1]*scale positional encoding for tensor  
        """
        assert tensor is not None or position is not None, "Either tensor or position must be provided"

        if position is None:
            position = torch.arange(tensor.size(1), device=tensor.device).unsqueeze(0).unsqueeze(-1).float()
        position = (position - position.min()) / (position.max() - position.min())
        position = position * self.pos_scale

        assert len(position.size()) == 3, "position must be [batch_size, seq_len, 1]"

        pe = torch.zeros(position.size(0), position.size(1), self.d_model, device=position.device)
        pe[:, :, 0::2] = torch.sin(position * self._div_term)
        pe[:, :, 1::2] = torch.cos(position * self._div_term)

        if tensor is None:
            return pe        
        return tensor + pe




class SinusoidalPositionEncodingmd(nn.Module):
    """Multi-dimensional Sinusoidal Position Encoding"""
    def __init__(self, d_model, pos_scale: float | List[float] = 1e2, n_dims=2):
        super(SinusoidalPositionEncodingmd, self).__init__()
        self.n_dims = n_dims
        self.d_model = d_model
        self.register_buffer('pos_scale', torch.tensor(pos_scale if isinstance(pos_scale, list) else [pos_scale] * n_dims))

        assert d_model % n_dims == 0, "d_model must be divisible by n_dims"
        self.register_buffer('_div_term', torch.exp(torch.arange(0, self.d_model // self.n_dims, 2).float() * -(math.log(10000.0) / self.d_model // self.n_dims)))

    
    def forward(self, tensor, position = None):
        """tensor:[batch_size, seq_len, d_model]  
        position:[batch_size, seq_len, n_dims]  
        tensor is None: return positional encoding   
        position is None: return [0-1]*scale positional encoding for tensor   
        """
        assert tensor is not None or position is not None, "Either tensor or position must be provided"

        if position is None:
            position = []
            for i in range(self.n_dims):
                pos = torch.arange(tensor.size(1), device=tensor.device).unsqueeze(0).unsqueeze(-1).float()
                position.append(pos)
            position = torch.cat(position, dim=-1)
        position = (position - position.min()) / (position.max() - position.min())
        position = position * self.pos_scale

        assert len(position.size()) == 3, "position must be [batch_size, seq_len, n_dims]"

        pe = torch.zeros(position.size(0), position.size(1), self.d_model, device=position.device)
        for i in range(self.n_dims): # [aaabbbccc]
            pe[:, :, i * self.d_model // self.n_dims:(i+1) * self.d_model // self.n_dims:2] = torch.sin(position[:, :, i:i+1] * self._div_term)
            pe[:, :, i * self.d_model // self.n_dims + 1:(i+1) * self.d_model // self.n_dims:2] = torch.cos(position[:, :, i:i+1] * self._div_term)

        if tensor is None:
            return pe
        return tensor + pe


class LearnablePositionEncoding(nn.Module):
    """Learnable Position Encoding"""
    def __init__(self, d_model, max_len=512):
        super(LearnablePositionEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, tensor):
        return tensor + self._pe[:tensor.size(0), :]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)
    

class MAE_Kradial(nn.Module):
    def __init__(self, 
                 mask_ratio = 0.5,
                 pts_readout = 640,
                 in_channels = 16, out_channels = 16,
                 d_model_encoder = 1024, d_model_decoder = 1024,
                 nlayers_encoder = 4, nlayers_decoder = 4,
                 nheads_encoder = 8, nheads_decoder = 8,
                 ):
        super().__init__()
        self.d_model_encoder = d_model_encoder
        self.d_model_decoder = d_model_decoder
        self.nlayer_encoder = nlayers_encoder
        self.nlayer_decoder = nlayers_decoder
        self.nheads_encoder = nheads_encoder
        self.nheads_decoder = nheads_decoder
        self.mask_ratio = mask_ratio
        self.pts_readout = pts_readout
        self.in_channels = in_channels
        self.out_channels = out_channels


        ### Encoder
        self.input_embed = nn.Sequential(
            mynn.IFFTn(dim = -1),
            mynn.Complex2Real(),
            Rearrange('batch channel phase readout -> batch phase (readout channel)'),
            nn.LayerNorm(pts_readout * in_channels * 2), 
            nn.Linear(pts_readout * in_channels * 2, d_model_encoder, bias=True),
            nn.LayerNorm(d_model_encoder)
        )

        self.norm_encoder = nn.LayerNorm(d_model_encoder)

        self.pe_encoder = SinusoidalPositionEncoding1d(d_model = d_model_encoder, pos_scale=100)

        self.encoder = Transformer(dim = d_model_encoder, depth = nlayers_encoder, heads = nheads_encoder, mlp_dim = 2048, dim_head = d_model_encoder // nheads_encoder)


        ### Decoder
        self.decoder_embed = nn.Linear(d_model_encoder, d_model_decoder, bias=True)

        self.pe_decoder = SinusoidalPositionEncoding1d(d_model = d_model_decoder, pos_scale=100)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model_decoder))

        self.decoder = Transformer(dim = d_model_decoder, depth = nlayers_decoder, heads = nheads_decoder, mlp_dim = 2048, dim_head = d_model_decoder // nheads_decoder)

        self.norm_decoder = nn.LayerNorm(d_model_decoder)

        self.output_embed = nn.Sequential(
            nn.Linear(d_model_decoder, pts_readout * out_channels * 2, bias=True),
            Rearrange('batch phase (readout channel) -> batch channel phase readout', channel = out_channels * 2, readout = pts_readout),
            mynn.Real2Complex(),
            mynn.FFTn(dim = -1),
        )



    def random_masking(self, x, mask_ratio):
        """
        https://github.com/facebookresearch/mae/tree/main

        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder_train(self, x, pos, mask_ratio):
        # x: [batch, phase, readout * channel * 2]
        # pos: [batch, phase, 1]

        ### Encoder
        # in training mode, the shape of kdata is matched with the shape of ktraj
        # kdata should be masked as same as ktraj
        x = self.pe_encoder(x, pos)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # cls token
        pass

        # encode
        
        x = self.encoder(x)
        x = self.norm_encoder(x)

        return x, mask, ids_restore
    
    def forward_decoder_train(self, x, pos, ids_restore):
        # x: [batch, phase, readout * channel * 2]
        # pos: [batch, phase, 1]

        ### Decoder
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # pe
        x = self.pe_decoder(x, pos)

        # apply Transformer blocks
        x = self.decoder(x)
        x = self.norm_decoder(x)

        # predictor projection
        x = self.output_embed(x)

        # remove cls token
        pass

        return x


    def forward(self, kdata, ktraj):
        # kdata: [batch, channel, phase, readout]
        # ktraj: [batch, phase, readout]
        kdata = self.input_embed(kdata) # [batch, phase, readout * channel]
        ktraj_angle = torch.angle(ktraj).mean(dim = -1).unsqueeze(-1) # [batch, angle]


        ktraj_angle = ktraj_angle[:,:kdata.shape[1]]
        # encode
        encoder_memory, mask, ids_restore = self.forward_encoder_train(kdata, ktraj_angle, self.mask_ratio)
        
        # decode
        kdata_pred = self.forward_decoder_train(encoder_memory, ktraj_angle, ids_restore)

        return kdata_pred, ktraj[:,:kdata_pred.shape[1]], mask

