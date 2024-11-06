import re
from torch import nn
import torch
import math
from typing import List
from einops import rearrange, repeat, pack, unpack

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


if __name__ == '__main__':
    # Test SinusoidalPositionEncoding1d
    pe = SinusoidalPositionEncoding1d(512, 100)
    x = torch.randn(100, 512, 1 )
    pos = torch.arange(100).unsqueeze(-1).unsqueeze(-1).float()
    y = pe(x, pos)
    print(y.shape)

    # Test SinusoidalPositionEncodingmd
    pe = SinusoidalPositionEncodingmd(512, [100, 200], 2)
    x = torch.randn(100, 512, 1 )
    pos = torch.rand(100, 512, 2)
    y = pe(x, pos)
    print(y.shape)

    # Test LearnablePositionEncoding
    pe = LearnablePositionEncoding(512, 100)
    x = torch.randn(100, 512)
    y = pe(x)
    print(y.shape)