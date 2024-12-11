# initialize environment for reproducibility

from email.mime import image
from turtle import pos
import torch
import random

random.seed(0)
torch.manual_seed(0)

# global variables

trainlistpath = "/home/liyy/data1/moco/datasets/.Fastmri_pics/trainlist.txt"
vallistpath = "/home/liyy/data1/moco/datasets/.Fastmri_pics/vallist.txt"
testlistpath = "/home/liyy/data1/moco/datasets/.Fastmri_pics/testlist.txt"

from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

expdir = f"./exp/{timestamp}/"
tensorboarddir = f"./.tb_logs/{timestamp}"

# mkdir

import os

os.makedirs(expdir, exist_ok=True)
os.makedirs(tensorboarddir, exist_ok=True)

print(f"expdir: {expdir}")
print(f"tensorboarddir: {tensorboarddir}")

# epochs = 300
iters = 12000
batch_size = 1

# format: filename index_of_slice


with open(trainlistpath, 'r') as f:
    trainlist = f.readlines()
    trainlist = [(line.split()[0], int(line.split()[1])) for line in trainlist]

with open(vallistpath, 'r') as f:
    vallist = f.readlines()
    vallist = [(line.split()[0], int(line.split()[1])) for line in vallist]

with open(testlistpath, 'r') as f:
    testlist = f.readlines()
    testlist = [(line.split()[0], int(line.split()[1])) for line in testlist]

print("file in trainlist: ", len(trainlist))
print("file in vallist: ", len(vallist))
print("file in testlist: ", len(testlist))

# dataset

import torch
from torchvision import transforms as tf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from models import mynn as mynn
from models.mynn import functional as myf

from random import shuffle

from einops import rearrange, repeat

from torchkbnufft import KbNufft, KbNufftAdjoint, calc_tensor_spmatrix, calc_density_compensation_function, ToepNufft

import numpy as np

import os


# ismrmrd_header rss csm kspace
class Fastmri_320p(Dataset):
    def __init__(self, filelist, num_subset = None, transform=None, lazy_memory=True, output_keys=["kspace", "after_transform"], output_type = dict, disk_cache = False):
        super(Fastmri_320p, self).__init__()
        self.filelist = filelist
        self.num_subset = num_subset
        self.transform = transform
        self.lazy_cache = lazy_memory
        self.output_keys = output_keys
        self.output_type = output_type
        self.disk_cache = disk_cache

        self.extension_disk_cache = "._cached"

        if self.num_subset is not None:
            self.filelist = self.filelist[:self.num_subset]

        self._cache = {}
        self._keys = ["kspace", "csm", "rss"]

        if not lazy_memory:
            try:
                for file in filelist:
                    path, index = file
                    if path not in self._cache:
                        self.load(path)
            except MemoryError as e: # The total size of the dataset is about 400GB
                print("Memory Error in preloading the dataset.", e)
                print("Lazy memory is recommended. Automatically switch to lazy memory.")
                self.release(ratio=0.75)
                self.lazy_cache = True

    def mnemonic_disk_cache(self, path):
        """
        Get the name of the disk cache file.
        """
        return path + self.extension_disk_cache

    def clean_disk_cache(self):
        """
        Clean the disk cache files.
        """
        print("WARNING: This will remove all disk cache files with the extension '._cached'.")
        for path, idx in self.filelist:
            cache_path = self.mnemonic_disk_cache(path)
            if os.path.exists(cache_path):
                os.remove(cache_path)

    def load(self, path):
        """
        Load the data from the file path.
        """
        cache_path = self.mnemonic_disk_cache(path)

        if self.disk_cache: # load from disk cache
            if os.path.exists(cache_path):
                self._cache[path] = torch.load(cache_path)
                return

        with np.load(path) as data:
            self._cache[path] = {key: torch.tensor(data[key]) for key in self._keys}
            self._cache[path].update(self.transform(self._cache[path]))
            for key in [key for key in self._cache[path] if key not in self.output_keys]:
                del self._cache[path][key]

            if self.disk_cache: # dump to disk cache
                torch.save(self._cache[path], cache_path)

    def release(self, ratio = 0.5):
        """
        Release the memory by deleting the cache with a ratio of the total cache.
        """
        assert 0 < ratio <= 1 , "The ratio must be a float between 0 and 1."

        keys_to_delete = shuffle(list(self._cache.keys()))[:int(len(self._cache) * ratio)]
        for key in keys_to_delete:
            del self._cache[key]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        path, index = self.filelist[idx]

        if path not in self._cache:
            try:
                self.load(path)
            except MemoryError as e:
                self.release()
                self.load(path) # try again

        if self.output_type == dict:
            return {key: self._cache[path][key][index] for key in self.output_keys}
        elif self.output_type == list:
            return [self._cache[path][key][index] for key in self.output_keys]
        

# loadtransform
from torch import nn


load_keys = ['kspace']

class Loadtransform(nn.Module):
    """Load the data to the memory"""
    def __init__(self):
        super().__init__()

    def forward(self, data):
        # input: ismrmrd_header rss csm kspace
        return {'kspace': data['kspace'].to(torch.complex64)}


# data enhancement transform

import torch
from torch import nn
from torchvision import transforms as tf

from models.mynn import functional as myf
from torchkbnufft import KbNufft, KbNufftAdjoint, calc_tensor_spmatrix, calc_density_compensation_function, ToepNufft

from einops import rearrange, repeat


class Motion(nn.Module):
    def __init__(self, image_size = (320, 320),
                 motion_ratio = [1, 1, 1, 1, 1], rot = 15, shift = (0.05, 0.05), scale = (0.01, 0.01), shear = (0.01, 0.01),
                 num_spokes_full = 500, num_spokes_partial = 20, num_pts_readout = 320, oversampling_factor = 2,
                 dtype = torch.complex64, device = torch.device('cuda')):
        super().__init__()
        self.image_size = image_size
        self.rot = rot
        self.shift = shift
        self.scale = scale
        self.shear = shear
        self.num_spokes_full = num_spokes_full
        self.num_spokes_partial = num_spokes_partial
        self.num_pts_readout = num_pts_readout
        self.oversampling_factor = oversampling_factor
        self.dtype = dtype
        self.device = device

        self.float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

        self.num_motion_states = len(motion_ratio)
        self.motion_partition = [0]
        for ratio in motion_ratio:
            self.motion_partition.append(self.motion_partition[-1] + ratio / sum(motion_ratio))
        
        self.random_motion = tf.RandomAffine(degrees=rot, translate=shift, scale=scale, shear=shear, fill=0).to(device)
        self._nufft_obj = KbNufft(im_size=image_size, dtype=dtype, device=device)
        self._inufft_obj = KbNufftAdjoint(im_size=image_size, dtype=dtype, device=device)

        self.traj_full = self.gatraj().to(device).to(self.float_dtype)
        self.traj_partial = self.traj_full[:self.num_spokes_partial]

    def move(self, image):
        """ input: batch, channel, height, width
            output: motion_state, batch, channel, height, width
        """
        image = myf.complex_to_real(image) # batch, channel * 2, height, width
        resl = torch.zeros((self.num_motion_states+1), *image.shape, dtype=image.dtype, device=image.device) # motion_state batch, channel, height, width
        resl[0] = image
        for i in range(self.num_motion_states):
            resl[i+1] = self.random_motion(image)
        resl = myf.real_to_complex(resl) # motion_state, batch, channel, height, width
        return resl
    
    def gatraj(self):
        """Get golden angle trajectory"""
        import sys
        import os
        sys.path.append(os.path.join(os.environ['BART_TOOLBOX_PATH'], 'python'))
        from bart import bart
        traj = bart(1, f'traj -x {self.num_pts_readout} -y {self.num_spokes_full} -r -G -o {self.oversampling_factor}')
        traj = torch.tensor(traj)[:2, :, :].real
        traj = rearrange(traj, 'pos readout phase -> phase readout pos')
        return traj

    def ft(self, image, ktraj):
        ktraj_shape = ktraj.shape
        ktraj = ktraj / self.num_pts_readout * 2 * torch.pi # bart normalization to torchkbnufft normalization
        ktraj = rearrange(ktraj, 'phase readout pos -> pos (phase readout)')

        # batched nufft
        original_shape = image.shape
        image = rearrange(image, '... channel phase readout -> (...) channel phase readout') # nufft only accept [b h w] input

        res = torch.cat([self._nufft_obj(image[i].unsqueeze(0), ktraj) for i in range(image.shape[0])], dim = 0).view(*original_shape[:-2], ktraj_shape[0], ktraj_shape[1]) # ... phase readout
        return res
    
    def ift(self, kspace, ktraj):
        ktraj_shape = ktraj.shape
        ktraj = ktraj / self.num_pts_readout * 2 * torch.pi
        ktraj = rearrange(ktraj, 'phase readout pos -> pos (phase readout)')

        interp_mats = calc_tensor_spmatrix(ktraj,im_size=self.image_size, table_oversamp=2)
        dcomp = calc_density_compensation_function(ktraj=ktraj, im_size=self.image_size)

        # batched inufft
        original_shape = kspace.shape

        kspace = rearrange(kspace, '... channel phase readout -> (...) channel (phase readout)') # inufft only accept [b pts] input

        res = torch.cat([self._inufft_obj(kspace[i].unsqueeze(0) * dcomp, ktraj, interp_mats) for i in range(kspace.shape[0])], dim=0).view(*original_shape[:-2], *self.image_size) # ... height width
        return res

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.move(image) # motion_state, batch, channel, height, width
        kspace = self.ft(image, self.traj_full) # motion_state, batch, channel, phase, readout

        indices = torch.randperm(kspace.shape[-2])
        kspace_mixed = torch.zeros_like(kspace[0]) # batch, channel, phase, readout
        for state in range(self.num_motion_states):
            kspace_mixed[:, :, indices[int(self.motion_partition[state] * indices.shape[0]):int(self.motion_partition[state+1] * indices.shape[0])], :] \
                = kspace[state][:, :, indices[int(self.motion_partition[state] * indices.shape[0]):int(self.motion_partition[state+1] * indices.shape[0])] , :]

        image = self.ift(kspace_mixed, self.traj_full)
        return image, kspace_mixed


class Dataenhance(nn.Module):
    """Computationally Intensive Transformations"""
    def __init__(self, motion_simulator: nn.Module):
        super().__init__()
        self.motion_simulator = motion_simulator

    def mean_std_norm_complex(self, data: torch.Tensor) -> torch.Tensor:
        real = data.real
        imag = data.imag
        real = (real - real.mean()) / real.std()
        imag = (imag - imag.mean()) / imag.std()
        return real + 1j * imag

    def forward(self, kspace):
        image = myf.ktoi(kspace)
        image = self.mean_std_norm_complex(image)
        kspace = myf.itok(image)
        image_after, kspace_after = self.motion_simulator(image)
        image_after = self.mean_std_norm_complex(image_after)

        pos_seq = torch.view_as_complex(self.motion_simulator.traj_full)
        pos_seq = repeat(pos_seq, 'phase readout-> batch phase readout', batch = image.shape[0])

        return {"kspace_before":kspace, "kspace_after":kspace_after, "image_before":image, "image_after":image_after, "pos_before": pos_seq, "pos_after": pos_seq}


# Model

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



import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

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


from torch import nn

from models import mynn as mynn
from models.mynn import functional as myf

class MyModel(nn.Module):
    def __init__(self, 
                 mask_ratio = 0.5,
                 pts_readout = 640,
                 in_channels = 16, out_channels = 16,
                 d_model_encoder = 1024, d_model_decoder = 1024,
                 nlayers_encoder = 4, nlayers_decoder = 4,
                 nheads_encoder = 8, nheads_decoder = 8,
                 mlp_dim_encoder = 2048, mlp_dim_decoder = 2048,
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
        self.mlp_dim_encoder = mlp_dim_encoder
        self.mlp_dim_decoder = mlp_dim_decoder


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

        self.encoder = Transformer(dim = d_model_encoder, depth = nlayers_encoder, heads = nheads_encoder, mlp_dim = mlp_dim_encoder, dim_head = d_model_encoder // nheads_encoder)


        ### Decoder
        self.decoder_embed = nn.Linear(d_model_encoder, d_model_decoder, bias=True)

        self.pe_decoder = SinusoidalPositionEncoding1d(d_model = d_model_decoder, pos_scale=100)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model_decoder))

        self.decoder = Transformer(dim = d_model_decoder, depth = nlayers_decoder, heads = nheads_decoder, mlp_dim = mlp_dim_decoder, dim_head = d_model_decoder // nheads_decoder)

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
    
    def forward_encoder_val(self, x, pos):
        # x: [batch, phase, readout * channel * 2]
        # pos: [batch, phase, 1]

        x = self.pe_encoder(x, pos)

        # cls token
        pass

        x = self.encoder(x)
        x = self.norm_encoder(x)
        return x
    
    def forward_decoder_val(self, x, pos):
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], pos.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)

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
        ktraj_angle = torch.angle(ktraj[:,:,1]-ktraj[:,:,0]).unsqueeze(-1) # [batch, angle]



        # if self.training:
        #     ktraj_angle = ktraj_angle[:,:kdata.shape[1]]
        #     # encode
        #     encoder_memory, mask, ids_restore = self.forward_encoder_train(kdata, ktraj_angle, self.mask_ratio)
            
        #     # decode
        #     kdata_pred = self.forward_decoder_train(encoder_memory, ktraj_angle, ids_restore)

        #     return kdata_pred

        # else:
        # in evaluation mode, the shape of kdata is matched with the shape of ktraj[:, :num_phase_kdata]
        encoder_memory= self.forward_encoder_val(kdata, ktraj_angle[:,:kdata.shape[1]])

        pred = self.forward_decoder_val(encoder_memory, ktraj_angle)
        return pred

        

def mean_std_norm_complex(data: torch.Tensor) -> torch.Tensor:
    real = data.real
    imag = data.imag
    real = (real - real.mean()) / real.std()
    imag = (imag - imag.mean()) / imag.std()
    return real + 1j * imag


def mean_std_norm_real(data: torch.Tensor) -> torch.Tensor:
    return (data - data.mean()) / data.std()

# train

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from models import mynn as mynn
from models.mynn import functional as myf

import os

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(tensorboarddir, flush_secs=1)

trainloader = DataLoader(Fastmri_320p(trainlist, transform = Loadtransform(), lazy_memory = True, output_keys = ['kspace'], output_type = dict, disk_cache = False), batch_size = batch_size, shuffle = True)
valloader = DataLoader(Fastmri_320p(vallist, transform = Loadtransform(), lazy_memory = True, output_keys = ['kspace'], output_type = dict, disk_cache = False), batch_size = batch_size, shuffle = True)

model = MyModel(in_channels = 16, out_channels = 16, d_model_encoder = 4096, d_model_decoder = 4096, nlayers_encoder = 2, nlayers_decoder = 2, nheads_encoder = 8, nheads_decoder = 8, mlp_dim_encoder=4096, mlp_dim_decoder=4096).to('cuda')

def loss_func(pred, target):
    pred = myf.complex_to_real(pred)
    target = myf.complex_to_real(target)
    return F.mse_loss(pred, target)

optimizer = optim.Adam(model.parameters(), lr = 1e-3)

motion = Motion(device = torch.device('cuda'))
dataenhance = Dataenhance(motion)

if iters is not None:
    epochs = iters // len(trainloader) + 1
    iter_cnter = 0

for epoch in range(epochs):
    if iters is not None and iter_cnter >= iters:
        break

    model.train()
    loss_epoch = 0
    for i, data in enumerate(trainloader):
        kspace = data['kspace'].to('cuda')
        with torch.no_grad():
            data_after = dataenhance(kspace)
        image_true = data_after['image_before'].to(torch.device('cuda'))
        kspace_after = data_after['kspace_after'].to(torch.device('cuda'))
        pos_seq = data_after['pos_before'].to(torch.device('cuda'))
 
        optimizer.zero_grad()
        kspace_pred = model(kspace_after, pos_seq)

        image_pred = torch.zeros_like(image_true)
        for ift_batch in range(image_pred.shape[0]):
            image_pred[ift_batch] = motion.ift(kspace_pred[ift_batch], torch.view_as_real(pos_seq[ift_batch]))

        image_pred = mean_std_norm_complex(image_pred)
        image_true = mean_std_norm_complex(image_true)
        
        loss = loss_func(image_pred, image_true)
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, batch: {i}, loss: {loss.item ()}")
        loss_epoch += loss.item()

        writer.add_scalar('Loss/train_iter', loss, iter_cnter)

        # every 1000 iterations, save the model
        if iters is not None and iter_cnter % 1000 == 0:
            torch.save(model.state_dict(), os.path.join(expdir, f"model_iter_{iter_cnter}.pth"))

        iter_cnter += 1

    writer.add_scalar('Loss/train', loss, epoch)

    # if epoch % 10 == 0:
    model.eval()
    loss_epoch = 0
    with torch.no_grad():
        for i, data in enumerate(valloader):
            kspace = data['kspace'].to('cuda')
            with torch.no_grad():
                data_after = dataenhance(kspace)
            image_true = data_after['image_before'].to(torch.device('cuda'))
            kspace_after = data_after['kspace_after'].to(torch.device('cuda'))
            pos_seq = data_after['pos_before'].to(torch.device('cuda'))
    
            kspace_pred = model(kspace_after, pos_seq)
            image_pred = torch.zeros_like(image_true)
            for ift_batch in range(image_pred.shape[0]):
                image_pred[ift_batch] = motion.ift(kspace_pred[ift_batch], torch.view_as_real(pos_seq[ift_batch]))
            image_pred = mean_std_norm_complex(image_pred)
            image_true = mean_std_norm_complex(image_true)
            loss = loss_func(image_pred, image_true)
            loss_epoch += loss.item()
            print(f"epoch: {epoch}, batch: {i}, loss: {loss.item ()}")

    writer.add_scalar('Loss/val', loss, epoch)
    torch.save(model.state_dict(), os.path.join(expdir, f"model_{epoch}.pth"))


