import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from models import mynn as mynn
from models.mynn import functional as myf


from einops import rearrange, repeat

from torchkbnufft import KbNufft, KbNufftAdjoint, calc_tensor_spmatrix, calc_density_compensation_function, ToepNufft

import numpy as np

import sys
import os
sys.path.append(os.path.join(os.environ['BART_TOOLBOX_PATH'], 'python'))
from bart import bart


class Transform(nn.Module):
    def __init__(self, image_size, num_spokes_full, num_spokes_partial):
        super(Transform, self).__init__()
        self.image_size = image_size
        self.num_spokes_full = num_spokes_full
        self.num_spokes_partial = num_spokes_partial

        num_readout_points = 320
        oversampling_ratio = 2

        gatraj = self.get_gatraj( num_readout_points = num_readout_points, oversampling_ratio = oversampling_ratio)


        self._nufft_obj = KbNufft(im_size=image_size, device=torch.device('cuda')).requires_grad_(False)
        self._inufft_obj = KbNufftAdjoint(im_size=image_size, device=torch.device('cuda')).requires_grad_(False)

        # toep_ob = ToepNufft().requires_grad_(False)

        self.traj_raw = gatraj.clone().to(torch.float32).to('cuda') # phase readout pos
        self.traj_raw = torch.view_as_complex(self.traj_raw) # phase readout
        # use complex number to represent the trajectory position
        # for example, traj[0,0] = 0.1 + 0.2j means the first point position in cartesian coordinate is (0.1, 0.2)
        # in other words, traj[0,0] = 0.1 + 0.2j means the first point position in polar coordinate is ( abs(0.1 + 0.2j), angle(0.1 + 0.2j) )

        self.traj_full = self.traj_raw[:num_spokes_full]
        self.traj_partial = self.traj_raw[:num_spokes_partial]

    def get_gatraj(self, num_readout_points, oversampling_ratio):
        points_per_phase = num_readout_points * oversampling_ratio
        gatraj = bart(1, f"traj -r -x{num_readout_points} -y800 -G -o{oversampling_ratio}")

        print(gatraj.shape)


        gatraj = torch.tensor(gatraj)
        gatraj = rearrange(gatraj, 'pos readout phase -> phase readout pos')
        gatraj = gatraj[:,:,:2]

        print(gatraj.shape)

        return gatraj


    def nufft(self, image, traj):
        device = image.device
        image = image.to(torch.device("cuda"))
        traj = traj.to(torch.device("cuda"))

        traj_shape = traj.shape
        traj = traj / 160 * torch.pi # bart normalization to torchkbnufft normalization
        traj = torch.view_as_real(traj)
        traj = rearrange(traj, 'phase readout pos -> pos (readout phase)')

        res = self._nufft_obj(image, traj)
        res = rearrange(res, '... (readout phase) ->... phase readout', phase=traj_shape[0], readout=traj_shape[1])

        res = res.to(device)
        return res
    
    def inufft(self, kspace, traj):
        device = kspace.device
        kspace = kspace.to(torch.device("cuda"))
        traj = traj.to(torch.device("cuda"))

        traj_shape = traj.shape
        traj = traj / 160 * torch.pi # bart normalization to torchkbnufft normalization
        traj = torch.view_as_real(traj)
        traj = rearrange(traj, 'phase readout pos -> pos (readout phase)')
        kspace = rearrange(kspace, '... phase readout ->... (readout phase)')

        interp_mats = calc_tensor_spmatrix(traj,im_size=self.image_size, table_oversamp=2)
        dcomp = calc_density_compensation_function(ktraj=traj, im_size=self.image_size)

        res = self._inufft_obj(kspace * dcomp, traj, interp_mats)
        res = res.to(device)
        return res

    def mean_std_norm_complex(self, data: torch.Tensor, dim = [-1, -2, -3]):
        real = data.real
        imag = data.imag
        real = (real - real.mean(dim=dim, keepdim = True)) / real.std(dim=dim, keepdim = True)
        imag = (imag - imag.mean(dim=dim, keepdim = True)) / imag.std(dim=dim, keepdim = True)
        return real + 1j * imag


    def forward(self, sample):
        device = sample["kspace"].device
        kspace = sample["kspace"].to(torch.complex64).to('cuda')

        image = myf.ktoi(kspace)
        image = self.mean_std_norm_complex(image)

        kspace_full = self.nufft(image, self.traj_full)
        # kspace_partial = self.nufft(image, self.traj_partial)
        pos_seq_full = repeat(self.traj_full, 'phase readout -> slice phase readout', slice=image.shape[0])
        # pos_seq_partial = repeat(self.traj_partial, 'phase readout -> slice phase readout', slice=image.shape[0])

        kspace = kspace.to(device)
        kspace_full = kspace_full.to(device)
        # kspace_partial = kspace_partial.to(device)
        pos_seq_full = pos_seq_full.to(device)
        # pos_seq_partial = pos_seq_partial.to(device)
        image = image.to(device)

        return {"kspace":kspace, "kspace_before": kspace_full, "kspace_after": kspace_full, "pos_seq_before": pos_seq_full, "pos_seq_after": pos_seq_full, "image": image}