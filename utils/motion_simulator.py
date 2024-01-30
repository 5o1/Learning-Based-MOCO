import torch
import h5py
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from numpy import *
from numpy.fft import *
import matplotlib.pyplot as plt


def interpolate3d(img: torch.Tensor, spacing: tuple[float, float, float]):
    """
    Uniform spacing of 3d.

    Parameters:
        img(Tensor): Input image which is 3d
        spacing(float, float, float): 3d spacing

    This function will uniform spacing of 3 dimention to the lowest spacing.
    """

    minspacing = min(spacing)

    scaling = (spacing[0] / minspacing, spacing[1] / minspacing, spacing[2] / minspacing)
    shape = img.shape
    # target_shape = torch.floor(shape[0] * scaling[0], shape[1] * scaling[1], shape[2] * scaling[2])

    img = img.view(1, 1, shape)
    output = nn.functional.interpolate(img, scale_factor=scaling)
    output = output[0][0]

    return output


def affine3d(img: torch.Tensor, translate: tuple[float, float], rotate: tuple[float, float, float]):
    """
    Using affine transformation on input matrix

    Parameters:
        img(Tensor): Input image which is 3d3
        translate(float, float): Translate along normal vector of sag. palne, normal vector of tra. plane.
        rotate(float, float, float): Rotate countclockwise about the normal vector of the tra. plane, cor. plane, sag. plane.
    """

    output = transforms


if __name__ == '__main__':
    h5path = '/mnt/nfs_datasets/fastMRI_brain/multicoil_train_sorted/size_320_640_4/file_brain_AXT1POST_203_6000594.h5'
    h5 = h5py.File(h5path, 'r')

    # kspace to img affine
    kspace = h5["kspace"][:, 0, :, :]
    print(kspace.shape)
    img = fftshift(ifft2(ifftshift(kspace), None, [1, 2]))
    print(kspace.dtype)

    # img affine
    # img = h5["reconstruction_rss"][:]
    # plt.figure()
    # plt.imshow(img, cmap='gray')

    # perform affine
    t = torch.from_numpy(img)
    t = t.unsqueeze(0).unsqueeze(0)
    # t = t.unsqueeze(0)
    # print(t.shape)
    ag = F.affine_grid(torch.tensor([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0]], dtype=torch.float).unsqueeze(0), t.size())
    t = F.grid_sample(t, ag, mode = 'nearest', padding_mode='zeros')
    # t = ag * t
    print(t)
