# initialize environment for reproducibility

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
batch_size = 4

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
        return {"kspace_before":kspace, "kspace_after":kspace_after, "image_before":image, "image_after":image_after}


# Model

from torch import nn
from torchvision import transforms as tf

from models import mynn as mynn


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


class Unet(nn.Module):

    def __init__(self, in_channels : int = 4 , out_channels : int = 4 , depth : int = 4, top_channels : int = 64, dtype = torch.float32, crop_res : bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.top_channels = top_channels
        self.dtype = dtype
        self.crop_res = crop_res
        self.padding = 0 if crop_res else 'same'

        if self.crop_res:
            self.pad = AdaptedPad(depth)
            self.crop = AdaptedCrop()
        self.checknan = mynn.CheckNan()
        self.checkinf = mynn.CheckInf()

        self._in = mynn.Boot(in_channels, top_channels, dtype = dtype,padding = self.padding) # in
        self.down_convs = nn.ModuleList() # Down convs
        channels = top_channels
        for _ in range(depth):
            self.down_convs.append(mynn.Down(channels, channels * 2, dtype = dtype, padding = self.padding))
            channels = channels * 2
        self.up_convs = nn.ModuleList() # Up convs
        for _ in range(depth):
            self.up_convs.append(mynn.Up(channels, channels // 2, dtype = dtype, padding = self.padding))
            channels = channels // 2
        self._out = mynn.Output(channels, out_channels, dtype = dtype, padding = self.padding)

    def forward(self, x: torch.Tensor):
        # double check
        x = self.checknan(x, 'in')
        x = self.checkinf(x, 'in')
        if self.crop_res:
            x = self.pad(x)
        x = self._in(x)
        if self.crop_res:
            cropped_x = self.crop(x, self.depth)
            res_x = [cropped_x]
        else:
            res_x = [x]
        # down path
        for i, f in enumerate(self.down_convs):
            x = f(x)
            if self.crop_res:
                cropped_x = self.crop(x,self.depth - 1 - i)
                res_x.append(cropped_x)
            else:
                res_x.append(x)
        # up path
        for i, f in enumerate(self.up_convs):
            x = f(x, res_x[-2 - i])
        if self.padding == 0:
            x = self.pad.crop(x)
        # out
        x = self._out(x)
        # double check
        x = self.checknan(x, 'out')
        x = self.checkinf(x, 'out')
        return x


from torch import nn

from models import mynn as mynn
from models.mynn import functional as myf

class MyModel(nn.Module):
    def __init__(self, in_channels : int = 16, out_channels : int = 16 , depth : int = 4, top_channels : int = 64, dtype = torch.float32, crop_res = True):
        super().__init__()
        self.unet_real = Unet(in_channels = in_channels, out_channels = out_channels, depth = depth, top_channels = top_channels, dtype = dtype, crop_res = crop_res)
        self.unet_imag = Unet(in_channels = in_channels, out_channels = out_channels, depth = depth, top_channels = top_channels, dtype = dtype, crop_res = crop_res)

    def forward(self, x: torch.Tensor):
        real = x.real
        imag = x.imag
        real = self.unet_real(real)
        imag = self.unet_imag(imag)
        return real + 1j * imag
        

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

model = MyModel(in_channels = 16, out_channels = 16, depth = 4, top_channels = 64, dtype = torch.float32, crop_res = True).to(torch.device('cuda'))

def loss_func(pred, target):
    pred = myf.complex_to_real(pred)
    target = myf.complex_to_real(target)
    return F.mse_loss(pred, target)

optimizer = optim.Adam(model.parameters(), lr = 1e-3)

dataenhance = Dataenhance(Motion(device = torch.device('cuda')))

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
        image_after = data_after['image_after'].to(torch.device('cuda'))
 
        optimizer.zero_grad()
        image_pred = model(image_after)
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
            image_after = data_after['image_after'].to(torch.device('cuda'))

            image_pred = model(image_after)
            loss = loss_func(image_pred, image_true)
            loss_epoch += loss.item()
            print(f"epoch: {epoch}, batch: {i}, loss: {loss.item ()}")

    writer.add_scalar('Loss/val', loss, epoch)
    torch.save(model.state_dict(), os.path.join(expdir, f"model_{epoch}.pth"))


