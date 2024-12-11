from numpy import mean
import torch
import typing

from torch.utils.data import Dataset

from skimage.data import shepp_logan_phantom
from skimage.transform import resize


class Phantom(Dataset):

    def __init__(self, image_size : tuple = (256,256), transforms: typing.Callable = lambda x:x, debug = False):
        super().__init__()
        self.transforms = transforms
        self.image_size = image_size


    def __getitem__(self, index: int):
        sample = shepp_logan_phantom()
        sample = resize(sample, self.image_size, anti_aliasing=True)
        sample = torch.tensor(sample).float().view(1, *self.image_size)
        sample = (sample - sample.mean()) / sample.std()

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
    


    def __len__(self):
        return 1
