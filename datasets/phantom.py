from ast import Tuple
import typing

from torch.utils.data import Dataset
from glob import glob
from torchvision.transforms.functional import affine

from pqdm.threads import pqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from skimage.data import shepp_logan_phantom
from skimage.transform import resize


class Phantom(Dataset):
    """"""

    def __init__(self, image_size : Tuple = (256,256), transforms: typing.Callable = None, debug = False):
        super().__init__()
        self.transforms = transforms
        self.debug = debug

        self.image_size = image_size


    def __getitem__(self, index: int):
        sample = shepp_logan_phantom()
        sample = resize(sample, self.image_size, anti_aliasing=True)

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
    


    def __len__(self):
        return 1
