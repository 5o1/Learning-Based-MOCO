import typing

from torch.utils.data import Dataset
from glob import glob
import h5py
from torchvision.transforms.functional import affine

import os
from pqdm.threads import pqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

class Fastmri_brain(Dataset):
    """get dataset fastmri_brain.
    This class is a subclass of torch.utils.datasets.Dataset and therefore accepted by DataLoader.
    Note that one item of fastmri_brain contains both kspace domain (multi-coil as one dimention) and image domain.
    However, you can only select one reasonable method to send to the models every time you extract this class,
    so that you must choose one transformation function (see utils.transforms package in this project!)
    and you can alse combine this with torchvision.transforms.Compose."""

    def __init__(self, path: str, transforms: typing.Callable, n_subset = None, debug = False, load_from_memory = False, n_jobs :int = 1):
        super().__init__()
        self.path = path
        self.transforms = transforms
        self._sample_list = glob(pathname=os.path.join(path, '*.h5'))
        self.n_subset = n_subset if n_subset is not None else len(self._sample_list)
        self.debug = debug
        self.load_from_memory = load_from_memory
        self.n_jobs = n_jobs

        self._sample_list = self._sample_list[:self.n_subset]

        if self.load_from_memory:
            self.cache = {}
            def job(ijob : int, item : str):
                sample = h5py.File(item, 'r')
                try:
                    sample = self.transforms(sample)
                except ValueError as e:
                    print(f"msg={repr(e)}, index={ijob}, file={self._sample_list[ijob]}")
                self.cache[ijob] = sample
            pqdm(enumerate(self._sample_list), job, n_jobs=self.n_jobs, desc = 'Data Preprocessing', argument_type='args')
            self.cache = [self.cache[i] for i in range(len(self.cache))]


    def __getitem__(self, index: int):
        if self.load_from_memory:
            sample = self.cache[index]
            return sample
        
        file = h5py.File(self._sample_list[index], 'r')
        try:
            sample = self.transforms(file)
        except ValueError as e:
            print(f"msg={repr(e)}, index={index}, file={self._sample_list[index]}")
        return sample
    

    def __getitems__(self, index_list: list):
        samples = []
        if self.load_from_memory:
            for index in index_list:
                sample = self.cache[index]
                samples.append(sample)
            return samples
        def job(index: int):
            file = h5py.File(self._sample_list[index], 'r')
            try:
                sample = self.transforms(file)
            except ValueError as e:
                print(f"msg={repr(e)}, index={index}, file={self._sample_list[index]}")
            return sample
        # with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        #     for sample in executor.map(job, index_list):
        #         samples.append(sample)
        for index in index_list:
            sample = job(index)
            samples.append(sample)
        return samples
    

    def __len__(self):
        return len(self._sample_list)
