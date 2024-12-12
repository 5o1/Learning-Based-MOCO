import torch
from torch.utils.data import Dataset

from random import shuffle
import numpy as np

import os

# ismrmrd_header rss csm kspace
class Fastmri_320p(Dataset):
    def __init__(self, filelist, num_subset = None, transform=None, lazy_memory=True, output_keys=["kspace"], output_type = dict, disk_cache = False):
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