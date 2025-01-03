import torch
from torch.utils.data import Dataset

from random import shuffle
import numpy as np

import time

import os
import uuid

def generate_random_uuid():
    return str(uuid.uuid4())

# ismrmrd_header rss csm kspace
class Fastmri_320p(Dataset):
    def __init__(self, filelist, num_subset = None, transform=None, memory_pin = None, disk_cache = False, debug = False, no_slices : list = ['ismrmrd_header']):
        """
        Note:   
        memory_pin must be None while it works with torch dataloader num_workers > 0.
        """
        super(Fastmri_320p, self).__init__()
        self.filelist = filelist
        self.num_subset = num_subset
        self.transform = transform
        self.memory_pin = memory_pin
        self.disk_cache = disk_cache
        self.debug = debug
        self.no_slices = no_slices

        self.extension_disk_cache = f".{self.disk_cache}._cached" if self.disk_cache and isinstance(self.disk_cache, str) else "._cached"

        if self.num_subset is not None:
            self.filelist = self.filelist[:self.num_subset]

        self._cache = {} if memory_pin is not None else None

        if memory_pin is not None:
            if memory_pin == "whole": # try to load all the data into memory
                try:
                    for file in filelist:
                        path, index = file
                        if path not in self._cache:
                            self.load_warm(path)
                except MemoryError as e: # The total size of the dataset is about 400GB
                    print("Memory Error in preloading the dataset.", e)
                    print("Lazy memory is recommended. Automatically switch to lazy memory.")
                    self.release(ratio=0.75)
                    self.memory_pin = "lazy" # switch to lazy memory
            elif memory_pin == "lazy":
                pass 

    def mnemonic_disk_cache(self, path):
        """
        Get the name of the disk cache file.
        """
        return path + self.extension_disk_cache

    def clean_disk_cache(self):
        """
        Clean the disk cache files.
        """
        if self.disk_cache and isinstance(self.disk_cache, str):
            print(f"WARNING: This operation will remove all disk cache files with the extension '.{self.disk_cache}._cached' in the filelist.")
        else:
            print(f"WARNING: This operation will remove all disk cache files with the extension '._cached' in the filelist.")
        for path, idx in self.filelist:
            cache_path = self.mnemonic_disk_cache(path)
            if os.path.exists(cache_path):
                os.remove(cache_path)

    def load_cold(self, path):
        """
        Load the data (or cache) directly from the disk without memory cache.
        """
        cache_path = self.mnemonic_disk_cache(path)

        if self.disk_cache:
            if os.path.exists(cache_path):
                _begin_time = time.time()
                try:
                    data = torch.load(cache_path, weights_only =False)
                except Exception as e:
                    print(f"Error in loading the cache {cache_path} without memory_pin.", e)
                    raise e
                if self.debug:
                    print(f'loading the cache {cache_path} without memory_pin, cost: {time.time() - _begin_time}')
                return data
            
        _begin_time = time.time()
        with np.load(path, 'r') as data:
            data = self.transform(data)

            if self.debug:
                print(f'loading the data {path} without memory_pin, cost: {time.time() - _begin_time}')

            if self.disk_cache:
                _begin_time = time.time()
                _tmp_path = cache_path + "." + generate_random_uuid() + ".tmp"
                torch.save(data, _tmp_path, _use_new_zipfile_serialization=False)
                if os.path.exists(cache_path): # in order to avoid two processes writing the same file
                    os.remove(_tmp_path)
                    if self.debug:
                        print(f"Warning: The cache file {cache_path} already exists. The temporary file {_tmp_path} is removed.")
                else:
                    os.rename(_tmp_path, cache_path)
                if self.debug:
                    print(f'saving the cache {cache_path} without memory_pin, cost: {time.time() - _begin_time}')

            return data

    def load_warm(self, path):
        """
        Load the data from the file path while caching the data in memory.
        Note: This function will cause Out of Memory Error if it works with torch dataloader num_workers > 0.
        """
        cache_path = self.mnemonic_disk_cache(path)

        if self.disk_cache: # load from disk cache
            if os.path.exists(cache_path):
                _begin_time = time.time()
                try:
                    self._cache[path] = torch.load(cache_path, weights_only =False)
                except Exception as e:
                    print(f"Error in loading the cache {cache_path}.", e)
                    raise e
                if self.debug:
                    print(f'loading the cache {cache_path} cost: {time.time() - _begin_time}')
                return self._cache[path]

        _begin_time = time.time()
        with np.load(path, 'r') as data:
            self._cache[path] = self.transform(data)

            if self.debug:
                print(f'loading the data {path} cost: {time.time() - _begin_time}')

            if self.disk_cache: # dump to disk cache
                _begin_time = time.time()
                _tmp_path = cache_path + "." + generate_random_uuid() + ".tmp"
                torch.save(self._cache[path], _tmp_path, _use_new_zipfile_serialization=False)
                if os.path.exists(cache_path): # in order to avoid two processes writing the same file
                    os.remove(_tmp_path)
                    if self.debug:
                        print(f"Warning: The cache file {cache_path} already exists. The temporary file {_tmp_path} is removed.")
                else:
                    os.rename(_tmp_path, cache_path)
                if self.debug:
                    print(f'saving the cache {cache_path} cost: {time.time() - _begin_time}')

            return self._cache[path]
        
    def load(self, path):
        """
        Load the data from the file path.
        """
        if self.memory_pin is None:
            return self.load_cold(path)
        elif self.memory_pin == "whole" or self.memory_pin == "lazy":
            if path in self._cache:
                return self._cache[path]
            else:
                try:
                    data = self.load_warm(path)
                except MemoryError as e:
                    self.release()
                    data = self.load_warm(path)
                return data
        else:
            raise ValueError("The memory_pin must be None, 'whole' or 'lazy'.")


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
        data = self.load(path)
        return {key: value[index] if key not in self.no_slices else value for key, value in data.items()}