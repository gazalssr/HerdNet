__copyright__ = \
    """
    Copyright (C) 2022 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the CC BY-NC-SA-4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/). 
    It is to be used for academic research purposes only, no commercial use is permitted.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 29, 2023
    """
__author__ = "Alexandre Delplanque"
__license__ = "CC BY-NC-SA 4.0"
__version__ = "0.2.0"
import numpy
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Sampler
from typing import List, Iterable, Iterator

from ..datasets import CSVDataset, BinaryFolderDataset
from ..data.utils import group_by_image

from ..utils.registry import Registry

SAMPLERS = Registry('samplers', module_key='animaloc.data.samplers')

__all__ = ['SAMPLERS', *SAMPLERS.registry_names]

@SAMPLERS.register()
# class BinaryBatchSampler(Sampler):
class BinaryBatchSampler(Sampler):
    def __init__(self, dataset, col, batch_size=16, shuffle=False):
        super().__init__(dataset)
        if not isinstance(dataset, BinaryFolderDataset):
            raise TypeError(f"dataset should be an instance of 'BinaryFolderDataset' class, but got '{type(dataset)}'")
        if batch_size % 2 != 0:
            raise ValueError(f"batch size should be even, but got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data = self.dataset.data
        if col not in self.data.columns:
            raise ValueError(f"'{col}' column is missing from the data")

        self.n = self.batch_size // 2
        self.c0_idx = self.data.loc[self.data[col] == 0].index.values.tolist()
        self.c1_idx = self.data.loc[self.data[col] == 1].index.values.tolist()

        # Ensure there are equal numbers of samples in each category by undersampling the majority class
        self.c0_idx = self._undersample(self.c0_idx, len(self.c1_idx))
        self.c1_idx = self._undersample(self.c1_idx, len(self.c0_idx))

        print(f"Initialized BinaryBatchSampler with batch_size: {self.batch_size}")
        print(f"After undersampling, c0_idx length: {len(self.c0_idx)}, c1_idx length: {len(self.c1_idx)}")

    def _undersample(self, indices, target_size):
        return indices[:target_size] if len(indices) > target_size else indices

    def __iter__(self):
        if self.shuffle:
            numpy.random.shuffle(self.c0_idx)
            numpy.random.shuffle(self.c1_idx)

        c0_batches = [self.c0_idx[i:i + self.n] for i in range(0, len(self.c0_idx), self.n)]
        c1_batches = [self.c1_idx[i:i + self.n] for i in range(0, len(self.c1_idx), self.n)]

        batches = []
        for c0_batch, c1_batch in zip(c0_batches, c1_batches):
            batch = c0_batch + c1_batch
            if len(batch) < self.batch_size:
                continue
            if self.shuffle:
                numpy.random.shuffle(batch)
            batches.append(batch)

        print(f"Generated {len(batches)} batches with batch_size: {self.batch_size}")
        return iter(batches)

    def __len__(self):
        return len(self.c0_idx) // self.n  # Number of batches

    def _undersample(self, indices: List[int], target_length: int) -> List[int]:
        if len(indices) <= target_length:
            return indices
        else:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            undersampled_indices = [indices[i] for i in torch.randperm(len(indices), generator=generator)[:target_length]]
            return undersampled_indices
######################## NEWWWWW
# class BinaryBatchSampler(Sampler):
#     def __init__(self, dataset, col, batch_size, shuffle=True):
#         self.dataset = dataset
#         self.col = col
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.c0_idx = [i for i, val in enumerate(dataset.data[col]) if val == 0]
#         self.c1_idx = [i for i, val in enumerate(dataset.data[col]) if val == 1]
        
#         self.valid_indices = set(range(len(self.dataset)))
#         self.c0_idx = [idx for idx in self.c0_idx if idx in self.valid_indices]
#         self.c1_idx = [idx for idx in self.c1_idx if idx in self.valid_indices]
    
#     def __iter__(self):
#         if self.shuffle:
#             numpy.random.shuffle(self.c0_idx)
#             numpy.random.shuffle(self.c1_idx)
        
#         c0_batches = [self.c0_idx[i:i+self.batch_size//2] for i in range(0, len(self.c0_idx), self.batch_size//2)]
#         c1_batches = [self.c1_idx[i:i+self.batch_size//2] for i in range(0, len(self.c1_idx), self.batch_size//2)]
        
#         balanced_batches = []
#         for c0_batch, c1_batch in zip(c0_batches, c1_batches):
#             batch = c0_batch + c1_batch
#             balanced_batches.append(batch)
        
#         if self.shuffle:
#             numpy.random.shuffle(balanced_batches)
        
#         return iter(balanced_batches)
    
#     def __len__(self):
#         return len(self.c0_idx) // (self.batch_size//2)

#     def _undersample(self, indices: List[int], target_length: int) -> List[int]:
#         if len(indices) <= target_length:
#             return indices
#         else:
#             generator = torch.Generator()
#             generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
#             undersampled_indices = [indices[i] for i in torch.randperm(len(indices), generator=generator)[:target_length]]
#             return undersampled_indices


class DataAnalyzer:
    def __init__(self):
        pass

    def analyze_batch_composition(self, dataloader, num_batches=10):
        empty_counts = []
        non_empty_counts = []
        for i, (images, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(targets, dict):
                targets = targets['binary'] 
            elif isinstance(targets, torch.Tensor):
                targets = targets
            else:
                raise ValueError("Unexpected target format")
            
            # Detailed debug prints
            print(f"Batch {i+1} targets: {targets.numpy()}")  # Debug print
            batch_empty_count = (targets == 0).sum().item()
            batch_non_empty_count = (targets == 1).sum().item()
            
            print(f"Batch {i+1}: Empty count = {batch_empty_count}, Non-empty count = {batch_non_empty_count}")  # Debug print
            
            empty_counts.append(batch_empty_count)
            non_empty_counts.append(batch_non_empty_count)
        return empty_counts, non_empty_counts

    def analyze_precision_trend(self, precision_values, save_path):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(precision_values, label='Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.title('Precision Trend Over Epochs')
        plt.legend()
        plt.savefig(save_path)
        plt.show()