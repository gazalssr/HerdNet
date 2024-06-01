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
#     ''' Samples elements from two image-level categories (C0 and C1) and returns batches
#     consisting of the same number of elements for each domain.
    
#     The batch size must be even and the csv file on which the dataset has been 
#     built must contain a column defining the two categories, i.e. C0 (0) and C1 (1).
#     '''

#     def __init__(
#         self, 
#         dataset: CSVDataset, 
#         col: str, 
#         batch_size: int = 2, 
#         shuffle: bool = False,
#         *args, **kwargs
#         ) -> None:
#         '''
#         Args:
#             dataset (CSVDataset): dataset from which to sample data. Must be a CSVDataset.
#             col (str): dataset's DataFrame column defining categories C0 and C1.
#             batch_size (int, optional): how many samples per batch to load. Defaults to 2.
#             shuffle (bool, optional): set to True to have the data reshuffled at every epoch.
#                 Defaults to False.
#         '''
#         super().__init__(dataset)  

#         if not isinstance(dataset, CSVDataset):
#             raise TypeError(
#                 f"dataset should be an instance of 'CSVDataset' class, but got '{type(dataset)}'"
#                 )
        
#         if batch_size % 2 != 0:
#             raise ValueError(f"batch size should be even, but got {batch_size}")

#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#         df = self.dataset.data.copy()
#         df = group_by_image(df)
#         self.col = col
#         if  self.col not in df.columns:
#             raise ValueError(f"'{col}' column is missing from the csv file")

#         self.n = self.batch_size // 2
#         self.c0_idx = df.loc[df[col]==0].index.values.tolist()
#         self.c1_idx = df.loc[df[col]==1].index.values.tolist()
        
#         c0_idx, c1_idx = self._grouped(self.c0_idx, n=self.n), self._grouped(self.c1_idx, n=self.n)
#         self.batch_idx = [[*c0, *c1] for c0, c1 in zip(c0_idx, c1_idx)]
    
#     def __iter__(self) -> Iterator:

#         if self.shuffle:
#             seed = int(torch.empty((), dtype=torch.int64).random_().item())
#             generator = torch.Generator()
#             generator.manual_seed(seed)

#             c0_idx = [self.c0_idx[i] for i in torch.randperm(len(self.c0_idx), generator=generator)]
#             c1_idx = [self.c1_idx[i] for i in torch.randperm(len(self.c1_idx), generator=generator)]
#             c0_idx, c1_idx = self._grouped(c0_idx, n=self.n), self._grouped(c1_idx, n=self.n)
#             batch_idx = [[*c0, *c1] for c0, c1 in zip(c0_idx, c1_idx)]

#             yield from batch_idx
        
#         else:
#             yield from self.batch_idx
    
#     def __len__(self) -> int:
#         return len(self.batch_idx)
    
#     def _grouped(self, iterable: Iterable, n: int) -> Iterable:
#         return zip(*[iter(iterable)]*n)
######################## NEWWWWW
class BinaryBatchSampler(Sampler):
    def __init__(self, dataset, col, batch_size=2, shuffle=False):
        super().__init__(dataset)
        if not isinstance(dataset, BinaryFolderDataset):  ##########
            raise TypeError(f"dataset should be an instance of 'BinaryFolderDataset' class, but got '{type(dataset)}'")
        if batch_size % 2 != 0:
            raise ValueError(f"batch size should be even, but got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        df = self.dataset.data.copy()
        df = group_by_image(df)
        self.col = col
        if self.col not in df.columns:
            raise ValueError(f"'{col}' column is missing from the csv file")

        self.n = self.batch_size // 2
        self.c0_idx = df.loc[df[col] == 0].index.values.tolist()
        self.c1_idx = df.loc[df[col] == 1].index.values.tolist()

        # Ensure there are equal numbers of samples in each category by undersampling the majority class
        self.c0_idx = self._undersample(self.c0_idx, len(self.c1_idx))
        self.c1_idx = self._undersample(self.c1_idx, len(self.c0_idx))

        print(f"After undersampling, c0_idx length: {len(self.c0_idx)}, c1_idx length: {len(self.c1_idx)}")

    def __iter__(self) -> Iterator:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

            c0_idx = [self.c0_idx[i] for i in torch.randperm(len(self.c0_idx), generator=generator)]
            c1_idx = [self.c1_idx[i] for i in torch.randperm(len(self.c1_idx), generator=generator)]
        else:
            c0_idx = self.c0_idx
            c1_idx = self.c1_idx

        # Create balanced batches
        batch_idx = []
        min_len = min(len(c0_idx), len(c1_idx))
        for i in range(0, min_len, self.n):
            batch_c0 = c0_idx[i:i+self.n]
            batch_c1 = c1_idx[i:i+self.n]
            if len(batch_c0) == self.n and len(batch_c1) == self.n:
                batch_idx.append(batch_c0 + batch_c1)

        if self.shuffle: 
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            batch_idx = [batch_idx[i] for i in torch.randperm(len(batch_idx), generator=generator)]

        print(f"Batch_idx length: {len(batch_idx)}")
        for batch in batch_idx:
            yield batch

    def __len__(self) -> int:
        return len(self.c0_idx) // self.n

    def _undersample(self, indices: List[int], target_length: int) -> List[int]:
        if len(indices) <= target_length:
            return indices
        else:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            undersampled_indices = [indices[i] for i in torch.randperm(len(indices), generator=generator)[:target_length]]
            return undersampled_indices


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