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

        # print(f"Initialized BinaryBatchSampler with batch_size: {self.batch_size}")
        # print(f"After undersampling, c0_idx length: {len(self.c0_idx)}, c1_idx length: {len(self.c1_idx)}")

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
####################################################
class BalancedMarginSampler(Sampler):
    def __init__(self, dataset, col, batch_size=16, shuffle=False, ignore_margins=True, black_threshold=0.02, white_threshold=0.02):
        super().__init__(dataset)
        if not isinstance(dataset, BinaryFolderDataset):
            raise TypeError(f"dataset should be an instance of 'BinaryFolderDataset', but got '{type(dataset)}'")
        if batch_size % 2 != 0:
            raise ValueError(f"batch size should be even, but got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ignore_margins = ignore_margins  # Whether to ignore patches with black/white margins
        self.black_threshold = black_threshold
        self.white_threshold = white_threshold

        self.data = self.dataset.data
        if col not in self.data.columns:
            raise ValueError(f"'{col}' column is missing from the data")

        self.n = self.batch_size // 2

        # Filter out margin patches if ignore_margins=True
        if self.ignore_margins:
            self.c0_idx, self.c1_idx = self._filter_margin_and_balance_patches()
        else:
            self.c0_idx = self.data.loc[self.data[col] == 0].index.values.tolist()
            self.c1_idx = self.data.loc[self.data[col] == 1].index.values.tolist()

        # Ensure equal samples in both categories (undersampling)
        self.c0_idx = self._undersample(self.c0_idx, len(self.c1_idx))
        self.c1_idx = self._undersample(self.c1_idx, len(self.c0_idx))

    def _filter_margin_and_balance_patches(self):
        """Filter out patches with large black/white areas and return balanced indices."""
        c0_idx_filtered = []
        c1_idx_filtered = []

        for idx in range(len(self.dataset)):
            image, target = self.dataset._get_single_item(idx)  # Load image and target
            binary_label = target['binary'].item()  # Get binary label (0 or 1)
            black_area_ratio = self._calculate_black_area(image)
            white_area_ratio = self._calculate_white_area(image)

            # Only include patches below black/white area thresholds
            if black_area_ratio <= self.black_threshold and white_area_ratio <= self.white_threshold:
                if binary_label == 0:
                    c0_idx_filtered.append(idx)
                elif binary_label == 1:
                    c1_idx_filtered.append(idx)

        return c0_idx_filtered, c1_idx_filtered

    def _calculate_black_area(self, image):
        """Calculate the percentage of black pixels in the image."""
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        if len(image.shape) == 3 and image.shape[0] == 3:
            image = numpy.transpose(image, (1, 2, 0))

        black_pixels = numpy.all(image == [0, 0, 0], axis=-1)
        black_area_ratio = numpy.mean(black_pixels)
        return black_area_ratio

    def _calculate_white_area(self, image):
        """Calculate the percentage of white pixels in the image."""
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        if len(image.shape) == 3 and image.shape[0] == 3:
            image = numpy.transpose(image, (1, 2, 0))

        white_pixels = numpy.all(image == [255, 255, 255], axis=-1)
        white_area_ratio = numpy.mean(white_pixels)
        return white_area_ratio

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

        return iter(batches)

    def __len__(self):
        return len(self.c0_idx) // self.n  # Number of batches