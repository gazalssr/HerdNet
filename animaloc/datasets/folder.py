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


import os
import PIL
import pandas 
import numpy 
import torch

from typing import Optional, List, Any, Dict, Tuple, Union
from ..data.types import BoundingBox
from ..data.utils import group_by_image

from .register import DATASETS

from .csv import CSVDataset
from PIL import Image
@DATASETS.register()
class FolderDataset(CSVDataset):
    ''' Class to create a dataset from a folder containing images only, and a CSV file 
    containing annotations.

    This dataset is built on the basis of CSV files containing box coordinates, in 
    [x_min, y_min, x_max, y_max] format, or point coordinates in [x,y] format.

    All images that do not have corresponding annotations in the CSV file are considered as 
    background images. In this case, the dataset will return the image and empty target 
    (i.e. empty lists).

    The type of annotations is automatically detected internally. The only condition 
    is that the file contains at least the keys ['images', 'x_min', 'y_min', 'x_max', 
    'y_max', 'labels'] for the boxes and, ['images', 'x', 'y', 'labels'] for the points. 
    Any additional information (i.e. additional columns) will be associated and returned 
    by the dataset.

    If no data augmentation is specified, the dataset returns the image in PIL format 
    and the targets as lists. If transforms are specified, the conversion to torch.Tensor
    is done internally, no need to specify this. 
    '''

    def __init__(
        self, 
        csv_file: str, 
        root_dir: str, 
        albu_transforms: Optional[list] = None,
        end_transforms: Optional[list] = None
        ) -> None:
        ''' 
        Args:
            csv_file (str): absolute path to the csv file containing 
                annotations
            root_dir (str) : path to the images folder
            albu_transforms (list, optional): an albumentations' transformations 
                list that takes input sample as entry and returns a transformed 
                version. Defaults to None.
            end_transforms (list, optional): list of transformations that takes
                tensor and expected target as input and returns a transformed
                version. These will be applied after albu_transforms. Defaults
                to None.
        '''

        super(FolderDataset, self).__init__(csv_file, root_dir, albu_transforms, end_transforms)

        self.folder_images = [i for i in os.listdir(self.root_dir) 
                                if i.endswith(('.JPG','.jpg','.JPEG','.jpeg'))]
    
        self._img_names = self.folder_images 
        self.anno_keys = self.data.columns
        self.data['from_folder'] = 0 # all images in the folder
        
        folder_only_images = numpy.setdiff1d(self.folder_images, self.data['images'].unique().tolist())
        folder_df = pandas.DataFrame(data=dict(images = folder_only_images))
        folder_df['from_folder'] = 1 # some have annotations

        self.data = pandas.concat([self.data, folder_df], ignore_index=True).convert_dtypes()

        self._ordered_img_names = group_by_image(self.data)['images'].values.tolist()

    def _load_image(self, index: int) -> PIL.Image.Image:
        img_name = self._ordered_img_names[index]
        img_path = os.path.join(self.root_dir, img_name)

        pil_img = PIL.Image.open(img_path).convert('RGB')
        pil_img.filename = img_name

        return pil_img

    def _load_target(self, index: int) -> Dict[str,List[Any]]:
        img_name = self._ordered_img_names[index]
        annotations = self.data[self.data['images'] == img_name]
        anno_keys = list(self.anno_keys)
        anno_keys.remove('images')

        target = {
        'image_id': [index], 
        'image_name': [img_name]
        }

        nan_in_annos =  annotations[anno_keys].isnull().values.any()
        if not nan_in_annos:
            for key in anno_keys:
                target.update({key: list(annotations[key])})

                if key == 'annos': 
                    target.update({key: [list(a.get_tuple) for a in annotations[key]]})

        else:
            for key in anno_keys:
                if self.anno_type == 'BoundingBox':
                    if key == 'annos':  
                        target.update({key: [[0,1,2,3]]})
                    elif key == 'labels':
                        target.update({key: [0]})
                else:        
                    target.update({key: []})
                    ##############
                    
        
        return target
############################# NEW Binaryfolderdataset that incorporates image_IDs into the targets #############

@DATASETS.register()
class BinaryFolderDataset(CSVDataset):
    def __init__(self, csv_file, root_dir, albu_transforms=None, end_transforms=None, preprocess=False):
        super(BinaryFolderDataset, self).__init__(csv_file, root_dir, albu_transforms, end_transforms)
        if preprocess:
            self.create_binary_dataframe()
        else:
            self.data = pandas.read_csv(csv_file)
        
        self.data.reset_index(drop=True, inplace=True)  # Reset the DataFrame index
        
        self.anno_type = 'binary'
        self._ordered_img_names = self.data['images'].values.tolist()

    def create_binary_dataframe(self):
        self.folder_images = [i for i in os.listdir(self.root_dir) if i.endswith(('.JPG', '.jpg', '.JPEG', '.jpeg'))]
        unmatched_images = [img for img in self.folder_images if img not in self.data['images'].tolist()]
        self.data = self.data.drop_duplicates(subset='images')
        self.data['binary'] = 1
        if 'base_images' not in self.data.columns:
            self.data['base_images'] = self.data['images'].apply(self.derive_base_image_id)
        folder_only_images = numpy.setdiff1d(self.folder_images, self.data['images'].tolist())
        folder_df = pandas.DataFrame({
            'images': folder_only_images,
            'binary': numpy.zeros(len(folder_only_images), dtype=int),
            'base_images': [self.derive_base_image_id(img) for img in folder_only_images]
        })
        self.data = pandas.concat([self.data, folder_df], ignore_index=True)
        self.data.drop(columns=['annos', 'subset', 'from_folder', 'labels', 'species'], errors='ignore', inplace=True)
        self.data.reset_index(drop=True, inplace=True)  # Reset the DataFrame index after modifications
        empty_patches_count = len(folder_only_images)
        non_empty_patches_count = len(self.data[self.data['binary'] == 1])
        print(f"Number of empty patches: {empty_patches_count}")
        print(f"Number of non-empty patches: {non_empty_patches_count}")

    def _load_image(self, index: int) -> Image.Image:
        img_name = self.data.at[index, 'images']
        img_path = os.path.join(self.root_dir, img_name)
        return Image.open(img_path).convert('RGB')
  
    def _load_target(self, index: int) -> Dict[str, Any]:
        img_name = self.data.at[index, 'images']
        binary_label = self.data.at[index, 'binary']
        return {
            'image_name': img_name,
            'binary': torch.tensor([binary_label], dtype=torch.int64)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(index, list):  ##### Check if the index is a list
            images = []
            targets = {'image_name': [], 'binary': []}
            for idx in index:
                image, target = self._get_single_item(idx)
                images.append(image)
                targets['image_name'].append(target['image_name'])
                targets['binary'].append(target['binary'])
            targets['binary'] = torch.stack(targets['binary'])
            return images, targets
        else:
            return self._get_single_item(index)

    def _get_single_item(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not isinstance(index, int):
            raise TypeError(f"Index {index} is not an integer")

        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} is out of bounds for dataset with length {len(self.data)}")

        image = self._load_image(index)
        target = self._load_target(index)
        if self.albu_transforms:
            augmented = self.albu_transforms(image=numpy.array(image))
            image = augmented['image']
        if self.end_transforms:
            for transform in self.end_transforms:
                image, target = transform(image, target)
        return image, target

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        binary_targets = torch.stack([item[1]['binary'] for item in batch])
        image_names = [item[1]['image_name'] for item in batch]
        return images, {'binary': binary_targets, 'image_name': image_names}

    def derive_base_image_id(self, image_name):
        if "_" in image_name:
            parts = image_name.rsplit("_", 1)
            return parts[0] + '.jpg'
        return image_name

    def save_binary_csv(self, save_path):
        self.data.to_csv(save_path, index=False)