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
##########################################
import os
import numpy as np
import pandas as pd
import PIL.Image
from torch.utils.data import Dataset

@DATASETS.register()
class BalancedFolderDataset(CSVDataset):
    def __init__(
        self, 
        csv_file: str, 
        root_dir: str, 
        albu_transforms: Optional[list] = None,
        end_transforms: Optional[list] = None
        ) -> None:
        super(BalancedFolderDataset, self).__init__(csv_file, root_dir, albu_transforms, end_transforms)

        self.folder_images = [i for i in os.listdir(self.root_dir) if i.endswith(('.JPG', '.jpg', '.JPEG', '.jpeg'))]
        self.anno_keys = self.data.columns
        self.data['from_folder'] = 0  # all images in the folder

        folder_only_images = np.setdiff1d(self.folder_images, self.data['images'].unique().tolist())
        folder_df = pd.DataFrame(data=dict(images=folder_only_images))
        folder_df['from_folder'] = 1  # images without annotations

        self.data = pd.concat([self.data, folder_df], ignore_index=True).convert_dtypes()
        # Use group_by_image to organize or group the image data
        self._ordered_img_names = group_by_image(self.data)['images'].values.tolist()
        # Separate empty and non-empty patches based on 'from_folder'
        self.non_empty_patches = self.data[self.data['from_folder'] == 0]['images'].tolist()
        self.empty_patches = self.data[self.data['from_folder'] == 1]['images'].tolist()

        # Initialize the dataset for the first epoch
        self.refresh_samples()

    def refresh_samples(self):
        if len(self.non_empty_patches) <= len(self.empty_patches):
            sampled_empty = np.random.choice(self.empty_patches, len(self.non_empty_patches), replace=False).tolist()
        else:
            sampled_empty = np.random.choice(self.empty_patches, len(self.non_empty_patches), replace=True).tolist()

        # Combine and shuffle
        self.sampled_patches = self.non_empty_patches + sampled_empty
        np.random.shuffle(self.sampled_patches)

    def _load_image(self, img_name: str) -> PIL.Image.Image:
        img_path = os.path.join(self.root_dir, img_name)
        try:
            pil_img = PIL.Image.open(img_path).convert('RGB')
            pil_img.filename = img_name
            return pil_img
        except (IOError, FileNotFoundError) as e:
            print(f"Error opening image {img_name}: {e}")
            # Handle error as appropriate (e.g., return a default image or skip this sample)

    def _load_target(self, img_name: str) -> Dict[str, List[Any]]:
        annotations = self.data[self.data['images'] == img_name]
        anno_keys = list(self.anno_keys)
        anno_keys.remove('images')   
        target = {
            'image_id': [self.sampled_patches.index(img_name)], 
            'image_name': [img_name]
        }

        nan_in_annos = annotations[anno_keys].isnull().values.any()
        if not nan_in_annos:
            for key in anno_keys:
                # Check if 'annos' key exists and handle accordingly
                if key == 'annos' and hasattr(annotations[key], 'get_tuple'): 
                    target.update({key: [list(a.get_tuple) for a in annotations[key]]})
                else:
                    target.update({key: list(annotations[key])})
        else:
            for key in anno_keys:
                if self.anno_type == 'BoundingBox':
                    if key == 'annos':  
                        target.update({key: [[0,1,2,3]]})
                    elif key == 'labels':
                        target.update({key: [0]})
                else:        
                    target.update({key: []})

        return target

    def __getitem__(self, index: int):
        img_name = self.sampled_patches[index]
        img = self._load_image(img_name)
        target = self._load_target(img_name)
        return img, target

    def __len__(self):
        return len(self.sampled_patches)

    def on_epoch_end(self):
        # Refresh samples at the end of each epoch
        self.refresh_samples()

    
########################## BinaryFolderDataset (new) #######################
class BinaryFolderDataset(CSVDataset):
    def __init__(self, csv_file, root_dir, albu_transforms=None, end_transforms=None):
        super(BinaryFolderDataset, self).__init__(csv_file, root_dir, albu_transforms, end_transforms)
        self.create_binary_dataframe()
        self.anno_type = 'binary'  # Specify the annotation type as 'Binary'
        self.anno_keys = ['binary']
        self._ordered_img_names = group_by_image(self.data)['images'].values.tolist()
    def create_binary_dataframe(self):
        # Get all image files in the root directory
        self.folder_images = [i for i in os.listdir(self.root_dir) if i.endswith(('.JPG', '.jpg', '.JPEG', '.jpeg'))]
        
        # Mark all current images with 'from_folder' = 0 
        self.data['from_folder'] = 0

        # Identify images in the folder that are not in the CSV annotations
        folder_only_images = numpy.setdiff1d(self.folder_images, self.data['images'].unique().tolist())
        
        # Create a new dataframe for folder-only images with 'from_folder' = 1
        folder_df = pandas.DataFrame({'images': folder_only_images, 'from_folder': 1, 'binary': 0})
        # Derive 'base_images' IDs for empty patches
        base_images_ids = [self.derive_base_image_id(image_name) for image_name in folder_only_images]
        # Create a DataFrame for folder-only images, assigning derived 'base_images' IDs
        folder_df = pandas.DataFrame({
            'images': folder_only_images,
            'base_images': base_images_ids,
            'from_folder': 1,
            'binary': 0
        })
        # Combine CSV data with folder-only images
        self.data = pandas.concat([self.data, folder_df], ignore_index=True)

        # Set 'binary' = 1 for all rows that are not marked as 'from_folder' = 1
        self.data['binary'] = numpy.where(self.data['from_folder'] == 1, 0, 1)

        # Remove specified columns that are not needed
        columns_to_remove = ['annos', 'subset', 'from_folder', 'labels', 'species']
        self.data.drop(columns=columns_to_remove, errors='ignore', inplace=True)

        # Handle duplicates for non-empty patches (binary = 1)
        non_empty_patches = self.data[self.data['binary'] == 1]
        unique_non_empty_patches = non_empty_patches.drop_duplicates(subset=['images'])

        # Reassemble the DataFrame to only include unique non-empty patches and all empty patches
        empty_patches = self.data[self.data['binary'] == 0]
        self.data = pandas.concat([unique_non_empty_patches, empty_patches], ignore_index=True)

    def _load_image(self, index: int) -> Image.Image:
        img_name = self.data.at[index, 'images']
        img_path = os.path.join(self.root_dir, img_name)
        pil_img = Image.open(img_path).convert('RGB')
        pil_img.filename = img_name  
        return pil_img
    
    # def _load_target(self, index: int) -> dict:
    #     # Always return a dictionary, even if it's for an image without annotations
        
    #     if self.data.at[index, 'binary'] == 1:
    #         # Return target for images with annotations
    #         target = {'binary': 1}  
    #     else:
    #         # Return empty or default target for images without annotations
    #         target = {'binary': 0}  
    #     return target
    def _load_target(self, index: int) -> dict:
        img_name = self._ordered_img_names[index]
        

        binary_label = self.data.loc[self.data['images'] == img_name, 'binary'].iloc[0]
        # Construct the target dictionary.
        target = {
            'image_id': index,
            'image_name': img_name,
            'binary': torch.tensor([binary_label], dtype=torch.int64)  # Ensure binary label is a tensor
        }

        return target

    
    def derive_base_image_id(self, image_name):
        # Remove the patch number from the end of 'images' filenames
        if "_" in image_name:
            parts = image_name.rsplit("_", 1)  # Split from the right on the last underscore
            base_image_id = parts[0] + '.jpg'  # Reattach '.jpg' to form the base image ID
        else:
            base_image_id = image_name  # Handle edge cases where the expected pattern does not match
        return base_image_id
    
    def save_binary_csv(self, save_path):
        # Save the DataFrame to a CSV file without the removed columns
        self.data.to_csv(save_path, index=False)