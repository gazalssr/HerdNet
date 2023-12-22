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

from typing import Optional, List, Any, Dict, Tuple, Union

from ..data.types import BoundingBox
from ..data.utils import group_by_image

from .register import DATASETS

from .csv import CSVDataset

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
        
        return target
###################################################################################
class BalancedFolderDataset(Dataset):
    def __init__(
        self, 
        csv_file: str, 
        root_dir: str, 
        gt_file: str,
        num_non_empty_samples: int,
        transform: Optional[callable] = None
    ) -> None:
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.gt_data = pd.read_csv(gt_file)
        self.num_non_empty_samples = num_non_empty_samples
        self.transform = transform

        # Separate empty and non-empty patches
        self.empty_patches = self.data[self.data['label'] == 0]
        self.non_empty_patches = self.data[self.data['label'] == 1]

        # Assuming equal number of empty and non-empty patches
        self.num_samples = min(len(self.empty_patches), len(self.non_empty_patches))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Randomly sample one empty and num_non_empty_samples non-empty patches in each iteration
        empty_patch = self.empty_patches.sample(1)
        non_empty_patches = self.non_empty_patches.sample(self.num_non_empty_samples)

        # Extract patch IDs
        empty_patch_id = empty_patch.iloc[0]['images']
        non_empty_patch_ids = non_empty_patches['images'].tolist()

        # Load images and targets
        empty_image = Image.open(os.path.join(self.root_dir, f"{empty_patch_id}.jpg")).convert('RGB')
        non_empty_images = [Image.open(os.path.join(self.root_dir, f"{patch_id}.jpg")).convert('RGB') for patch_id in non_empty_patch_ids]

        # Extract targets from the ground truth file
        empty_target = self.gt_data[self.gt_data['images'] == empty_patch_id]
        non_empty_targets = self.gt_data[self.gt_data['images'].isin(non_empty_patch_ids)]

        # Apply initial transformations if specified
        if self.transform:
            empty_image = self.transform(empty_image)
            non_empty_images = [self.transform(image) for image in non_empty_images]

        return {
            'empty_image': empty_image,
            'non_empty_images': non_empty_images,
            'empty_target': empty_target,
            'non_empty_targets': non_empty_targets
        }

# Example usage:
csv_file = 'path/to/your/csv/file.csv'
root_dir = 'path/to/your/patches/folder'
gt_file = 'path/to/your/ground/truth/file.csv'
num_non_empty_samples = 1000
transform = transforms.Compose([transforms.ToTensor()])

dataset = BalancedFolderDataset(csv_file, root_dir, gt_file, num_non_empty_samples, transform=transform)

# Use DataLoader as usual
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
