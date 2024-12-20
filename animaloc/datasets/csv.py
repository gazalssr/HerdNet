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
import os
import PIL
import numpy
import albumentations
import torchvision
import torchvision.transforms
from torch.utils.data import Dataset
import albumentations as A
from typing import Any, Dict, List, Optional, Tuple, Union

from .register import DATASETS

from ..data.annotations import AnnotationsFromCSV
from ..data.transforms import SampleToTensor

from ..data import transforms
# Convert labels into tensors
def dict_to_tensor(d: dict) -> Tuple[dict, dict]:
    tensor_params = {}
    types = {}

    for k, v in d.items():
        if isinstance(v, bool):
            tensor_params.update({k: v})
        elif isinstance(v, (int, float)):
            tensor_params.update({k: torch.tensor(v, dtype=torch.float64)})
        else:
            tensor_params.update({k: v})

        types.update({k: type(v)})
    
    return tensor_params, types
# Convert a torch.Tensor back into its original numerical type (int or float)
def retrieve_num_type(num: torch.Tensor, type: type) -> Union[int, float]:
    assert isinstance(num, torch.Tensor)
    if type == int:
        return int(torch.round(num))
    elif type == float:
        return float(num)


@DATASETS.register()
class CSVDataset(Dataset):
    ''' Class to create a Dataset from a CSV file 
    
    This dataset is built on the basis of CSV files containing box coordinates, in 
    [x_min, y_min, x_max, y_max] format, or point coordinates in [x,y] format.

    The type of annotations is automatically detected internally. The only condition 
    is that the file contains at least the keys ['images', 'x_min', 'y_min', 'x_max', 
    'y_max', 'labels'] for the boxes and, ['images', 'x', 'y', 'labels'] for the points
    and 'images', 'binary' for binary annoataions. 
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
        # albu_transforms: Optional[list] = None,
        albu_transforms: Optional[Union[list, A.Compose]] = None,
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
        #################### Added A.compose to assert ######
        assert isinstance(albu_transforms, (list,  A.Compose, type(None))), \
            f'albumentations-transformations must be a list or an A.Compose, got {type(albu_transforms)}'
            # f'albumentations-transformations must be a list, got {type(albu_transforms)}'
            
        assert isinstance(end_transforms, (list, type(None))), \
            f'end-transformations must be a list, got {type(end_transforms)}'

        self.csv_file = csv_file
        self.root_dir = root_dir
        self.albu_transforms = albu_transforms
        self.end_transforms = end_transforms
        # store end parameters for adaloss
        self._store_end_params()

        self.annotations = AnnotationsFromCSV(self.csv_file) #already changed for binary
        self.data = self.annotations.dataframe

        self.anno_type = self.data.annos[0].atype

        used = set()
        self._img_names = [x for x in self.annotations.images 
            if x not in used and (used.add(x) or True)]
    
    def _load_image(self, index: int) -> PIL.Image.Image:
        img_name = self._img_names[index]
        img_path = os.path.join(self.root_dir, img_name)

        return PIL.Image.open(img_path).convert('RGB')
 
    def _load_target(self, index: int) -> Dict[str, Any]:
        img_name = self._img_names[index]
        annotations = self.data[self.data['images'] == img_name]
        annotations = annotations.drop(columns='images')

        target = {
            'image_id': index, 
            'image_name': img_name,
        }

        if self.anno_type == 'binary':
        
            binary_labels = annotations['binary'].tolist() if 'binary' in annotations.columns else [0]
            target['labels'] = binary_labels
        else:
        # For non-binary annotations, iterating over the DataFrame columns and updating the target dict
            for key in annotations.columns:
                if key == 'annos':
                
                    target[key] = [anno.get_tuple() for anno in annotations[key]]
                else:
                # Directly updating the target with other columns as lists.
                    target[key] = annotations[key].tolist()

        return target

    def _transforms(self, image: PIL.Image.Image, target: dict) -> Tuple[torch.Tensor, dict]:
        label_fields = target.copy()
        for key in ['annos', 'image_id', 'image_name']:
            label_fields.pop(key, None)  # Remove keys not needed for transformations

        transformed = {'image': image}  # Default to untransformed image

        if self.albu_transforms:
            albumentations_params = {}
            if 'annos' in target:
                if self.anno_type == 'BoundingBox':
                    albumentations_params['bbox_params'] = albumentations.BboxParams(format='pascal_voc', label_fields=['labels'])
                    transform_pipeline = albumentations.Compose(self.albu_transforms, **albumentations_params)
                elif self.anno_type == 'Point':
                    albumentations_params['keypoint_params'] = albumentations.KeypointParams(format='xy', label_fields=['labels'])
                    transform_pipeline = albumentations.Compose(self.albu_transforms, **albumentations_params)
                elif self.anno_type == 'Binary':

                    transform_pipeline = albumentations.Compose(self.albu_transforms)
                
                # Prepare data for transformation
                albumentations_input = {'image': numpy.array(image)}
                if self.anno_type in ['BoundingBox', 'Point']:
                    annos_type = 'bboxes' if self.anno_type == 'BoundingBox' else 'keypoints'
                    albumentations_input[annos_type] = target['annos']
                
                transformed = transform_pipeline(**albumentations_input)
                transformed['image'] = numpy.asarray(transformed['image'])
                # print("Applied Transformations")
        else:
            transformed['image'] = torchvision.transforms.ToTensor()(image)

        # Convert transformed image and target to tensors
        tr_image, tr_target = SampleToTensor()(transformed['image'], target, anno_type=self.anno_type.lower())

        # Apply any specified end transformations
        if self.end_transforms is not None:
            for trans in self.end_transforms:
                tr_image, tr_target = trans(tr_image, tr_target)

        return tr_image, tr_target

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:        
        img = self._load_image(index)
        target = self._load_target(index)

        tr_img, tr_target = self._transforms(img, target)

        return tr_img, tr_target
    
    def load_end_param(self, end_param: str, value: float) -> None:
        self.end_params[end_param] = value

    def update_end_transforms(self) -> None:
        new_transforms = []
        up_params = self._update_end_params()
        for trans, params in zip(self.end_transforms, up_params):
            name = type(trans).__name__
            new_transforms.append(transforms.__dict__[name](**params))

        self.end_transforms = new_transforms
        self._store_end_params()
    
    def __len__(self) -> int:
        return len(self._img_names)
    
    def _store_end_params(self) -> None:
        self.end_params = {}
        self._end_params_types = []

        if self.end_transforms is not None:
            for trans in self.end_transforms:
                tensor_params, types = dict_to_tensor(trans.__dict__)
                self._end_params_types.append(types)
                self.end_params.update(tensor_params)
    
    def _update_end_params(self) -> list:
        up_params = []
        for trans in self._end_params_types:
            up_dict = {}
            for k, v in trans.items():
                up_num = self.end_params[k]
                if isinstance(self.end_params[k], torch.Tensor):
                    up_num = retrieve_num_type(self.end_params[k], v)

                up_dict.update({k: up_num})
            
            up_params.append(up_dict)
        
        return up_params


