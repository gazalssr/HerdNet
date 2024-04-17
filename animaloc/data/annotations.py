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


import pandas
import warnings

from typing import List, Optional, Union
from operator import itemgetter

from .types import Point, BoundingBox,BinaryAnnotation

__all__ = ['Annotations', 'AnnotationsFromCSV','objects_from_df', 'dict_from_objects']

def objects_from_df(df: pandas.DataFrame) -> List[Union[Point, BoundingBox, BinaryAnnotation]]:
    ''' Function to convert object coordinates to an appropried object type

    Args:
        df (pandas.DataFrame): DataFrame with 
            a header of this type for Point:
                | 'x' | 'y' |
            a header of this type for BoundingBox
                | 'x_min' | 'y_min' | 'x_max' | 'y_max' |
            a header of this type for Binary:
                | 'binary' |
    
    Returns:
        list:
            list of objects (Point, BoundingBox, or int for binary)
    '''

    # Point
    if {'x','y'}.issubset(df.columns):
        data = df[['x','y']]
        objects = [Point(r.x, r.y) for i,r in data.iterrows()]

    # BoundingBox
    elif {'x_min','y_min','x_max','y_max'}.issubset(df.columns):
        data = df[['x_min','y_min','x_max','y_max']]
        objects = [BoundingBox(r.x_min, r.y_min, r.x_max, r.y_max) for i,r in data.iterrows()]
        
    # Binary
    elif 'binary' in df.columns:
        data = df[['binary']]
        objects=[BinaryAnnotation(int(r['binary'])) for i, r in data.iterrows()]
    else:
        raise Exception('Wrong columns\' names for defining the objects in DataFrame. ' \
                        'Define x and y columns for Point object ' \
                         'define binary column for patch-based binary object' \
                        'or define x_min, y_min, x_max and y_max columns for BoundingBox object.')
    
    return objects

def dict_from_objects(obj_list: List[Union[Point, BoundingBox, BinaryAnnotation]]) -> List[dict]:
    ''' Function to convert a list of objects to corresponding coordinates, 
    stored in a list of dict

    Args:
        obj_list (list): list of objects (Point Binary or BoundingBox)

    Returns:
        List[dict]:
            List of dict with a header keys of this type for Point:
                | 'x' | 'y' |
            and a header keys of this type for BoundingBox
                | 'x_min' | 'y_min' | 'x_max' | 'y_max' |
            integers (0 or 1) for binary annotations 'binary'
    '''

    assert all(isinstance(o, (Point, BoundingBox)) for o in obj_list) is True, \
        'Objects must be Point or BoundingBox instances.'
    
    # Point
    if isinstance(obj_list[0], Point):
        data = [{'x': o.x, 'y': o.y} for o in obj_list]

    # BoundingBox
    elif isinstance(obj_list[0], BoundingBox):
        data = [
            {'x_min': o.x_min, 'y_min': o.y_min, 'x_max': o.x_max, 'y_max': o.y_max}
            for o in obj_list
        ]
    # Binary
    elif isinstance(obj_list[0], BinaryAnnotation):
        data = [{'binary': o} for o in obj_list]
    
    return data

class Annotations:
    ''' Class to create an Annotations object '''

    def __init__(
        self, 
        images: Union[str, List[str]], 
        annos: List[Union[Point, BoundingBox, BinaryAnnotation]], 
        labels: Optional[List[int]] = None,  # Now conditional based on anno type
        **kwargs
        ) -> None:
        '''
        Initialize the Annotations object with images, annotations, and optionally labels.
        Labels are considered necessary for Point and BoundingBox but not for BinaryAnnotation.
        '''

        self.images = images
        self.annos = annos
        self.labels = labels
        self.__dict__.update(kwargs)

        # Validate the provided data
        if any(len(v) > 0 for v in [self.images, self.annos] + list(kwargs.values())):
            # Ensure all annotations are of supported types
            assert all(isinstance(o, (Point, BoundingBox, BinaryAnnotation)) for o in self.annos), \
                'annos must be composed of Point, BoundingBox, or BinaryAnnotation instances.'

            # Determine if labels are required
            labels_required = any(isinstance(o, (Point, BoundingBox)) for o in self.annos)

            # Validate labels if required
            if labels_required:
                assert self.labels is not None and all(isinstance(lab, int) for lab in self.labels), \
                    'labels are required for Point and BoundingBox annotations and must be integers.'
            else:
                # Optional for BinaryAnnotation, ensure they're integers if provided
                if self.labels is not None:
                    assert all(isinstance(lab, int) for lab in self.labels), \
                        'When provided, labels must be a list composed of integers only.'

            # Validate images
            if isinstance(self.images, list):
                assert all(isinstance(im, str) for im in self.images), \
                    'images must be a list composed of strings only.'
            elif not isinstance(self.images, str):
                raise ValueError('images must be a string or a list of strings')

            # Ensure uniform length for images and annos 
            provided_lists = [self.images, self.annos] + [v for k, v in kwargs.items() if isinstance(v, list)]
            assert len(set(len(lst) for lst in provided_lists if lst)) == 1, \
                'images, annos, and any additional provided lists must have the same length'

        else:
            warnings.warn('Empty or incomplete Annotations object created. Check your inputs.')
    
    @property
    def dataframe(self) -> pandas.DataFrame:
        ''' To get annotations in Pandas DataFrame 

        Returns:
            pandas.DataFrame
        '''
        
        return pandas.DataFrame(data = self.__dict__)
    
    def sort(
        self, 
        attr: str, 
        keep: Optional[str] = None, 
        reverse: bool = False
        ) -> None:
        ''' Sort the object attributes while keeping the values of an attribute 
        grouped or not.

        Args:
            attr (str): attribute to sort
            keep (str, optional): attribute to keep grouped. Defaults to None
            reverse (bool, optional): set to True for descending sort. Defaults to
                False
        '''

        assert attr in self.__dict__.keys(), \
            f'{attr} is not an attribute of the object'
        
        if isinstance(keep, str):
            assert keep in self.__dict__.keys(), \
                f'{keep} is not an attribute of the object'

        all_attr = [a for a in self.__iter__()]
        
        # sort attr first, then the attribute to keep (ascending order)
        specs = [(attr, reverse)]
        if keep is not None : 
            specs = [(attr, reverse), (keep, False)]

        for key, reverse in specs:
            all_attr.sort(key=itemgetter(key), reverse=reverse)
        
        # update
        keys = self.__dict__.keys()
        for key in keys:
            sorted_list = [row[key] for row in all_attr]
            self.__dict__.update({key: sorted_list})
    
    def sub(self, image_name: str):
        ''' Returns an Annotations sub-object by selecting the items that 
        contain the specified image name

        Args: 
            image_name (str): the image name with extension
        
        Returns:
            Annotations
        '''

        new_kwargs = {}

        image_idx = [i for i, _ in enumerate(self.images) if self.images[i]==image_name]
        for key, values in self.__dict__.items():
            new_values = [values[i] for i in image_idx]
            new_kwargs.update({key: new_values})
        
        return Annotations(**new_kwargs)
    
    def get_supp_args_names(self):
        supp_args_names = []
        for key, values in self.__dict__.items():
            if key not in ['annos','images','labels']:
                supp_args_names.append(key)
        
        return supp_args_names

    def __iter__(self) -> dict:
        for i in range(len(self.images)):
            out_dict = {}
            for key in self.__dict__.keys():
                out_dict.update({key: self.__dict__[key][i]})
            
            yield out_dict
    
    def __getitem__(self, index) -> dict:
        out_dict = {}
        for key in self.__dict__.keys():
            out_dict.update({key: self.__dict__[key][index]})

        return out_dict
    
    def __len__(self) -> int:
        return len(self.images)

class AnnotationsFromCSV(Annotations):
    ''' Class to create annotations object from a CSV file
    
    Inheritance of Annotations class.

    The CSV file must have, at least, a header of this type for points:
    | 'images' | 'x' | 'y' | 'labels' |

    and of this type for bounding box:
    | 'images' | 'x_min' | 'y_min' | 'x_max' | 'y_max' | 'labels' |

    Other columns containing other information may be present. 
    In such a case, these will be kept and linked to the necessary basic content.
    '''

    def __init__(self, csv: Union[str,pandas.DataFrame]) -> None:
        '''
        Args:
            csv (str or pandas.DataFrame): absolute path to the CSV file (with extension),
                or DataFrame object.
        '''

        assert isinstance(csv, (str, pandas.DataFrame)), \
            'csv argument must be a string (absolute path with extension) ' \
            'or a pandas.DataFrame.'
        
        data_df = csv
        if isinstance(csv, str):
            data_df = pandas.read_csv(csv)

        assert {'images'}.issubset(data_df.columns), \
            'File must contain at least images and labels columns name'
        
        images = list(data_df['images'])
        labels = list(data_df['labels']) if 'labels' in data_df.columns else None
        annos = objects_from_df(data_df)
###### Previous version ########
        # # get other information
        # supp_arg = {}
        # for column, content in data_df.items():
        #     if column not in ['images','labels'] and column.startswith(('x','y')) is False:
        #         supp_arg.update({column: list(content)})

        # super(AnnotationsFromCSV, self).__init__(images, annos, labels)
        # if supp_arg:
        #     super(AnnotationsFromCSV, self).__init__(images, annos, labels, **supp_arg)\
##### New Version ######
# Identify supplementary arguments beyond the core annotations and known columns
        core_columns = ['images', 'x', 'y', 'x_min', 'y_min', 'x_max', 'y_max', 'binary']
        if labels is not None:
            core_columns.append('labels')
        supp_arg = {column: list(content) for column, content in data_df.items() if column not in core_columns}

        # Initialize the Annotations object with the extracted data and supplementary arguments
        super().__init__(images=images, annos=annos, labels=labels, **supp_arg)