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


import math
import copy
import sklearn.neighbors
import numpy
import torch
from sklearn.metrics import confusion_matrix
from itertools import tee
from typing import Optional, List

from ..data import BoundingBox
from .utils import bboxes_iou

from ..utils.registry import Registry

METRICS = Registry('metrics', module_key='animaloc.eval.metrics')

__all__ = ['METRICS', *METRICS.registry_names]

@METRICS.register()
class Metrics:
    '''
    Class to accumulate classification, detection and counting metrics, i.e.: 

        - Precision
        - Recall
        - F-beta score
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Average Precision (AP)
        - Interclass confusion
        - Classification accuracy

    for binary or multiclass model.

    First, instanciate with a data matching threshold (e.g. IoU, radius),
    then feed the object with ground truth and predictions. You can retrieve 
    any metric at any time by calling the corresponding attribute.
    '''
    
    def __init__(self, threshold: float, num_classes: int = 2) -> None:
        '''
        Args:
            threshold (float): data matching threshold
            num_classes (int, optional): number of classes, background included. 
                Defaults to 2 (binary case).
        '''
        self.binary_annotations = False
        self.threshold = threshold
        self.num_classes = num_classes

        self.detections = []
        self.idx = 0

        self.tp = self._init_attr()
        self.fp = self._init_attr()
        self.fn = self._init_attr()
        self.tn = self._init_attr()
        
        self._sum_absolute_error = self._init_attr()
        self._sum_squared_error = self._init_attr()
        self._n_calls = self._init_attr()
        self._agg_sum_absolute_error = 0
        self._agg_sum_squared_error = 0
        self._total_calls = 0
        self._total_count = self._init_attr()

        self._ap_tables = self._init_attr(val=[])

        self.confusion_matrix = numpy.zeros((2, 2), dtype=int)
        
        self.confusion_matrix = numpy.zeros((self.num_classes-1,self.num_classes-1))
        self._confusion_matrix = self.confusion_matrix

    def feed(self, gt: dict, preds: dict, est_count: Optional[list] = None) -> None:
        ''' Feed the object with ground truth and predictions and returns
        specified metrics optionally.

        Args:
            gt (dict): ground truth containing a dict with 'loc' and 'labels' 
                keys and list as values.
            preds (dict): predictions containing a dict with 'loc' and 'labels' 
                keys and list as values. Can contain an optional 'scores' key. 
            est_count (list, optional): list containing estimated count for each
                class, background excluded. Defaults to None.
        '''
        
        self.binary_annotations = 'binary' in gt and 'binary' in preds
        if self.binary_annotations:
            self.matching(gt, preds)  # Direct comparison as in ImageLevelMetrics
        else:
            assert isinstance(est_count, (type(None), list))
            for o in [gt, preds]:
                assert isinstance(o, dict)
                assert len(o['loc']) == len(o['labels'])
            
            self.score_flag = 0
            if 'scores' in preds.keys():
                self.score_flag = 1
                assert len(preds['scores']) == len(preds['loc'])
            
            if len(gt['loc']) == 0:
                self._no_gt(gt, preds)
            
            if len(preds['loc']) == 0:
                self._no_preds(gt, preds)
            
            if len(gt['loc']) > 0 and len(preds['loc']) > 0:
                self.matching(gt, preds)
        if est_count is None:
            non_empty_preds = (preds['binary'] == 1).sum().item()  # Count of non-empty patches
            empty_preds = (preds['binary'] == 0).sum().item()  # Count of empty patches
            est_count = [empty_preds, non_empty_preds]
        
     
        else:
            gt_count = self._init_attr(0)
            if len(gt['loc']) > 0:
                gt_count = [gt['labels'].count(i+1) for i in range(self.num_classes-1)]

            self._update_errors(gt_count, est_count)
            self._update_calls(gt_count, est_count)
        
            self._total_calls += 1
            self._total_count = [self._total_count[i] + count for i, count in enumerate(est_count)]
        
        self._store_detections(preds, est_count)
        self.idx += 1
    
    def matching(self, gt: dict, preds: dict) -> None:
        ''' Method to match ground truth and predictions.
        To be overriden by subclasses
        
        Args:
            gt (dict): ground truth containing a dict with 'loc' and 'labels' 
                keys and list as values.
            preds (dict): predictions containing a dict with 'loc' and 'labels' 
                keys and list as values. Can contain an optional 'scores' key. 
        '''
        pass

    def copy(self):
        clone = copy.deepcopy(self)
        return clone
    
    def flush(self) -> None:
        ''' Flush the object '''

        self.detections = []
        self.idx = 0
        self._sum_absolute_error = self._init_attr()
        self._sum_squared_error = self._init_attr()
        self._n_calls = self._init_attr()
        self._agg_sum_absolute_error = 0
        self._agg_sum_squared_error = 0
        self._total_calls = 0
        self._total_count = self._init_attr()

        self._ap_tables = self._init_attr(val=[])

        # self.confusion_matrix = numpy.zeros((self.num_classes-1,self.num_classes-1))
        self._confusion_matrix = self.confusion_matrix
    
    def aggregate(self) -> None:
        ''' Aggregate the metrics.

        By default, the classes are aggregated into a single class and the metrics are 
        therefore relative to the object vs. background configuration.
        '''

        inter = int(self._confusion_matrix.sum()) - sum(self.tp)

        self.fp = [sum(self.fp) - inter]
        self.fn = [sum(self.fn) - inter]
        self.tp = [int(self._confusion_matrix.sum())]
        self._sum_absolute_error = [self._agg_sum_absolute_error]
        self._sum_squared_error = [self._agg_sum_squared_error]
        self._n_calls = [self._total_calls]
        self._ap_tables = [[[1,*x[1:]] for x in sum(self._ap_tables, [])]]
        self._confusion_matrix = numpy.array([[1.]])
        self._total_count = [sum(self._total_count)]

    ################## NEW Adapted Precision Code ###################
    def precision(self, c: int = 1) -> float:
        ''' Precision 
        Args:
            c (int, optional): For object detection, class id starting from 1. For binary classification, 0 for negative class and 1 for positive class. Defaults to 1 for binary.
        
        Returns:
            float: Precision value.
        '''
        # Binary classification case
        if hasattr(self, 'binary_annotations') and self.binary_annotations:
            # Assuming binary classification (0: negative, 1: positive)
            tp = self.tp[0] if c == 1 else 0  # True positives for binary are stored at index 0, assuming c=1 for positive class
            fp = self.fp[0] if c == 1 else 0  # False positives for binary are also stored at index 0
            total = tp + fp
            return float(tp) / total if total > 0 else 0.0

        # Object detection or multiclass classification case (unchanged)
        else:
            c = c - 1  
            if self.tp[c] > 0:
                return float(self.tp[c] / (self.tp[c] + self.fp[c]))
            else:
                return float(0)


    ######################## NEW code for Recall ###########
    def recall(self, c: int = 1) -> float:
        ''' Recall 
        Args:
            c (int, optional): For object detection, class id starting from 1. For binary classification, 0 for negative class and 1 for positive class. Defaults to 1 for binary.
        
        Returns:
            float: Recall value.
        '''
        # Binary classification case
        if hasattr(self, 'binary_annotations') and self.binary_annotations:
            # Assuming binary classification (0: negative, 1: positive)
    
            tp = self.tp[0] if c == 1 else 0  # True positives for binary are stored at index 0, assuming c=1 for positive class
            fn = self.fn[0] if c == 1 else 0  # False negatives for binary are also stored at index 0
            total = tp + fn
            return float(tp) / total if total > 0 else 0.0

        # Object detection or multiclass classification case (unchanged)
        else:
            c = c - 1  # Adjust for 0-based indexing
            if self.tp[c] > 0:
                return float(self.tp[c] / (self.tp[c] + self.fn[c]))
            else:
                return float(0)
   
    ############## New f-beta code ##############
    def fbeta_score(self, c: int = 1, beta: int = 1) -> float:
        ''' F-beta score 
        Args:
            c (int, optional): For object detection, class id starting from 1. For binary classification, 0 for negative class and 1 for positive class. Defaults to 1 for binary.
            beta (int, optional): Beta value, which determines the weight of recall in the combined score. Defaults to 1.
        
        Returns:
            float: The F-beta score.
        '''
        # Binary classification case
        if hasattr(self, 'binary_annotations') and self.binary_annotations:
            # Assuming binary classification (0: negative, 1: positive)
            precision = self.precision(c)
            recall = self.recall(c)
            if precision + recall > 0:
                return float(
                    (1 + beta**2) * precision * recall / 
                    ((beta**2) * precision + recall)
                )
            else:
                return float(0)
        # Object detection or multiclass classification case (unchanged)
        else:
            c = c - 1  # Adjust for 0-based indexing
            if self.tp[c] > 0:
                precision = self.precision(c + 1)  # Adjusting c back for the precision and recall calls
                recall = self.recall(c + 1)
                return float(
                    (1 + beta**2) * precision * recall / 
                    ((beta**2) * precision + recall)
                )
            else:
                return float(0)

    ############# new mae code #######
    def mae(self, c: int = 1) -> float:
        ''' Mean Absolute Error
        Args:
            c (int, optional): class id. Defaults to 1 for binary, but this parameter is ignored in binary cases as MAE is not calculated.
            
        Returns:
            float: The mean absolute error for object detection tasks, or a predefined value (e.g., None) for binary classification tasks.
        '''
        # Binary classification case: Skip calculation and return None or 0.0
        if hasattr(self, 'binary_annotations') and self.binary_annotations:
            return 0.0  

        # Object detection or multiclass classification case: Perform usual MAE calculation
        else:
            c = c - 1  # Adjust for 0-based indexing
            if c < 0 or c >= len(self._sum_absolute_error):
                raise ValueError("Class index out of range.")
            if self._n_calls[c] > 0:
                return float(self._sum_absolute_error[c] / self._n_calls[c])
            else:
                return 0.0

    
    ############ NEW MSE code ##################
    def mse(self, c: int = 1) -> float:
        ''' Mean Squared Error
        Args:
            c (int, optional): For object detection, class id starting from 1. For binary classification, ignored since MSE is not typically calculated.
            
        Returns:
            float: The mean squared error for object detection tasks, or a predefined value (e.g., 0.0) for binary classification tasks.
        '''
        if hasattr(self, 'binary_annotations') and self.binary_annotations:
            return 0.0  # Return a default float value

        else:
            c = c - 1
            if c < 0 or c >= len(self._sum_squared_error):
                raise ValueError("Class index out of range.")
            return float(self._sum_squared_error[c] / self._n_calls[c]) if self._n_calls[c] else 0.0

    ################## New RMSE Code ########
    def rmse(self, c: int = 1) -> float:
        ''' Root Mean Squared Error
        Args:
            c (int, optional): For object detection, class id starting from 1. For binary classification, ignored since RMSE is not typically calculated.
        
        Returns:
            float: The root mean squared error for object detection tasks, or 0.0 for binary classification tasks.
        '''
        # Binary classification case: Skip calculation and return 0.0
        if hasattr(self, 'binary_annotations') and self.binary_annotations:
            return 0.0  # Bypass RMSE calculation for binary classification

        # Object detection or multiclass classification case: Perform usual RMSE calculation
        else:
    
            mse_value = self.mse(c)
            return float(math.sqrt(mse_value)) if mse_value is not None else 0.0

    
    def ap(self, c: int = 2) -> float:
        '''Average Precision
        Args: 
            c (int, optional): class id. Defaults to 1.
            
        Returns:
            float: Calculated average precision, or a placeholder if not applicable.
        '''
        recalls, precisions = self.rec_pre_lists(c)
        
        # For binary classification scenarios where rec_pre_lists is not implemented
        if hasattr(self, 'binary_annotations') and self.binary_annotations:
            
            return 0.0  
        
        # For non-binary tasks, proceed with AP calculation if recall and precision values exist
        elif len(recalls) == 0 or len(precisions) == 0:
            return 0.
        else:
            return self._compute_AP(recalls, precisions)

    
    ################################## NEW rec_pre_lists ###########################
    def rec_pre_lists(self, c: int = 1) -> tuple:
        '''Recalls and Precisions lists for both binary and object detection tasks.
        
        Args: 
            c (int, optional): class id. Defaults to 1 for binary and specific class IDs for object detection/multiclass.
        
        Returns:
            tuple: recalls and precisions lists.
        '''
        # Check if we're in a binary classification scenario
        if hasattr(self, 'binary_annotations') and self.binary_annotations:

            return [], []

        else:  # Object detection or multiclass classification case
            c = c - 1  # Adjust for 0-based indexing

            if len(self._ap_tables[c]) == 0:
                return [], []

            # Proceed with the original logic for non-binary tasks
            n_gt = self.fn[c] + self.tp[c]
            sorted_table = sorted(self._ap_tables[c], key=lambda x: x[1], reverse=True)
            sorted_table = numpy.array(sorted_table)
            sorted_table[:, 2] = numpy.cumsum(sorted_table[:, 2], axis=0)  # Cumulative true positives
            sorted_table[:, 3] = numpy.cumsum(sorted_table[:, 3], axis=0)  # Cumulative false positives

            precisions = sorted_table[:, 2] / (sorted_table[:, 2] + sorted_table[:, 3])
            recalls = sorted_table[:, 2] / n_gt

            return recalls.tolist(), precisions.tolist()

    ##################### New Confusion Code #############################
    def confusion(self, c: int = 1) -> float:
        ''' Construct the confusion matrix for binary classification tasks
        or return interclass confusion for a specified class in object detection tasks.

        Args:
            c (int, optional): The class id for which to calculate interclass confusion in non-binary tasks. Defaults to 1 for binary.
        
        Returns:
            For binary classification: A dictionary representing the confusion matrix with TP, TN, FP, FN.
            For object detection or multiclass classification: Float representing interclass confusion for the specified class.
        '''
        # Binary classification case
        if hasattr(self, 'binary_annotations') and self.binary_annotations:
            # Construct and return a confusion matrix dictionary for binary classification
            return {
                'TP': self.tp[0],  # True Positives
                'TN': self.tn[0],  # True Negatives
                'FP': self.fp[0],  # False Positives
                'FN': self.fn[0],  # False Negatives
            }

        # Object detection or multiclass classification case
        else:
            c = c - 1  # Adjust for 0-based indexing
            if c < 0 or c >= len(self._confusion_matrix):
                raise ValueError("Class index out of range.")
            cm_row = self._confusion_matrix[c]
            p = cm_row[c] / sum(cm_row) if sum(cm_row) else 0.
            return 1 - p
############ No Need To Change ##########
    def accuracy(self) -> float:
        ''' Classification accuracy 
        
        Returns:
            float
        '''

        N = self.confusion_matrix.sum()
        tp = self.confusion_matrix.diagonal().sum()
        if N > 0:
            return tp / N
        else:
            return 0.
    
   
    ######## New total count function ###########
    def total_count(self, c: int = 1) -> int:
        ''' Total class count
        Args: 
            c (int, optional): For object detection, class id starting from 1. For binary classification, this can be ignored or set to 1 for positive class.
        
        Returns:
            int: The total count of instances for the specified class (in object detection) or the total count of positive instances (in binary classification).
        '''
        # Binary classification case
        if hasattr(self, 'binary_annotations') and self.binary_annotations:
            # count of positive instances
            # Assuming index 0 stores the count for the positive class (class 1)
            return self._total_count[0]

        # Object detection or multiclass classification case
        else:
            c = c - 1  # Adjust for 0-based indexing
            if c < 0 or c >= len(self._total_count):
                raise ValueError("Class index out of range.")
            return self._total_count[c]

        
    def _init_attr(self, val: int = 0) -> list:
        return [val] * (self.num_classes - 1)
    
    def _update_calls(self, gt_count: list, est_count: list):

        for i, _ in enumerate(self._n_calls):
            if gt_count[i] != 0 or est_count[i] != 0:
                self._n_calls[i] += 1

    def _update_errors(self, gt_count: list, est_count: list):

        for i, (count, est) in enumerate(zip(gt_count, est_count)):
            error = abs(count - est)
            squared_error = error**2

            self._sum_absolute_error[i] += error
            self._sum_squared_error[i] += squared_error
        
        agg_error = abs(sum(gt_count) - sum(est_count))
        agg_squared_error = agg_error**2
        self._agg_sum_absolute_error += agg_error
        self._agg_sum_squared_error += agg_squared_error
    
    def _no_gt(self, gt: dict, preds: dict) -> None:

        for c in range(1, self.num_classes):
            n_pred = len([lab for lab in preds['labels'] if lab == c])

            self.tp[c-1] += 0
            self.fp[c-1] += n_pred
            self.fn[c-1] += 0

            if self.score_flag:
                preds_fp = [[preds['labels'][i],preds['scores'][i],0,1]
                                  for i, _ in enumerate(preds['labels'])
                                  if preds['labels'][i] == c]

                self._ap_tables[c-1] = [*self._ap_tables[c-1], *preds_fp]
    
    def _no_preds(self, gt: dict, preds: dict) -> None:

        for c in range(1, self.num_classes):
            n_gt = len([lab for lab in gt['labels'] if lab == c])

            self.tp[c-1] += 0
            self.fp[c-1] += 0
            self.fn[c-1] += n_gt
    
    def _compute_AP(self, recalls: list, precisions: list) -> float:
        '''
        Compute the VOC Average Precision
        Code from: https://github.com/Cartucho/mAP
        (adapted from official matlab code VOC2012)
        '''

        recalls.insert(0, 0.0)
        recalls.append(1.0)
        precisions.insert(0, 0.0) 
        precisions.append(0.0) 

        mrec, mpre = recalls[:], precisions[:]

        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) 

        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        
        return ap
    
    def _store_detections(self, preds: dict, est_count: Optional[list] = None) -> None:
        ''' Store detections internally (1 row = 1 detection) '''

        m = map(dict, zip( * [
            [(k, v) for v in value]
            for k, value in preds.items()
            ]))
        m, m_copy = tee(m)
        
        counts = {}
        if est_count is not None:
            counts = {f'count_{i+1}': x for i, x in enumerate(est_count)}

        if len([x for x in m_copy]) > 0:
            for det in m:
                self.detections.append({'images': self.idx, **det, **counts})
        else:
            self.detections.append({'images': self.idx, **counts})
    
@METRICS.register()
class PointsMetrics(Metrics):
    ''' Metrics class for points (must be in (x,y) format) '''

    def __init__(self, radius: float, num_classes: int = 2) -> None:
        '''
        Args:
            radius (float): distance between ground truth and predicted point
                from which a point is characterizd as true positive
            num_classes (int, optional): number of classes, background included. 
                Defaults to 2 (binary case).
        '''
        super().__init__(threshold=radius, num_classes=num_classes)
    
    def matching(self, gt: dict, preds: dict) -> None:
        
        # matching
        dist = sklearn.neighbors.NearestNeighbors(n_neighbors=1, metric='euclidean').fit(preds['loc'])
        dist, idx = dist.kneighbors(gt['loc'])
        match_gt = [(k, d, i) for k, (d, i) in enumerate(zip(dist[:,0], idx[:,0]))]

        # sort according to distance
        match_gt = sorted(match_gt, key = lambda tup: tup[1])              
        # discard duplicates
        k_discard, i_discard = [], []
        filter_match_gt = []
        for k, d, i in match_gt:
            if k not in k_discard and i not in i_discard:
                filter_match_gt.append((k,d,i))
                k_discard.append(k), i_discard.append(i)
        # threshold
        filter_match_gt = [(k, d, i) for k, d, i in filter_match_gt if d <= self.threshold]

        # confusion matrix
        y_true = [gt['labels'][k] for k, d, i in filter_match_gt]
        y_pred = [preds['labels'][i] for k, d, i in filter_match_gt]

        self._confusion_matrix += confusion_matrix(
            y_true, y_pred, labels=list(range(1, self.num_classes)))

        for c in range(1, self.num_classes):
            n_gt = len([lab for lab in gt['labels'] if lab == c])
            n_pred = len([lab for lab in preds['labels'] if lab == c])

            lab_match = [(d, i) for k, d, i in filter_match_gt 
                            if gt['labels'][k] == preds['labels'][i] == c]

            tp = len(lab_match)
            self.tp[c-1] += tp
            self.fp[c-1] += (n_pred - tp)
            self.fn[c-1] += (n_gt - tp)

            if self.score_flag:
                tp_ids = [i for d, i in lab_match]
                preds_tp = [[preds['labels'][i],preds['scores'][i],1,0]
                              for _, i in lab_match]
                preds_fp = [[preds['labels'][i],preds['scores'][i],0,1]
                              for i, _ in enumerate(preds['labels'])
                              if preds['labels'][i] == c and i not in tp_ids]

                self._ap_tables[c-1] = [*self._ap_tables[c-1], *preds_tp, *preds_fp]
    
    def _store_detections(self, preds: dict, est_count: Optional[list] = None) -> None:

        m = map(dict, zip( * [
            [(k, v) for v in value]
            for k, value in preds.items()
            ]))
        m, m_copy = tee(m)
        
        counts = {}
        if est_count is not None:
            counts = {f'count_{i+1}': x for i, x in enumerate(est_count)}

        if len([x for x in m_copy]) > 0:
            for det in m:
                y, x = det['loc']
                det.update(dict(x=x, y=y))
                _ = det.pop('loc')
                self.detections.append({'images': self.idx, **det, **counts})
        else:
            self.detections.append({'images': self.idx, **counts})

@METRICS.register()
class BoxesMetrics(Metrics):
    ''' Metrics class for bounding boxes 
    (must be in (x_min, y_min, x_max, y_max) format '''

    def __init__(self, iou: float, num_classes: int = 2) -> None:
        '''
        Args:
            iou (float): Intersect-over-Union (IoU) threshold used to define a true
                positive.
            num_classes (int, optional): number of classes, background included. 
                Defaults to 2 (binary case).
        '''
        super().__init__(threshold=iou, num_classes=num_classes)
    
    def matching(self, gt: dict, preds: dict) -> None:

        ious, idx = self._most_overlapping_boxes(gt['loc'], preds['loc'])
        match_gt = [(k, iou, i) for k, (iou, i) in enumerate(zip(ious, idx)) 
                        if iou >= self.threshold]
        
        # confusion matrix
        y_true = [gt['labels'][k] for k, d, i in match_gt]
        y_pred = [preds['labels'][i] for k, d, i in match_gt]

        self._confusion_matrix += confusion_matrix(
            y_true, y_pred, labels=list(range(1, self.num_classes)))
        
        for c in range(1, self.num_classes):
            n_gt = len([lab for lab in gt['labels'] if lab == c])
            n_pred = len([lab for lab in preds['labels'] if lab == c])

            lab_match = [(d, i) for k, d, i in match_gt 
                            if gt['labels'][k] == preds['labels'][i] == c]

            tp = len(lab_match)
            self.tp[c-1] += tp
            self.fp[c-1] += (n_pred - tp)
            self.fn[c-1] += (n_gt - tp)

            if self.score_flag:
                tp_ids = [i for d, i in lab_match]
                preds_tp = [[preds['labels'][i],preds['scores'][i],1,0]
                              for _, i in lab_match]
                preds_fp = [[preds['labels'][i],preds['scores'][i],0,1]
                              for i, _ in enumerate(preds['labels'])
                              if preds['labels'][i] == c and i not in tp_ids]

                self._ap_tables[c-1] = [*self._ap_tables[c-1], *preds_tp, *preds_fp]
    
    def _most_overlapping_boxes(
        self, 
        gt_boxes: List[tuple], 
        preds_boxes: List[tuple], 
        ) -> tuple:
        
        gt_boxes = [BoundingBox(*coord) for coord in gt_boxes]
        preds_boxes = [BoundingBox(*coord) for coord in preds_boxes]

        iou_matrix = bboxes_iou(gt_boxes, preds_boxes)

        match_idx = []
        ious = []
        for row in iou_matrix:
            filt_row = [(k, elem) for k, elem in enumerate(row) if k not in match_idx]
            if len(filt_row) > 0:
                idx, iou_max = max(filt_row, key=lambda item:item[1])

                match_idx.append(idx)
                ious.append(iou_max)
        
        return ious, match_idx
    
    def _store_detections(self, preds: dict, est_count: Optional[list] = None) -> None:

        m = map(dict, zip( * [
            [(k, v) for v in value]
            for k, value in preds.items()
            ]))
        
        m, m_copy = tee(m)
        
        counts = {}
        if est_count is not None:
            counts = {f'count_{i+1}': x for i, x in enumerate(est_count)}

        if len([x for x in m_copy]) > 0:
            for det in m:
                x_min, y_min, x_max, y_max = det['loc']
                det.update(dict(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))
                _ = det.pop('loc')
                self.detections.append({'images': self.idx, **det, **counts})
        else:
            self.detections.append({'images': self.idx, **counts})
   
@METRICS.register()

class ImageLevelMetrics(Metrics):
    '''Metrics class for image-level classification for binary tasks.'''
####### updated ##### there was no init in this class before 
# (for the problem of the repeated detections in batchsize more than 1)
    def __init__(self, img_names, num_classes: int = 2):
        '''Initialize metrics for binary classification tasks.'''
        num_classes = num_classes + 1  # Adjust for binary classification
        super().__init__(threshold=None, num_classes=num_classes)
        self.img_names = img_names  # Passed image names directly
        
    def feed(self, gt: dict, preds: dict):
        '''Feed metrics with ground truth and predictions.'''
        super().feed(gt, preds)

    def matching(self, gt: dict, preds: dict):
        '''Compare binary ground truth labels to binary predictions.'''
        gt_binary = gt['binary'][0].squeeze()
        pred_binary = preds['binary'].squeeze()
        # print('gt_binary:',gt_binary)
        # Convert to boolean for logical operations
        gt_binary_bool = gt_binary.bool()
        pred_binary_bool = pred_binary.bool()

        # Calculate True Positives, False Positives, False Negatives, True Negatives
        self.tp[0] += ((gt_binary_bool & pred_binary_bool).sum().item())
        self.fp[0] += ((~gt_binary_bool & pred_binary_bool).sum().item())
        self.fn[0] += ((gt_binary_bool & ~pred_binary_bool).sum().item())
        self.tn[0] += ((~gt_binary_bool & ~pred_binary_bool).sum().item())
        # print("TP:", self.tp[0], "FP:", self.fp[0], "FN:", self.fn[0], "TN:", self.tn[0])
    
        
        # Update the confusion matrix
        self.confusion_matrix[0, 0] = self.tp[0]
        self.confusion_matrix[0, 1] = self.fp[0]
        self.confusion_matrix[1, 0] = self.fn[0]
        self.confusion_matrix[1, 1] = self.tn[0]
        # print(self.confusion_matrix)
    def _store_detections(self, preds: dict, est_count: Optional[list] = None) -> None:
        ''' Store detections internally (1 row = 1 detection), convert tensor to int. '''

        # Convert tensor predictions to integers before storing them
        preds = {k: [v.item() if torch.is_tensor(v) else v for v in vals] for k, vals in preds.items()}

        m = map(dict, zip(*[
            [(k, v) for v in value]
            for k, value in preds.items()
        ]))
        m, m_copy = tee(m)

        counts = {}
        if est_count is not None:
            counts = {f'count_{i+1}': x for i, x in enumerate(est_count)}

        if len([x for x in m_copy]) > 0:
            for det in m:
                self.detections.append({'images': self.idx, **det, **counts})
        else:
            self.detections.append({'images': self.idx, **counts})
 
    def _store_detections_binary(self, preds: dict, est_count: Optional[list] = None):
        ''' Store detections for binary classification internally '''

        # Calculate estimated counts if not provided
        if est_count is None:
            non_empty_preds = (preds['binary'] == 1).sum().item()  # Count of positive detections
            empty_preds = (preds['binary'] == 0).sum().item()      # Count of negative detections
            est_count = [empty_preds, non_empty_preds]

        counts = {'empty_count': est_count[0], 'non_empty_count': est_count[1]}

        # Flatten binary values
        binary_values = preds['binary'].view(-1).cpu().tolist() if isinstance(preds['binary'], torch.Tensor) else [int(x) for x in preds['binary']]

        # Process each binary detection value
        for idx, binary_value in enumerate(binary_values):
            if idx < len(self.img_names):
                image_name = self.img_names[idx]
                self.detections.append({'images': image_name, 'binary': binary_value, **counts})
            else:
                print(f"Skipping idx {idx}: Out of range for img_names with length {len(self.img_names)}")
                break  # Stop processing if index exceeds the range of img_names




@METRICS.register()
class RegressionMetrics(Metrics):
    ''' Metrics class for regression type tasks '''

    def __init__(self, num_classes: int = 2) -> None:
        num_classes = num_classes + 1 # for convenience
        super().__init__(0, num_classes)
    
    def feed(self, gt: float, pred: float) -> tuple:
        '''
        Args:
            gt (float): numeric ground truth value
            pred (float): numeric predicted value
        '''

        gt = dict(labels=[gt], loc=[(0,0)])
        preds = dict(labels=[pred], loc=[(0,0)])
        
        super().feed(gt, preds)
    
    def matching(self, gt: dict, pred: dict) -> None:
        gt_lab = gt['labels'][0]
        p_lab = pred['labels'][0]

        diff= math.abs(gt_lab-p_lab) # L1-loss
