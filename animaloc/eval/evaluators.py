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
import pandas
import os
import numpy
import wandb
import matplotlib
from itertools import chain

matplotlib.use('Agg')

from typing import Any, Optional, Dict, List, Callable, Type

import torch.nn.functional as F
from animaloc.models import DLAEncoderDecoder
from ..utils.logger import CustomLogger

from .stitchers import Stitcher
from .metrics import Metrics
from .lmds import HerdNetLMDS

from ..utils.registry import Registry

EVALUATORS = Registry('evaluators', module_key='animaloc.eval.evaluators')

__all__ = ['EVALUATORS', *EVALUATORS.registry_names]

@EVALUATORS.register()
class Evaluator:
    ''' Base class for evaluators '''

    def __init__(
        self,
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        metrics: Metrics,
        device_name: str = 'cuda', 
        print_freq: int = 10,
        stitcher: Optional[Stitcher] = None,
        vizual_fn: Optional[Callable] = None,
        work_dir: Optional[str] = None,
        header: Optional[str] = None
        
        ):
        '''
        Args:
            model (torch.nn.Module): CNN detection model to evaluate, that takes as 
                input tensor image and returns output and loss as tuple.
            dataloader (torch.utils.data.DataLoader): a pytorch's DataLoader that returns a tensor
                image and target.
            metrics (Metrics): Metrics instance used to compute model performances.
            device_name (str): the device name on which tensors will be allocated ('cpu' or  
                'cuda').
                Defaults to 'cuda'.
            print_freq (int, optional): define the frequency at which the logs will be
                printed and/or recorded. 
                Defaults to 10.
            stitcher (Stitcher, optional): optional Stitcher class instance to evaluate over
                large images. The specified dataloader should thus be composed of large images 
                for the use of this algorithm to make sense.
                Defaults to None.
            vizual_fn (callable, optional): a model specific function that will be use for plotting
                samples in Weights & Biases during validation. It must take 'image', 'target', and 
                'output' as arguments and return a matplotlib figure. Defaults to None.
            work_dir (str, optional): directory where logs (and results) will be saved. If
                None is given, results and logs files will be saved in current working 
                directory.
                Defaults to None.
            header (str, optional): string put at the beginning of the printed logs
                Defaults to None
        '''
        
        assert isinstance(model, torch.nn.Module), \
            'model argument must be an instance of nn.Module'
        
        assert isinstance(dataloader, torch.utils.data.DataLoader), \
            'dataset argument must be an instance of torch.utils.data.DataLoader'

        assert isinstance(metrics, Metrics), \
            'metrics argument must be an instance of Metrics'
        
        assert isinstance(stitcher, (type(None), Stitcher)), \
            'stitcher argument must be an instance of Stitcher class'
        
        assert callable(vizual_fn) or isinstance(vizual_fn, type(None)), \
            f'vizual_fn argument must be a callable function, got \'{type(vizual_fn)}\''
        
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.device = torch.device(device_name)
        self.print_freq = print_freq
        self.stitcher = stitcher
        self.vizual_fn = vizual_fn
        
        self.work_dir = work_dir
        if self.work_dir is None:
            self.work_dir = os.getcwd()

        self.header = header

        self._stored_metrics = None

        self.logs_filename = 'evaluation'

    def prepare_data(self, images: Any, targets: Any) -> tuple:
        ''' Method to prepare the data before feeding to the model. 
        Can be overriden by subclasses.

        Args:
            images (Any)
            targets (Any)
        
        Returns:
            tuple
        '''
        if isinstance(targets, dict):
            # Move each tensor within the target dictionary to the device
            
            if len(targets.keys())>1:
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = [v.to(self.device) for k, v in targets.items()]
        elif isinstance(targets, (list, tuple)):
            # If targets is a list or tuple, move each item to the device
            targets = [tar.to(self.device) for tar in targets]
            print("Targets are in a list or tuple.")
        else:
            print("Unexpected targets format.")
            
        return images, targets
    
    def prepare_feeding(self, targets: Any, output: Any) -> dict:
        ''' Method to prepare targets and output before feeding to the Metrics instance. 
        Can be overriden by subclasses.

        Args:
            targets (Any)
            output (Any)
        
        Returns:
            dict
        '''
        
        return dict(gt = targets, preds = output)
    
    def post_stitcher(self, output: torch.Tensor) -> Any:
        ''' Method to post-treat the output of the stitcher.
        Can be overriden by subclasses.

        Args:
            output (torch.Tensor): output of Stitcher call
        
        Returns:
            Any
        '''
        return output
    
    @torch.no_grad()
    def evaluate(self, returns: str = 'recall', wandb_flag: bool = False, viz: bool = False,
        log_meters: bool = True) -> float:
        ''' Evaluate the model
        
        Args:
            returns (str, optional): metric to be returned. Possible values are:
                'recall', 'precision', 'f1_score', 'mse', 'mae', 'rmse', 'accuracy'
                and 'mAP'. Defauts to 'recall'
            wandb_flag (bool, optional): set to True to log on Weight & Biases. 
                Defaults to False.
            viz (bool, optional): set to True to save vizual predictions on original
                images. Defaults to False.
            log_meters (bool, optional): set to False to disable meters logging. 
                Defaults to True.
        
        Returns:
            float
        '''
        
        self.model.eval()

        self.metrics.flush()

        logger = CustomLogger(delimiter=' ', filename=self.logs_filename, work_dir=self.work_dir)
        iter_metrics = self.metrics.copy()

        for i, (images, targets) in enumerate(logger.log_every(self.dataloader, self.print_freq, self.header)):

            images, targets = self.prepare_data(images, targets)

            if self.stitcher is not None:
                output = self.stitcher(images[0])
                # output = self.post_stitcher(output)
            else:
                output, _ = self.model(images, targets)  
                output, _ = self.model(images)

            if viz and self.vizual_fn is not None:
                if i % self.print_freq == 0 or i == len(self.dataloader) - 1:
                    fig = self._vizual(image = images, target = targets, output = output)
                    wandb.log({'validation_vizuals': fig})

            output = self.prepare_feeding(targets, output)

            iter_metrics.feed(**output)
            iter_metrics.aggregate()
            if log_meters:
                logger.add_meter('n', sum(iter_metrics.tp) + sum(iter_metrics.fn))
                logger.add_meter('recall', round(iter_metrics.recall(),2))
                logger.add_meter('precision', round(iter_metrics.precision(),2))
                logger.add_meter('f1-score', round(iter_metrics.fbeta_score(),2))
                logger.add_meter('MAE', round(iter_metrics.mae(),2))
                logger.add_meter('MSE', round(iter_metrics.mse(),2))
                logger.add_meter('RMSE', round(iter_metrics.rmse(),2))

            if wandb_flag:
                wandb.log({
                    'n': sum(iter_metrics.tp) + sum(iter_metrics.fn),
                    'recall': iter_metrics.recall(),
                    'precision': iter_metrics.precision(),
                    'f1_score': iter_metrics.fbeta_score(),
                    'MAE': iter_metrics.mae(),
                    'MSE': iter_metrics.mse(),
                    'RMSE': iter_metrics.rmse()
                    })

            iter_metrics.flush()

            self.metrics.feed(**output)
        
        self._stored_metrics = self.metrics.copy()

        mAP = numpy.mean([self.metrics.ap(c) for c in range(1, self.metrics.num_classes)]).item()
        
        self.metrics.aggregate()

        if wandb_flag:
            wandb.run.summary['recall'] =  self.metrics.recall()
            wandb.run.summary['precision'] =  self.metrics.precision()
            wandb.run.summary['f1_score'] =  self.metrics.fbeta_score()
            wandb.run.summary['MAE'] =  self.metrics.mae()
            wandb.run.summary['MSE'] =  self.metrics.mse()
            wandb.run.summary['RMSE'] =  self.metrics.rmse()
            wandb.run.summary['accuracy'] =  self.metrics.accuracy()
            wandb.run.summary['mAP'] =  mAP
            wandb.run.finish()

        if returns == 'recall':
            return self.metrics.recall()
        elif returns == 'precision':
            return self.metrics.precision()
        elif returns == 'f1_score':
            return self.metrics.fbeta_score()
        elif returns == 'mse':
            return self.metrics.mse()
        elif returns == 'mae':
            return self.metrics.mae()
        elif returns == 'rmse':
            return self.metrics.rmse()
        elif returns == 'accuracy':
            return self.metrics.accuracy()
        elif returns == 'mAP':
            return mAP


        
    @property
    def results(self) -> pandas.DataFrame:
        ''' Returns metrics by class (recall, precision, f1_score, mse, mae, and rmse) 
        in a pandas dataframe '''
        
        assert self._stored_metrics is not None, \
            'No metrics have been stored, please use the evaluate method first.'
        
        metrics_cpy = self._stored_metrics.copy()
        
        res = []
        for c in range(1, metrics_cpy.num_classes):
            metrics = {
                'class': str(c),
                'n': metrics_cpy.tp[c-1] + metrics_cpy.fn[c-1],
                'recall': metrics_cpy.recall(c),
                'precision': metrics_cpy.precision(c),
                'f1_score': metrics_cpy.fbeta_score(c),
                'confusion': metrics_cpy.confusion(c), 
                'mae': metrics_cpy.mae(c),
                'mse': metrics_cpy.mse(c),
                'rmse': metrics_cpy.rmse(c),
                'ap': metrics_cpy.ap(c),
            }
            res.append(metrics)
        return pandas.DataFrame(data = res)
    
    @property
   
    def detections(self) -> pandas.DataFrame:
        assert self._stored_metrics is not None, \
            'No detections have been stored, please use the evaluate method first.'

        img_names = self.dataloader.dataset._img_names
        dets = self._stored_metrics.detections

        print("Number of detections:", len(dets))
        # print("Image names from dataloader:", img_names)

        for det in dets:
            index = det['images']
            # Check if 'index' is actually an integer index; if not, assume it's a valid image name
            if isinstance(index, int):
                if index < len(img_names):
                    det['images'] = img_names[index]
                else:
                    print(f"IndexError: {index} is out of bounds for image names with length {len(img_names)}")
                    det['images'] = 'InvalidIndex'
            elif isinstance(index, str):
                # 'index' is already the image name, so no need to change it
                continue
            else:
                print(f"TypeError: {index} is not a valid index or image name")
                det['images'] = 'InvalidIndexOrName'

        # print("Updated detections with image names:", dets)
        return pandas.DataFrame(data=dets)

@EVALUATORS.register()
class HerdNetEvaluator(Evaluator):

    def __init__(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, metrics: Metrics, 
        lmds_kwargs: dict = {'kernel_size': (3,3)}, device_name: str = 'cuda', print_freq: int = 10, 
        stitcher: Optional[Stitcher] = None, vizual_fn: Optional[Callable] = None, work_dir: Optional[str] = None, 
        header: Optional[str] = None
        ) -> None:
        super().__init__(model, dataloader, metrics, device_name=device_name, print_freq=print_freq, 
            vizual_fn=vizual_fn, stitcher=stitcher, work_dir=work_dir, header=header)

        self.lmds_kwargs = lmds_kwargs

    def prepare_data(self, images: Any, targets: Any) -> tuple:        
        return images.to(self.device), targets
    
    def post_stitcher(self, output: torch.Tensor) -> Any:
        heatmap = output[:,:1,:,:]
        clsmap = output[:,1:,:,:]
        return heatmap, clsmap

    def prepare_feeding(self, targets: Dict[str, torch.Tensor], output: List[torch.Tensor]) -> dict:

        gt_coords = [p[::-1] for p in targets['points'].squeeze(0).tolist()]
        gt_labels = targets['labels'].squeeze(0).tolist()
        
        gt = dict(
            loc = gt_coords,
            labels = gt_labels
        )

        up = True
        if self.stitcher is not None:
            up = False

        lmds = HerdNetLMDS(up=up, **self.lmds_kwargs)
        counts, locs, labels, scores, dscores = lmds(output)
        
        preds = dict(
            loc = locs[0],
            labels = labels[0],
            scores = scores[0],
            dscores = dscores[0]
        )
        
        return dict(gt = gt, preds = preds, est_count = counts[0])

@EVALUATORS.register()
class DensityMapEvaluator(Evaluator):
  
    def prepare_data(self, images: Any, targets: Any) -> tuple:        
        return images.to(self.device), targets

    def prepare_feeding(self, targets: Dict[str, torch.Tensor], output: torch.Tensor) -> dict:

        gt_coords = [p[::-1] for p in targets['points'].squeeze(0).tolist()]
        gt_labels = targets['labels'].squeeze(0).tolist()
        
        gt = dict(loc = gt_coords, labels = gt_labels)
        preds = dict(loc = [], labels = [], scores = [])

        _, idx = torch.max(output, dim=1)
        masks = F.one_hot(idx, num_classes=output.shape[1]).permute(0,3,1,2)
        output = (output * masks)
        est_counts = output[0].sum(2).sum(1).tolist()
        
        return dict(gt = gt, preds = preds, est_count = est_counts)

@EVALUATORS.register()
class DensityMapEvaluator(Evaluator):
  
    def prepare_data(self, images: Any, targets: Any) -> tuple:        
        return images.to(self.device), targets

    def prepare_feeding(self, targets: Dict[str, torch.Tensor], output: torch.Tensor) -> dict:

        gt_coords = [p[::-1] for p in targets['points'].squeeze(0).tolist()]
        gt_labels = targets['labels'].squeeze(0).tolist()
        
        gt = dict(loc = gt_coords, labels = gt_labels)
        preds = dict(loc = [], labels = [], scores = [])

        _, idx = torch.max(output, dim=1)
        masks = F.one_hot(idx, num_classes=output.shape[1]).permute(0,3,1,2)
        output = (output * masks)
        est_counts = output[0].sum(2).sum(1).tolist()
        
        return dict(gt = gt, preds = preds, est_count = est_counts)
    
@EVALUATORS.register()
class FasterRCNNEvaluator(Evaluator):

    def prepare_data(self, images: List[torch.Tensor], targets: List[dict]) -> tuple:
        images = list(image.to(self.device) for image in images)    
        targets = [{k: v.to(self.device) for k, v in t.items() if torch.is_tensor(v)} 
                            for t in targets]
        return images, targets
    
    def post_stitcher(self, output: dict) -> list:
        return [output]
    
    def prepare_feeding(self, targets: List[dict], output: List[dict]) -> dict:

        targets, output = targets[0], output[0]

        gt = dict(
            loc = targets['boxes'].tolist(),
            labels = targets['labels'].tolist()
            )
        
        preds = dict(
            loc = output['boxes'].tolist(),
            labels = output['labels'].tolist(),
            scores = output['scores'].tolist()
            )
        
        num_classes = self.metrics.num_classes - 1
        counts = [preds['labels'].count(i+1) for i in range(num_classes)]

        return dict(gt = gt, preds = preds, est_count = counts)
    
@EVALUATORS.register()
class TileEvaluator(Evaluator):
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        metrics_class: Type[Metrics],
        num_classes: int,
        device_name: str = 'cuda',
        print_freq: int = 10,
        stitcher: Optional[Stitcher] = None,
        vizual_fn: Optional[Callable] = None,
        work_dir: Optional[str] = None,
        header: Optional[str] = None,
        threshold: float = 0.3
    ):
        # Extract img_names from the dataloader's dataset
        img_names = dataloader.dataset._img_names

        # Create an instance of the metrics using the modified constructor
        metrics = metrics_class(img_names=img_names, num_classes=num_classes)

        super().__init__(
            model=model,
            dataloader=dataloader,
            metrics=metrics,
            device_name=device_name,
            print_freq=print_freq,
            stitcher=stitcher,
            vizual_fn=vizual_fn,
            work_dir=work_dir,
            header=header
        )
        self.threshold = threshold
    def prepare_data(self, images: Any, targets: Any) -> tuple: 
        if isinstance(targets, dict):
            # Check if any of the values in the targets dictionary are lists
            new_targets = {}
            for k, v in targets.items():
                if isinstance(v, list):
                    # Convert each element in the list to the device
                    new_targets[k] = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in v]
                else:
                    new_targets[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
            targets = new_targets
        elif isinstance(targets, (list, tuple)):
            # If targets is a list or tuple, move each item to the device
            targets = [tar.to(self.device) for tar in targets if isinstance(tar, torch.Tensor)]
            print("Targets are in a list or tuple.")
        else:
            print("Unexpected targets format.")
            
        images = images.to(self.device) if isinstance(images, torch.Tensor) else images
        
        return images, targets


    def prepare_feeding(self, targets: Any, output: torch.Tensor) -> dict:
        """
        Adjust targets and output for feeding into metrics.
        This version assumes targets are provided in a suitable format for binary classification.
        """
        # Check if output is a tuple and extract tensor
        if isinstance(output, tuple):
            output = output[0]

        # Ensure targets are in the correct format
        if isinstance(targets, dict):
            targets = {k: v.float().to(self.device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
        elif isinstance(targets, list):
            targets = [v.float().to(self.device) if isinstance(v, torch.Tensor) else v for v in targets]

        # Convert model output to probabilities via sigmoid
        scores = torch.sigmoid(output).float()

        # Print scores after sigmoid
        print(f"Scores after sigmoid: {scores.detach().cpu().numpy()}")

        # Convert probabilities to binary predictions (threshold)
        pred_binary = (scores > self.threshold).float()

        # Print thresholded predictions
        print(f"Thresholded predictions: {pred_binary.detach().cpu().numpy()}")

        # Prepare dictionary for feeding into metrics
        feeding_dict = {
            'gt': targets,
            'preds': {'binary': pred_binary}
        }
         
        return feeding_dict
    
    def evaluate(self, returns: str = 'recall', wandb_flag: bool = False, viz: bool = False,
                log_meters: bool = True) -> float:
        ''' Evaluate the model '''
        
        self.model.eval()

        self.metrics.flush()
        logger = CustomLogger(delimiter=' ', filename=self.logs_filename, work_dir=self.work_dir)
        iter_metrics = self.metrics.copy()

        for i, (images, targets) in enumerate(logger.log_every(self.dataloader, self.print_freq, self.header)):

            images, targets = self.prepare_data(images, targets)

            if self.stitcher is not None:
                output = self.stitcher(images[0])
                output = self.post_stitcher(output)
            else:
                if isinstance(self.model, DLAEncoderDecoder):
                    output, _ = self.model(images, targets)
                else:
                    output = self.model(images)  # Use only images for other models

            # Handle output depending on its type (tuple or tensor)
            if isinstance(output, tuple):
                output = output[0]  # Extract the first element from the tuple

            # Print raw model output before applying sigmoid
            print(f"Raw model output (before sigmoid): {output.detach().cpu().numpy()}")

            # Print the corresponding ground truth labels
            print(f"Ground truth labels: {targets}")

            # Prepare the output for metrics
            output = self.prepare_feeding(targets, output)

            # Feed the output to metrics
            iter_metrics.feed(**output)

            # Log metrics if enabled
            if log_meters:
                logger.add_meter('threshold', self.threshold)
                logger.add_meter('n', sum(iter_metrics.tp) + sum(iter_metrics.fn))
                logger.add_meter('recall', round(iter_metrics.recall(),2))
                logger.add_meter('precision', round(iter_metrics.precision(),2))
                logger.add_meter('f1-score', round(iter_metrics.fbeta_score(),2))
                logger.add_meter('MAE', round(iter_metrics.mae(),2))
                logger.add_meter('MSE', round(iter_metrics.mse(),2))
                logger.add_meter('RMSE', round(iter_metrics.rmse(),2))

            if wandb_flag:
                wandb.log({
                    'threshold': self.threshold,
                    'n': sum(iter_metrics.tp) + sum(iter_metrics.fn),
                    'recall': iter_metrics.recall(),
                    'precision': iter_metrics.precision(),
                    'f1_score': iter_metrics.fbeta_score(),
                    'MAE': iter_metrics.mae(),
                    'MSE': iter_metrics.mse(),
                    'RMSE': iter_metrics.rmse()
                })

            iter_metrics.flush()

            self.metrics.feed(**output)

        # Print final stored metrics
        self._stored_metrics = self.metrics.copy()
        print("Stored metrics:", self._stored_metrics)    
        print("Final Recall:", self._stored_metrics.recall())
        print("Final Precision:", self._stored_metrics.precision())
        mAP = numpy.mean([self.metrics.ap(c) for c in range(1, self.metrics.num_classes)]).item()

        # Log final metrics if using wandb
        if wandb_flag:
            wandb.run.summary['threshold'] = self.threshold
            wandb.run.summary['recall'] = self.metrics.recall()
            wandb.run.summary['precision'] = self.metrics.precision()
            wandb.run.summary['f1_score'] = self.metrics.fbeta_score()
            wandb.run.summary['MAE'] = self.metrics.mae()
            wandb.run.summary['MSE'] = self.metrics.mse()
            wandb.run.summary['RMSE'] = self.metrics.rmse()
            wandb.run.summary['accuracy'] = self.metrics.accuracy()
            wandb.run.summary['mAP'] = mAP
            wandb.run.finish()

        if returns == 'recall':
            return self.metrics.recall()
        elif returns == 'precision':
            return self.metrics.precision()
        elif returns == 'f1_score':
            return self.metrics.fbeta_score()
        elif returns == 'mse':
            return self.metrics.mse()
        elif returns == 'mae':
            return self.metrics.mae()
        elif returns == 'rmse':
            return self.metrics.rmse()
        elif returns == 'accuracy':
            return self.metrics.accuracy()
        elif returns == 'mAP':
            return mAP
def detection_test(self) -> pandas.DataFrame:
        """
        Specific test-time detection function to output predictions and ground truth.
        Saves results for each image in a DataFrame.

        Returns:
            pd.DataFrame containing 'image_name', 'true_binary', and 'predicted_binary' columns.
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            for images, targets in self.dataloader:
                images = images.to(self.device)
                targets['binary'] = targets['binary'].to(self.device)

                # Inference
                outputs = self.model(images)

                # If outputs is a tuple, assume the first element is the prediction tensor
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # Apply sigmoid to the output to obtain probabilities
                outputs = torch.sigmoid(outputs)

                # Collect predictions and ground truth
                for i in range(images.size(0)):
                    image_name = targets['image_name'][i]
                    predicted_binary = 1 if outputs[i].item() >= self.threshold else 0
                    true_binary = targets['binary'][i].item()

                    results.append({
                        'image_name': image_name,
                        'true_binary': true_binary,
                        'predicted_binary': predicted_binary
                    })

                    # print(f"Image: {image_name}, True Binary: {true_binary}, Predicted Binary: {predicted_binary}")

        # Convert results to DataFrame and return
        results_df = pandas.DataFrame(results)
        return results_df
