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

from typing import Optional

from .register import LOSSES

@LOSSES.register()
class FocalLoss(torch.nn.Module):
    ''' Focal Loss module '''

    def __init__(
        self, 
        alpha: int = 2, 
        beta: int = 4, 
        reduction: str = 'sum',
        weights: Optional[torch.Tensor] = None,
        density_weight: Optional[str] = None,
        normalize: bool = False,
        eps: float = 1e-6
        ) -> None:
        '''
        Args:
            alpha (int, optional): alpha parameter. Defaults to 2
            beta (int, optional): beta parameter. Defaults to 4
            reduction (str, optional): batch losses reduction. Possible
                values are 'sum' and 'mean'. Defaults to 'sum'
            weights (torch.Tensor, optional): channels weights, if specified
                must be a torch Tensor. Defaults to None
            density_weight (str, optional): to weight each sample by objects density 
                (high factor for high density). Possible values are: 'linear', 'squared', 
                or 'cubic' for choosing a linear, squared or cubic exponent to apply to
                the number of locations. Defaults to None
            normalize (bool, optional): set to True to normalize the loss according to 
                the number of positive samples. Defaults to False
            eps (float, optional): for numerical stability. Defaults to 1e-6.
        '''

        super().__init__()

        assert reduction in ['mean', 'sum'], \
            f'Reduction must be either \'mean\' or \'sum\', got {reduction}'

        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.weights = weights
        self.density_weight = density_weight
        self.normalize = normalize
        self.eps = eps
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            output (torch.Tensor): [B,C,H,W]
            target (torch.Tensor): [B,C,H,W]
        
        Returns:
            torch.Tensor
        '''

        return self._neg_loss(output, target)

    def _neg_loss(self, output: torch.Tensor, target: torch.Tensor):
        ''' Focal loss, adapted from CenterNet 
        https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
        Args:
            output (torch.Tensor): [B,C,H,W]
            target (torch.Tensor): [B,C,H,W]
        
        Returns:
            torch.Tensor
        '''

        B, C, _, _ = target.shape

        if self.weights is not None:
            assert self.weights.shape[0] == C, \
                'Number of weights must match the number of channels, ' \
                    f'got {C} channels and {self.weights.shape[0]} weights'

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, self.beta)

        loss = torch.zeros((B,C))

         # avoid NaN when net output is 1.0 or 0.0
        output = torch.clamp(output, min=self.eps, max=1-self.eps)

        pos_loss = torch.log(output) * torch.pow(1 - output, self.alpha) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(output, self.alpha) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum(3).sum(2)
        pos_loss = pos_loss.sum(3).sum(2)
        neg_loss = neg_loss.sum(3).sum(2)

        for b in range(B):
            for c in range(C):
                density = torch.tensor([1]).to(neg_loss.device)
                if self.density_weight == 'linear':
                    density = num_pos[b][c]
                elif self.density_weight == 'squared':
                    density = num_pos[b][c] ** 2
                elif self.density_weight == 'cubic':
                    density = num_pos[b][c] ** 3

                if num_pos[b][c] == 0:
                    loss[b][c] = loss[b][c] - neg_loss[b][c]
                else:
                    loss[b][c] = density * (loss[b][c] - (pos_loss[b][c] + neg_loss[b][c]))
                    if self.normalize:
                         loss[b][c] =  loss[b][c] / num_pos[b][c]
        
        if self.weights is not None:
            loss = self.weights * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
###########################BinaryFocal loss####################

import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(
        self, 
         alpha_pos=0.85, 
         alpha_neg=0.15, 
         beta=4, 
         reduction='mean', 
         weights=None, 
         eps=1e-6):
        super(BinaryFocalLoss, self).__init__()
        self.alpha_pos = alpha_pos  # Alpha for controlling emphasis on positive samples
        self.alpha_neg = alpha_neg  # Alpha for controlling emphasis on negative samples
        self.beta = beta    # beta focuses more on hard examples
        self.reduction = reduction
        self.weights = weights  # Weights for each class
        self.eps = eps
    

    def forward(self, outputs, targets):
        # Ensuring outputs are clamped to avoid log(0)
        outputs = torch.clamp(outputs, min=self.eps, max=1 - self.eps)

        # Calculate the basic binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        probas = torch.sigmoid(outputs)
        p_t = torch.where(targets == 1, probas, 1 - probas)
        focal_factor = (1 - p_t) ** self.beta
        focal_loss = focal_factor * bce_loss
        alpha_factor = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)
        focal_loss = alpha_factor * focal_loss

        # Apply class-specific weights if provided
        if self.weights is not None:
            weight_factor = torch.where(targets == 1, self.weights[1], self.weights[0])
            focal_loss = weight_factor * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
####################################### Focal Combo Loss with regular dice loss ###########################################
# class FocalComboLoss(nn.Module):
#     def __init__(
#         self, 
#         alpha_pos=0.75, 
#         alpha_neg=0.25, 
#         beta=4, 
#         reduction='mean', 
#         weights=None, 
#         dice_weight=0.5,  # Parameter to balance between focal loss and dice loss
#         eps=1e-6):
#         super(FocalComboLoss, self).__init__()
#         self.alpha_pos = alpha_pos
#         self.alpha_neg = alpha_neg
#         self.beta = beta
#         self.reduction = reduction
#         self.weights = weights
#         self.dice_weight = dice_weight  # Weight for dice loss component
#         self.eps = eps

#     def forward(self, outputs, targets):

#         outputs = torch.clamp(outputs, min=self.eps, max=1 - self.eps)
#         bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
#         probas = torch.sigmoid(outputs)
#         p_t = torch.where(targets == 1, probas, 1 - probas)
#         focal_factor = (1 - p_t) ** self.beta
#         alpha_factor = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)
#         focal_loss = alpha_factor * focal_factor * bce_loss

#         if self.weights is not None:
#             weight_factor = torch.where(targets == 1, self.weights[1], self.weights[0])
#             focal_loss = weight_factor * focal_loss

#         # Calculate the Dice Loss
#         smooth = 1e-6
#         inputs = torch.sigmoid(outputs)
#         intersection = (inputs * targets).sum()
#         total = (inputs + targets).sum()
#         dice_score = (2. * intersection + smooth) / (total + smooth)
#         dice_loss = 1 - dice_score

#         # Combine the losses
#         combined_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss

#         if self.reduction == 'mean':
#             combined_loss = combined_loss.mean()
#         elif self.reduction == 'sum':
#             combined_loss = combined_loss.sum()

#         return combined_loss
########################### Focal Combo loss with modified dice loss( paper) ################
class FocalComboLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,  # Balance between focal and dice loss components
        beta=2,  # Modifier for the Dice Loss
        gamma=2,  # Focusing parameter for the Focal Loss
        reduction='mean', 
        weights=None,  # Weights for empty and non-empty (or positive and negative)
        eps=1e-6):
        super(FocalComboLoss, self).__init__()
        self.alpha = alpha  # Overall balance between focal and dice loss
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights  # Array or tuple with [weight_for_negative, weight_for_positive]
        self.eps = eps

    def forward(self, outputs, targets):
        outputs = torch.clamp(outputs, min=self.eps, max=1 - self.eps)
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        
        probas = torch.sigmoid(outputs)
        p_t = torch.where(targets == 1, probas, 1 - probas)
        focal_factor = (1 - p_t) ** self.gamma
        alpha_factor = torch.where(targets == 1, self.weights[1], self.weights[0])
        focal_loss = alpha_factor * focal_factor * bce_loss

        # Calculate the modified Dice Loss
        inputs = torch.sigmoid(outputs)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        dice_score = (2. * intersection + self.eps) / (total + self.eps)
        modified_dice_score = dice_score ** (1 / self.beta)
        dice_loss = 1 - modified_dice_score

        # Combine the losses using alpha as the balancing factor
        combined_loss = (1 - self.alpha) * focal_loss.mean() + self.alpha * dice_loss.mean()

        if self.reduction == 'mean':
            combined_loss = combined_loss.mean()
        elif self.reduction == 'sum':
            combined_loss = combined_loss.sum()

        return combined_loss