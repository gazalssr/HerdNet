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
class BinaryFocalLoss(torch.nn.Module):
    def __init__(
        self, 
        alpha_pos=2, 
        alpha_neg=1, 
        beta=4, 
        reduction='mean', 
        weights=None, 
        eps=1e-6
    ):
        super(BinaryFocalLoss, self).__init__()
        self.alpha_pos = alpha_pos  # Emphasis on positive samples
        self.alpha_neg = alpha_neg  # Emphasis on negative samples
        self.beta = beta  # Focusing parameter for hard examples
        self.reduction = reduction
        self.weights = weights  # Class weights
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw logits output from the model (not probabilities).
            targets (torch.Tensor): Ground truth labels (0 or 1).
        """
        # Ensure logits and targets are of the same size
        if targets.size() != logits.size():
            targets = targets.view_as(logits)

        # Binary cross-entropy loss with logits (no sigmoid applied yet)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')

        # Compute probabilities using logits
        probas = torch.sigmoid(logits)

        # Compute p_t based on targets
        p_t = torch.where(targets == 1, probas, 1 - probas)

        # Apply focal factor
        focal_factor = (1 - p_t) ** self.beta
        focal_loss = focal_factor * bce_loss

        # Apply alpha scaling
        alpha_factor = torch.where(
            targets == 1,
            torch.tensor(self.alpha_pos, dtype=logits.dtype, device=logits.device),
            torch.tensor(self.alpha_neg, dtype=logits.dtype, device=logits.device)
        )
        focal_loss = alpha_factor * focal_loss

        # Apply class-specific weights if provided
        if self.weights is not None:
            weight_factor = torch.where(targets == 1, self.weights[1], self.weights[0])
            focal_loss = weight_factor * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


####################################### My Modified Focal Combo Loss with regular dice loss ###########################################
class FocalComboLoss_M(torch.nn.Module):
    def __init__(
        self, 
        alpha_pos=0.75, 
        alpha_neg=0.25, 
        gamma=4, 
        reduction='mean', 
        weights=None, 
        dice_weight=0.5,  # Parameter to balance between focal loss and dice loss (alpha in the paper)
        eps=1e-6):
        super(FocalComboLoss_M, self).__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights
        self.dice_weight = dice_weight  # Weight for dice loss component
        self.eps = eps

    def forward(self, outputs, targets):

        outputs = torch.clamp(outputs, min=self.eps, max=1 - self.eps)
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        probas = torch.sigmoid(outputs)
        p_t = torch.where(targets == 1, probas, 1 - probas)
        focal_factor = (1 - p_t) ** self.gamma
        alpha_factor = torch.where(targets == 1, self.alpha_pos, self.alpha_neg) # The cahnge compared to satndard Fcal Loss
        focal_loss = alpha_factor * focal_factor * bce_loss

        if self.weights is not None:
            weight_factor = torch.where(targets == 1, self.weights[1], self.weights[0])
            focal_loss = weight_factor * focal_loss

        # Calculate the Dice Loss
        smooth = 1e-6
        inputs = torch.sigmoid(outputs)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        dice_score = (2. * intersection + smooth) / (total + smooth)
############################## Modified dice score has to be placed here (if u want it for test) ##########
        dice_loss = 1 - dice_score

        # Combine the losses
        combined_loss = self.dice_weight * focal_loss + (1-self.dice_weight) * dice_loss

        if self.reduction == 'mean':
            combined_loss = combined_loss.mean()
        elif self.reduction == 'sum':
            combined_loss = combined_loss.sum()

        return combined_loss
########################### Focal Combo loss with modified dice loss( paper) ################

import torch.nn.functional as F
class FocalComboLoss_P(torch.nn.Module):
    def __init__(
        self,
        alpha=0.25,
        beta=2,
        gamma=1,
        reduction='mean', 
        weights=None,
        eps=1e-6):
        super(FocalComboLoss_P, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights
        self.eps = eps
        
        if self.weights is None:
            self.weights = [1.0, 1.0]

    def forward(self, outputs, targets):
        if self.weights is None:
            raise ValueError("Weights are not defined.")
        # Ensure outputs and targets have the same shape
        if outputs.shape != targets.shape:
            targets = targets.view_as(outputs)
        
        # Cast targets to float to match the type of outputs
        targets = targets.float()

        outputs = torch.clamp(outputs, min=self.eps, max=1 - self.eps)
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        
        # Compute focal loss component
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Compute dice loss component
        smooth = 1.
        iflat = outputs.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) /
                         (iflat.sum() + tflat.sum() + smooth))
        
        # Combine the two loss components
        combined_loss = self.weights[0] * focal_loss + self.weights[1] * dice_loss

        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:
            return combined_loss
    ########### Density loss for the decoder #########
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss
    
import torch

class DensityLoss(torch.nn.Module):
    def __init__(
        self,
        reduction: str = 'mean',
        eps: float = 1e-6
        ) -> None:

        super().__init__()

        assert reduction in ['mean', 'max'], \
            f'Reduction must be either \'mean\' or \'max\', got {reduction}'

        self.reduction = reduction
        self.eps = eps
        self.th= 1e-6
        # take a single input feature and produces a single output feature
        self.binary_head= torch.nn.Linear(1,1)
        self.loss = BinaryFocalLoss(alpha_pos=2, alpha_neg=1, beta=4, reduction=reduction)
        # self.loss = BCEWithLogitsLoss()
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            output (torch.Tensor): [B,C,H,W]
            target (torch.Tensor): [B,C,H,W]

        Returns:
            torch.Tensor
        '''

        return self._neg_loss(output, target)
 ################# neg_loss based on Max values in heatmap #####
    # def _neg_loss(self, output: torch.Tensor, target: torch.Tensor):
    #     output = torch.clamp(output, min=self.eps, max=1-self.eps)
        
    #     # Use max over spatial dimensions to determine if there's a significant activation
    #     if output.dim() == 4:  # [batch_size, channels, height, width]
    #         m = output.max(dim=[1, 2, 3])[0]  # max over spatial dimensions
    #     elif output.dim() == 3:  # [batch_size, channels, spatial_dim]
    #         m = output.max(dim=[1, 2])[0]  # max over channels and spatial dimensions
    #     elif output.dim() == 2:  # [batch_size, channels]
    #         m = output.max(dim=1)[0]  # max over channels
    #     else:
    #         raise ValueError(f"Unexpected output tensor with shape: {output.shape}")
        
    #     # Apply threshold to filter out low activations
    #     m = torch.where(m > self.eps, m, torch.zeros_like(m))
        
    #     # Ensure m has shape [batch_size, 1]
    #     m = m.unsqueeze(1)

    #     # Move m and target to the same device as binary_head
    #     device = self.binary_head.weight.device
    #     m = m.to(device)
    #     target = target.to(device)

    #     logits = self.binary_head(m)
    #     return self.loss(logits, target)

    
 ###############################################################
    def _neg_loss(self, output: torch.Tensor, target: torch.Tensor):
        # output = torch.clamp(output, min=self.eps, max=1-self.eps)

        # If target is a dictionary, extract the tensor (assuming a 'binary' key)
        if isinstance(target, dict):
            target = target.get('binary', None)
            if target is None:
                raise ValueError("The target dictionary does not contain a 'binary' key.")

        # Check the number of dimensions and compute the mean accordingly
        if output.dim() == 4:  # [batch_size, channels, height, width]
            m = output.mean(dim=[1, 2, 3])  # density mean over the spatial dimensions
        elif output.dim() == 3:  # [batch_size, channels, spatial_dim]
            m = output.mean(dim=[1, 2])  # density mean over channels and spatial dimensions
        elif output.dim() == 2:  # [batch_size, channels]
            m = output.mean(dim=1)  # density mean over channels
        else:
            raise ValueError(f"Unexpected output tensor with shape: {output.shape}")

        # Ensure m has shape [batch_size, 1]
        m = m.unsqueeze(1)

        # Move m to the same device as binary_head
        m = m.to(self.binary_head.weight.device)

        # Compute logits using the binary head
        logits = self.binary_head(m)

        # Move the target to the same device as logits
        target = target.to(logits.device)

        # Reshape or squeeze target to match logits' shape
        if target.shape != logits.shape:
            target = target.view_as(logits)  # Reshape target to match logits

        # Convert target to float (same as logits)
        target = target.float()

        # Compute the loss
        return self.loss(logits, target)


