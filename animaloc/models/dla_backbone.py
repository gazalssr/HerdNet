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

import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torch.nn import BCEWithLogitsLoss
from typing import Optional

from .register import MODELS

from . import dla as dla_modules


@MODELS.register()
#DLA encoder with batch normalization
class DLAEncoder(nn.Module):
    ''' DLA encoder architecture '''

    def __init__(
        self,
        num_layers: int = 34,
        num_classes: int = 2,
        pretrained: bool = True, 
    ):
        '''
        Args:
            num_layers (int, optional): number of layers of DLA. Defaults to 34.
            num_classes (int, optional): number of output classes, background included. 
                Defaults to 2.
            pretrained (bool, optional): set False to disable pretrained DLA encoder parameters
                from ImageNet. Defaults to True.
        '''
        super(DLAEncoder, self).__init__()
        
        base_name = 'dla{}'.format(num_layers)
        self.num_classes = num_classes

        # backbone
        base = dla_modules.__dict__[base_name](pretrained=pretrained, return_levels=True)
        setattr(self, 'base_0', base)
        setattr(self, 'channels_0', base.channels)

        channels = self.channels_0

        # Add batch normalization to each convolutional layer
        self.base_0_bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features) for num_features in channels
        ])

        # bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            channels[-1], channels[-1], 
            kernel_size=1, stride=1, 
            padding=0, bias=True
        )

        # Add batch normalization after the bottleneck convolution
        self.bottleneck_bn = nn.BatchNorm2d(channels[-1])
        
        self.pooling = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)  # we take the average of each filter
        self.cls_head = nn.Linear(512, 1)  # binary head

    def load_custom_pretrained_weights(self, weight_path):
        ''' Load custom pretrained weights into the model. '''
        # print(f"Loading weights from: {weight_path}")
        pretrained_dict = torch.load(weight_path, map_location='cuda')
        adapted_pretrained_dict = self.adapt_keys(pretrained_dict)

        model_dict = self.state_dict()
        # print("Model keys:", model_dict.keys())  #print model keys
        # print("Adapted pretrained keys:", adapted_pretrained_dict.keys())  #print adapted keys

        # Filter out unmatched keys and size mismatches
        pretrained_dict = {k: v for k, v in adapted_pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        if not pretrained_dict:
            print("No matching keys found or size mismatch.")
        else:
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            print("Custom pretrained weights loaded successfully.")
    
    
    def forward(self, input: torch.Tensor):
        encode = self.base_0(input)  # Nx512x16x16
        
        # Apply batch normalization to each encoded layer
        for i in range(len(encode)):
            encode[i] = self.base_0_bn_layers[i](encode[i])
        
        bottleneck = self.bottleneck_conv(encode[-1])
        
        # Apply batch normalization
        bottleneck = self.bottleneck_bn(bottleneck)
        
        bottleneck = self.pooling(bottleneck)
        bottleneck = torch.reshape(bottleneck, (bottleneck.size()[0], -1))  # keeping the first dimension (samples)
        encode[-1] = bottleneck  # Nx512
        cls = self.cls_head(encode[-1])
        
        # cls = nn.functional.sigmoid(cls)
        return cls

    def freeze(self, layers: list) -> None:
        ''' Freeze all layers mentioned in the input list '''
        for layer in layers:
            self._freeze_layer(layer)
    
    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False
    def adapt_keys(self, pretrained_dict):
        ''' Adapt keys from the pretrained model to fit the current model structure. '''
        new_keys = {}
        for k in pretrained_dict:
            # Adapt the keys based on the structure of model_dict
            new_key = 'base_0.' + k 
            new_keys[new_key] = pretrained_dict[k]
        return new_keys
    
###################################### DLA Autoencoder ############################

@MODELS.register()
class DLAEncoderDecoder(nn.Module):
    ''' DLA Encoder-Decoder architecture for binary patch detection '''

    def __init__(
        self,
        num_layers: int = 34,
        num_classes: int = 2,
        pretrained: bool = True, 
        down_ratio: Optional[int] = 2, 
        head_conv: int = 64,
    ):
        super(DLAEncoderDecoder, self).__init__()

        # Initialize the model
        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'
        
        base_name = 'dla{}'.format(num_layers)
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv

        # Calculate first level
        self.first_level = int(np.log2(down_ratio))

        # Backbone
        base = dla_modules.__dict__[base_name](pretrained=pretrained, return_levels=True)
        setattr(self, 'base_0', base)
        setattr(self, 'channels_0', base.channels)

        channels = self.channels_0
    # Upsampling and rescaling in different scales in decoder
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = dla_modules.DLAUp(channels[self.first_level:], scales=scales)

        # Bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            channels[-1], channels[-1], 
            kernel_size=1, stride=1, 
            padding=0, bias=True
        )

        # Localization head (similar to HerdNet)
        self.loc_head = nn.Sequential(
            nn.Conv2d(channels[self.first_level], head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, 1, 
                kernel_size=1, stride=1, 
                padding=0, bias=True
                )
            # nn.Sigmoid()
        )

        self.loc_head[-1].bias.data.fill_(0.00)
    def load_custom_pretrained_weights(self, weight_path):
        ''' Load custom pretrained weights into the model. '''
        # print(f"Loading weights from: {weight_path}")
        pretrained_dict = torch.load(weight_path, map_location='cuda')
        adapted_pretrained_dict = self.adapt_keys(pretrained_dict)

        model_dict = self.state_dict()
        # print("Model keys:", model_dict.keys())  #print model keys
        # print("Adapted pretrained keys:", adapted_pretrained_dict.keys())  #print adapted keys

        # Filter out unmatched keys and size mismatches
        pretrained_dict = {k: v for k, v in adapted_pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        if not pretrained_dict:
            print("No matching keys found or size mismatch.")
        else:
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            print("Custom pretrained weights loaded successfully.")


    def forward(self, input: torch.Tensor):
        # Encoder
        encode = self.base_0(input)    
        bottleneck = self.bottleneck_conv(encode[-1])
        encode[-1] = bottleneck

        # Decoder
        decode_hm = self.dla_up(encode[self.first_level:])
        heatmap = self.loc_head(decode_hm)
        # print("Heatmap values (before activation):", heatmap.detach().cpu().numpy())
        # print("Heatmap size:", heatmap.size())
        # Compute the mean of the heatmap
        density_mean = heatmap.mean(dim=[2, 3])  # Average over height and width

        return density_mean
###### to vizualize the heatmaps
    # def forward(self, input: torch.Tensor):
    #     # Encoder
    #     encode = self.base_0(input)    
    #     bottleneck = self.bottleneck_conv(encode[-1])
    #     encode[-1] = bottleneck

    #     # Decoder
    #     decode_hm = self.dla_up(encode[self.first_level:])
    #     heatmap = self.loc_head(decode_hm)  # This should be the heatmap you want to visualize

    #     # Compute the mean of the heatmap
    #     density_mean = heatmap.mean(dim=[2, 3])  # Average over height and width

    #     return heatmap, density_mean  # Return both the heatmap and the final output
    def freeze(self, layers: list) -> None:
        ''' Freeze all layers mentioned in the input list '''
        for layer in layers:
            self._freeze_layer(layer)
    
    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False
   

    ########################################################
    def adapt_keys(self, pretrained_dict):
        ''' Adapt keys from the pretrained model to fit the current model structure. '''
        new_keys = {}
        for k in pretrained_dict:
            # Adapt the keys based on the structure of model_dict
            new_key = 'base_0.' + k 
            new_keys[new_key] = pretrained_dict[k]
        return new_keys

  

########################################################################

    def freeze(self, layers: list) -> None:
        ''' Freeze all layers mentioned in the input list '''
        for layer in layers:
            self._freeze_layer(layer)
    
    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False

