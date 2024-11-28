import random
import animaloc
import pandas as pd
import os

# Set the seed
from animaloc.utils.seed import set_seed
import numpy as np
set_seed(9292)

import matplotlib.pyplot as plt
from animaloc.datasets import FolderDataset, CSVDataset, BinaryFolderDataset
from animaloc.data.batch_utils import show_batch, collate_fn
from animaloc.data.samplers import BinaryBatchSampler, DataAnalyzer
from torch.utils.data import DataLoader
import torch
import albumentations as A
from animaloc.data.transforms import  BinaryMultiTransformsWrapper,MultiTransformsWrapper, DownSample, PointsToMask, BinaryTransform
import wandb
import torch
from animaloc.models import DLAEncoder, DLAEncoderDecoder
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss, BinaryFocalLoss, FocalComboLoss_M, FocalComboLoss_P, DensityLoss
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss
from albumentations.pytorch import ToTensorV2
from animaloc.vizual.plots import PlotTradeOff
from animaloc.data.samplers import DataAnalyzer
NUM_WORKERS= 2
import albumentations as A



binary=True
preprocess=False
patch_size = 512
num_classes = 2
batch_size=8
down_ratio = 2
train_dataset = BinaryFolderDataset(
    preprocess=preprocess,
    csv_file='/herdnet/DATASETS/TRAIN_patches_no_margin_5/TRAIN_gt_no_margin.csv',
    root_dir='/herdnet/DATASETS/TRAIN_patches_no_margin_5',
    albu_transforms=A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Blur(blur_limit=15, p=0.2),
        A.Normalize(p=1.0),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        ToTensorV2()
    ]),
    end_transforms=[BinaryMultiTransformsWrapper([
        BinaryTransform(),
    ])]
)
# train_dataset.data.to_csv('/herdnet/DATASETS/TRAIN_Patches_5/Train_binary_gt.csv', index=False)

import os
from PIL import Image
val_dataset = BinaryFolderDataset(
    preprocess=preprocess,
    csv_file='/herdnet/DATASETS/VAL_patches_no_margin_5/VAL_gt_no_margin.csv',
    root_dir='/herdnet/DATASETS/VAL_patches_no_margin_5/',
    
    albu_transforms=A.Compose([
        A.Normalize(p=1.0),
        ToTensorV2()
    ]),
    end_transforms=[BinaryMultiTransformsWrapper([
        BinaryTransform(),
    ])]
)
# val_dataset.data.to_csv('/herdnet/DATASETS/VAL_Patches_5/Val_binary_gt.csv', index=False)

test_dataset = BinaryFolderDataset(
    preprocess=preprocess,
    csv_file='/herdnet/DATASETS/TEST_patches_no_margin_5/TEST_gt_no_margin.csv',
    root_dir='/herdnet/DATASETS/TEST_patches_no_margin_5/',
    
    albu_transforms=A.Compose([
        A.Normalize(p=1.0),
        ToTensorV2()
    ]),
    end_transforms=[BinaryMultiTransformsWrapper([
        BinaryTransform(),
    ])]
)

# Initialize BinaryBatchSampler for training
train_sampler = BinaryBatchSampler(
    dataset=train_dataset,
    col='binary',  
    batch_size=16,  # Even batch_size
    shuffle=True
)
val_sampler = BinaryBatchSampler(
    dataset=val_dataset,
    col='binary',  
    batch_size=8,  # Even batch_size
    shuffle=False
)
# test_dataset.data.to_csv('/herdnet/DATASETS/All_herds_30_no_margins/Test_density/Test_binary_gt.csv', index=False)

# Dataloaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset = train_dataset, sampler=train_sampler, collate_fn=BinaryFolderDataset.collate_fn, num_workers=NUM_WORKERS)

# num_workers= 2
val_dataloader = DataLoader(dataset = val_dataset, sampler=val_sampler, collate_fn=BinaryFolderDataset.collate_fn, num_workers=NUM_WORKERS)

test_dataloader= DataLoader(dataset = test_dataset, batch_size=1 , shuffle= False, collate_fn=BinaryFolderDataset.collate_fn)
# torch.cuda.empty_cache()
num_classes=2
image= torch.ones([1,3,512,512]).cuda()

######### Imagenet initializations ########################
dla_encoder_decoder = DLAEncoderDecoder(num_classes=num_classes, pretrained=False, down_ratio=down_ratio).cuda()
dla_encoder_decoder.load_custom_pretrained_weights('/herdnet/pth_files/dla34-ba72cf86.pth')

######### Random initialization ########################
# Define a function for weight initialization (random weights) ###########
import torch.nn as nn
def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
# Initialize model without pretrained weights
# dla_encoder_decoder = DLAEncoderDecoder(num_classes=num_classes, pretrained=False, down_ratio=down_ratio).cuda()
# # Apply the weight initialization function to the model
# dla_encoder_decoder.apply(initialize_weights)
######## Density Loss ############
losses = [
    {'loss': DensityLoss(reduction='mean', eps=1e-6).to("cuda"), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'density_loss'},
]
# for loss in losses:
#     for name, param in loss['loss'].named_parameters():
#         print(f"Loss parameter {name} is on device {param.device}")

######## Density Loss ############
dla_encoder_decoder = LossWrapper(dla_encoder_decoder, losses=losses)
#Freezing the layers in the network
def get_parameter_names(model): # getting the model layers
  param_dict= dict()
  for l, (name,param) in enumerate(model.named_parameters()):
    #print(l,":\t",name,type(param),param.requires_grad)
    param_dict[name]= l
  return param_dict

def freeze_parts(model, get_parameter_names, layers_to_freeze, lr, unfreeze=False):
    params_to_update = []

    for l, (name, param) in enumerate(model.named_parameters()):
        res = any(ele in name for ele in layers_to_freeze)
        param.requires_grad = unfreeze if res else not unfreeze

        if param.requires_grad == True:
            params_to_update.append({
                "params": param,
                "lr": lr,
            })

        # # Print parameters to update
        # if param.requires_grad:
        #      print(f"Trainable parameter: {name}")
        # else:
        #     print(f"Frozen parameter: {name}")

    return params_to_update
layers_to_freeze= []

###### Layers to Freeze ##############
#AUTO_ENCODER
param_dict= get_parameter_names(dla_encoder_decoder)

# print(param_dict)
lr = 2e-4 # learning rate
params_to_update = freeze_parts(dla_encoder_decoder,param_dict, layers_to_freeze,lr,False)
#### Create the Trainer #######
from torch.optim import Adam

from animaloc.train import Trainer
from animaloc.eval import ImageLevelMetrics, HerdNetStitcher, TileEvaluator
from animaloc.utils.useful_funcs import mkdir

work_dir = '/herdnet/val_output'
# mkdir(work_dir)
weight_decay = 1e-4
epochs = 10

optimizer = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)
lr_milestones = [100, 180, 250]  # Example milestones
auto_lr = {'mode': 'min', 'factor': 0.1, 'patience': 10, 'verbose': True}
img_names = val_dataloader.dataset._img_names
metrics = ImageLevelMetrics(img_names=img_names, num_classes=2)

metrics.binary_annotations = True

evaluator = TileEvaluator(
    metrics_class=ImageLevelMetrics,
    threshold=0.5,
    model=dla_encoder_decoder,
    # model=model,
    dataloader=val_dataloader,
    num_classes=num_classes,
    stitcher=None,
    work_dir=work_dir,
    header='validation',
    )

trainer = Trainer(
    model=dla_encoder_decoder,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    num_epochs=epochs,
    loss_fn=None,
    loss_dicts=losses,
    evaluator=evaluator, 
    lr_milestones=lr_milestones,  # Pass the milestones
    auto_lr=auto_lr,
    best_model_path='/herdnet/pth_files',
    val_dataloader= val_dataloader, # loss evaluation
    patience=15,
    work_dir=work_dir
    )

if wandb.run is not None:
  wandb.finish()

wandb.init(project="herdnet_pretrain")
# ####### Ploting Heatmaps ############
# output_dir = '/herdnet/val_output/before_training'
# os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# def generate_heatmaps(val_dataloader, model, epoch, work_dir):
#     output_dir = os.path.join(work_dir, f'epoch_{epoch}_heatmaps')
#     os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

#     # Ensure the model is in evaluation mode
#     model.eval()

#     # Get a batch from the validation dataloader
#     images, targets = next(iter(val_dataloader))  
#     images = images.to('cuda')  # Move images to GPU

#     # Get heatmaps
#     with torch.no_grad():  # Disable gradient computation
#         output = model(images)  # Get the output from the model

#     # Extract heatmaps from the output tuple
#     heatmaps = output[0][0]  # The heatmaps tensor
#     heatmaps = heatmaps.cpu().numpy()  # Move the heatmaps to CPU and convert to numpy

#     # Get image IDs from the targets
#     image_ids = targets['image_name'][0]  # Extract the list of image names

#     # Ensure the lengths match before proceeding
#     assert len(image_ids) == heatmaps.shape[0], "Mismatch between the number of heatmaps and image IDs!"

#     # Plot and save the heatmaps for each image in the batch
#     for j in range(heatmaps.shape[0]):
#         # Normalize the heatmap values to range [0, 1]
#         heatmap = heatmaps[j, 0]
#         heatmap_min = heatmap.min()
#         heatmap_max = heatmap.max()
#         normalized_heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-6)  # Avoid division by zero
        
#         plt.figure(figsize=(10, 10))
#         plt.imshow(normalized_heatmap, cmap='hot', interpolation='nearest')  # Plot normalized heatmap
#         plt.title(f'Heatmap for Image ID {image_ids[j]} (Epoch {epoch})')
        
#         # Save the plot with the image ID and epoch number in the filename
#         output_path = os.path.join(output_dir, f'heatmap_image_{image_ids[j]}_epoch_{epoch}.png')
#         plt.savefig(output_path)
#         plt.close()

#     print(f"Heatmaps saved for epoch {epoch} in {output_dir}")

# def train_with_heatmap_generation(train_dataloader, val_dataloader, model, num_epochs, checkpoint_epoch, work_dir, warmup_iters=None):
#     optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-3)
#     density_loss = DensityLoss()  # Assuming DensityLoss is defined elsewhere in your code

# # ###### Ploting Heatmaps ############
# output_dir = '/herdnet/val_output/before_training'
# os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# def generate_heatmaps(val_dataloader, model, epoch, work_dir):
#     output_dir = os.path.join(work_dir, f'epoch_{epoch}_heatmaps')
#     os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

#     # Ensure the model is in evaluation mode
#     model.eval()

#     # Get a batch from the validation dataloader
#     images, targets = next(iter(val_dataloader))  
#     images = images.to('cuda')  # Move images to GPU
    
#     # Generate heatmaps (with gradients disabled)
#     with torch.no_grad():  # Disable gradient computation for regular heatmaps
#         output = model(images)  # Get the output from the model

#     # Extract heatmaps from the output tuple
#     heatmaps = output[0][0]  # The heatmaps tensor
    
#     # Detach the tensor from the computation graph and convert to numpy
#     heatmaps = heatmaps.detach().cpu().numpy()

#     # Get image IDs from the targets
#     image_ids = targets['image_name'][0]  # Extract the list of image names

#     # Ensure the lengths match before proceeding
#     assert len(image_ids) == heatmaps.shape[0], "Mismatch between the number of heatmaps and image IDs!"

#     # Plot and save the heatmaps for each image in the batch
#     for j in range(heatmaps.shape[0]):
#         # Extract the heatmap for the current image
#         heatmap = heatmaps[j, 0]

#         # Clip and normalize the heatmap values
#         # clipped_heatmap = np.clip(heatmap, np.percentile(heatmap, 0.5), np.percentile(heatmap, 99.5))
#         #### Change the following 3 lines to clipped_hetamap if u want to clip the heatmap
#         heatmap_min = heatmap.min()
#         heatmap_max = heatmap.max()
#         normalized_heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-6)  # Avoid division by zero

#         # Plot the normalized heatmap
#         plt.figure(figsize=(10, 10))
#         plt.imshow(normalized_heatmap, cmap='jet', interpolation='nearest')
#         plt.title(f'Heatmap for Image ID {image_ids[j]} (Epoch {epoch})')

#         # Add the colorbar without inversion
#         cbar = plt.colorbar()
#         cbar.set_label('Normalized Intensity')

#         # Save the plot with the image ID and epoch number in the filename
#         output_path = os.path.join(output_dir, f'heatmap_image_{image_ids[j]}_epoch_{epoch}.png')
#         plt.savefig(output_path)
#         plt.close()

#     print(f"Heatmaps saved for epoch {epoch} in {output_dir}")

# def train_with_heatmap_generation(train_dataloader, val_dataloader, model, num_epochs, checkpoint_epoch, work_dir, warmup_iters=None):
#     optimizer = Adam(params=model.parameters(), lr=4e-4, weight_decay=1e-4)
#     density_loss = DensityLoss()  
#     print("Generating heatmaps before training...")
#     generate_heatmaps(val_dataloader, model, 0, work_dir)  # Epoch 0 for before training

#     for epoch in range(1, num_epochs + 1):
#         print(f"Epoch {epoch}/{num_epochs}: Training...")
#         model.train()  # Set the model to training mode
        
#         for batch_idx, (images, targets) in enumerate(train_dataloader):
#             images = images.to('cuda')
#             targets = targets['binary'].to('cuda')

#             optimizer.zero_grad()  # Zero the gradients
#             outputs = model(images)

#             density_mean = outputs[1]
#             loss = density_loss(density_mean, targets)

#             loss.backward()

#             # Apply warmup iterations if specified
#             if warmup_iters and epoch == 1 and batch_idx < warmup_iters:
#                 warmup_factor = float(batch_idx + 1) / warmup_iters
#                 for param_group in optimizer.param_groups:
#                     for param in param_group['params']:
#                         if param.grad is not None:
#                             param.grad *= warmup_factor  # Scale the gradients during warmup
            
#             optimizer.step()

#             # Log gradients and loss
#             for name, param in model.named_parameters():
#                 if param.grad is not None:
#                     print(f"{name}: Grad mean = {param.grad.abs().mean()}")
#                 else:
#                     print(f"{name}: No gradient computed")
                    
#             print(f"Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item()}")

#         # Generate heatmaps at specified intervals
#         if epoch > 0 and epoch % checkpoint_epoch == 0:
#             generate_heatmaps(val_dataloader, model, epoch, work_dir)

#         # Ensure to switch back to training mode after evaluation
#         model.train()

#     print("Training completed.")
    

# num_epochs = 100
# checkpoint_epoch = 100  # Generate heatmaps every 10 epochs
# work_dir = '/herdnet/val_output/after_training'
# warmup_iters=100
# train_with_heatmap_generation(train_dataloader, val_dataloader, dla_encoder_decoder, num_epochs, checkpoint_epoch, work_dir,warmup_iters=warmup_iters)
########################################################################

#             # Print gradient mean and loss for debugging
#             # print(f"Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item()}")
#             # for name, param in model.named_parameters():
#             #     if param.grad is not None:
#             #         print(f"{name}: Grad mean = {param.grad.abs().mean().item()}")
#             #     else:
#             #         print(f"{name}: No gradient computed")
                    
#         # Generate heatmaps at specified intervals
#         if epoch > 0 and epoch % checkpoint_epoch == 0:
#             print(f"Generating validation heatmaps and Grad-CAM at epoch {epoch}...")
#             generate_heatmaps(val_dataloader, model, epoch, work_dir)  # Generate standard heatmaps

#             # Grad-CAM for bottleneck and dla_up layers
#             grad_cam_bottleneck = GradCAM(model, target_layer_name='bottleneck_conv')
#             grad_cam_dla_up = GradCAM(model, target_layer_name='dla_up')

#             # Process a batch of validation images for Grad-CAM heatmaps
#             model.eval()  # Set model to evaluation mode for Grad-CAM
#             with torch.no_grad():  # Disable gradients for validation set
#                 images, targets = next(iter(val_dataloader))
#                 images = images.to('cuda')

#                 for j in range(images.shape[0]):  # Loop through images in the batch
#                     input_image = images[j:j+1]  # Use a single image from the batch

#                     # Generate Grad-CAM heatmaps for bottleneck and dla_up
#                     heatmap_bottleneck = grad_cam_bottleneck(input_image, target_class=1)
#                     heatmap_dla_up = grad_cam_dla_up(input_image, target_class=1)

#                     # Plotting or saving logic for heatmap_bottleneck and heatmap_dla_up
#                     plt.figure(figsize=(10, 5))
#                     plt.subplot(1, 2, 1)
#                     plt.imshow(heatmap_bottleneck, cmap='jet')
#                     plt.title(f'Grad-CAM Bottleneck - Image {j} (Epoch {epoch})')
#                     plt.colorbar()
                    
#                     plt.subplot(1, 2, 2)
#                     plt.imshow(heatmap_dla_up, cmap='jet')
#                     plt.title(f'Grad-CAM DLA Up - Image {j} (Epoch {epoch})')
#                     plt.colorbar()

#                     # Save plots
#                     output_path_bottleneck = os.path.join(work_dir, f'grad_cam_bottleneck_image_{j}_epoch_{epoch}.png')
#                     output_path_dla_up = os.path.join(work_dir, f'grad_cam_dla_up_image_{j}_epoch_{epoch}.png')
#                     plt.savefig(output_path_bottleneck)
#                     plt.savefig(output_path_dla_up)
#                     plt.close()

#             # Ensure to switch back to training mode after evaluation
#             model.train()

#     print("Training completed.")

# # Call the training function with parameters
# num_epochs = 200
# checkpoint_epoch = 25  # Generate heatmaps every 25 epochs
# work_dir = '/herdnet/val_output/after_training'
# warmup_iters = 100
# train_with_heatmap_generation(train_dataloader, val_dataloader, dla_encoder_decoder, num_epochs, checkpoint_epoch, work_dir, warmup_iters=warmup_iters)

# trainer.resume(pth_path='/herdnet/herdnet/Binary_pth/binary_20240829.pth', checkpoints='best', select='max', validate_on='f1_score', load_optim=True, wandb_flag=False)
trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='f1_score', wandb_flag =True)
from animaloc.models import load_model
pth_path = trainer.best_model_file 
herdnet = load_model(dla_encoder_decoder, pth_path=pth_path)
###### Evaluator ######
val_f1_score=evaluator.evaluate(returns='f1_score', viz=True, wandb_flag =False)

#Detections and Results
df_val_r=evaluator.results
df_val_r.to_csv('/herdnet/new_Validation_Binary_results.csv')
df_val_d=evaluator.detections
val_detections_path='/herdnet/new_Val_Binary_detections.csv'
df_val_d.to_csv(val_detections_path)

################ Plot Beta- Recall - Precision tradeoff in Focal Loss regarding different beta values ###########
# eval_epochs = 50
# fine_tune_epochs = 10
# def fine_tune_and_evaluate(beta, alpha_pos, alpha_neg, train_dataloader, val_dataloader, model, eval_epochs, fine_tune_epochs, work_dir):
#     # Check if val_dataloader is not None and contains data
#     if val_dataloader is None:
#         raise ValueError("val_dataloader is None")
#     if len(val_dataloader) == 0:
#         raise ValueError("val_dataloader is empty")

#     # Make sure the model is not already wrapped
#     if not isinstance(model, LossWrapper):
#         density_loss = DensityLoss(reduction='mean', eps=1e-6)
#         # Update BinaryFocalLoss parameters
#         density_loss.loss = BinaryFocalLoss(alpha_pos=alpha_pos, alpha_neg=alpha_neg, beta=beta, reduction='mean')

#         # Wrap the model with LossWrapper
#         model = LossWrapper(model, losses=[{'loss': density_loss, 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'density_loss'}])
#         print(f"Wrapped Model Type: {type(model)}")  ####### Debugging Print
    
#     optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-3)

#     evaluator = TileEvaluator(
#         metrics_class=ImageLevelMetrics,
#         threshold=0.5,
#         model=model,
#         dataloader=val_dataloader,
#         num_classes=2,
#         stitcher=None,
#         work_dir=work_dir,
#         header='validation',
#     )
#     best_model_path = os.path.join(work_dir, '/herdnet/herdnet/augmented_images')
#     fine_tuner = Trainer(
#         model=model,
#         train_dataloader=train_dataloader,
#         optimizer=optimizer,
#         num_epochs=fine_tune_epochs,
#         evaluator=evaluator,
#         val_dataloader=val_dataloader,
#         work_dir=work_dir,
#         loss_dicts=model.losses,
#         best_model_path=best_model_path
#     )

#     # Start fine-tuning with logging
#     print(f"Starting fine-tuning for beta={beta}, alpha_pos={alpha_pos}, alpha_neg={alpha_neg}")
#     fine_tuner.start()  # Use start method of Trainer

#     # Evaluation phase
#     precision_list = []
#     recall_list = []

#     for epoch in range(eval_epochs):
#         precision = evaluator.evaluate(returns='precision')
#         recall = evaluator.evaluate(returns='recall')
#         precision_list.append(precision)
#         recall_list.append(recall)
#         print(f"Evaluation Epoch [{epoch+1}/{eval_epochs}] completed: Precision={precision}, Recall={recall}")

#     avg_precision = sum(precision_list[-20:]) / min(len(precision_list), 20)  # Average of last 20 epochs
#     avg_recall = sum(recall_list[-20:]) / min(len(recall_list), 20)

#     return avg_precision, avg_recall


# ###### plot alpha_pos and beta in one plot
# # Parameters to test
# import numpy as np
# beta_values = [2, 2.5, 3, 3.5, 4]
# alpha_pos_values = [1, 1.5, 2, 2.5, 3]
# alpha_neg = 1  # Keep alpha_neg constant, or you can vary it similarly
# precisions = []
# recalls = []

# for beta in beta_values:
#     for alpha_pos in alpha_pos_values:
#         print(f"Evaluating for beta: {beta}, alpha_pos: {alpha_pos}, alpha_neg: {alpha_neg}")
#         precision, recall = fine_tune_and_evaluate(beta, alpha_pos, alpha_neg, train_dataloader, val_dataloader, dla_encoder_decoder, eval_epochs, fine_tune_epochs, work_dir)
#         precisions.append((beta, alpha_pos, precision))
#         recalls.append((beta, alpha_pos, recall))
#         print(f"Results for beta={beta}, alpha_pos={alpha_pos}, alpha_neg={alpha_neg}: Precision: {precision}, Recall: {recall}")

# # Convert the list of tuples to a numpy array for easier manipulation
# precisions = np.array(precisions)
# recalls = np.array(recalls)
# plt.figure(figsize=(12, 6))

# # Precision plot
# for alpha_pos in np.unique(precisions[:, 1]):
#     mask = precisions[:, 1] == alpha_pos
#     plt.plot(precisions[mask, 0], precisions[mask, 2], marker='o', label=f'Precision (alpha_pos={alpha_pos})')

# # Recall plot
# for alpha_pos in np.unique(recalls[:, 1]):
#     mask = recalls[:, 1] == alpha_pos
#     plt.plot(recalls[mask, 0], recalls[mask, 2], marker='x', linestyle='--', label=f'Recall (alpha_pos={alpha_pos})')

# plt.title('Precision and Recall vs Beta for different Alpha_pos values')
# plt.xlabel('Beta')
# plt.ylabel('Score')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('/herdnet/Decoder_precision_recall_vs_beta_alpha_pos_line_plot.pdf')
# plt.show()

#################### Test data and evaluation ###########
# Create output folder
test_dir = '/herdnet/test_output'
mkdir(test_dir)

# Create an Evaluator
test_evaluator = TileEvaluator(
    model=dla_encoder_decoder,
    dataloader=test_dataloader,
    metrics_class=ImageLevelMetrics,
   num_classes=num_classes,
    stitcher=None,
    work_dir=test_dir,
    threshold=0.5,
    header='test'
    )

if wandb.run is not None:
  wandb.finish()
wandb.init(project="herdnet_pretrain", reinit=False)

# Start testing
test_f1_score = test_evaluator.evaluate(returns='f1_score', wandb_flag=False)
# Print global F1 score (%)
print(f"F1 score = {test_f1_score * 100:0.0f}%")

# Get the test results
test_results = test_evaluator.results
test_results.to_csv('/herdnet/test_output/new_Binary_test_results.csv', index=False)

# Get the test detections and save to CSV with renamed columns
test_detections = test_evaluator.detection_test()
# Rename columns directly in the DataFrame
test_detections.rename(columns={'true_binary': 'Ground truth', 'predicted_binary': 'Prediction'}, inplace=True)
# Save to CSV
test_detections.to_csv('/herdnet/test_output/new_Binary_test_detections.csv', index=False)


# Rename columns directly in the DataFrame
test_detections.rename(columns={'true_binary': 'Ground truth', 'predicted_binary': 'Prediction'}, inplace=True)

# Save to CSV
test_detections.to_csv('/herdnet/test_output/new_Binary_test_detections.csv', index=False)

########################################## Threshold Tuning ################################################
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# # Define threshold range
# thresholds = np.linspace(0.1, 0.9, 9)

# # Initialize dictionaries to store metric values
# f1_scores = {}
# precision_scores = {}
# recall_scores = {}

# # Evaluate at each threshold
# for threshold in thresholds:
#     evaluator.threshold = threshold  # Update the threshold in the evaluator
    
#     # Calculate metrics
#     precision = evaluator.evaluate(returns='precision')
#     recall = evaluator.evaluate(returns='recall')
#     f1_score = evaluator.evaluate(returns='f1_score')
    
#     # Store values
#     f1_scores[threshold] = f1_score
#     precision_scores[threshold] = precision
#     recall_scores[threshold] = recall
    
#     # Print metrics for each threshold
#     print(f"Threshold: {threshold:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

# # Save plots to a PDF
# pdf_path = '/herdnet/test_output/metrics_vs_threshold.pdf'
# with PdfPages(pdf_path) as pdf:
#     # Plot F1 Score vs. Threshold
#     plt.figure(figsize=(10, 5))
#     plt.plot(list(f1_scores.keys()), list(f1_scores.values()), marker='o', linestyle='-', color='b')
#     plt.title('F1 Score vs. Threshold')
#     plt.xlabel('Threshold')
#     plt.ylabel('F1 Score')
#     plt.grid(True)
#     pdf.savefig()  # Save this figure as a page in the PDF
#     plt.close()

#     # Plot Precision vs. Threshold
#     plt.figure(figsize=(10, 5))
#     plt.plot(list(precision_scores.keys()), list(precision_scores.values()), marker='o', linestyle='-', color='g')
#     plt.title('Precision vs. Threshold')
#     plt.xlabel('Threshold')
#     plt.ylabel('Precision')
#     plt.grid(True)
#     pdf.savefig()  # Save this figure as a page in the PDF
#     plt.close()

#     # Plot Recall vs. Threshold
#     plt.figure(figsize=(10, 5))
#     plt.plot(list(recall_scores.keys()), list(recall_scores.values()), marker='o', linestyle='-', color='r')
#     plt.title('Recall vs. Threshold')
#     plt.xlabel('Threshold')
#     plt.ylabel('Recall')
#     plt.grid(True)
#     pdf.savefig()  # Save this figure as a page in the PDF
#     plt.close()

# print(f"Plots have been saved to {pdf_path}")

