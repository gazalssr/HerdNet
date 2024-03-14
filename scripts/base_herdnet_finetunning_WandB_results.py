import random
import animaloc

# Set the seed
from animaloc.utils.seed import set_seed

set_seed(9292)

import matplotlib.pyplot as plt
from animaloc.datasets import FolderDataset, CSVDataset
from animaloc.data.batch_utils import show_batch, collate_fn
from torch.utils.data import DataLoader
import torch
import albumentations as A
from animaloc.data.transforms import MultiTransformsWrapper, DownSample, PointsToMask, FIDT, TRANSFORMS
######################## Setting the env file for Wandb ###################################################################
import yaml
import wandb
import torch

def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load the configuration file for train
if __name__ == '__main__':
    config_path_1 = '/herdnet/configs/train/herdnet.yaml'  
    config = load_config(config_path_1)
 # Load the configuration file for test
if __name__ == '__main__':
    config_path_2 = '/herdnet/configs/test/herdnet.yaml'  # Update with the actual path to your YAML file
    config = load_config(config_path_2)   

#Extracting wandb config info

batch_size = 8
NUM_WORKERS= 8
# csv_path = '/herdnet/DATASETS/Train_patches_stratified/gt.csv'
csv_path = '/herdnet/DATASETS/Train_patches_stratified/gt.csv'
image_path = '/herdnet/DATASETS/Train_patches_stratified'
dataset = FolderDataset(csv_path, image_path, [A.Normalize()])

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers= NUM_WORKERS)

sample_batch = next(iter(dataloader))
for i in range(len(sample_batch[1])):
  points = sample_batch[1][i]['points'].numpy()
  bbox= []
  for pt in points:
      bbox.append([pt[0]-2,pt[1]-2,pt[0]+2,pt[1]+2])
  print(len(sample_batch[1][i]['labels']))
  sample_batch[1][i]['annotations']=torch.tensor(bbox)
plt.figure(figsize=(16,2))
show_batch(sample_batch)
plt.savefig('/herdnet/show_annotation_patch.pdf')

# Training, validation and test datasets
import albumentations as A
patch_size = 512
num_classes = 2
down_ratio = 2

train_dataset = CSVDataset(
    csv_file = '/herdnet/DATASETS/Train_patches_stratified/gt.csv',
    root_dir = '/herdnet/DATASETS/Train_patches_stratified',
    albu_transforms = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Blur(blur_limit=15, p=0.2),
        A.Normalize(p=1.0)
        ],
    end_transforms = [MultiTransformsWrapper([
        FIDT(num_classes=num_classes, down_ratio=down_ratio),
        PointsToMask(radius=2, num_classes=num_classes, squeeze=True, down_ratio=int(patch_size//16))
        ])
    ]
    )

val_dataset = CSVDataset(
    csv_file = '/herdnet/DATASETS/val_patches_stratified/gt.csv',
    root_dir = '/herdnet/DATASETS/val_patches_stratified',
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
    )

test_dataset = CSVDataset(
    csv_file = '/herdnet/DATASETS/test_patches_stratified/gt.csv',
    root_dir = '/herdnet/DATASETS/test_patches_stratified',
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
    )

# Dataloaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset = train_dataset, batch_size = 4,shuffle = True, num_workers= NUM_WORKERS)

val_dataloader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = False, num_workers= NUM_WORKERS)

test_dataloader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False, num_workers= NUM_WORKERS)

## Define HerdNet for training

# Path to your .pth file (initial pth file)

import torch
pth_path = None #'/herdnet/output/best_model.pth'
from pathlib import Path

dir_path = Path('/herdnet/output')  
dir_path.mkdir(parents=True, exist_ok=True)
pth_path= '/herdnet/DATASETS/20220413_herdnet_model.pth'
if not pth_path:
    gdown.download(
        'https://drive.google.com/uc?export=download&id=1-WUnBC4BJMVkNvRqalF_HzA1_pRkQTI_',
        '/herdnet/output/20220413_herdnet_model.pth'
        )
    pth_path = '/herdnet/output/20220413_herdnet_model.pth'

from animaloc.models import HerdNet
from torch import Tensor
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss
pretrained= True

herdnet = HerdNet(pretrained= pretrained, num_classes=num_classes, down_ratio=down_ratio).cuda()
if not pretrained:
    pretrained_dict = torch.load(pth_path)['model_state_dict']
    #herdnet_dict = herdnet.state_dict()
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in herdnet_dict}
    #herdnet.load_state_dict(pretrained_dict, strict=False)

losses = [
    {'loss': FocalLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
    {'loss': CrossEntropyLoss(reduction='mean'), 'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
    ]

herdnet = LossWrapper(herdnet, losses=losses)

#############Get model layers ###########################
def get_parameter_names(model): # getting the model layers
  param_dict= dict()
  for l, (name,param) in enumerate(model.named_parameters()):
    #print(l,":\t",name,type(param),param.requires_grad)
    param_dict[name]= l
  return param_dict
result = get_parameter_names(herdnet)
print(result)

"""# Freeze the alyers (different options)
1. half of a layer and other layers
"""

#Freeze half of a specified layer
def freeze_parts(model, get_parameter_names, layers_to_freeze, freeze_layer_half=None, lr=0.0001, unfreeze=False):
    params_to_update = []

    for l, (name, param) in enumerate(model.named_parameters()):
        res = any(ele in name for ele in layers_to_freeze)
        param.requires_grad = unfreeze if res else not unfreeze

        # Check if the current layer is the specified layer to freeze half of its parameters
        if freeze_layer_half is not None and freeze_layer_half in name:
            total_params = param.numel()
            half_params = total_params // 2
            param.requires_grad = unfreeze if l < half_params else not unfreeze

        if param.requires_grad:
            params_to_update.append({
                "params": param,
                "lr": lr,
            })

        # Print parameters to update
        if param.requires_grad:
            print(f"Trainable parameter: {name}")
        else:
            print(f"Frozen parameter: {name}")

    return params_to_update

#freezing half of one lyer+ other layers
params_to_update = freeze_parts(herdnet.model, get_parameter_names, layers_to_freeze=['base_layer','level0','level1','level2','level3'], freeze_layer_half='level_4', lr=0.0001, unfreeze=False)

## Freeze a complete layer ##

# #Freeze the layers
# def freeze_parts(model, get_parameter_names, layers_to_freeze, lr, unfreeze=False):
#     params_to_update = []

#     for l, (name, param) in enumerate(model.named_parameters()):
#         res = any(ele in name for ele in layers_to_freeze)
#         param.requires_grad = unfreeze if res else not unfreeze

#         if param.requires_grad == True:
#             params_to_update.append({
#                 "params": param,
#                 "lr": lr,
#             })

#         # Print parameters to update
#         if param.requires_grad:
#             print(f"Trainable parameter: {name}")
#         else:
#             print(f"Frozen parameter: {name}")

# #     return params_to_update
# layers_to_freeze=['base_layer','level0','level1','level2','level3','level4']
# params_to_update=freeze_parts(herdnet.model,get_parameter_names,layers_to_freeze,lr,unfreeze=False)
## Create the Trainer

from torch.optim import Adam
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator
from animaloc.utils.useful_funcs import mkdir
import pandas as pd
work_dir = '/herdnet/output'
mkdir(work_dir)

lr = 1e-4
weight_decay = 1e-3
epochs = 3

optimizer = Adam(params_to_update, lr=lr, weight_decay=weight_decay)

metrics = PointsMetrics(radius=20, num_classes=num_classes)

stitcher = HerdNetStitcher(
    model=herdnet,
    size=(patch_size,patch_size),
    overlap=160,
    down_ratio=down_ratio,
    reduction='mean'
    )

evaluator = HerdNetEvaluator(
    model=herdnet,
    dataloader=val_dataloader,
    metrics=metrics,
    stitcher= None, # stitcher,
    work_dir=work_dir,
    header='validation'
    )

trainer = Trainer(
    model=herdnet,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    num_epochs=epochs,
    evaluator=evaluator,
    # val_dataloader= val_dataloader      #loss evaluation
    work_dir=work_dir
    )

## Start training

import wandb
if wandb.run is not None:
  wandb.finish()
# wandb.init(dir='herdnet/DATASETS')
wandb.init(project="herdnet-finetuning")

trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='f1_score', wandb_flag =True)


#save and load finetunned parameters
# herdnet = HerdNet()
# torch.save(herdnet.state_dict(), 'fine_tuned_base.pth')
# herdnet.load_state_dict(torch.load('fine_tuned_base.pth'))
# pth_path = '/herdnet/fine_tuned_base.pth'
# torch.save(herdnet.state_dict(), pth_path)

if 0:
    herdnet = HerdNet()
    herdnet.load_state_dict(torch.load(pth_path))
# Load trained parameters
if 0:
    from animaloc.models import load_model
    checkpoint = torch.load(pth_path, map_location=map_location)
    herdnet.load_state_dict(checkpoint['model_state_dict'])
    herdnet = load_model(herdnet, pth_path=pth_path)
    
    
############################# MODEL EVALUATION ##########################################

###Evaluation on train data #####
# Create output folder
# train_dir = '/herdnet/train_output'
# mkdir(train_dir)


#Create Train dataloder for evaluation

train_dataset = CSVDataset(
    csv_file = '/herdnet/DATASETS/Train_patches_stratified/gt.csv',
    root_dir = '/herdnet/DATASETS/Train_patches_stratified',
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=2, anno_type='point')]
    )
train_dataloader = DataLoader(dataset = train_dataset, batch_size = 1, shuffle = False, num_workers= NUM_WORKERS)

# Create a train_Evaluator
train_evaluator = HerdNetEvaluator(
    model=herdnet,
    dataloader=train_dataloader,
    metrics=metrics,
    stitcher=None,
    work_dir=work_dir,
    header='train'
    )
#Start train Evaluation
train_f1_score=train_evaluator.evaluate(returns='f1_score')

#Save and plot detection results on train data
df_train_r=train_evaluator.results
df_train_r.to_csv('/herdnet/train_output/train_results.csv')
#Save and plot detection file
df_train_d=train_evaluator.detections
df_train_d.to_csv('/herdnet/train_output/train_detections.csv')
root_dir = '/herdnet/DATASETS/Train_patches_stratified'
detections_df='/herdnet/train_output/train_detections.csv'
detections_df=pd.read_csv(detections_df)
##### Plot the detections in Wandb #################
import wandb
from PIL import Image
import os

# Initialize your WandB run
wandb.init(project='HerdNet', entity='ghazaleh-serati')

# Assuming detections_df and root_dir are defined as before
# Function to convert point detections to small bounding boxes
def convert_points_to_wandb_boxes(detection):
    x, y = detection['x'], detection['y']
    offset = 1  # 1 pixel bounding_box size=2 pixels
    box_data = {
        "position": {
            "minX": x - offset,
            "maxX": x + offset,
            "minY": y - offset,
            "maxY": y + offset
        },
        "class_id": int(detection['labels']),
        "box_caption": f"Label: {detection['labels']}, Score: {detection['scores']}",
        "scores": {"detection_score": detection['dscores']},
        "domain": "pixel",
    }
    return box_data
detections_df['full_image_path'] = detections_df['images'].apply(lambda x: os.path.join(root_dir, x))

# Log aggregated data to WandB
aggregated_images = []
aggregated_boxes = []
desired_batch_size=3
for image_path in detections_df['full_image_path'].unique():
    # Open the image
    image = Image.open(image_path)
    current_detections = detections_df[detections_df['full_image_path'] == image_path]
    boxes_for_image = []
    
    for _, detection in current_detections.iterrows():
        box_data = convert_points_to_wandb_boxes(detection)
        boxes_for_image.append(box_data)
    
    # Prepare the data for logging
    aggregated_images.append(wandb.Image(image, boxes={"predictions": {"box_data": boxes_for_image, "class_labels": {0: "Detection"}}}))

    # Check if it's time to log the data
    if len(aggregated_images) >= desired_batch_size:
        wandb.log({"patch_detections": aggregated_images})
        aggregated_images = []  # Reset for the next batch

# Log any remaining images
if aggregated_images:
    wandb.log({"patch_detections": aggregated_images})

# Finish the WandB run
wandb.finish()
########## printing the image list and detection list to ensure that WANDB passes through all the patches ##########
# print(f"Processing {len(unique_paths)} unique image paths...")
# unique_paths = detections_df['full_image_path'].unique()
    # for image_path in unique_paths:
    # current_detections = detections_df[detections_df['full_image_path'] == image_path]
    # # print(f"{image_path} has {len(current_detections)} detections")
    # if len(current_detections) == 0:
    #     print(f"No detections for {image_path}")
    # print(f"Logging {image_path} with {len(current_detections)} detections")
# print("Unique images:", len(detections_df['images'].unique()))

########################### Evaluation on validation data ##########################
# Create output folder
# val_dir = '/herdnet/val_output'
# mkdir(val_dir)

#Create validation dataloder for evaluation
val_dataset = CSVDataset(
    csv_file = '/herdnet/DATASETS/val_patches_stratified/gt.csv',
    root_dir = '/herdnet/DATASETS/val_patches_stratified',
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
    )
val_dataloader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = True, num_workers= NUM_WORKERS)
# Create a val_Evaluator
val_evaluator = HerdNetEvaluator(
    model=herdnet,
    dataloader=val_dataloader,
    metrics=metrics,
    stitcher=None,
    work_dir=work_dir,
    header='validation'
    )
#Start validation Evaluation
val_f1_score=val_evaluator.evaluate(returns='f1_score')

#Save and plot detection results on validation data
df_val_r=val_evaluator.results
df_val_r.to_csv('/herdnet/val_output/val_results.csv')
# Patch file and detection file
df_val_d=val_evaluator.detections
df_val_d.to_csv('/herdnet/val_output/val_detections.csv')


# Validation Datasets
detections_df=pd.read_csv('/herdnet/val_output/val_detections.csv')
root_dir = '/herdnet/DATASETS/val_patches_stratified'

##### Plot the validation data detections in Wandb #################
import wandb
from PIL import Image
import os

# Initialize your WandB run
wandb.init(project='HerdNet', entity='ghazaleh-serati')

# Assuming detections_df and root_dir are defined as before
# Function to convert point detections to small bounding boxes
def convert_points_to_wandb_boxes(detection):
    x, y = detection['x'], detection['y']
    offset = 1  # 1 pixel bounding_box size=2 pixels
    box_data = {
        "position": {
            "minX": x - offset,
            "maxX": x + offset,
            "minY": y - offset,
            "maxY": y + offset
        },
        "class_id": int(detection['labels']),
        "box_caption": f"Label: {detection['labels']}, Score: {detection['scores']}",
        "scores": {"detection_score": detection['dscores']},
        "domain": "pixel",
    }
    return box_data
detections_df['full_image_path'] = detections_df['images'].apply(lambda x: os.path.join(root_dir, x))

# Log aggregated data to WandB
aggregated_images = []
aggregated_boxes = []
desired_batch_size=3
for root_dir in detections_df['full_image_path'].unique():
    # Open the image
    image = Image.open(root_dir)
    current_detections = detections_df[detections_df['full_image_path'] == root_dir]
    boxes_for_image = []
    
    for _, detection in current_detections.iterrows():
        box_data = convert_points_to_wandb_boxes(detection)
        boxes_for_image.append(box_data)
    
    # Prepare the data for logging
    aggregated_images.append(wandb.Image(image, boxes={"predictions": {"box_data": boxes_for_image, "class_labels": {0: "Detection"}}}))

    # Check if it's time to log the data
    if len(aggregated_images) >= desired_batch_size:
        wandb.log({"patch_detections": aggregated_images})
        aggregated_images = []  # Reset for the next batch

# Log any remaining images
if aggregated_images:
    wandb.log({"patch_detections": aggregated_images})

# Finish the WandB run
wandb.finish()

    ########## printing the image list and detection list to ensure that WANDB passes through all the patches ##########
# print(f"Processing {len(unique_paths)} unique image paths...")
# unique_paths = detections_df['full_image_path'].unique()
    # for root_dir in unique_paths:
    # current_detections = detections_df[detections_df['full_image_path'] == image_path]
    # # print(f"{root_dir} has {len(current_detections)} detections")
    # if len(current_detections) == 0:
    #     print(f"No detections for {root_dir}")
    # print(f"Logging {root-dir} with {len(current_detections)} detections")
# print("Unique images:", len(detections_df['images'].unique()))
########################### Evaluation on test data ##########################
# Create output folder
# test_dir = '/herdnet/test_output'
# mkdir(test_dir)

# Create an Evaluator
test_evaluator = HerdNetEvaluator(
    model=herdnet,
    dataloader=test_dataloader,
    metrics=metrics,
    stitcher=stitcher,
    work_dir=work_dir,
    header='test'
    )

# Start testing
test_f1_score = test_evaluator.evaluate(returns='f1_score')

# Print global F1 score (%)
print(f"F1 score = {test_f1_score * 100:0.0f}%")

# Get the test results
test_results = test_evaluator.results
test_results.to_csv('/herdnet/test_output/test_results.csv', index=False)

#Get the test detections
test_detections=test_evaluator.detections
test_detections.to_csv('/herdnet/test_output/test_detections.csv', index=False)

# TEST_DATASETS
detections_df=pd.read_csv('/herdnet/test_output/test_detections.csv')
root_dir = '/herdnet/DATASETS/test_patches_stratified'
##### Plot the test data detections in Wandb #################
import wandb
from PIL import Image
import os

# Initialize your WandB run
wandb.init(project='HerdNet', entity='ghazaleh-serati')

# Assuming detections_df and root_dir are defined as before
# Function to convert point detections to small bounding boxes
def convert_points_to_wandb_boxes(detection):
    x, y = detection['x'], detection['y']
    offset = 1  # 1 pixel bounding_box size=2 pixels
    box_data = {
        "position": {
            "minX": x - offset,
            "maxX": x + offset,
            "minY": y - offset,
            "maxY": y + offset
        },
        "class_id": int(detection['labels']),
        "box_caption": f"Label: {detection['labels']}, Score: {detection['scores']}",
        "scores": {"detection_score": detection['dscores']},
        "domain": "pixel",
    }
    return box_data
detections_df['full_image_path'] = detections_df['images'].apply(lambda x: os.path.join(root_dir, x))

# Log aggregated data to WandB
aggregated_images = []
aggregated_boxes = []
desired_batch_size=3
for root_dir in detections_df['full_image_path'].unique():
    # Open the image
    image = Image.open(root_dir)
    current_detections = detections_df[detections_df['full_image_path'] == root_dir]
    boxes_for_image = []
    
    for _, detection in current_detections.iterrows():
        box_data = convert_points_to_wandb_boxes(detection)
        boxes_for_image.append(box_data)
    
    # Prepare the data for logging
    aggregated_images.append(wandb.Image(image, boxes={"predictions": {"box_data": boxes_for_image, "class_labels": {0: "Detection"}}}))

    # Check if it's time to log the data
    if len(aggregated_images) >= desired_batch_size:
        wandb.log({"patch_detections": aggregated_images})
        aggregated_images = []  # Reset for the next batch

# Log any remaining images
if aggregated_images:
    wandb.log({"patch_detections": aggregated_images})

# Finish the WandB run
wandb.finish()
