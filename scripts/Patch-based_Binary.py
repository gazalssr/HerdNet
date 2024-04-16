import random
import animaloc
import pandas as pd
# Set the seed
from animaloc.utils.seed import set_seed

set_seed(9292)

import matplotlib.pyplot as plt
from animaloc.datasets import FolderDataset, CSVDataset, BinaryFolderDataset
from animaloc.data.batch_utils import show_batch, collate_fn
from torch.utils.data import DataLoader
import torch
import albumentations as A
from animaloc.data.transforms import  BinaryMultiTransformsWrapper,MultiTransformsWrapper, DownSample, PointsToMask, BinaryTransform
import wandb
import torch
from animaloc.models import DLAEncoder
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss
wandb.init(project='HerdNet', entity='ghazaleh-serati')

NUM_WORKERS= 8
import albumentations as A
binary=True
patch_size = 512
num_classes = 2
batch_size=32
down_ratio = 2
train_dataset = BinaryFolderDataset(
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
    end_transforms = [BinaryMultiTransformsWrapper([
        BinaryTransform(),
        ])]
    )
train_dataset.data.to_csv('/herdnet/DATASETS/Train_patches_stratified/Train_binary_gt.csv', index=False)


val_dataset = BinaryFolderDataset(
    csv_file = '/herdnet/DATASETS/val_patches_stratified/gt.csv',
    root_dir = '/herdnet/DATASETS/val_patches_stratified',
    albu_transforms = [
        A.Normalize(p=1.0)
        ],
    end_transforms = [BinaryMultiTransformsWrapper([
        BinaryTransform(),
        ])]
    )
val_dataset.data.to_csv('/herdnet/DATASETS/val_patches_stratified/Val_binary_gt.csv', index=False)


test_dataset = BinaryFolderDataset(
    csv_file = '/herdnet/DATASETS/test_patches_stratified/gt.csv',
    root_dir = '/herdnet/DATASETS/test_patches_stratified',
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [BinaryMultiTransformsWrapper([
        BinaryTransform(),
        ])]
    )
test_dataset.data.to_csv('/herdnet/DATASETS/test_patches_stratified/Test_binary_gt.csv', index=False)
# Dataloaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset = train_dataset, batch_size= 32 , num_workers= 2, shuffle= True)

val_dataloader = DataLoader(dataset = val_dataset, batch_size=1 , num_workers= 2, shuffle= True)

test_dataloader= DataLoader(dataset = test_dataset, batch_size=1 , num_workers= 2, shuffle= False)
num_classes=2
dla_encoder = DLAEncoder(num_classes=num_classes).cuda()

# Define DLAENCODER for training
image= torch.ones([1,3,512,512]).cuda()
print(torch.cuda.mem_get_info())

cls= dla_encoder(image)

losses = [
      {'loss': BCEWithLogitsLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'bce_loss'}
      ]

    
# else:
#   losses = [
#       {'loss': L1Loss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'bce_loss'},
#       ]
    
dla_encoder = LossWrapper(dla_encoder, losses=losses)


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

        # Print parameters to update
        if param.requires_grad:
            print(f"Trainable parameter: {name}")
        else:
            print(f"Frozen parameter: {name}")

    return params_to_update

###### Layers to Freeze ##############
param_dict= get_parameter_names(dla_encoder.model)
print(param_dict)
lr = 1e-4 # learning rate
layers_to_freeze= [] # nothing frozen
#layers_to_freeze= ['base_layer','level0','level1','level2','level3','level4'] # we are feezing all the levels below level5
# layers_to_freeze= ['base_layer','level0','level1','level2','level3','level4','level5','fc','bottleneck_conv'] # we are feezing everything except cls_head

params_to_update = freeze_parts(dla_encoder.model,param_dict, layers_to_freeze,lr,False)


#### Create the Trainer #######
from torch.optim import Adam

from animaloc.train import Trainer
from animaloc.eval import ImageLevelMetrics, HerdNetStitcher, TileEvaluator
from animaloc.utils.useful_funcs import mkdir

work_dir = '/content/drive/MyDrive/output'
mkdir(work_dir)

lr = 1e-4
weight_decay = 1e-3
epochs = 100

optimizer = Adam(params=dla_encoder.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)
####### Adding Binrya Option #######
metrics = ImageLevelMetrics(num_classes=num_classes)
metrics.binary_annotations = True
evaluator = TileEvaluator(
    model=dla_encoder,
    dataloader=val_dataloader,
    metrics=metrics,
    stitcher=None,
    work_dir=work_dir,
    header='validation',
    )

trainer = Trainer(
    model=dla_encoder,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    num_epochs=epochs,
    evaluator=evaluator,             # metric evaluation
    # val_dataloader= val_dataloader, # loss evaluation
    work_dir=work_dir
    )

if wandb.run is not None:
  wandb.finish()
wandb.init(project="herdnet_pretrain")

####### Trainer ############
trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='f1_score', wandb_flag =True)

##### Evaluator on validation ############
if wandb.run is not None:
  wandb.finish()
wandb.init(project="herdnet_pretrain")

val_f1_score=evaluator.evaluate(returns='f1_score', viz=False, wandb_flag =True )

#Detections and Results
df_val_r=evaluator.results
df_val_r.to_csv('/herdnet/Binary_results.csv')
df_train_d=evaluator.detections
df_train_d.rename(columns={'count_1': 'empty_count', 'count_2': 'non_empty_count'}, inplace=True)
df_train_d.to_csv('/herdnet/Binary_detections.csv')

############### Comparison of the detections and gt #########
detections_df = pd.read_csv('/herdnet/Binary_detections.csv')
gt_df = pd.read_csv('/herdnet/DATASETS/val_patches_stratified/Val_binary_gt.csv')
# Create a new column 'Ground_truth' in df_detection and initialize with NaN
detections_df['Ground_truth'] = pd.NA

# Create a dictionary from the gt DataFrame for quick lookup
gt_dict = pd.Series(gt_df['binary'].values, index=gt_df['images']).to_dict()

# Iterate over each row in the detection DataFrame
for index, row in detections_df.iterrows():
    image_id = row['images']
    # Check if the current image_id is in the gt_dict and assign the corresponding binary value
    if image_id in gt_dict:
        detections_df.at[index, 'Ground_truth'] = gt_dict[image_id]

# Save the updated DataFrame back to a new CSV (optional)
detections_df.to_csv('/herdnet/VAL_updated_detection_file.csv', index=False)
#################### Test data and evaluation ###########
# Create output folder
test_dir = '/herdnet/test_output'
mkdir(test_dir)

# Create an Evaluator
test_evaluator = TileEvaluator(
    model=dla_encoder,
    dataloader=test_dataloader,
    metrics=metrics,
    stitcher=None,
    work_dir=test_dir,
    header='test'
    )

# Start testing
test_f1_score = test_evaluator.evaluate(returns='f1_score')

# Print global F1 score (%)
print(f"F1 score = {test_f1_score * 100:0.0f}%")

# Get the test results
test_results = test_evaluator.results
test_results.to_csv('/herdnet/test_output/Binary_test_results.csv', index=False)

#Get the test detections
test_detections=test_evaluator.detections
test_detections.to_csv('/herdnet/test_output/Binary_test_detections.csv', index=False)

# TEST_DATASETS
detections_df=pd.read_csv('/herdnet/test_output/test_detections.csv')
root_dir = '/herdnet/DATASETS/test_patches_stratified'


