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

# torch.cuda.empty_cache()

binary=True
preprocess=False
patch_size = 512
num_classes = 2
batch_size=8
down_ratio = 2
train_dataset = BinaryFolderDataset(
    preprocess=preprocess,
    csv_file='/herdnet/DATASETS/CAH_no_margins_30/train/Train_binary_gt.csv',
    root_dir='/herdnet/DATASETS/CAH_no_margins_30/train/',
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

# train_dataset.data.to_csv('/herdnet/DATASETS/CAH_Complete_FCH_101114_STRATIFIED/train_CF/Train_binary_gt.csv', index=False)
import os
from PIL import Image
val_dataset = BinaryFolderDataset(
    preprocess=preprocess,
    csv_file='/herdnet/DATASETS/CAH_no_margins_30/val/Val_binary_gt.csv',
    root_dir='/herdnet/DATASETS/CAH_no_margins_30/val/',
    albu_transforms=A.Compose([
        A.Normalize(p=1.0),
        ToTensorV2()
    ]),
    end_transforms=[BinaryMultiTransformsWrapper([
        BinaryTransform(),
    ])]
)
# val_dataset.data.to_csv('/herdnet/DATASETS/CAH_Complete_FCH_101114_STRATIFIED/val_CF/Val_binary_gt.csv', index=False)


test_dataset = BinaryFolderDataset(
    preprocess=preprocess,
    csv_file='/herdnet/DATASETS/CAH_no_margins_30/test/Test_binary_gt.csv',
    root_dir='/herdnet/DATASETS/CAH_no_margins_30/test/',
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
    shuffle=True
)
# test_dataset.data.to_csv('/herdnet/DATASETS/CAH_Complete_FCH_101114_STRATIFIED/test_W/Test_binary_gt.csv', index=False)
# Dataloaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset = train_dataset, sampler=train_sampler, collate_fn=BinaryFolderDataset.collate_fn)

# num_workers= 2
val_dataloader = DataLoader(dataset = val_dataset, sampler=val_sampler, collate_fn=BinaryFolderDataset.collate_fn)

test_dataloader= DataLoader(dataset = test_dataset, batch_size=1 , shuffle= False, collate_fn=BinaryFolderDataset.collate_fn)
num_classes=2
image= torch.ones([1,3,512,512]).cuda()
# Number of patches per class (train datset)
total_patches = len(train_dataset)
# empty_patches = 7960
empty_patches=1502
non_empty_patches = 1502
# Class weights
weight_for_empty = total_patches / (2 * empty_patches)
weight_for_non_empty = total_patches / (2 * non_empty_patches)
weights = [weight_for_empty, weight_for_non_empty]
# Define the model(with saved imagenet weights)
# dla_encoder = DLAEncoder(num_classes=num_classes, pretrained=False).cuda()
# dla_encoder.load_custom_pretrained_weights('/herdnet/pth_files/dla34-ba72cf86.pth')
##### DLA AutoEncoder 
dla_encoder_decoder = DLAEncoderDecoder(num_classes=num_classes, pretrained=False, down_ratio=down_ratio).cuda()
dla_encoder_decoder.load_custom_pretrained_weights('/herdnet/pth_files/dla34-ba72cf86.pth')
######## Density Loss ############
losses = [
    {'loss': DensityLoss(reduction='mean', eps=1e-6), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'density_loss'},
]

# herdnet = LossWrapper(dla_encoder_decoder, losses=losses)
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

        # Print parameters to update
        if param.requires_grad:
            print(f"Trainable parameter: {name}")
        else:
            print(f"Frozen parameter: {name}")

    return params_to_update
layers_to_freeze= ['cls_head']

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
mkdir(work_dir)
weight_decay = 1e-4
epochs = 100


optimizer = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)
# lr_milestones = [30, 60, 90]  # Example milestones
# auto_lr = {'mode': 'min', 'factor': 0.1, 'patience': 10, 'verbose': True}
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
    # lr_milestones=lr_milestones,  # Pass the milestones
    # auto_lr=auto_lr,
    best_model_path='/herdnet/pth_files',
    val_dataloader= val_dataloader, # loss evaluation
    patience=70,
    work_dir=work_dir
    )


if wandb.run is not None:
  wandb.finish()

wandb.init(project="herdnet_pretrain")
####### Ploting Heatmaps ############
output_dir = '/herdnet/val_output/before_training'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Get a batch from the validation dataloader
images, targets = next(iter(val_dataloader))  
images = images.to('cuda')  # Move images to GPU

# Ensure the model is in evaluation mode
dla_encoder_decoder.eval()

# Get heatmaps
with torch.no_grad():  # Disable gradient computation
    output = dla_encoder_decoder(images)  # Get the output from the model

# Extract heatmaps from the output tuple
heatmaps = output[0][0]  # The heatmaps tensor
heatmaps = heatmaps.cpu().numpy()  # Move the heatmaps to CPU and convert to numpy

# Get image IDs from the targets
image_ids = targets['image_name'][0]  # Extract the list of image names

# Print the length of image_ids and shape of heatmaps to debug
print(f"Number of image IDs: {len(image_ids)}")
print(f"Number of heatmaps: {heatmaps.shape[0]}")

# Ensure the lengths match before proceeding
assert len(image_ids) == heatmaps.shape[0], "Mismatch between the number of heatmaps and image IDs!"

# Plot and save the heatmaps for each image in the batch
for j in range(heatmaps.shape[0]):
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmaps[j, 0], cmap='hot', interpolation='nearest')
    plt.title(f'Heatmap for Image ID {image_ids[j]}')
    
    # Save the plot with the image ID in the filename
    output_path = os.path.join(output_dir, f'heatmap_image_{image_ids[j]}_before_training.png')
    plt.savefig(output_path)
    plt.close()

print(f"Heatmaps saved in {output_dir}")
############ After training;
output_dir = '/herdnet/val_output/after_training'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
# Initialize your custom loss function
density_loss = DensityLoss()  # Assuming DensityLoss is defined elsewhere in your code

# Training loop
num_epochs = 50  # Example: Train for 50 epochs
checkpoint_epoch = 50  # Plot heatmaps every n epochs

# Inside the training loop, only use training data for model updates
for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}/{num_epochs}: Training...")
    
    # Training loop
    dla_encoder_decoder.train()  # Set the model to training mode
    
    for batch_idx, (images, targets) in enumerate(train_dataloader):
        images = images.to('cuda')
        targets = targets['binary'].to('cuda')
        
        optimizer.zero_grad()  # Zero the gradients
        
        outputs = dla_encoder_decoder(images)
        
        density_mean = outputs[1]
        
        loss = density_loss(density_mean, targets)
        
        loss.backward()
        optimizer.step()
        
        # Log gradients and loss
        for name, param in dla_encoder_decoder.named_parameters():
            if param.grad is not None:
                print(f"{name}: Grad mean = {param.grad.abs().mean()}")
            else:
                print(f"{name}: No gradient computed")
                
        print(f"Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item()}")
    
    # Evaluate on validation data without affecting gradients
    if epoch > 0 and epoch % checkpoint_epoch == 0:
        print(f"Epoch {epoch}: Generating heatmaps...")

        dla_encoder_decoder.eval()
        with torch.no_grad():
            images, targets = next(iter(val_dataloader))
            images = images.to('cuda')
            
            output = dla_encoder_decoder(images)
            
            heatmaps = output[0][0].cpu().numpy()
            image_ids = targets['image_name'][0]
            
            for j in range(heatmaps.shape[0]):
                plt.figure(figsize=(10, 10))
                plt.imshow(heatmaps[j, 0], cmap='hot', interpolation='nearest')
                plt.title(f'Heatmap for Image ID {image_ids[j]} (Epoch {epoch})')
                output_path = os.path.join(output_dir, f'heatmap_image_{image_ids[j]}_epoch_{epoch}.png')
                plt.savefig(output_path)
                plt.close()
                
        print(f"Heatmaps saved for epoch {epoch}")
    
    # Ensure to switch back to training mode after evaluation
    dla_encoder_decoder.train()


    # Ensure to switch back to training mode after heatmap generation
    dla_encoder_decoder.train()

print("Training completed.")


# trainer.resume(pth_path='/herdnet/herdnet/Binary_pth/binary_20240829.pth', checkpoints='best', select='max', validate_on='f1_score', load_optim=True, wandb_flag=False)
trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='f1_score', wandb_flag =True)
###### Evaluator ######
val_f1_score=evaluator.evaluate(returns='f1_score', viz=True, wandb_flag =False)

#Detections and Results
df_val_r=evaluator.results
df_val_r.to_csv('/herdnet/new_Validation_Binary_results.csv')
df_val_d=evaluator.detections
val_detections_path='/herdnet/new_Val_Binary_detections.csv'
df_val_d.to_csv(val_detections_path)
df_val_d.rename(columns={'binary': 'Ground truth','count_1': 'empty_count', 'count_2': 'non_empty_count'}, inplace=True)
# ############### Comparison of the detections and gt #########
detections_df = pd.read_csv(val_detections_path)
gt_df = pd.read_csv('/herdnet/new_Val_Binary_detections.csv')
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
detections_df.to_csv('/herdnet/new_Val_Final_detection_file.csv', index=False)

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
wandb.init(project="herdnet_pretrain", reinit=True)

# Start testing
test_f1_score = test_evaluator.evaluate(returns='f1_score', wandb_flag=True)
# Print global F1 score (%)
print(f"F1 score = {test_f1_score * 100:0.0f}%")

# Get the test results
test_results = test_evaluator.results
test_results.to_csv('/herdnet/test_output/new_Binary_test_results.csv', index=False)

#Get the test detections
test_detections=test_evaluator.detections
test_detections.to_csv('/herdnet/test_output/new_Binary_test_detections.csv', index=False)

# TEST_DATASETS
test_detections.rename(columns={'binary': 'Ground truth','count_1': 'empty_count', 'count_2': 'non_empty_count'}, inplace=True)
detections_df=pd.read_csv('/herdnet/test_output/new_Binary_test_detections.csv')
############### Comparison of the detections and gt #########
gt_df = pd.read_csv('/herdnet/DATASETS/CAH_no_margins_30/test/Test_binary_gt.csv')
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

# Save the updated DataFrame back to a new CSV 
detections_df.to_csv('/herdnet/test_output/new_Test_Final_detection_file.csv', index=False)