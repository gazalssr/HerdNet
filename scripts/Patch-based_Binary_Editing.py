import random
import animaloc
import pandas as pd
import os
# Set the seed
from animaloc.utils.seed import set_seed
import numpy
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
from animaloc.train.losses import FocalLoss, BinaryFocalLoss, FocalComboLoss_M, FocalComboLoss_P
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
batch_size=32
down_ratio = 2
train_dataset = BinaryFolderDataset(preprocess=preprocess,
    csv_file = '/herdnet/DATASETS/CAH_no_margins_30/train/Train_binary_gt.csv',
    root_dir = '/herdnet/DATASETS/CAH_no_margins_30/train/',
    albu_transforms = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Blur(blur_limit=15, p=0.2),
        A.Normalize(p=1.0)
        ]),
    end_transforms = [BinaryMultiTransformsWrapper([
        BinaryTransform(),
        ])]
    )
# train_dataset.data.to_csv('/herdnet/DATASETS/CAH_Complete_FCH_101114_STRATIFIED/train_CF/Train_binary_gt.csv', index=False)
import os
from PIL import Image
val_dataset = BinaryFolderDataset(preprocess=preprocess,
    csv_file = '/herdnet/DATASETS/CAH_no_margins_30/val/Val_binary_gt.csv',
    root_dir = '/herdnet/DATASETS/CAH_no_margins_30/val/',
    albu_transforms = [
        A.Normalize(p=1.0),
        ToTensorV2()
        ],
    end_transforms = [BinaryMultiTransformsWrapper([
        BinaryTransform(),
        ])]
    )
# val_dataset.data.to_csv('/herdnet/DATASETS/CAH_Complete_FCH_101114_STRATIFIED/val_CF/Val_binary_gt.csv', index=False)


test_dataset = BinaryFolderDataset(preprocess=preprocess,
    
    csv_file = '/herdnet/DATASETS/CAH_no_margins_30/test/Test_binary_gt.csv',
    root_dir = '/herdnet/DATASETS/CAH_no_margins_30/test/',
    albu_transforms = [A.Normalize(p=1.0)], 
    end_transforms = [BinaryMultiTransformsWrapper([
        BinaryTransform(),
        ])]
    )

# Initialize BinaryBatchSampler for training
train_sampler = BinaryBatchSampler(
    dataset=train_dataset,
    col='binary',  
    batch_size=32,  # Even batch_size
    shuffle=True
)
# test_dataset.data.to_csv('/herdnet/DATASETS/CAH_Complete_FCH_101114_STRATIFIED/test_W/Test_binary_gt.csv', index=False)
# Dataloaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset = train_dataset,batch_size=16, num_workers= 2,shuffle= True, collate_fn=BinaryFolderDataset.collate_fn)

val_dataloader = DataLoader(dataset = val_dataset, batch_size=1 , num_workers= 2, shuffle= False)

test_dataloader= DataLoader(dataset = test_dataset, batch_size=1 , num_workers= 2, shuffle= False)
num_classes=2
dla_encoder = DLAEncoder(num_classes=num_classes, pretrained=True).cuda()
##### TESTING #############
original_save_dir = './herdnet/original_images'
augmented_save_dir = './herdnet/augmented_images'
os.makedirs(original_save_dir, exist_ok=True)  # Ensure the directory exists
os.makedirs(augmented_save_dir, exist_ok=True)  # Ensure the directory exists

# Save a few samples to verify
# Save a few samples to verify
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return numpy.array(image)

# Function to save images as PDF
def save_image_as_pdf(image, save_path):
    plt.imsave(save_path, image, format='pdf')

# Function to clip image values to valid range
def clip_image(image):
    image = numpy.clip(image, 0, 1)
    return image

# Save a few samples to verify
for i, (images, targets) in enumerate(train_dataloader):
    if i >= 1:
        break
    # Save original and augmented images
    for j in range(len(images)):
        image_name = targets['image_name'][j]
        image_path = os.path.join('/herdnet/DATASETS/CAH_no_margins_30/train/', image_name)
        
        # Load the original image
        img_original = load_image(image_path)

        # Save the original image as PDF
        original_save_path = os.path.join(original_save_dir, f'original_image_batch{i+1}_image{j+1}_name{image_name}.pdf')
        save_image_as_pdf(img_original, original_save_path)

        # Process the augmented image if it's a tensor
        img_augmented = images[j]
        if isinstance(img_augmented, torch.Tensor):
            img_augmented = img_augmented.numpy()

        # Ensure the augmented image has the correct shape (height, width, channels)
        if img_augmented.shape[0] == 3:
            img_augmented = numpy.transpose(img_augmented, (1, 2, 0))

        # Clip the image to the valid range
        img_augmented = clip_image(img_augmented)

        # Save the augmented image as PDF
        augmented_save_path = os.path.join(augmented_save_dir, f'augmented_image_batch{i+1}_image{j+1}_name{image_name}.pdf')
        save_image_as_pdf(img_augmented, augmented_save_path)

    # print(f"Saved original and augmented images for batch {i+1}")

################### PRINT DATALOADER INFO ##################
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torchvision import transforms

# Function to visualize and save augmented images
import torch
import matplotlib.pyplot as plt
import os
# Instantiate the dataset
train_dataset = BinaryFolderDataset(
    preprocess=preprocess,
    csv_file='/herdnet/DATASETS/CAH_no_margins_30/train/Train_binary_gt.csv',
    root_dir='/herdnet/DATASETS/CAH_no_margins_30/train/',
    albu_transforms=[
        A.VerticalFlip(p=0.9),
        A.HorizontalFlip(p=0.9),
        A.RandomRotate90(p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Blur(blur_limit=15, p=0.9),
        A.Normalize(p=0.9),
        ToTensorV2()
    ],
    end_transforms=[BinaryMultiTransformsWrapper([BinaryTransform()])]
)

# Define DLAENCODER for training
import torch.nn as nn
image= torch.ones([1,3,512,512]).cuda()

################ Make a test phase for training loop ########################
# Number of patches per class (train datset)
total_patches = len(train_dataset)
empty_patches = 7960
non_empty_patches = 1502
# Class weights
weight_for_empty = total_patches / (2 * empty_patches)
weight_for_non_empty = total_patches / (2 * non_empty_patches)
weights = [weight_for_empty, weight_for_non_empty]
###############
import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DLAEncoder().to(device)

# Define the optimizer
learning_rate = 0.0001  # Adjust if necessary
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the class weights based on your dataset
total_patches = len(train_dataset)
empty_patches = 7960
non_empty_patches = 1502

# Class weights
weight_for_empty = total_patches / (2 * empty_patches)
weight_for_non_empty = total_patches / (2 * non_empty_patches)
pos_weight = torch.tensor([weight_for_non_empty], device=device)  ##### Define pos_weight for BCEWithLogitsLoss

# Initialize the BCEWithLogitsLoss with pos_weight
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
weights = [weight_for_empty, weight_for_non_empty]

# Initialize the custom loss function with weights
criterion = FocalComboLoss_P(weights=weights)

# Define the accuracy calculation function
def calculate_accuracy(probabilities, targets, threshold=0.5):
    preds = (probabilities > threshold).float()
    correct = (preds == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

# Convert list of numpy arrays to list of tensors and stack
def convert_and_stack(inputs):
    tensor_list = [torch.tensor(img).to(device).permute(2, 0, 1) for img in inputs]  ##### Convert each numpy array to tensor and permute
    return torch.stack(tensor_list)  ##### Stack the list of tensors

# Get a batch of data for initial predictions
inputs, targets = next(iter(train_dataloader))
targets = targets['binary'].to(device)
inputs = convert_and_stack(inputs)  ##### Convert and stack inputs to tensor

# Forward pass: get logits from the model
outputs = model(inputs)

# Apply sigmoid to convert logits to probabilities
probabilities = torch.sigmoid(outputs)

# Convert probabilities to binary predictions
threshold = 0.5  # Adjust threshold if needed
predictions = (probabilities > threshold).float()

# Print initial logits, probabilities, binary predictions, and targets
print("Initial Predictions")
print("Logits (raw outputs):")
print(outputs)
print("Probabilities after sigmoid:")
print(probabilities)
print("Binary predictions:")
print(predictions)
print("Targets:")
print(targets)

# Calculate and print initial loss
initial_loss = criterion(outputs, targets)
print("Initial Loss:", initial_loss.item())

# Calculate and print initial accuracy
initial_accuracy = calculate_accuracy(probabilities, targets)
print("Initial Accuracy:", initial_accuracy)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0

    for inputs, targets in train_dataloader:
        targets = targets['binary'].to(device)
        inputs = convert_and_stack(inputs)  ##### Convert and stack inputs to tensor

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        probabilities = torch.sigmoid(outputs)
        accuracy = calculate_accuracy(probabilities, targets)
        total_loss += loss.item()
        total_accuracy += accuracy

    avg_loss = total_loss / len(train_dataloader)
    avg_accuracy = total_accuracy / len(train_dataloader)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    # Add a condition to check for NaN or very high loss values
    if torch.isnan(loss):
        print("Loss is NaN, check the model and data.")
        break

# Get a batch of data for final predictions
inputs, targets = next(iter(train_dataloader))
targets = targets['binary'].to(device)
inputs = convert_and_stack(inputs)  ##### Convert and stack inputs to tensor

# Forward pass: get logits from the model
outputs = model(inputs)

# Apply sigmoid to convert logits to probabilities
probabilities = torch.sigmoid(outputs)

# Convert probabilities to binary predictions
predictions = (probabilities > threshold).float()

# Print final logits, probabilities, binary predictions, and targets
print("Final Predictions")
print("Logits (raw outputs):")
print(outputs)
print("Probabilities after sigmoid:")
print(probabilities)
print("Binary predictions:")
print(predictions)
print("Targets:")
print(targets)

# Calculate and print final loss
final_loss = criterion(outputs, targets)
print("Final Loss:", final_loss.item())

# Calculate and print final accuracy
final_accuracy = calculate_accuracy(probabilities, targets)
print("Final Accuracy:", final_accuracy)
########################    
# def print_dataloader_info(dataloader):
#     total_batches = len(dataloader)
#     total_images = len(dataloader.dataset)
#     batch_size = dataloader.batch_size

#     print(f"Total number of images: {total_images}")
#     print(f"Batch size: {batch_size}")
#     print(f"Total number of batches: {total_batches}")
#     for i, (images, targets) in enumerate(train_dataloader):
#         print(f"Batch {i+1}, Number of images: {len(images)}")
#         batch_index = 0
#         for batch in dataloader:
#             images, targets = batch
#         #     print(f"Batch {batch_index}:")
#             # print(f"Number of images in this batch: {len(images)}")
#             # print("List of images:")
#             # for i, image in enumerate(images):
#             #     print(f"Image {i+1}: {image}")  # Assuming image is a file path or similar
#             # batch_index += 1
            
#             if isinstance(targets, dict) and 'binary' in targets:
#                 print(f"Number of targets in this batch: {len(targets['binary'])}")
#                 print("Targets tensor:")
#                 print(targets['binary'])
#             elif isinstance(targets, torch.Tensor):  
#                 print(f"Number of targets in this batch: {len(targets)}")
#                 print("Targets tensor:")
#                 print(targets)
#             else:
#                 print("Targets not found or not in expected format.")

# print("Train DataLoader:")
# print_dataloader_info(train_dataloader)
# print("Val DataLoader:")
# print_dataloader_info(val_dataloader)
# print("Test DataLoader:")
# print_dataloader_info(test_dataloader)
#####################

################## BCEWithLogitsLoss ############################

# # # Create a tensor of weights for use in BCEWithLogitsLoss
# # # weights = torch.tensor([weight_for_empty, weight_for_non_empty], dtype=torch.float32).cuda()

# # # Pos_weight for BCEWithLogitsLoss
# pos_weight = torch.tensor([weight_for_non_empty / weight_for_empty], dtype=torch.float32).cuda()

# # # Initialize the loss function with pos_weight
# bce_loss_with_weights = BCEWithLogitsLoss(pos_weight=pos_weight)
# ###### Scenario1: using bceloss loss with weights
# losses = [
#     {'loss': bce_loss_with_weights, 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'bce_loss'}
# ]
######## Scenario2: using BCELoss without weights
# losses = [
#       {'loss': BCEWithLogitsLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'bce_loss'}
#       ]
###### Scenario 3: using FCLoss and BCELoss with weights 
# Initialize FocalLoss with appropriate alpha and weights
# focal_loss = BinaryFocalLoss(alpha_pos=0.85, alpha_neg=0.15, beta=3, reduction='mean')

# # Configuration of losses with lambda set to 1.0 as previously used
# losses = [
#     {'loss': focal_loss, 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
#     {'loss': bce_loss_with_weights, 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'bce_loss'}
# ]
###### Scenario 4: using weighted Focal-loss only
# focal_loss = BinaryFocalLoss(alpha_pos=0.75, alpha_neg=0.25, beta=4, reduction='mean', weights=[weight_for_empty, weight_for_non_empty])
# losses = [
#     {'loss': focal_loss, 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'}]
# ################## Scenario 5: Focal combo loss with weights (My modified FCloss) 
# focal_combo_loss = FocalComboLoss_M(alpha_pos=0.75, alpha_neg=0.25, gamma=4, reduction='mean', weights=[weight_for_empty, weight_for_non_empty], dice_weight=0.3  # Adjust the balance between focal and dice loss components
# )
# losses = [{'loss': focal_combo_loss, 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_combo_loss'}]
################## Scenario 6: Focal combo loss with weights with 1/beta as denominator (FCLoss in the paper) 
focal_combo_loss = FocalComboLoss_P(alpha=0.35,beta=3, gamma=2, reduction='mean', weights=[weight_for_empty, weight_for_non_empty] # Adjust the balance between focal and dice loss components
)
losses = [{'loss': focal_combo_loss, 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_combo_loss'}]
# else:
#   losses = [
#       {'loss': L1Loss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'bce_loss'},
#       ]
    
dla_encoder = LossWrapper(dla_encoder, losses=losses)

######################################################################################################################
# DLA Auto_Encoder
##### DLA AutoEncoder 
# dla_encoder_decoder = DLAEncoderDecoder(num_classes=num_classes, pretrained=True).cuda()
# dla_encoder_decoder.load_custom_pretrained_weights('/herdnet/pth_files/dla34-ba72cf86.pth')
# dla_encoder_decoder = LossWrapper(dla_encoder_decoder, losses=losses)

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
#AUTO_ENCODER
# param_dict= get_parameter_names(dla_encoder_decoder.model)
#ENCODER
param_dict= get_parameter_names(dla_encoder.model)
print(param_dict)
lr = 1e-4 # learning rate
layers_to_freeze= [] # nothing frozen
#layers_to_freeze= ['base_layer','level0','level1','level2','level3','level4'] # we are feezing all the levels below level5
# layers_to_freeze= ['base_layer','level0','level1','level2','level3','level4','level5','fc','bottleneck_conv'] # we are feezing everything except cls_head

# params_to_update = freeze_parts(dla_encoder_decoder.model,param_dict, layers_to_freeze,lr,False)
params_to_update = freeze_parts(dla_encoder.model,param_dict, layers_to_freeze,lr,False)


#### Create the Trainer #######
from torch.optim import Adam

from animaloc.train import Trainer
from animaloc.eval import ImageLevelMetrics, HerdNetStitcher, TileEvaluator, AutoTileEvaluator
from animaloc.utils.useful_funcs import mkdir

work_dir = '/content/drive/MyDrive/output'
mkdir(work_dir)

lr = 1e-4
weight_decay = 1e-3
epochs =100
# optimizer = Adam(params=dla_encoder_decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)
lr_milestones = [30, 60, 90]  # Example milestones
auto_lr = {'mode': 'min', 'factor': 0.1, 'patience': 10, 'verbose': True}
img_names = val_dataloader.dataset._img_names
metrics = ImageLevelMetrics(img_names=img_names, num_classes=2)
# metrics = ImageLevelMetrics(num_classes=num_classes)
metrics.binary_annotations = True
#AutotileEvaluator for autoencoder and tileEvaluator for DLA encoder
evaluator = TileEvaluator(
    metrics_class=ImageLevelMetrics,
    threshold=0.3,
    model=dla_encoder,
    dataloader=val_dataloader,
    num_classes=num_classes,
    stitcher=None,
    work_dir=work_dir,
    header='validation',
    )

trainer = Trainer(
    model=dla_encoder,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    num_epochs=epochs,
    evaluator=evaluator, 
    lr_milestones=lr_milestones,  # Pass the milestones
    auto_lr=auto_lr,# metric evaluation
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

val_f1_score=evaluator.evaluate(returns='f1_score', viz=True, wandb_flag =True )

#Detections and Results
df_val_r=evaluator.results
df_val_r.to_csv('/herdnet/Validation_Binary_results.csv')
df_val_d=evaluator.detections
# df_val_d.rename(columns={'count_1': 'empty_count', 'count_2': 'non_empty_count'}, inplace=True)
val_detections_path='/herdnet/CAH_COMPLETE_Val_Binary_detections.csv'
df_val_d.to_csv(val_detections_path)

# ############### Comparison of the detections and gt #########
detections_df = pd.read_csv(val_detections_path)
gt_df = pd.read_csv('/herdnet/CAH_COMPLETE_Val_Binary_detections.csv')
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
detections_df.to_csv('/herdnet/CAH_COMPLETE_Val_Final_detection_file.csv', index=False)
################################## Analyze batch composition for the sampler ##################
data_analyzer = DataAnalyzer()
# Collect precision values
precision_values = [evaluator.metrics.precision() for _ in range(trainer.epochs)]
train_empty_counts, train_non_empty_counts = data_analyzer.analyze_batch_composition(train_dataloader)
val_empty_counts, val_non_empty_counts = data_analyzer.analyze_batch_composition(val_dataloader)

# Plot precision trend
data_analyzer.analyze_precision_trend(precision_values,save_path='herdnet/precision_trend.pdf')
##################################### Plot gamma-beta tradeoff in focalcomboloss ############
# beta_values = [2, 2.5, 3, 3.5, 4]
# gamma_values = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
# plot_trade_off = PlotTradeOff()
# eval_epochs = 40
# def train_and_evaluate_model(beta, gamma, train_dataloader, val_dataloader, model, epochs, work_dir):
#     print(f"Training with beta={beta} and gamma={gamma}")
#     focal_combo_loss = FocalComboLoss_P(alpha=0.35, beta=beta, gamma=gamma, reduction='mean', weights=[weight_for_empty, weight_for_non_empty])
#     losses = [{'loss': focal_combo_loss, 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_combo_loss'}]

#     # Initialize the model with the given loss
#     model = LossWrapper(model, losses=losses)
    
#     optimizer = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-3)
#     evaluator = TileEvaluator(
#         metrics_class=ImageLevelMetrics,
#         threshold=0.3,
#         model=model,
#         dataloader=val_dataloader,
#         num_classes=2,
#         stitcher=None,
#         work_dir=work_dir,
#         header='validation',
#     )
    
#     trainer = Trainer(
#         model=model,
#         train_dataloader=train_dataloader,
#         optimizer=optimizer,
#         num_epochs=eval_epochs,
#         evaluator=evaluator,
#         work_dir=work_dir
#     )
    
#     trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='f1_score', wandb_flag=False)
    
#     f1_score = evaluator.evaluate(returns='f1_score')
#     return f1_score

# for beta in beta_values:
#     f1_scores = []
#     for gamma in gamma_values:
#         f1_score = train_and_evaluate_model(beta, gamma, train_dataloader, val_dataloader, dla_encoder, epochs, work_dir)
#         f1_scores.append(f1_score)
#     plot_trade_off.feed(beta, gamma_values, f1_scores)
#     print(f"Results for beta={beta}:")
#     print(f"F1 Scores: {f1_scores}")

# plot_trade_off.plot()
# plot_trade_off.save('/herdnet/trade_off_f1_score.pdf')


########################################## Threshold Tuning ################################################
# import numpy as np
# thresholds = np.linspace(0.1, 0.9, 9)
# f1_scores = {}

# for threshold in thresholds:
#     evaluator.threshold = threshold  # Update the threshold attribute of the evaluator
#     f1_score = evaluator.evaluate(returns='f1_score', viz=True)
#     print(f"Threshold: {threshold}, F1 Score: {f1_score}")
#     f1_scores[threshold] = f1_score

# # Plotting F1 Score vs. Threshold
# plt.figure(figsize=(10, 5))
# plt.plot(list(f1_scores.keys()), list(f1_scores.values()), marker='o', linestyle='-', color='b')
# plt.title('F1 Score vs. Threshold')
# plt.xlabel('Threshold')
# plt.ylabel('F1 Score')
# plt.grid(True)
# plt.show()
# plt.savefig('/herdnet/f1score_Confidence.pdf')
#################### Test data and evaluation ###########
# Create output folder
test_dir = '/herdnet/test_output'
mkdir(test_dir)

# Create an Evaluator
test_evaluator = TileEvaluator(
    model=dla_encoder,
    dataloader=test_dataloader,
    metrics_class=ImageLevelMetrics,
   num_classes=num_classes,
    stitcher=None,
    work_dir=test_dir,
    threshold=0.3,
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
test_results.to_csv('/herdnet/test_output/Binary_test_results_WAH.csv', index=False)

#Get the test detections
test_detections=test_evaluator.detections
test_detections.to_csv('/herdnet/test_output/Binary_test_detections_WAH.csv', index=False)

# TEST_DATASETS
df_val_d.rename(columns={'binary': 'Ground truth','count_1': 'empty_count', 'count_2': 'non_empty_count'}, inplace=True)
detections_df=pd.read_csv('/herdnet/test_output/Binary_test_detections_WAH.csv')
############### Comparison of the detections and gt #########
gt_df = pd.read_csv('/herdnet/DATASETS/test_complete_CAH/Test_binary_gt.csv')
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
detections_df.to_csv('/herdnet/WAH_Test_Final_detection_file.csv', index=False)
######
