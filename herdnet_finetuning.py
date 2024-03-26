#Turn on GPU access
import os
os.system('nvidia-smi')


# Dataset paths
#train
train_root_path='/home/ghazaleh/HerdNet/data/train_m'
train_gt_path='/home/ghazaleh/HerdNet/data/train_m/train.csv'
#Validation
val_root_path='/home/ghazaleh/HerdNet/data/val_m'
val_gt_path='/home/ghazaleh/HerdNet/data/val_m/val.csv'
#test
test_root_path='/home/ghazaleh/HerdNet/data/test_m'
test_gt_path='/home/ghazaleh/HerdNet/data/test_m/test.csv'


#Set the seed
from animaloc.utils.seed import set_seed
set_seed(9292)


#Showing an image that is patched
#%matplotlib inline
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
from animaloc.data.patches import ImageToPatches, AnnotatedImageToPatches
img_path ='/home/ghazaleh//HerdNet/whole_images_stratified/train/CAH_5_50_2017.jpg'
sample_img=Image.open(img_path)
patcher = ImageToPatches(sample_img, (512,512), overlap = 100)
patches = patcher.make_patches()
print(f'Number of patches: {len(patches)}')
plt.figure(figsize=(15,10))
_ = patcher.show()


# Showing some samples of patches and the annotations
#%matplotlib inline
import matplotlib.pyplot as plt
from animaloc.datasets import CSVDataset
from animaloc.data.batch_utils import show_batch, collate_fn
from torch.utils.data import DataLoader
import torch
import albumentations as A
batch_size = 8
csv_path = train_gt_path
image_path = train_root_path
dataset = FolderDataset(csv_path, image_path, [A.Normalize()])
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
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


# Training, validation and test datasets
import albumentations as A
from animaloc.datasets import FolderDataset
from animaloc.data.transforms import MultiTransformsWrapper, DownSample, PointsToMask, FIDT

patch_size = 512
num_classes = 2
down_ratio = 2

train_dataset = FolderDataset(
    csv_file = train_gt_path,
    root_dir = train_root_path,
    # Data Augmentation
    albu_transforms = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(p=1.0),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Blur(blur_limit=15, p=0.2),
        ],
    end_transforms = [MultiTransformsWrapper([
        FIDT(num_classes=num_classes, down_ratio=down_ratio),
        PointsToMask(radius=2, num_classes=num_classes, squeeze=True, down_ratio=int(patch_size//16))
        ])]
    )

val_dataset = FolderDataset(

    csv_file = val_gt_path,
    root_dir = val_root_path,
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
    )

test_dataset = FolderDataset(
    csv_file = test_gt_path,
    root_dir = test_root_path,
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
    )


# Dataloaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset = train_dataset, batch_size = 1, shuffle = True)

val_dataloader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = False)

test_dataloader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)


# Define HerdNet for training
# 1. Path to your .pth file (initial pth file)
import torch
pth_path = '/home/ghazaleh/HerdNet/pth_files/Binary.pth'
pretrained_dict=torch.load(pth_path)


# 2. Transfer weights to Herdnet ###
from animaloc.models import HerdNet
herdnet = HerdNet(num_classes=num_classes, down_ratio=down_ratio).cuda()
herdnet_dict = herdnet.state_dict()
# Match and load pre-trained weights
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in herdnet_dict}
herdnet_dict.update(pretrained_dict)
herdnet.load_state_dict(herdnet_dict)


#3. Define loss functions
from animaloc.models import HerdNet
from torch import Tensor
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss

herdnet = HerdNet(num_classes=num_classes, down_ratio=down_ratio).cuda()
weight = Tensor([1.28,4.54]).cuda()
losses = [
    {'loss': FocalLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
    {'loss': CrossEntropyLoss(reduction='mean'), 'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
    ]

herdnet = LossWrapper(herdnet, losses=losses)


# 4. Get model layers ###########################
def get_parameter_names(model): # getting the model layers
  param_dict= dict()
  for l, (name,param) in enumerate(model.named_parameters()):
    #print(l,":\t",name,type(param),param.requires_grad)
    param_dict[name]= l
  return param_dict
result = get_parameter_names(herdnet)
print(result)



# 5. Freeze half of a specified layer
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


# 6. Specifying half of one layer+ other layers to freeze
params_to_update = freeze_parts(herdnet.model, get_parameter_names, layers_to_freeze=['base_layer','level0','level1','level2','level3'], freeze_layer_half='level_4', lr=0.0001, unfreeze=False)


# 7. Create the trainer
from torch.optim import Adam
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator
from animaloc.utils.useful_funcs import mkdir

work_dir = '/home/ghazaleh/HerdNet/output'
mkdir(work_dir)

lr = 1e-4
weight_decay = 1e-3
epochs =1

optimizer = Adam(params=herdnet.parameters(), lr=lr, weight_decay=weight_decay)

metrics = PointsMetrics(radius=20, num_classes=num_classes)

stitcher = HerdNetStitcher(
    model=herdnet,
    size=(patch_size,patch_size),
    overlap=256,
    down_ratio=down_ratio,
    reduction='mean'
    )

evaluator = HerdNetEvaluator(
    model=herdnet,
    dataloader=val_dataloader,
    metrics=metrics,
    stitcher=None,
    work_dir=work_dir,
    header='validation'
    )

trainer = Trainer(
    model=herdnet,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    num_epochs=epochs,
    evaluator=evaluator,  # metric evaluation (original)
    # val_dataloader= val_dataloader, # loss evaluation
    work_dir=work_dir
 )


# 8. Start training and saving parameters
#wandb
import wandb
if wandb.run is not None:
  wandb.finish()
wandb.init(project="herdnet-finetuning")

# 9. Trainer
trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='f1_score', wandb_flag =True)


########## Evaluating the model on training set
# 1. Load trained parameters
pth_path='/home/ghazaleh/HerdNet/output/best_model.pth'
from animaloc.models import load_model
herdnet = load_model(herdnet, pth_path=pth_path)
torch.save(herdnet.state_dict(), 'fine_tuned_model.pth')  

# #Define a new train dataloader for evaluation
# train_dataset = FolderDataset(

#     csv_file = train_gt_path,
#     root_dir = train_root_path,
#     # Data Augmentation
#     albu_transforms = [A.Normalize(p=1.0)],
#     end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
# )

# train_dataloader = DataLoader(dataset = train_dataset, batch_size = 1, shuffle = False)
# train_evaluator = HerdNetEvaluator(
#     model=herdnet,
#     dataloader=train_dataloader,
#     metrics=metrics,
#     stitcher=None,
#     work_dir=train_dir,
#     header='train'
#     )


# # Start train evaluation
# train_f1_score = train_evaluator.evaluate(returns='f1_score')


# #Get evaluation results
# df= train_evaluator.results # getting the validation results in pandas format
# df.head()
# df= train_evaluator.detections
# df.head()
# df.to_csv('/home/ghazaleh/HerdNet/output/train_detections.csv', index=False)

# #Plot thge detections on the patches...........
# # from animaloc.data.batch_utils import 