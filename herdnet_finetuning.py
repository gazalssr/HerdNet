# #%#% [markdown]
# <a href="https://colab.research.google.com/github/gazalssr/HerdNet/blob/main/Base_Herdnet_Finetunning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# #%#% [markdown]
# # DEMO - Training and testing HerdNet on nadir aerial images

# #%#% [markdown]
# ## Installations


# #%#%



# #%#% [markdown]
# ## Create datasets

# #%#%
# Download some of the data of Delplanque et al. (2021) as an example
#!gdown 1z9cg2eQs9Xy32ug_OJcbDXIE6A-pEwQM -O /content/data.zip
#!unzip -oq /content/data.zip -d /content

# #%#%
# Set the seed
from animaloc.utils.seed import set_seed

set_seed(9292)

# #%#%
#!python /herdnet/tools/patcher.py -h

# #%#%
# From bbox to points (no need to do this)
import pandas as pd

train_df = pd.read_csv('/content/data/train.csv')
train_df['x'] = (train_df['x_min'] + train_df['x_max']) / 2
train_df['y'] = (train_df['y_min'] + train_df['y_max']) / 2
train_df = train_df[['images','x','y','labels']]
train_df.to_csv('/content/data/train.csv', index=False)

val_df = pd.read_csv('/content/data/val.csv')
val_df['x'] = (val_df['x_min'] + val_df['x_max']) / 2
val_df['y'] = (val_df['y_min'] + val_df['y_max']) / 2
val_df = val_df[['images','x','y','labels']]
val_df.to_csv('/content/data/val.csv', index=False)

# #%#%
#%matplotlib inline
import matplotlib.pyplot as plt
import PIL.Image as Image

from animaloc.data.patches import ImageToPatches, AnnotatedImageToPatches

img_path = '/content/drive/MyDrive/images/CAH_GRP_1.jpg'

sample_img = Image.open(img_path)

patcher = ImageToPatches(sample_img, (512,512), overlap = 100)
patches = patcher.make_patches()

print(f'Number of patches: {len(patches)}')

plt.figure(figsize=(15,10))
_ = patcher.show()

# #%#%
# Create training  patches using the patcher tool
from animaloc.utils.useful_funcs import mkdir
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/CAH_small_samples/Train_CAH_Sample/Train_CAH_Sample.csv')
df = df.to_csv('/content/drive/MyDrive/CAH_small_samples/Train_CAH_Sample/Train_CAH_Sample.csv', index=False)
# mkdir('/content/drive/MyDrive/Train_patches')
mkdir('/content/Train_patches') # local directory
# !python /content/HerdNet/tools/patcher.py /content/drive/MyDrive/images/ 512 512 256 /content/Train_patches -csv /content/drive/MyDrive/images/TRAIN_FCH_8.csv -min 0.0 -all True
#!python /content/HerdNet/tools/patcher.py /content/drive/MyDrive/CAH_small_samples/Train_CAH_Sample/ 512 512 256 /content/Train_patches -csv /content/drive/MyDrive/CAH_small_samples/Train_CAH_Sample/Train_CAH_Sample.csv -min 0.0 -all False


# #%#%
# Create validation patches
#mkdir('/content/drive/MyDrive/val_patches')
mkdir('/content/val_patches')
# !python /content/HerdNet/tools/patcher.py /content/drive/MyDrive/valFCH/ 512 512 256 /content/validation_patches -csv /content/drive/MyDrive/valFCH/FCH_val_8.csv -min 0.0 -all True
#!python /content/HerdNet/tools/patcher.py /content/drive/MyDrive/CAH_small_samples/Val_CAH_Sample/ 512 512 256 /content/val_patches -csv /content/drive/MyDrive/CAH_small_samples/Val_CAH_Sample/VAL_CAH_Sample.csv -min 0.0 -all False

# #%#%
from google.colab import drive
drive.mount('/content/drive')

# #%#%
# Create test patches
from animaloc.utils.useful_funcs import mkdir
import pandas as pd
mkdir('/content/drive/MyDrive/test_patches')
# mkdir('/content/test_patches')
# !python /content/HerdNet/tools/patcher.py /content/drive/MyDrive/CAH_training_session_1/TestCAH 512 512 256 /content/drive/MyDrive/test_patches -csv /content/drive/MyDrive/CAH_training_session_1/TestCAH/TestCAH_2017_1.csv -min 0.0 -all True
#!python /content/HerdNet/tools/patcher.py /content/drive/MyDrive/CAH_small_samples/Test_CAH_Sample/ 512 512 256 /content/test_patches -csv /content/drive/MyDrive/CAH_small_samples/Test_CAH_Sample/Test_CAH_Sample.csv -min 0.0 -all False

# #%#% [markdown]
# Ziping the patches and transfer to google drive

# #%#%
#make a zip file from the patch folders (Only run this when ypou add a new dataset)
#Zip file path (destination)       source file path
import shutil
shutil.make_archive('/content/drive/MyDrive/Train_patches_S', 'zip', '/content/drive/MyDrive/Train_patches')
shutil.make_archive('/content/drive/MyDrive/val_patches_S', 'zip', '/content/drive/MyDrive/val_patches')
shutil.make_archive('/content/drive/MyDrive/test_patches_S', 'zip', '/content/drive/MyDrive/test_patches')

# #%#%
#### Downloading and unziping the files
#zip file download (destination link)                  Zip file saving location (same as the link)
#!gdown https://drive.google.com/uc?id=1-QF9YQHWrlfFxeQcRWYhU0cljm4CvoVz -O /content/Train_patches_S.zip
#Zip file location           uzipped file saving location
#!unzip -oq /content/Train_patches_S.zip -d /content/Train_patches

#!gdown https://drive.google.com/uc?id=18OfeLVDbgGFAeo09LAd8zMyD16kTLAQ0 -O /content/Val_patches_S.zip
#!unzip -oq /content/Val_patches_S.zip -d /content/val_patches

#!gdown https://drive.google.com/uc?id=1-79l91SDG5nCiohdvhbIzar3Zab_r60e -O /content/test_patches_S.zip
#!unzip -oq /content/test_patches_S.zip -d /content/test_patches


# #%#%
#%matplotlib inline
# Showing some samples of patches and the annotations
import matplotlib.pyplot as plt
from animaloc.datasets import CSVDataset
from animaloc.data.batch_utils import show_batch, collate_fn
from torch.utils.data import DataLoader
import torch
import albumentations as A
batch_size = 8

csv_path = '/content/Train_patches/gt.csv'
image_path = '/content/Train_patches'
dataset = CSVDataset(csv_path, image_path, [A.Normalize()])
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

# #%#%
# Training, validation and test datasets
import albumentations as A

from animaloc.datasets import CSVDataset
from animaloc.data.transforms import MultiTransformsWrapper, DownSample, PointsToMask, FIDT

patch_size = 512
num_classes = 2
down_ratio = 2

train_dataset = CSVDataset(
    # csv_file = '/content/drive/MyDrive/Train_patches/gt.csv',
    # root_dir = '/content/drive/MyDrive/Train_patches',
    csv_file = '/content/Train_patches/gt.csv',
    root_dir = '/content/Train_patches',
    # Data Augmentation
    albu_transforms = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(p=1.0),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        # A.Blur(blur_limit=15, p=0.2),
        ],
    end_transforms = [MultiTransformsWrapper([
        FIDT(num_classes=num_classes, down_ratio=down_ratio),
        PointsToMask(radius=2, num_classes=num_classes, squeeze=True, down_ratio=int(patch_size//16))
        ])]
    )

val_dataset = CSVDataset(
    # csv_file = '/content/drive/MyDrive/val_patches/gt.csv',
    # root_dir = '/content/drive/MyDrive/val_patches',
    csv_file = '/content/val_patches/gt.csv',
    root_dir = '/content/val_patches',
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
    )

test_dataset = CSVDataset(
    # csv_file = '/content/drive/MyDrive/test_patches/gt.csv',
    # root_dir = '/content/drive/MyDrive/test_patches',
    csv_file = '/content/test_patches/gt.csv',
    root_dir = '/content/test_patches',
    albu_transforms = [A.Normalize(p=1.0)],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
    )

# #%#%
# Dataloaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset = train_dataset, batch_size = 4, shuffle = True)

val_dataloader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = False)

test_dataloader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)

# #%#%
sample_batch[0].size()


# #%% [markdown]
# ## Define HerdNet for training

# %%
# Path to your .pth file (initial pth file)
import gdown
import torch
pth_path = 'output/best_model.pth'

if not pth_path:
    gdown.download(
        'https://drive.google.com/uc?export=download&id=1-WUnBC4BJMVkNvRqalF_HzA1_pRkQTI_',
        '/content/20220413_herdnet_model.pth'
        )
    pth_path = '/content/20220413_herdnet_model.pth'

# %%
from animaloc.models import HerdNet
from torch import Tensor
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss

herdnet = HerdNet(num_classes=num_classes, down_ratio=down_ratio).cuda()

losses = [
    {'loss': FocalLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
    {'loss': CrossEntropyLoss(reduction='mean'), 'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
    ]

herdnet = LossWrapper(herdnet, losses=losses)


# %%
#############Get model layers ###########################
def get_parameter_names(model): # getting the model layers
  param_dict= dict()
  for l, (name,param) in enumerate(model.named_parameters()):
    #print(l,":\t",name,type(param),param.requires_grad)
    param_dict[name]= l
  return param_dict
result = get_parameter_names(herdnet)
print(result)

# %%
#Freeze the layers
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

# %%
#Selecting the freezed layers (change the freezed ayers and record the results)
lr = 1e-3 # learning rate
# layers_to_freeze= [] # nothing frozen
layers_to_freeze= ['base_layer','level0','level1','level2','level3','level4'] # we are feezing all the levels below level5
# layers_to_freeze= ['base_layer','level0','level1','level2','level3','level4','level5','fc','bottleneck_conv'] # we are feezing everything except cls_head

params_to_update= freeze_parts(herdnet.model,get_parameter_names,layers_to_freeze,lr,unfreeze=False)
# optimizer = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)


# %% [markdown]
# ## Create the Trainer

# %%
from torch.optim import Adam
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator
from animaloc.utils.useful_funcs import mkdir

work_dir = '/content/drive/MyDrive/output'
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

# %%
#!ls /content/drive/MyDrive/val_patches/*.jpg
# csv_data= pd.read_csv('/content/drive/MyDrive/Train_patches/gt.csv')
# print(csv_data)

# %% [markdown]
# ## Inintial training

# %%
import wandb
if wandb.run is not None:
  wandb.finish()
wandb.init(project="herdnet-finetuning")

# %%
trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='f1_score', wandb_flag =True)

# %%
##Unfreeze the layers
layers_to_unfreeze = ['level5', 'fc']
params_to_update_UF = freeze_parts(herdnet.model, get_parameter_names, layers_to_unfreeze, lr=1e-3, unfreeze=True)

# %%
work_dir = '/content/drive/MyDrive/output'
mkdir(work_dir)

lr = 1e-3
weight_decay = 1e-3
epochs = 1
parameters=params_to_update_UF
# optimizer = Adam(params=params_to_update_UF, lr=lr, weight_decay=)
optimizer = Adam(params=herdnet.parameters(), lr=lr, weight_decay=weight_decay)


metrics = PointsMetrics(radius=20, num_classes=num_classes)

stitcher = HerdNetStitcher(
    model=herdnet,
    size=(patch_size,patch_size),
    overlap=0,
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
    evaluator=evaluator,
    work_dir=work_dir
    )

# %% [markdown]
# ## continue traing with unfreezing the freezed layers
# 

# %%
import wandb
if wandb.run is not None:
  wandb.finish()
wandb.init(project="herdnet")

# %%
trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='accuracy', wandb_flag =True)


# %%
torch.save(herdnet.state_dict(), '/content/drive/MyDrive/fine_tuned_base_herdnet.pth')

# %% [markdown]
# ## Test the model

# %%
# Path to your .pth file
import gdown

pth_path = '/content/drive/MyDrive/fine_tuned_base_herdnet.pth'

# if not pth_path:
#     gdown.download(
#         'https://drive.google.com/uc?export=download&id=1-WUnBC4BJMVkNvRqalF_HzA1_pRkQTI_',
#         '/content/20220413_herdnet_model.pth'
#         )
#     pth_path = '/content/20220413_herdnet_model.pth'

# %%
# Create output folder
from animaloc.utils.useful_funcs import mkdir
test_dir = '/content/drive/MyDrive/test_output_base_finetune'
mkdir(test_dir)

# %%
 # Load trained parameters
from animaloc.models import load_model
pth_path= '/content/drive/MyDrive/output/best_model.pth'
herdnet = load_model(herdnet, pth_path=pth_path)

# %%
# Create an Evaluator
test_evaluator = HerdNetEvaluator(
    model=herdnet,
    dataloader=test_dataloader,
    metrics=metrics,
    stitcher=stitcher,
    work_dir=test_dir,
    header='test'
    )


# %%
# Start testing
test_f1_score = test_evaluator.evaluate(returns='f1_score')

# %%
# Print global F1 score (%)
print(f"F1 score = {test_f1_score * 100:0.0f}%")

# %%
# Get the detections (Metrics)
detections = test_evaluator.results
detections.to_csv('/content//drive/MyDrive/test_output_base_finetune/metrics.csv')

# %%
# precision-recall curves
#%matplotlib inline
from animaloc.vizual import PlotPrecisionRecall
pr_curve = PlotPrecisionRecall(legend=True)
metrics = test_evaluator._stored_metrics
for c in range(1, metrics.num_classes):
    rec, pre = metrics.rec_pre_lists(c)
    pr_curve.feed(rec, pre, str(c))

pr_curve.plot()


