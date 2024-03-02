import torch
import random
import sys
import time
import cv2
import matplotlib
import os
import pickle
import platform

import nibabel as nib
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms as transforms
import statsmodels.api as sm
import matplotlib.pyplot as plt
import albumentations as A

from torch import nn
from torch.utils.data import DataLoader
from sklearn import preprocessing  # pip install scikit-learn

from Training.dataset import MyDataset

from parameters import MetaParameters
from Preprocessing.dirs_logs import create_dir, create_dir_log, log_stats
from Model.unet2D import UNet_2D, UNet_2D_AttantionLayer


########################################################################################################################
# Show software and harware
########################################################################################################################
print(f"Python Platform: {platform.platform()}")
print(f'python version: {sys.version}')
print(f'torch version: {torch.__version__}')
print(f'numpy version: {np.__version__}')
print(f'pandas version: {pd.__version__}')


global device


def device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


device = device()
print(device)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight = None, gamma = 2,reduction = 'mean'):    #reduction='sum'
        super(FocalLoss, self).__init__(weight,reduction = reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

########################################################################################################################
# COMMENTS
########################################################################################################################
meta = MetaParameters()


create_dir_log(meta.UNET1_PROJ_NAME)
projec_name = meta.UNET1_PROJ_NAME

if meta.PRETRAIN:    
    try:
        model = torch.load(f'{projec_name}/{meta.MODEL_NAME}.pth').to(device=device)
        model.eval()
        print(f'model loaded: {projec_name}/{meta.MODEL_NAME}.pth')
    except:
        print('no trained models')
        model = UNet_2D_AttantionLayer().to(device=device)
else:
    # model = UNet_2D_AttantionLayer().to(device=device)
    model = UNet_2D().to(device=device)


if meta.FREEZE_BN is True:
    for name, child in model.named_children(): 
        if name in ['decoder2', 'decoder1', 'upconv1', 'upconv2', 'conv', 'Att2', 'Att1']: 
            print(name + ' has been unfrozen.') 
            
            for param in child.parameters(): 
                param.requires_grad = True 
        else: 
            for param in child.parameters(): 
                param.requires_grad = False

# loss_function = nn.CrossEntropyLoss(weight=meta.CE_WEIGHTS).to(device)
loss_function = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=meta.LR, weight_decay=meta.WDC)

scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=meta.TMAX, eta_min=0, last_epoch=-1, verbose=True)


########################################################################################################################
## Main image transforms in Dataloder
########################################################################################################################
default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

transform_01 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation((-10, 10), expand=False),
    transforms.RandomHorizontalFlip(0.7),
    transforms.RandomVerticalFlip(0.7),
    transforms.ToTensor(),
])

transform_04 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomAffine(degrees=(-2, 2), translate=(0.05, 0.25), scale=(0.75, 1.25)),
    transforms.ToTensor(),
])

def aug_transforms():
    return [
        A.ElasticTransform(alpha=20, sigma=50, alpha_affine=8,
                           interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=None,
                           mask_value=None, always_apply=False, approximate=False, p=1),
    ]

transform_05 = A.Compose(A.ElasticTransform(alpha=20, sigma=50, alpha_affine=8,
                           interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=None,
                           mask_value=None, always_apply=False, approximate=False, p=1))

transform_06 = A.Compose(A.GridDistortion(num_steps=10, distort_limit=0.05, interpolation=cv2.INTER_NEAREST,
                            border_mode=cv2.BORDER_CONSTANT, value=None, mask_value=None,
                            always_apply=False, p=1))



