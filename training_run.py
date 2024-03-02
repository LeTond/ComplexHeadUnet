 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1
Date: 02-04-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""

from parameters import *
from configuration import *
from Training.train import *
from Training.dataset import *
from Preprocessing.split_dataset import *


########################################################################################################################
# Creating loaders for training and validating network
########################################################################################################################
train_ds = GetData(train_list, meta.AUGMENTATION).generated_data_list()
valid_ds = GetData(valid_list, False).generated_data_list()

train_ds_origin = train_ds[0]
train_ds_mask = train_ds[1]
train_ds_names = train_ds[2]

valid_ds_origin = valid_ds[0]
valid_ds_mask = valid_ds[1]
valid_ds_names = valid_ds[2]

kernel_sz = meta.KERNEL

train_set = MyDataset(meta.NUM_CLASS, train_ds_origin, train_ds_mask, train_ds_names, kernel_sz, default_transform)
for i in range(3):
    train_set += MyDataset(meta.NUM_CLASS, train_ds_origin, train_ds_mask, train_ds_names, kernel_sz, transform_04)
    # train_set += MyDataset(meta.NUM_CLASS, train_ds_origin, train_ds_mask, train_ds_names, kernel_sz, transform_01)

train_loader = DataLoader(train_set, meta.BT_SZ, drop_last=True, shuffle=True, pin_memory=False)

valid_set = MyDataset(meta.NUM_CLASS, valid_ds_origin, valid_ds_mask, valid_ds_names, kernel_sz, default_transform)

valid_batch_size = len(valid_set)
valid_loader = DataLoader(valid_set, meta.BT_SZ, drop_last=True, shuffle=True, pin_memory=False)


print(f'Train size: {len(train_set)} | Valid size: {len(valid_set)}')
model = TrainNetwork(model, optimizer, loss_function, train_loader, valid_loader, meta, ds).train()




