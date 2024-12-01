#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
import config
import os
import numpy as np
import random

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import time
import glob
from sklearn.cluster import KMeans
import MODEL
import DATA_LOADER as DATA_LOADER
print("hello world")
if not os.path.exists(os.path.join(config.MODEL_DIR)):
    os.makedirs(os.path.join(config.MODEL_DIR))
print(config.MODEL_DIR)

def mse_loss(input_image, target, ignored_index, reduction):
#     print(input_image.shape,target.shape)
    mask = input_image == ignored_index
    out = (input_image[~mask]-target[~mask])**2
#     print(out)
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out

def cl_criterion(reps,no_max_reps):
    no_of_reps = reps.shape[0]
    torch_arr = torch.randint(0, no_of_reps, (2*no_max_reps,))
    torch_arr_2 = torch.randint(0, no_of_reps, (2*no_max_reps,))
    total_log_sum = 0
    count_rep = 0
    
    for i in range(torch_arr.shape[0]): 
        if(torch_arr[i] == torch_arr_2[i]):
            continue
        mag1 = torch.sqrt(torch.sum(torch.square(reps[torch_arr[i]])))
        mag2 = torch.sqrt(torch.sum(torch.square(reps[torch_arr_2[i]])))
        prod12 = mag1*mag2
        dot12 = torch.abs(torch.dot(reps[torch_arr[i]],reps[torch_arr_2[i]]))
        cos12 = torch.div(dot12,prod12)
        log12 = -torch.log(cos12)
        if(torch.isnan(log12)):
            print(log12,mag1,mag2,prod12,dot12,cos12)
    
        total_log_sum += log12
        count_rep += 1

    if(count_rep == 0):
        count_rep = 1
        
    avg_log = torch.div(total_log_sum,count_rep)

    return avg_log

        
model = MODEL.SpatialtemporalAutoencoder(in_channels_spatial=config.channels, out_channels_spatial=config.channels,in_channels_temp= config.channels,out_channels_temp = config.channels)
model= model.to(config.device)
# model.load_state_dict(torch.load(os.path.join(config.ae_model_dir, config.ae_MODEL_NAME+".pt")),strict = False)
# model.eval()

image_patches_spatial_list = []
label_patches_spatial_list = []
image_patches_temp_list = []
label_patches_temp_list = []
label_IDs_list = []
IDs_list = []
# OLD PATH
# all_label_array = np.load('/home/kumarv/pravirat/Realsat_labelling/all_IDs_labels.npy')
# NEW PATH
all_label_array = np.load('/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/all_IDs_labels_realsat.npy')

print('Updated All label array  Shape:',all_label_array.shape, ' Count:',np.bincount(all_label_array))   

train_dataset = np.load(os.path.join(config.DATASET_DIR,config.DATASET_NAME_TRAIN))
# test_dataset = np.load(os.path.join(config.DATASET_DIR,config.DATASET_NAME_TEST))
print('TRAIN DATASET LENGTH:', train_dataset.shape)


for ID_no in train_dataset:

    image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp, label_ID,ID = DATA_LOADER.get_data_spatial_temporal_label_ID_load_arrays(ID_no,all_label_array)
    
    if(len(image_patches_spatial_list) == config.no_train_images):
            break
            
    if(len(image_patches_spatial_list) % 1000 == 0):
            print(len(image_patches_spatial_list))
            
    image_patches_spatial_list.append(image_patches_spatial)
    label_patches_spatial_list.append(label_patches_spatial)    
    image_patches_temp_list.append(image_patches_temp)
    label_patches_temp_list.append(label_patches_temp)
    label_IDs_list.append(label_ID)
    IDs_list.append(ID)

print(len(image_patches_spatial_list))
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list)))
# print(count_zero)

data = DATA_LOADER.SEGMENTATION_SLTL_PRED(image_patches_spatial_list,label_patches_spatial_list,image_patches_temp_list,label_patches_temp_list,label_IDs_list,IDs_list)
data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True, num_workers=0)

all_reps = []
valid_paths = []
valid_outs_s = []
valid_outs_t = []

criterion = torch.nn.MSELoss(reduction = 'none')
# criterion = torch.nn.MSELoss()
# criterion = mse_loss()

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# %%
print("#######################################################################")
print("TRAIN MODEL")
train_loss = []

for epoch in range(1,config.n_epochs+1):
    print('## EPOCH {} ##'.format(epoch))
    model.train()

    train_time_start = time.time()
    epoch_loss = 0
    epoch_loss_s = 0
    epoch_loss_t = 0
    epoch_loss_kl = 0
    epoch_loss_cl = 0
    epoch_loss_farm_log = 0
    epoch_loss_river_log = 0
    epoch_loss_stable_lakes_log = 0
    epoch_loss_mod_seas_lakes_log = 0
    epoch_loss_eph_lakes_log = 0
    
    for batch, [image_patch_s, label_patch_s,image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(data_loader):
        
        optimizer.zero_grad()
        code_vec, out_s, out_t = model(image_patch_s.to(config.device).float(), image_patch_t.to(config.device).float())
    
        label_patch_s = label_patch_s.to(config.device).float()
        label_patch_t = label_patch_t.to(config.device).float()

        code_vec_farm = code_vec[label_batch == 1]
        code_vec_river = code_vec[label_batch == 3]
        code_vec_stable_lakes = code_vec[label_batch == 4]
        code_vec_mod_seas_lakes = code_vec[label_batch == 5]
        code_vec_eph_lakes = code_vec[label_batch == 7]
        min_class_labels = np.amin([code_vec_farm.shape[0],code_vec_river.shape[0],code_vec_stable_lakes.shape[0],code_vec_mod_seas_lakes.shape[0]])
        # print(min_class_labels)
        
        if(code_vec_farm.shape[0] > 2):
            farm_batch_loss_log = cl_criterion(code_vec_farm,min_class_labels)
        else:
            farm_batch_loss_log = torch.zeros(1, requires_grad=True)
            
        if(code_vec_river.shape[0] > 2):
            river_batch_loss_log = cl_criterion(code_vec_river,min_class_labels)
        else:
            river_batch_loss_log = torch.zeros(1, requires_grad=True)
            
        if(code_vec_stable_lakes.shape[0] > 2):
            stable_lakes_batch_loss_log = cl_criterion(code_vec_stable_lakes,min_class_labels)
        else:
            stable_lakes_batch_loss_log = torch.zeros(1, requires_grad=True)
            
        if(code_vec_mod_seas_lakes.shape[0] > 2):
            mod_seas_lakes_batch_loss_log = cl_criterion(code_vec_mod_seas_lakes,min_class_labels)
        else:
            mod_seas_lakes_batch_loss_log = torch.zeros(1, requires_grad=True)
            
        if(code_vec_eph_lakes.shape[0] > 2):
            eph_lakes_batch_loss_log = cl_criterion(code_vec_eph_lakes,min_class_labels)
        else:
            eph_lakes_batch_loss_log = torch.zeros(1, requires_grad=True)
        
        
        batch_loss_s = torch.mean(torch.sum(mse_loss(input_image = out_s, target = label_patch_s,ignored_index = config.ignore_index,reduction = 'None')))
        batch_loss_t = torch.mean(torch.sum(criterion(out_t, label_patch_t),dim=[1]))

        print(batch_loss_s)
        
        if epoch < 1000 :
            batch_loss = batch_loss_s * 0.01 + batch_loss_t
        else:
            batch_loss = batch_loss_s * 0.01 + batch_loss_t + (river_batch_loss_log + farm_batch_loss_log + stable_lakes_batch_loss_log + mod_seas_lakes_batch_loss_log) * 0.25


        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        epoch_loss_s += batch_loss_s.item()
        epoch_loss_t += batch_loss_t.item()
        epoch_loss_farm_log += farm_batch_loss_log.item()
        epoch_loss_river_log += river_batch_loss_log.item()
        epoch_loss_stable_lakes_log += stable_lakes_batch_loss_log.item()
        epoch_loss_mod_seas_lakes_log += mod_seas_lakes_batch_loss_log.item()
        epoch_loss_eph_lakes_log += eph_lakes_batch_loss_log.item()

    epoch_loss = epoch_loss/(batch+1)
    epoch_loss_s = epoch_loss_s/(batch+1)
    epoch_loss_t = epoch_loss_t/(batch+1)
    epoch_loss_farm_log = epoch_loss_farm_log/(batch+1)
    epoch_loss_river_log = epoch_loss_river_log/(batch+1)
    epoch_loss_stable_lakes_log = epoch_loss_stable_lakes_log/(batch+1)
    epoch_loss_mod_seas_lakes_log = epoch_loss_mod_seas_lakes_log/(batch+1)
    epoch_loss_eph_lakes_log = epoch_loss_eph_lakes_log/(batch+1)
    print(epoch_loss)
#     print(epoch_loss,epoch_loss_s,epoch_loss_t,epoch_loss_farm_log,epoch_loss_river_log,epoch_loss_stable_lakes_log,epoch_loss_mod_seas_lakes_log,epoch_loss_eph_lakes_log)
    print('\n')

    model.eval()
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, str(config.experiment_id) + "_epoch_" + str(epoch) + ".pt"))