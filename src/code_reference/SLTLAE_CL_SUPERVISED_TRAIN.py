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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report, precision_score, recall_score
import time
import glob
from sklearn.cluster import KMeans
import MODEL
import DATA_CODE.DATA_LOADER as DATA_LOADER

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

print("#######################################################################")
print("BUILD MODEL")       
model = MODEL.SpatialtemporalAutoencoder_supervised(in_channels_spatial=config.channels, out_channels_spatial=config.channels,in_channels_temp= config.channels,out_channels_temp = config.channels)
model= model.to('cuda')

for name, param in model.named_parameters():
    print(name,param.requires_grad)

if(config.load_model == 1):
    print("LOADING MODEL")
    print(config.load_MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(config.load_MODEL_DIR, config.load_MODEL_NAME+".pt")),strict = False)

if(config.freeze_layers == 1):
    for name, param in model.named_parameters():
        if(name not in config.not_freeze_list):
            param.requires_grad = False
            
for name, param in model.named_parameters():
    print(name,param.requires_grad)
    

criterion = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# %%
print("#######################################################################")
print("TRAIN MODEL")
train_loss = []

image_patches_spatial_list = []
label_patches_spatial_list = []
image_patches_temp_list = []
label_patches_temp_list = []
label_IDs_list = []
IDs_list = []

# paths_list = glob.glob(os.path.join(config.PAD_DIR,config.PAD_FOLDER, '*.npy'))
paths_list = glob.glob(os.path.join(config.WARP_DIR,config.WARP_FOLDER, '*.npy'))

conti_path_list = []

if(config.sub_conti == 1):
    continent_info = np.load('/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/continent_info.npy')
    for path in paths_list:
        ID = path.split('/')[-1].split('_')[-4]
        if(continent_info[int(ID)] == config.continent_no):
            conti_path_list.append(path)

if(config.sub_conti == 1):
    paths_list = conti_path_list

# paths_list = paths_list[random.sample(range(len(paths_list)), len(paths_list))]
print(len(paths_list))

ID_count_array = np.zeros((1000_000)).astype(np.int64)
path_dict = {}
for i,path in enumerate(paths_list):
#     print(path)
    ID = int(path.split('/')[-1].split('_')[1])
#     print(ID)
    ID_count_array[int(ID)] += 1
    path_dict[ID] = path

all_label_array = np.load('/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/all_IDs_labels.npy')
print('Updated All label array  Shape:',all_label_array.shape, ' Count:',np.bincount(all_label_array))   

# train_dataset = np.load(os.path.join(config.DATASET_DIR,config.DATASET_NAME_TRAIN_4CL_OVERSAMPLED))
# train_dataset = np.load(os.path.join(config.DATASET_DIR,config.DATASET_NAME_TRAIN_4CL_UNDERSAMPLED))
train_dataset = np.load(os.path.join(config.DATASET_DIR,config.DATASET_NAME_TRAIN_4CL_SAMESAMPLED))


print('TRAIN DATASET LENGTH:', train_dataset.shape)


for ID_no in train_dataset:
    if(all_label_array[int(ID_no)] != 0 and all_label_array[int(ID_no)] != 2 and all_label_array[int(ID_no)] != 6 and all_label_array[int(ID_no)] != 7 and all_label_array[int(ID_no)] != 8 and all_label_array[int(ID_no)] != 0):

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
# print(count_zero)
label_IDs_list_updated = []
for label_value in label_IDs_list:
    label_IDs_list_updated.append(config.label_dict[label_value])
                                  
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list)))
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list_updated)))

data = DATA_LOADER.SEGMENTATION_SLTL_PRED(image_patches_spatial_list,label_patches_spatial_list,image_patches_temp_list,label_patches_temp_list,label_IDs_list_updated,IDs_list)
data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size_labelled_only, shuffle=True, num_workers=0)

all_reps = []
valid_paths = []
valid_outs_s = []
valid_outs_t = []


for epoch in range(1,config.no_epochs_train+1):
    print('## EPOCH {} ##'.format(epoch))
    model.train()

    train_time_start = time.time()
    epoch_loss = 0
    
    for batch, [image_patch_s, label_patch_s,image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(data_loader):
        
        optimizer.zero_grad()
        code_vec, out = model(image_patch_s.to('cuda').float(), image_patch_t.to('cuda').float())
    
        label_batch = label_batch.type(torch.long).to('cuda')

        batch_loss = criterion(out, label_batch)

        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        
    epoch_loss = epoch_loss/(batch+1)
    print(epoch_loss)
    print('\n')  
    train_loss.append(epoch_loss)

    model.eval()
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, str(config.experiment_id) + "_epoch_" + str(epoch) + "_train_loss_" + str("{:.4f}".format(train_loss[-1]))+".pt"))
    
    
    
test_dataset = np.load(os.path.join(config.DATASET_DIR,config.DATASET_NAME_TEST))
print('TEST DATASET LENGTH:', test_dataset.shape)

image_patches_spatial_list_test = []
label_patches_spatial_list_test = []
image_patches_temp_list_test = []
label_patches_temp_list_test = []
label_IDs_list_test = []
IDs_list_test = []

for ID_no in test_dataset:
    # path = path_dict[int(ID_no)]
    if(all_label_array[int(ID_no)] != 0 and all_label_array[int(ID_no)] != 2 and all_label_array[int(ID_no)] != 6 and all_label_array[int(ID_no)] != 7 and all_label_array[int(ID_no)] != 8 and all_label_array[int(ID_no)] != 0):
        image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp, label_ID,ID = DATA_LOADER.get_data_spatial_temporal_label_ID_load_arrays(ID_no,all_label_array)
    #     print('Shapes', image_patches.shape,label_patches.shape,len(image_patches_list))

        if(len(image_patches_spatial_list_test) == config.no_train_images):
                break

        if(len(image_patches_spatial_list_test) % 1000 == 0):
                print(len(image_patches_spatial_list_test))

        image_patches_spatial_list_test.append(image_patches_spatial)
        label_patches_spatial_list_test.append(label_patches_spatial)
        image_patches_temp_list_test.append(image_patches_temp)
        label_patches_temp_list_test.append(label_patches_temp)
        label_IDs_list_test.append(label_ID)
        IDs_list_test.append(ID)

print(len(image_patches_spatial_list_test))
# print(np.bincount(np.array(label_IDs_list_test)))
# print(count_zero)
label_IDs_list_updated_test = []
for label_value in label_IDs_list_test:
    label_IDs_list_updated_test.append(config.label_dict[label_value])
                                  
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list_test)))
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list_updated_test)))
                                  
# data = DATA_LOADER.SEGMENTATION_PRED(image_patches_list, label_patches_list, label_IDs_list, IDs_list)
data_test = DATA_LOADER.SEGMENTATION_SLTL_PRED(image_patches_spatial_list_test, label_patches_spatial_list_test, image_patches_temp_list_test, label_patches_temp_list_test, label_IDs_list_updated_test, IDs_list_test)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=config.batch_size_labelled_only, shuffle=False, num_workers=0)

loss = 0
preds = []
labels = []
IDs_all = []

for batch, [image_patch_s, label_patch_s, image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(data_loader_test):
    
    optimizer.zero_grad()

    code_vec, out = model(image_patch_s.to('cuda').float(), image_patch_t.to('cuda').float())

    label_batch = label_batch.type(torch.long).to('cuda')
    batch_loss = criterion(out, label_batch)
    loss += batch_loss.item()

    out_label_batch = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
    out_label_batch_cpu = out_label_batch.detach().cpu().numpy()
    label_batch_cpu = label_batch.detach().cpu().numpy()
        
    preds.append(out_label_batch_cpu)
    labels.append(label_batch_cpu)

    del out
    del code_vec

loss = loss/(batch+1)
print('Test Loss:{} '.format(loss), end="\n")
print("\n")

pred_array = np.concatenate(preds, axis=0)
label_array = np.concatenate(labels, axis=0)

print(pred_array.shape)
print(label_array.shape)

# print(classification_report(label_array, pred_array, digits=4))
# print(f1_score(y_true=label_array, y_pred=pred_array, average='macro'))

print(classification_report(label_array, pred_array, digits=4))
print("\nCONFUSION MATRIX:")
confusion_matrix_set = confusion_matrix(label_array,pred_array)
print(confusion_matrix_set)
f1_score_array = f1_score(label_array, pred_array, average = None)
precision_score_array = precision_score(label_array, pred_array, average = None)
recall_score_array = recall_score(label_array, pred_array, average = None)
# f1_score_array = f1_score(label_array, pred_array, labels = config.labels_for_cl, average = None)
# precision_score_array = precision_score(label_array, pred_array, labels = config.labels_for_cl, average = None)
# recall_score_array = recall_score(label_array, pred_array, labels = config.labels_for_cl, average = None)
for r in range(f1_score_array.shape[0]):
    print(f1_score_array[r],end = ' ')
print('')
for r in range(precision_score_array.shape[0]):
    print(precision_score_array[r],end = ' ')
print('')
for r in range(recall_score_array.shape[0]):
    print(recall_score_array[r],end = ' ')
print('')