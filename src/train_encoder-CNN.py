'''
Script to run the training loop for the encoder-CNN
'''

'''####################################################### Imports #######################################################''' 
import numpy as np
import time
import os

import torch
from torch.utils.data import random_split
from scipy.special import logsumexp

from model import EncoderCNN
from load_data import get_training_dataset,get_testing_dataset, SEGMENTATION_SLTL_PRED
import config

'''####################################################### Functions #######################################################''' 

criterion = torch.nn.CrossEntropyLoss()

'''####################################################### Element Definitions #######################################################''' 
model = EncoderCNN(in_channels_spatial=config.channels, in_channels_temp= config.channels)
print('Created Model')

model = model.to(config.device)
print('Sent model to device')

for name, param in model.named_parameters():
    print(name,param.requires_grad)

if(config.load_model == 1):
    print("LOADING MODEL")
    print(config.load_MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(config.load_MODEL_DIR, config.load_MODEL_NAME+".pt")),strict = False)

# if(config.freeze_layers == 1):
#     for name, param in model.named_parameters():
#         if(name not in config.not_freeze_list):
#             param.requires_grad = False
            
for name, param in model.named_parameters():
    print(name,param.requires_grad)
    

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
print('Created optimizer')

label_info_array = np.load(config.label_info_filepath)
print('Uploaded label array')

image_patches_spatial_list, label_patches_spatial_list, image_patches_temp_list, label_patches_temp_list, label_IDs_list, IDs_list = get_training_dataset()
print('Getting training dataset')

data = SEGMENTATION_SLTL_PRED(image_patches_spatial_list,label_patches_spatial_list,image_patches_temp_list,label_patches_temp_list,label_IDs_list,IDs_list)

# Determine split sizes
dataset_size = len(data)
train_size = int(0.8 * dataset_size)  # 80% for training
test_size = dataset_size - train_size

# Split the dataset
train_dataset, test_dataset = random_split(data, [train_size, test_size])

print('Call data loaders')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

all_reps = []
valid_paths = []
valid_outs_s = []
valid_outs_t = []
'''####################################################### Encoder-CNN Training Loop #######################################################''' 
epoch_losses = np.array([])
train_loss = []

for epoch in range(1, config.num_epochs+1):
    model.train()

    train_timer = time.time()
    epoch_loss = 0

    for batch, [image_patch_s, label_patch_s,image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(train_loader):
        
        optimizer.zero_grad()
        code_vec, out = model(image_patch_s.to(config.device).float(), image_patch_t.to(config.device).float())

        label_batch = label_batch.type(torch.long).to(config.device)
        
        batch_loss = criterion(out, label_batch)
 
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

    epoch_loss = epoch_loss/(batch+1)

    model.eval()
    if epoch % 100 == 0:
        print(f'\nEpoch {epoch} loss: {epoch_loss}')
        model_weights = os.path.join(config.model_dir_e_CNN, str(config.experiment_id_e_CNN) + "_epoch_" + str(epoch) + ".pt")
        torch.save(model.state_dict(), model_weights)
    epoch_losses = np.append(epoch_losses, epoch_loss)

np.save('epoch_losses_e_CNN.npy', epoch_losses)

'''####################################################### Encoder-CNN Testing Code  #######################################################''' 

def testing_e_CNN(test_data_loader, optim):
    loss = 0
    preds = []
    labels = []

    for batch, [image_patch_s, label_patch_s, image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(test_data_loader):
        
        optim.zero_grad()
        code_vec, out = model(image_patch_s.to(config.device).float(), image_patch_t.to(config.device).float())

        label_batch = label_batch.type(torch.long).to(config.device)
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

    return pred_array, label_array


'''####################################################### Encoder-CNN Testing Loop Region 1 data #######################################################''' 

pred_array, label_array = testing_e_CNN(test_data_loader=test_loader, optim=optimizer)

print('Pred array shape', pred_array.shape)
print('Label array shape', label_array.shape)

np.save('pred_array_e_CNN.npy', pred_array)
np.save('label_array_e_CCNN.npy', label_array)



'''####################################################### Encoder-CNN Testing Loop Region 2 data #######################################################''' 
image_patches_spatial_list, label_patches_spatial_list, image_patches_temp_list, label_patches_temp_list, label_IDs_list, IDs_list = get_testing_dataset()
print('Getting testing dataset from unseen region')

unseen_data = SEGMENTATION_SLTL_PRED(image_patches_spatial_list,label_patches_spatial_list,image_patches_temp_list,label_patches_temp_list,label_IDs_list,IDs_list)
unseen_test_loader = torch.utils.data.DataLoader(dataset=unseen_data, batch_size=config.batch_size, shuffle=False, num_workers=0)

u_pred_array, u_label_array = testing_e_CNN(test_data_loader=test_loader, optim=optimizer)

np.save('pred_array_e_CNN_unseen_data.npy', u_pred_array)
np.save('label_array_e_CCNN_unseen_data.npy', u_label_array)