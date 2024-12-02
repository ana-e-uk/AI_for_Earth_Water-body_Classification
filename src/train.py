'''
Script to run the training loop
'''

'''####################################################### Imports #######################################################''' 
import numpy as np
import time
import os

import torch
print(torch.__version__)  # Displays the PyTorch version
print(torch.cuda.is_available())  # Should return False
from torch.utils.data import random_split

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report, precision_score, recall_score

from model import SpatialtemporalAutoencoder, SpatialCNN, TemporalLSTM, EmbeddingSpace, SpatialDecoder
from load_data import get_training_dataset, SEGMENTATION_SLTL_PRED
import config
'''####################################################### Functions #######################################################''' 

def spatial_mse_loss(orignial_images, reconstructed_images, num_images, image_dim):
    """
    Mean squared error function for spatial data.

    Args:
        original_images: Array of original images.
        reconstructed_images: Array of reconstructed images.
        num_images: Number of images.
        image_dim: Height and width of images.
    """
    if orignial_images.shape != reconstructed_images.shape:
        raise ValueError("The shapes of original images and reconstructed images are not the same.")
    
    squared_diff = (orignial_images - reconstructed_images)**2

    avg = np.mean(squared_diff)

    return avg

def mse_loss(input_image, target, ignored_index, reduction):
#     print(input_image.shape,target.shape)
    mask = input_image == ignored_index
    out = (input_image[~mask]-target[~mask])**2
#     print(out)
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out

def time_mse_loss(original_ts, reconstricted_ts, num_timestamps):
    """
    Mean squared error function for time series data.

    Args:
        original_ts: Array of original time series.
        reconstructed_ts: Array of reconstructed time series.
        num_timestamps: Number of timestamps.
    """
    if original_ts.shape != reconstricted_ts.shape:
        raise ValueError("The shape of the original and reconstructed time series are not the same. ")
    
    squared_diff = (original_ts - reconstricted_ts)**2

    avg = np.mean(squared_diff)

    return avg

def constrained_loss(embeddings, labels):
    """
    Constrained loss function comparing labeled embeddings.

    Args:
        embeddings: Embeddings from spatial and temporal encoders.
        labels: Class labels for each embedding.
    """
    unique_classes  = np.unique(labels)
    total_loss = 0.0
    count = 0

    for c in unique_classes:
        # Select embeddings belonging to current class
        class_indices = np.where(labels == c)[0]
        class_embeddings = embeddings[class_indices]

        # Calculate pairwise cosine similarities
        for i in range(len(class_embeddings)):
            for j in range(i + 1, len(class_embeddings)):
                e1 = class_embeddings[i]
                e2 = class_embeddings[j]

                # compute cosine similarity
                cos = torch.div(torch.matmul(e1, e2), torch.norm(e1) * torch.norm(e2) + 1e-8)     # Add small number for stability

                # Compute log of the cosine similarity
                if cos > 0:
                    total_loss += torch.log(cos)
            count += count

    avg_loss = torch.div(total_loss, count)

    return avg_loss

criterion = torch.nn.MSELoss(reduction = 'none')

'''####################################################### Element Definitions #######################################################''' 
model = SpatialtemporalAutoencoder(in_channels_spatial=config.channels, out_channels_spatial=config.channels,in_channels_temp= config.channels,out_channels_temp = config.channels)
print('Created Model')

model = model.to(config.device)
print('Sent model to device')

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

# data loader for all the data
# data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True, num_workers=0)

all_reps = []
valid_paths = []
valid_outs_s = []
valid_outs_t = []
'''####################################################### Training Loop #######################################################''' 

train_loss = []

for epoch in range(1, config.num_epochs+1):
    print("\nEpoch {}".format(epoch))
    model.train()

    train_timer = time.time()
    epoch_loss = 0
    epoch_loss_s = 0
    epoch_loss_t = 0
    epoch_loss_cl = 0
    epoch_loss_farm_log = 0
    epoch_loss_river_log = 0
    epoch_loss_stable_log = 0
    epoch_loss_seasonal_log = 0

    for batch, [image_patch_s, label_patch_s,image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(train_loader):
        if(batch % 10 == 0):
            print(f'\tBatch {batch}')

        optimizer.zero_grad()
        code_vec, out_s, out_t = model(image_patch_s.to(config.device).float(), image_patch_t.to(config.device).float())
        print('\t1')
        label_patch_s = label_patch_s.to(config.device).float()
        label_patch_t = label_patch_t.to(config.device).float()
        print('\t2')
        code_vec_farm = code_vec[label_batch == 1]
        code_vec_river = code_vec[label_batch == 3]
        code_vec_stable_lakes = code_vec[label_batch == 4]
        code_vec_mod_seas_lakes = code_vec[label_batch == 5]
        print('\t3')
        min_class_labels = np.amin([code_vec_farm.shape[0],code_vec_river.shape[0],code_vec_stable_lakes.shape[0],code_vec_mod_seas_lakes.shape[0]])
        # print(min_class_labels)
        print('\t4')
        if(code_vec_farm.shape[0] > 2):
            farm_batch_loss_log = constrained_loss(code_vec_farm,min_class_labels)
        else:
            farm_batch_loss_log = torch.zeros(1, requires_grad=True)
        print('\t5')
        if(code_vec_river.shape[0] > 2):
            river_batch_loss_log = constrained_loss(code_vec_river,min_class_labels)
        else:
            river_batch_loss_log = torch.zeros(1, requires_grad=True)
        print('\t6')
        if(code_vec_stable_lakes.shape[0] > 2):
            stable_lakes_batch_loss_log = constrained_loss(code_vec_stable_lakes,min_class_labels)
        else:
            stable_lakes_batch_loss_log = torch.zeros(1, requires_grad=True)
        print('\t7')
        if(code_vec_mod_seas_lakes.shape[0] > 2):
            mod_seas_lakes_batch_loss_log = constrained_loss(code_vec_mod_seas_lakes,min_class_labels)
        else:
            mod_seas_lakes_batch_loss_log = torch.zeros(1, requires_grad=True)
        print('\tStarting spatial loss')
        batch_loss_s = torch.mean(torch.sum(mse_loss(input_image = out_s, target = label_patch_s,ignored_index = config.ignore_index,reduction = 'None')))
        print('\tEnded spatial loss')
        print('\tStarted time loss')
        batch_loss_t = torch.mean(torch.sum(criterion(out_t, label_patch_t),dim=[1]))
        print('\tEnded time loss')

        print(f'\tBatch loss: {batch_loss_s}')
        
        if epoch < config.num_epochs :
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
        epoch_loss_stable_log += stable_lakes_batch_loss_log.item()
        epoch_loss_seasonal_log += mod_seas_lakes_batch_loss_log.item()

    epoch_loss = epoch_loss/(batch+1)
    epoch_loss_s = epoch_loss_s/(batch+1)
    epoch_loss_t = epoch_loss_t/(batch+1)
    epoch_loss_farm_log = epoch_loss_farm_log/(batch+1)
    epoch_loss_river_log = epoch_loss_river_log/(batch+1)
    epoch_loss_stable_log = epoch_loss_stable_log/(batch+1)
    epoch_loss_seasonal_log = epoch_loss_seasonal_log/(batch+1)

    print(epoch_loss)
#     print(epoch_loss,epoch_loss_s,epoch_loss_t,epoch_loss_farm_log,epoch_loss_river_log,epoch_loss_stable_lakes_log,epoch_loss_mod_seas_lakes_log,epoch_loss_eph_lakes_log)
    print('\n')

    model.eval()
    model_weights = os.path.join(config.model_dir, str(config.experiment_id) + "_epoch_" + str(epoch) + ".pt")
    torch.save(model.state_dict(), model_weights)

# '''####################################################### Testing Loop #######################################################''' 

# # Load the saved model
# model_path = model_weights
# model.load_state_dict(torch.load(model_path))
# model.eval()  # Set the model to evaluation mode

# loss = 0
# preds = []
# labels = []
# IDs_all = []

# for batch, [image_patch_s, label_patch_s, image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(test_loader):
    
#     optimizer.zero_grad()

#     code_vec, out = model(image_patch_s.to(config.device).float(), image_patch_t.to(config.device).float())

#     label_batch = label_batch.type(torch.long).to(config.device)
#     batch_loss = criterion(out, label_batch)
#     loss += batch_loss.item()

#     out_label_batch = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
#     out_label_batch_cpu = out_label_batch.detach().cpu().numpy()
#     label_batch_cpu = label_batch.detach().cpu().numpy()
        
#     preds.append(out_label_batch_cpu)
#     labels.append(label_batch_cpu)

#     del out
#     del code_vec

# loss = loss/(batch+1)
# print('Test Loss:{} '.format(loss), end="\n")
# print("\n")

# pred_array = np.concatenate(preds, axis=0)
# label_array = np.concatenate(labels, axis=0)

# print(pred_array.shape)
# print(label_array.shape)

# # print(classification_report(label_array, pred_array, digits=4))
# # print(f1_score(y_true=label_array, y_pred=pred_array, average='macro'))

# print(classification_report(label_array, pred_array, digits=4))
# print("\nCONFUSION MATRIX:")
# confusion_matrix_set = confusion_matrix(label_array,pred_array)
# print(confusion_matrix_set)
# f1_score_array = f1_score(label_array, pred_array, average = None)
# precision_score_array = precision_score(label_array, pred_array, average = None)
# recall_score_array = recall_score(label_array, pred_array, average = None)
# # f1_score_array = f1_score(label_array, pred_array, labels = config.labels_for_cl, average = None)
# # precision_score_array = precision_score(label_array, pred_array, labels = config.labels_for_cl, average = None)
# # recall_score_array = recall_score(label_array, pred_array, labels = config.labels_for_cl, average = None)
# for r in range(f1_score_array.shape[0]):
#     print(f1_score_array[r],end = ' ')
# print('')
# for r in range(precision_score_array.shape[0]):
#     print(precision_score_array[r],end = ' ')
# print('')
# for r in range(recall_score_array.shape[0]):
#     print(recall_score_array[r],end = ' ')
# print('')

