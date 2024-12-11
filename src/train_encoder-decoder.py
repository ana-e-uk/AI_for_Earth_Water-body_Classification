'''
Script to run the training loop for the encoder-decoder
'''

'''####################################################### Imports #######################################################''' 
import numpy as np
import time
import os
<<<<<<< HEAD
import math
=======
import matplotlib.pyplot as plt



>>>>>>> 37587ba (Reconstruction visualization)

import torch
print(torch.__version__)  # Displays the PyTorch version
print(torch.cuda.is_available())  # Should return False
from torch.utils.data import random_split

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report, precision_score, recall_score

from model import EncoderDecoder, SpatialCNN, TemporalLSTM, EmbeddingSpace, SpatialDecoder
from load_data import get_training_dataset, SEGMENTATION_SLTL_PRED
import config
'''####################################################### Functions #######################################################''' 
def get_magnitude(num):
    if num == 0:
        return 0  # Special case for zero
    return int(math.log10(abs(num)))

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
                log1 = -torch.log(cos) 
                # Compute log of the cosine similarity
                if cos > 0:
                    if(torch.isnan(log1)):
                        print(log1,e1,e2,cos)
                    total_loss += (log1)    
            count += count

    avg_loss = torch.div(total_loss, count)

    return avg_loss



def save_spatial_patches(true_patches, reconstructed_patches, epoch, batch_index, save_dir="spatial_patches"):
    """
    Save spatial reconstructions for a batch with epoch and patch information.
    
    Args:
        true_patches: Ground truth spatial data.
        reconstructed_patches: Reconstructed spatial data.
        epoch: Current epoch number.
        batch_index: Index of the batch being processed.
        save_dir: Directory to save the spatial reconstructions.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(true_patches.shape[0]):  # Loop over each patch
        true_img = true_patches[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC
        recon_img = reconstructed_patches[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("True Patch")
        plt.imshow(true_img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Patch")
        plt.imshow(recon_img)
        plt.axis("off")

        # Save file with epoch, batch, and patch index
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_index}_patch_{i}.png"))
        plt.close()


def save_temporal_patches(true_ts, reconstructed_ts, epoch, batch_index, save_dir="temporal_patches"):
    """
    Save temporal reconstructions for a batch with epoch and patch information.
    
    Args:
        true_ts: Ground truth temporal data.
        reconstructed_ts: Reconstructed temporal data.
        epoch: Current epoch number.
        batch_index: Index of the batch being processed.
        save_dir: Directory to save the temporal reconstructions.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(true_ts.shape[0]):  # Loop over each time-series
        true_series = true_ts[i].detach().cpu().numpy()
        recon_series = reconstructed_ts[i].detach().cpu().numpy()

        plt.figure(figsize=(8, 4))
        plt.plot(true_series, label="True", linestyle="--", alpha=0.7)
        plt.plot(recon_series, label="Reconstructed", alpha=0.7)
        plt.title(f"Temporal Patch {i} in Epoch {epoch}, Batch {batch_index}")
        plt.legend()

        # Save file with epoch, batch, and patch index
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_index}_patch_{i}.png"))
        plt.close()


def save_epoch_reconstructions(model, data_loader, epoch, save_dir="epoch_reconstructions", selected_batches=[0, 5, 10], num_patches=10):
    """
    Save spatial and temporal reconstructions for specific batches and samples across epochs.
    
    Args:
        model: The trained model.
        data_loader: DataLoader to fetch data.
        epoch: Current epoch number.
        save_dir: Directory to save the reconstructions.
        selected_batches: List of batch indices to save.
        num_patches: Number of patches per batch to save.
    """
    model.eval()  # Set the model to evaluation mode
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (image_patch_s, label_patch_s, image_patch_t, label_patch_t, _, _) in enumerate(data_loader):
            if batch_idx in selected_batches:
                # Forward pass
                image_patch_s = image_patch_s.to(config.device).float()
                image_patch_t = image_patch_t.to(config.device).float()
                label_patch_s = label_patch_s.to(config.device).float()
                label_patch_t = label_patch_t.to(config.device).float()

                _, out_s, out_t = model(image_patch_s, image_patch_t)

                # Limit to the specified number of patches
                selected_true_s = label_patch_s[:num_patches].cpu()
                selected_recon_s = out_s[:num_patches].cpu()
                selected_true_t = label_patch_t[:num_patches].cpu()
                selected_recon_t = out_t[:num_patches].cpu()

                # Define directories
                spatial_save_dir = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}_spatial")
                temporal_save_dir = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}_temporal")

                # Save patches
                save_spatial_patches(selected_true_s, selected_recon_s, epoch, batch_idx, save_dir=spatial_save_dir)
                save_temporal_patches(selected_true_t, selected_recon_t, epoch, batch_idx, save_dir=temporal_save_dir)


criterion = torch.nn.MSELoss(reduction = 'none')

'''####################################################### Element Definitions #######################################################''' 
model = EncoderDecoder(in_channels_spatial=config.channels, out_channels_spatial=config.channels,in_channels_temp= config.channels,out_channels_temp = config.channels)
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

epoch_losses = np.array([])
'''####################################################### Encoder-Decoder Training Loop #######################################################''' 

train_loss = []

# for epoch in range(1, config.num_epochs + 1):
#     print("\nEpoch {}".format(epoch), flush=True)
#     model.train()

#     train_timer = time.time()
#     epoch_loss = 0
#     epoch_loss_s = 0
#     epoch_loss_t = 0

#     for batch, [image_patch_s, label_patch_s, image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(train_loader):
#         # Print for selected batches (e.g., every 10th batch)
#         if batch % 10 == 0:
#             print(f'\tProcessing Batch {batch}', flush=True)

#         # Forward pass
#         code_vec, out_s, out_t = model(image_patch_s.to(config.device).float(), image_patch_t.to(config.device).float())



#         # Compute losses
#         label_patch_s = label_patch_s.to(config.device).float()
#         label_patch_t = label_patch_t.to(config.device).float()

#         batch_loss_s = torch.mean(torch.sum(mse_loss(input_image=out_s, target=label_patch_s, ignored_index=config.ignore_index, reduction='None')))
#         batch_loss_t = torch.mean(torch.sum(criterion(out_t, label_patch_t), dim=[1]))

#         batch_loss = batch_loss_s * 0.01 + batch_loss_t
#         batch_loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         # Accumulate losses
#         epoch_loss += batch_loss.item()
#         epoch_loss_s += batch_loss_s.item()
#         epoch_loss_t += batch_loss_t.item()

#     # Average epoch losses
#     epoch_loss = epoch_loss / (batch + 1)
#     epoch_loss_s = epoch_loss_s / (batch + 1)
#     epoch_loss_t = epoch_loss_t / (batch + 1)

#     print(f'Epoch {epoch} completed: Loss={epoch_loss:.4f}, Spatial Loss={epoch_loss_s:.4f}, Temporal Loss={epoch_loss_t:.4f}', flush=True)
#     # Save reconstructions for selected batches and patches
#     save_epoch_reconstructions(model, test_loader, epoch, save_dir="epoch_reconstructions", selected_batches=[0, 5, 10, 20, 40], num_patches=10)
#     # Save model weights every 100 epochs
#     # if epoch % 100 == 0:
#     #     model_weights = os.path.join(config.model_dir, f"{config.experiment_id}_epoch_{epoch}.pt")
#     #     torch.save(model.state_dict(), model_weights)

#     if epoch % 100 == 0:
#         model_weights = os.path.join(config.model_dir, f"{config.experiment_id}_epoch_{epoch}.pt")
#         print(os.path)
#         os.makedirs(os.path.dirname(model_weights), exist_ok=True)  # Ensure directory exists
#         torch.save(model.state_dict(), model_weights)




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

        # Forward pass
        code_vec, out_s, out_t = model(image_patch_s.to(config.device).float(), image_patch_t.to(config.device).float())
        


        optimizer.zero_grad()
        code_vec, out_s, out_t = model(image_patch_s.to(config.device).float(), image_patch_t.to(config.device).float())
        if (epoch%1)==0:
            np.save(f'out_s_{epoch}.npy', out_s.cpu().numpy())
            np.save(f'out_t_{epoch}.npy', out_t.cpu().numpy())
            np.save(f'image_patch_s_{epoch}.npy', image_patch_s)
            np.save(f'image_patch_t_{epoch}.npy', image_patch_t)

        # print('\t1')
        label_patch_s = label_patch_s.to(config.device).float()
        label_patch_t = label_patch_t.to(config.device).float()
        # print('\t2')
        code_vec_farm = code_vec[label_batch == 1]
        code_vec_river = code_vec[label_batch == 3]
        code_vec_stable_lakes = code_vec[label_batch == 4]
        code_vec_mod_seas_lakes = code_vec[label_batch == 5]
        # print('\t3')
        min_class_labels = np.amin([code_vec_farm.shape[0],code_vec_river.shape[0],code_vec_stable_lakes.shape[0],code_vec_mod_seas_lakes.shape[0]])
        # print(min_class_labels)
        # print('\t4')
        if(code_vec_farm.shape[0] > 2):
            farm_batch_loss_log = constrained_loss(code_vec_farm,min_class_labels)
        else:
            farm_batch_loss_log = torch.zeros(1, requires_grad=True)
        # print('\t5')
        if(code_vec_river.shape[0] > 2):
            river_batch_loss_log = constrained_loss(code_vec_river,min_class_labels)
        else:
            river_batch_loss_log = torch.zeros(1, requires_grad=True)
        # print('\t6')
        if(code_vec_stable_lakes.shape[0] > 2):
            stable_lakes_batch_loss_log = constrained_loss(code_vec_stable_lakes,min_class_labels)
        else:
            stable_lakes_batch_loss_log = torch.zeros(1, requires_grad=True)
        # print('\t7')
        if(code_vec_mod_seas_lakes.shape[0] > 2):
            mod_seas_lakes_batch_loss_log = constrained_loss(code_vec_mod_seas_lakes,min_class_labels)
        else:
            mod_seas_lakes_batch_loss_log = torch.zeros(1, requires_grad=True)
        # print('\tStarting spatial loss')
        batch_loss_s = torch.mean(torch.sum(mse_loss(input_image = out_s, target = label_patch_s,ignored_index = config.ignore_index,reduction = 'None')))
        # print('\tEnded spatial loss')
        # print('\tStarted time loss')
        batch_loss_t = torch.mean(torch.sum(criterion(out_t, label_patch_t),dim=[1]))
        # print('\tEnded time loss')

        # check scales of spatial and temporal losses are comparable
        if epoch ==20:        
            print(f'\tBatch loss_s: {batch_loss_s}')
            print(f'\tBatch loss_t: {batch_loss_t}')
        
        # scale the spatial and temporal losses
        if get_magnitude(batch_loss_s) > get_magnitude(batch_loss_t):
            if get_magnitude(batch_loss_s) >= 4:
                scaled_batch_loss_s = batch_loss_s * 0.005
            if get_magnitude(batch_loss_s) >= 3:
                scaled_batch_loss_s = batch_loss_s * 0.05
        else:
            scaled_batch_loss_s = batch_loss_s

        if epoch < 1000:
            batch_loss = scaled_batch_loss_s + batch_loss_t     # TODO: check they are of the same scale after epoch 10, change multiplier as needed
        else:
            batch_loss = scaled_batch_loss_s + batch_loss_t + (river_batch_loss_log + farm_batch_loss_log + stable_lakes_batch_loss_log + mod_seas_lakes_batch_loss_log) * 0.25
        # TODO: use else as batch loss after 1000 epochs (like paper pg 491) <-- play with number [500,1000,15000 etc.]

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

    print(f'epoch loss: {epoch_loss}')
#     print(epoch_loss,epoch_loss_s,epoch_loss_t,epoch_loss_farm_log,epoch_loss_river_log,epoch_loss_stable_lakes_log,epoch_loss_mod_seas_lakes_log,epoch_loss_eph_lakes_log)
    print('\n')
    print(f'Epoch {epoch} completed: Loss={epoch_loss:.4f}, Spatial Loss={epoch_loss_s:.4f}, Temporal Loss={epoch_loss_t:.4f}', flush=True)
    # Save reconstructions for selected batches and patches
    save_epoch_reconstructions(model, test_loader, epoch, save_dir="epoch_reconstructions", selected_batches=[0, 5, 10, 20, 40], num_patches=10)
#     # Save model weights every 100 epochs
#     # if epoch % 100 == 0:
#     #     model_weights = os.path.join(config.model_dir, f"{config.experiment_id}_epoch_{epoch}.pt")
#     #     torch.save(model.state_dict(), model_weights)
    model.eval()
    if epoch % 100 == 0:
        model_weights = os.path.join(config.model_dir, str(config.experiment_id) + "_epoch_" + str(epoch) + ".pt")
        torch.save(model.state_dict(), model_weights)
    epoch_losses = np.append(epoch_losses, epoch_loss)
    
np.save('epoch_losses.npy', epoch_losses)

