'''
Script to run the training loop for the encoder-decoder
'''

'''####################################################### Imports #######################################################''' 
import numpy as np
import time
import os
import math
import matplotlib.pyplot as plt

import torch
from torch.utils.data import random_split
from scipy.special import logsumexp
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
                log1 = -torch.log(np.abs(cos)) 
                # Compute log of the cosine similarity
                if cos > 0:
                    if(torch.isnan(log1)):
                        print(f"Torch == nan: {log1,e1,e2,cos}")
                    total_loss += (log1)    
            count += count

    avg_loss = torch.div(total_loss, count)

    return avg_loss

def cl_criterion(reps,no_max_reps):
    no_of_reps = reps.shape[0]
    torch_arr = torch.randint(0, no_of_reps, (2*no_max_reps,))
    torch_arr_2 = torch.randint(0, no_of_reps, (2*no_max_reps,))
    total_log_sum = np.array([])
    count_rep = 0
    # print(f"no_of_reps: {no_of_reps}")
    
    for i in range(torch_arr.shape[0]): 
        if(torch_arr[i] == torch_arr_2[i]):
            continue
        mag1 = torch.sqrt(torch.sum(torch.square(reps[torch_arr[i]])))
        # print(f'mag1\t{mag1}')
        mag2 = torch.sqrt(torch.sum(torch.square(reps[torch_arr_2[i]])))
        # print(f'mag2\t{mag2}')
        prod12 = mag1*mag2
        dot12 = torch.abs(torch.dot(reps[torch_arr[i]],reps[torch_arr_2[i]]))
        # print(f'dot12\t{dot12}')
        cos12 = torch.div(dot12,prod12)
        # print(f'cos12\t{cos12}')
        log12 = -torch.log(cos12)
        log12_np = log12.detach().cpu().numpy()
        if(torch.isnan(log12)):
            print(f"\t\tlog12 is NAN")
        
        # if i == 1:
        #     print(f'mag1\t{mag1}')
        #     print(f'mag2\t{mag2}')
        #     print(f'dot12\t{dot12}')
        #     print(f'cos12\t{cos12}')
        #     print(f'log12\t{log12}')
    
        # total_log_sum += log12  # log sum exp (rescale number)
        total_log_sum = np.append(total_log_sum, log12_np)
        # print(f"\nTOTAL_LOG_SUM: \t{total_log_sum}")
        count_rep += 1

    if(count_rep == 0):
        count_rep = 1
    
    total_log_sum_calculated = logsumexp(total_log_sum)

    avg_log = torch.div(total_log_sum_calculated,count_rep)
    # print(f"\t\tCL_criterion returning {avg_log}")

    return avg_log



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

""" Aylar code """
# def save_epoch_reconstructions(model, data_loader, epoch, save_dir="epoch_reconstructions", selected_batches=[0, 5, 10], num_patches=10):
#     """
#     Save spatial and temporal reconstructions for specific batches and samples across epochs.
    
#     Args:
#         model: The trained model.
#         data_loader: DataLoader to fetch data.
#         epoch: Current epoch number.
#         save_dir: Directory to save the reconstructions.
#         selected_batches: List of batch indices to save.
#         num_patches: Number of patches per batch to save.
#     """
#     model.eval()  # Set the model to evaluation mode
#     os.makedirs(save_dir, exist_ok=True)
    
#     with torch.no_grad():
#         for batch_idx, (image_patch_s, label_patch_s, image_patch_t, label_patch_t, _, _) in enumerate(data_loader):
#             if batch_idx in selected_batches:
#                 # Forward pass
#                 image_patch_s = image_patch_s.to(config.device).float()
#                 image_patch_t = image_patch_t.to(config.device).float()
#                 label_patch_s = label_patch_s.to(config.device).float()
#                 label_patch_t = label_patch_t.to(config.device).float()

#                 _, out_s, out_t = model(image_patch_s, image_patch_t)

#                 # Limit to the specified number of patches
#                 selected_true_s = label_patch_s[:num_patches].cpu()
#                 selected_recon_s = out_s[:num_patches].cpu()
#                 selected_true_t = label_patch_t[:num_patches].cpu()
#                 selected_recon_t = out_t[:num_patches].cpu()

#                 # Define directories
#                 spatial_save_dir = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}_spatial")
#                 temporal_save_dir = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}_temporal")

#                 # Save patches
#                 save_spatial_patches(selected_true_s, selected_recon_s, epoch, batch_idx, save_dir=spatial_save_dir)
#                 save_temporal_patches(selected_true_t, selected_recon_t, epoch, batch_idx, save_dir=temporal_save_dir)

'''Ana code'''
def save_epoch_reconstructions(epoch, image_patch_s, image_patch_t, label_patch_s, label_patch_t, save_dir="epoch_reconstructions"):
    """
    Save spatial and temporal reconstructions for specific batches and samples across epochs.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Define directories
    spatial_save_dir = os.path.join(save_dir, f"epoch_{epoch}_spatial")
    temporal_save_dir = os.path.join(save_dir, f"epoch_{epoch}_temporal")
    os.makedirs(spatial_save_dir, exist_ok=True)
    os.makedirs(temporal_save_dir, exist_ok=True)

    # Convert tensors to numpy arrays
    image_patch_s = image_patch_s.detach().cpu().numpy()  # Shape: (batch_size, 1, height, width)
    label_patch_s = label_patch_s.detach().cpu().numpy()  # Shape: (batch_size, 1, height, width)

    image_patch_t = image_patch_t.detach().cpu().numpy()  # Shape: (batch_size, sequence_length)
    label_patch_t = label_patch_t.detach().cpu().numpy()  # Shape: (batch_size, sequence_length)

    # Process spatial patches
    for i in range(min(5, image_patch_s.shape[0])):  # Save up to 5 examples
        true_image = np.squeeze(label_patch_s[i], axis=0)  # Remove the channel dimension
        reconstructed_image = np.squeeze(image_patch_s[i], axis=0)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("True Patch")
        plt.imshow(true_image)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Patch")
        plt.imshow(reconstructed_image)
        plt.axis("off")
        plt.savefig(os.path.join(spatial_save_dir, f"epoch_{epoch}_spatial_patch_{i}.png"))
        plt.close()

    # Process temporal patches
    for i in range(min(5, image_patch_t.shape[0])):  # Save up to 5 examples
        true_temporal = label_patch_t[i]
        reconstructed_temporal = image_patch_t[i]

        plt.figure(figsize=(8, 4))
        plt.plot(true_temporal, label="True", linestyle="--", alpha=0.7)
        plt.plot(reconstructed_temporal, label="Reconstructed", alpha=0.7)
        plt.title(f"Temporal Patch {i} from Epoch {epoch}")
        plt.legend()
        plt.savefig(os.path.join(temporal_save_dir, f"epoch_{epoch}_temporal_patch_{i}.png"))
        plt.close()


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

all_reps = []
valid_paths = []
valid_outs_s = []
valid_outs_t = []

'''####################################################### Encoder-Decoder Training Loop #######################################################''' 
epoch_losses = np.array([])
train_loss = []

for epoch in range(1, config.num_epochs+1):
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
        # Forward pass
        optimizer.zero_grad()
        code_vec, out_s, out_t = model(image_patch_s.to(config.device).float(), image_patch_t.to(config.device).float())

        label_patch_s = label_patch_s.to(config.device).float()
        label_patch_t = label_patch_t.to(config.device).float()

        code_vec_farm = code_vec[label_batch == 1]
        code_vec_river = code_vec[label_batch == 3]
        code_vec_stable_lakes = code_vec[label_batch == 4]
        code_vec_mod_seas_lakes = code_vec[label_batch == 5]

        batch_loss_s = torch.mean(torch.sum(mse_loss(input_image = out_s, target = label_patch_s,ignored_index = config.ignore_index,reduction = 'None')))
        batch_loss_t = torch.mean(torch.sum(criterion(out_t, label_patch_t),dim=[1]))

        # # scale the spatial and temporal losses
        # min_loss = min(batch_loss_s, batch_loss_t)
        # max_loss = min(batch_loss_s, batch_loss_t)

        # scale_dif = get_magnitude(max_loss) - get_magnitude(min_loss)
        # if scale_dif > 0:
        #     if scale_dif >= 3:
        #         scaled_max_loss = max_loss * 0.0005
        #     elif scale_dif >= 2:
        #         scaled_max_loss = max_loss * 0.005
        #     elif scale_dif >= 1:
        #         scaled_max_loss = max_loss * 0.05
        #     else:
        #         print(f"\tLOSS SCALES OFF (epoch {epoch}):max {max_loss}\t min {min_loss}")
        # batch_loss = scaled_max_loss + min_loss

        scale_dif = get_magnitude(batch_loss_s) - get_magnitude(batch_loss_t)
        if scale_dif > 0:
            if scale_dif >= 3:
                scaled_batch_loss_s = batch_loss_s * 0.0005
            elif scale_dif >= 2:
                scaled_batch_loss_s = batch_loss_s * 0.005
            elif scale_dif >= 1:
                scaled_batch_loss_s = batch_loss_s * 0.05
            else:
                print(f"\tLOSS SCALES OFF (epoch {epoch}): batch loss s {batch_loss_s}\t batch loss t {batch_loss_t}")

        # calculate final batch loss
        batch_loss = scaled_batch_loss_s + batch_loss_t

        # if epoch >= 1000:
        #     min_class_labels = np.amin([code_vec_farm.shape[0],code_vec_river.shape[0],code_vec_stable_lakes.shape[0],code_vec_mod_seas_lakes.shape[0]])

        #     farm_batch_loss_log = torch.zeros(1, device=config.device, requires_grad=True) if code_vec_farm.shape[0] <= 2 else constrained_loss(code_vec_farm, min_class_labels)
        #     river_batch_loss_log = torch.zeros(1, device=config.device, requires_grad=True) if code_vec_river.shape[0] <= 2 else constrained_loss(code_vec_river, min_class_labels)
        #     stable_lakes_batch_loss_log = torch.zeros(1, device=config.device, requires_grad=True) if code_vec_stable_lakes.shape[0] <= 2 else constrained_loss(code_vec_stable_lakes, min_class_labels)
        #     mod_seas_lakes_batch_loss_log = torch.zeros(1, device=config.device, requires_grad=True) if code_vec_mod_seas_lakes.shape[0] <= 2 else constrained_loss(code_vec_mod_seas_lakes, min_class_labels)

        #     batch_loss += (farm_batch_loss_log + river_batch_loss_log + stable_lakes_batch_loss_log + mod_seas_lakes_batch_loss_log) * 0.25
        if epoch >= 1000:#1000:
            # Find the minimum number of class labels
            min_class_labels = np.amin([
                code_vec_farm.shape[0],
                code_vec_river.shape[0],
                code_vec_stable_lakes.shape[0],
                code_vec_mod_seas_lakes.shape[0]
            ])

            # Compute class-specific losses or set to scalar zeros
            farm_batch_loss_log = torch.tensor(0.0, device=config.device, requires_grad=True) \
                if code_vec_farm.shape[0] <= 2 else cl_criterion(code_vec_farm, min_class_labels)

            river_batch_loss_log = torch.tensor(0.0, device=config.device, requires_grad=True) \
                if code_vec_river.shape[0] <= 2 else cl_criterion(code_vec_river, min_class_labels)

            stable_lakes_batch_loss_log = torch.tensor(0.0, device=config.device, requires_grad=True) \
                if code_vec_stable_lakes.shape[0] <= 2 else cl_criterion(code_vec_stable_lakes, min_class_labels)

            mod_seas_lakes_batch_loss_log = torch.tensor(0.0, device=config.device, requires_grad=True) \
                if code_vec_mod_seas_lakes.shape[0] <= 2 else cl_criterion(code_vec_mod_seas_lakes, min_class_labels)

            # Ensure all loss components are scalars
            farm_batch_loss_log = farm_batch_loss_log.squeeze()
            river_batch_loss_log = river_batch_loss_log.squeeze()
            stable_lakes_batch_loss_log = stable_lakes_batch_loss_log.squeeze()
            mod_seas_lakes_batch_loss_log = mod_seas_lakes_batch_loss_log.squeeze()

            # Add the losses and scale
            batch_loss += (farm_batch_loss_log + river_batch_loss_log +
                        stable_lakes_batch_loss_log + mod_seas_lakes_batch_loss_log) * 0.25

        # print(f"\tbatch loss {batch_loss}")
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        # epoch_loss_s += scaled_batch_loss_s.item()
        # epoch_loss_t += batch_loss_t.item()
        # epoch_loss_farm_log += farm_batch_loss_log.item()
        # epoch_loss_river_log += river_batch_loss_log.item()
        # epoch_loss_stable_log += stable_lakes_batch_loss_log.item()
        # epoch_loss_seasonal_log += mod_seas_lakes_batch_loss_log.item()

    epoch_loss = epoch_loss/(batch+1)
    # epoch_loss_s = epoch_loss_s/(batch+1)
    # epoch_loss_t = epoch_loss_t/(batch+1)
    # epoch_loss_farm_log = epoch_loss_farm_log/(batch+1)
    # epoch_loss_river_log = epoch_loss_river_log/(batch+1)
    # epoch_loss_stable_log = epoch_loss_stable_log/(batch+1)
    # epoch_loss_seasonal_log = epoch_loss_seasonal_log/(batch+1)

    print(f"EPOCH {epoch}\t {epoch_loss}")
    model.eval()
    if epoch%500==0:
        print(f'\tScaled s: {scaled_batch_loss_s:.4f}\t t: {batch_loss_t:.4f}')
        # print(f'\tScaled max: {scaled_max_loss:.4f}\t min: {min_loss:.4f}')
        # Save reconstructions for selected batches and patches
        # save_epoch_reconstructions(model, test_loader, epoch, save_dir="epoch_reconstructions", selected_batches=[0, 5, 10, 20, 40], num_patches=10)  # Aylar code
        save_epoch_reconstructions(epoch, image_patch_s=out_s, image_patch_t=out_t, label_patch_s=label_patch_s, label_patch_t=label_patch_t, save_dir="epoch_reconstructions") # Ana attempt
    
        model_weights = os.path.join(config.model_dir, str(config.experiment_id) + "_epoch_" + str(epoch) + ".pt")
        torch.save(model.state_dict(), model_weights)
    epoch_losses = np.append(epoch_losses, epoch_loss)
    
np.save('epoch_losses_e_d.npy', epoch_losses)
print(f'saved epoch losses to epoch_losses_e_d.npy')