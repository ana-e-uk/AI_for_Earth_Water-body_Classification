'''
Script to run the training loop
'''
<<<<<<< HEAD

'''####################################################### Imports #######################################################''' 
import numpy as np
import torch

from model import SpatialCNN, TemporalLSTM, EmbeddingSpace, SpatialDecoder
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

'''####################################################### Element Definitions #######################################################''' 
model = SpatialCNN()

optimizer = torch.optim.Adam()

label_info_array = np.load(config.label_info_filepath)

train_dataset = np.load(config.data_train_baseline)
'''####################################################### Training Loop #######################################################''' 
=======
import torch
print(torch.__version__)  # Displays the PyTorch version
print(torch.cuda.is_available())  # Should return False
>>>>>>> bf31550 (Fixing some errors with Praveen)
