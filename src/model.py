'''
Code for defining and training models
'''

'''####################################################### Imports #######################################################''' 
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
'''####################################################### SLTLAE_CL #######################################################''' 
class SpatialCNN(nn.Module):
    def __init__(self):
        super(SpatialCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=config.channels, out_channels=16, kernel_size=3, stride=1, padding=1) # 1 -> 16 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 16 -> 32 channels
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 32 -> 64 channels
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 64 -> 128 channels
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # 128 -> 256 channels

        # Fully connected layer
        self.fc1 = nn.Linear(256 * (config.patch_size // 2**5)**2, 256)  # Considering pooling reduces size by 2^5 (1024)
        self.fc2 = nn.Linear(256, config.num_classes)  # 4 for original num classes, 5 when including reservoir

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = x.view(-1, config.channels, config.patch_size, config.patch_size)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class TemporalLSTM(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, bidirectional=True):
        super(TemporalLSTM, self).__init__()
        self.encode = nn.LSTM(
            input_size = in_size, # Features in the input
            hidden_size = hidden_size,   # Num features in hidden state
            num_layers = num_layers,     # Num LSTM layers
            bidirectional = bidirectional,
            batch_first = False
        )

    def forward(self, x):
        # Output shape: [seq_len, batch_size, directions * hidden_size]
        # h_n, c_n: [num_layers * directions, batch_size, hidden_size]
        output, (h_n, c_n) = self.encode(x)
        return output, (h_n, c_n)
    
class EmbeddingSpace(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, embedding_dim):
        """
        Combine spatial and temporal features into a single embedding.
        
        Args:
            spatial_dim: Dimension of the spatial features (from CNN output).
            temporal_dim: Dimension of the temporal features (from LSTM output).
            embedding_dim: Dimension of the final embedding.
        """
        super(EmbeddingSpace, self).__init__()
        self.fc = nn.Linear(spatial_dim + temporal_dim, embedding_dim)  # combine features
        self.relu = nn.ReLu()   # ReLU activation
        self.l2_norm = nn.LayerNorm(embedding_dim)  # l_2 normalization

    def forward(self, spatial_features, temporal_features):
        """
        Args:
            spatial_features: Tensor of shape [batch_size, spatial_dim]
            temporal_features: Tensor of shape [batch_size, temporal_dim]
        Returns:
            embeddings: Tensor of shape [batch_size, embedding_dim]
        """
        # Concatenate spatial and temporal features along the last dimension
        combined_features = torch.cat([spatial_features, temporal_features], dim=-1)
        
        # Apply the transformation, activation, and normalization
        embeddings = self.fc(combined_features)  # Linear transformation
        embeddings = self.relu(embeddings)  # ReLU activation
        embeddings = self.l2_norm(embeddings)  # l2 normalization
        return embeddings 

class SpatialDecoder(nn.Module):
    def __init__(self):
        super(SpatialDecoder, self).__init__()
        self.latent_dim = config.latent_dim

        # Fully connected layer to reshape latent vector to start of deconvolution
        self.fc = nn.Linear(self.latent_dim, 64 * (config.output_size // 16) * (config.output_size // 16))  # 256, 2048

        # Deconvolution layers (up-sampling layers)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # Upsample 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Upsample 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(16, config.output_channels, kernel_size=4, stride=2, padding=1),  # Upsample to (output_size)
        )

    def forward(self, latent_vector):
        """
        Args:
            latent_vector: Output of encoder. Tensor of shape [batch_size, latent_dim]
        Returns:
            reconstructed_x: Reconstructed input tensor of shape [batch_size, output_channels, output_size, output_size]
        """
        # Fully connected layer to reshape latent vector
        x = self.fc(latent_vector)

        # Pass through deconvolution layers
        reconstructed_x = self.deconv_layers(x)
        reconstructed_x = reconstructed_x.view(-1, config.output_channels, config.output_size, config.output_size)
        return reconstructed_x
'''####################################################### M_2, M_2_r #######################################################''' 
