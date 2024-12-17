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
        print('\treconstructing x')
        # Pass through deconvolution layers
        reconstructed_x = self.deconv_layers(x)
        reconstructed_x = reconstructed_x.view(-1, config.output_channels, config.output_size, config.output_size)
        return reconstructed_x
    
class EncoderDecoder(torch.nn.Module):
    def __init__(self, in_channels_spatial, out_channels_spatial, in_channels_temp, out_channels_temp):
        super(EncoderDecoder, self).__init__()
        
        self.code_dim = config.batch_size
        self.device = config.device
        self.in_channels_temp = in_channels_temp
        
        self.conv1_1 = torch.nn.Conv2d(in_channels_spatial, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc = torch.nn.Linear(4096, self.code_dim)
        self.fc_s = torch.nn.Linear(self.code_dim, self.code_dim)
        self.fc_t = torch.nn.Linear(self.code_dim, self.code_dim)

        self.upfc = torch.nn.Linear(self.code_dim, 4096)
        
        self.unpool3 = torch.nn.ConvTranspose2d(64 , 64, kernel_size=2, stride=2)
        self.upconv3_1 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.upconv3_2 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.unpool2 = torch.nn.ConvTranspose2d(32 , 32, kernel_size=2, stride=2)
        self.upconv2_1 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.upconv2_2 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.unpool1 = torch.nn.ConvTranspose2d(16 , 16, kernel_size=2, stride=2)
        self.upconv1_1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.upconv1_2 = torch.nn.Conv2d(16, out_channels_spatial, 3, padding=1)
        
        self.instance_encoder = torch.nn.Linear(in_features=in_channels_temp, out_features=self.code_dim)
        self.temporal_encoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True)
        self.temporal_decoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True)
        self.instance_decoder = torch.nn.Linear(in_features=self.code_dim, out_features=out_channels_temp) 

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        
    def forward(self,x_s,x_t):
        # print('\tstarting forward training')
        x_s = x_s.view(-1, config.channels, config.patch_size, config.patch_size)
        # Convolutional layer
        x_s = self.relu(self.conv1_1(x_s))
        x_s = self.relu(self.conv1_2(x_s))
        x_s = self.maxpool(x_s)
        # Convolutional layer
        x_s = self.relu(self.conv2_1(x_s))
        x_s = self.relu(self.conv2_2(x_s))
        x_s = self.maxpool(x_s)
        # Convolutional layer
        x_s = self.relu(self.conv3_1(x_s))
        x_s = self.relu(self.conv3_2(x_s))
        x_s = self.maxpool(x_s)

        # Spatial encoder
        # print('\tspatiaal encoder')
        enc_s = self.relu(self.fc(x_s.view(-1,4096)))
        enc_s = self.relu(self.fc_s(enc_s))
        
        enc_s_norm = torch.nn.functional.normalize(enc_s, p=2.0, dim=1, eps=1e-12)
        
        batch, window, _ = x_t.shape

        x_encoder = self.instance_encoder(x_t)
        _, x_encoder = self.temporal_encoder(x_encoder)

        # Temporal encoder
        # print('\ttemporal encoder')
        enc_t = x_encoder[0].squeeze()
        enc_t = self.relu(self.fc_t(enc_t))
        enc_t_norm = torch.nn.functional.normalize(enc_t, p=2.0, dim=1, eps=1e-12)
        
        # Combined embeddings
        combined_emb = (enc_s_norm + enc_t_norm)
        
        out_t = torch.zeros(batch, window, self.in_channels_temp).to(self.device)

        # Decoders
        # print('\ttemporal decoder')
        input = torch.unsqueeze(torch.zeros_like(combined_emb), dim=1)
        h = x_encoder
        for step in range(window):
            input, h = self.temporal_decoder(input, h)
            out_t[:,step] = self.instance_decoder(input.squeeze())

        # print('\tspatial decoder')
        up = self.relu(self.upfc(combined_emb))
        # Up-pooling to recover spatial vector
        up = self.unpool3(up.view(-1,64,8,8))
        up = self.relu(self.upconv3_1(up))
        up = self.relu(self.upconv3_2(up))
        # Up-pooling to recover spatial vector
        up = self.unpool2(up)
        up =self.relu(self.upconv2_1(up))
        up = self.relu(self.upconv2_2(up))
        # Up-pooling to recover spatial vector
        up = self.unpool1(up)
        up = self.relu(self.upconv1_1(up))
        up = self.upconv1_2(up)

        out_s = up.view(-1, config.channels, config.patch_size, config.patch_size)
            
        return combined_emb, out_s, out_t
    
class EncoderCNN(torch.nn.Module):
    def __init__(self, in_channels_spatial, in_channels_temp):
        super(EncoderCNN, self).__init__()
        
        self.code_dim = config.batch_size
        self.device = config.device
        self.in_channels_temp = in_channels_temp
        
        self.conv1_1 = torch.nn.Conv2d(in_channels_spatial, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc = torch.nn.Linear(4096, self.code_dim)
        self.fc_s = torch.nn.Linear(self.code_dim, self.code_dim)
        self.fc_t = torch.nn.Linear(self.code_dim, self.code_dim)

        self.out_1 = torch.nn.Linear(256,256)
        self.out_2 = torch.nn.Linear(256,4)
             
        self.instance_encoder = torch.nn.Linear(in_features=in_channels_temp, out_features=self.code_dim)
        self.temporal_encoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True)
      
        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        
    def forward(self,x_s,x_t):
        # print('\tstarting forward training')
        x_s = x_s.view(-1, config.channels, config.patch_size, config.patch_size)
        # Convolutional layer
        x_s = self.relu(self.conv1_1(x_s))
        x_s = self.relu(self.conv1_2(x_s))
        x_s = self.maxpool(x_s)
        # Convolutional layer
        x_s = self.relu(self.conv2_1(x_s))
        x_s = self.relu(self.conv2_2(x_s))
        x_s = self.maxpool(x_s)
        # Convolutional layer
        x_s = self.relu(self.conv3_1(x_s))
        x_s = self.relu(self.conv3_2(x_s))
        x_s = self.maxpool(x_s)

        # Spatial encoder
        # print('\tspatial encoder')
        encoded_s = x_s.view(-1,4096)
        encoded_s = self.relu(self.fc_s(encoded_s))
        encoded_s_norm = torch.nn.functional.normalize(encoded_s, p=2.0, dim=1, eps=1e-12)
        
        # Temporal encoder
        # print('\ttemporal encoder')
        x_encoder = self.instance_encoder(x_t)
        _, x_encoder = self.temporal_encoder(x_encoder)
        encoded_t = x_encoder[0].squeeze()
        encoded_t = self.relu(self.fc_t(encoded_t))
        encoded_t_norm = torch.nn.functional.normalize(encoded_t, p=2.0, dim=1, eps=1e-12)
        
        # Combined embeddings
        combined_emb = (encoded_s_norm + encoded_t_norm)
        
        out_labels_1 = self.relu((self.out1(combined_emb.view(-1,256))))
        out_labels_2 = self.out_2(out_labels_1.view(-1,256))

        return combined_emb, out_labels_2
'''####################################################### #######################################################''' 
