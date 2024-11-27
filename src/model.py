'''
Code for defining and training models
'''

'''####################################################### Imports #######################################################''' 
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
'''####################################################### SLTLAE_CL #######################################################''' 
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=config.channels, out_channels=16, kernel_size=3, stride=1, padding=1) # 1 -> 16 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 16 -> 32 channels
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 32 -> 64 channels
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 64 -> 128 channels
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # 128 -> 256 channels

        # Fully connected layer
        self.fc1 = nn.Linear(256 * (config.patch_size // 2**5)**2, 256)  # Considering pooling reduces size by 2^5
        self.fc2 = nn.Linear(256, 10)  # Assuming 10 output classes (adjust as needed)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
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

'''####################################################### M_2, M_2_r #######################################################''' 