'''
Constants for AI for earth
'''

'''####################################################### Imports #######################################################''' 
import os

'''####################################################### Directories #######################################################''' 


'''####################################################### Constants #######################################################''' 
# spatial CNN
channels = 1
patch_size = 64

# spatial decoder
latent_dim = 256
output_size = patch_size
output_channels = channels

# temporal bi-directional LSTM
time_steps = 442

# general
num_classes = 4 # 5 when doing reservoir

# training params
batch_size = 256
lambda_val = 0.01
gamma_val = 1
num_epochs = 1  # change to 2000
learning_rate = 0.001
