'''
Constants for AI for earth
'''

'''####################################################### Imports #######################################################''' 
import os
import numpy as np

'''####################################################### Directories #######################################################''' 
data_dir = "../../../../../panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT"

# padded_data_dir = os.path.join(data_dir, "350_400_stage2_padded")
# warped_data_dir = os.path.join(data_dir, "350_400_stage2_warped_64x64")

frac_map_data_dir = os.path.join(data_dir, "350_400_stage2_warped_64x64_frac_map")  # fraction maps of equal size
timeseries_data_dir = os.path.join(data_dir, "350_400_stage2_padded_time_series")   # time series

continent_info_filepath = os.path.join(data_dir, "continent_info.npy")
label_info_filepath = os.path.join(data_dir, "all_IDs_labels_realsat.npy")

split_data_dir = "../data"

data_train_baseline = ""    # .npy file with training data
data_test_baseline = ""

data_train_new = ""
data_test_new = ""


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


continent_dict = {0: 'Unknown',
                  1: 'Asia',
                  2: 'North America',
                  3: 'Europe',
                  4: 'Africa',
                  5: 'South America',
                  6: 'Oceania',
                  7: 'Australia',
                  8: 'Antartica'}

labels_dict = {0: 'Unknown',
               1: 'Farm',
               2: 'Reservoir',
               3: 'River',
               4: 'Stable Lakes',
               5: 'Seasonal Lakes',
               6: 'Highly Seasonal Lakes',
               7: 'Ephemeral Lakes',
               8: 'Rover runoff/oxbow'}
