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

model_dir = '/users/6/uribe055/AI_for_Earth_Water-body_Classification/src/supervised_training_out'
model_dir_e_CNN = '/users/6/uribe055/AI_for_Earth_Water-body_Classification/src/classification_training_out'

RESULTS = '/users/6/uribe055/AI_for_Earth_Water-body_Classification/figures'


split_data_dir = "../data"

data_train_baseline = ""    # .npy file with training data
data_test_baseline = ""

data_train_new = ""
data_test_new = ""

# params for the Supervised learning model
freeze_layers = 0
load_MODEL_NAME = 'SLTLAE_CL_epoch_1200'
load_MODEL_DIR = '/users/6/uribe055/AI_for_Earth_Water-body_Classification/src/supervised_training_out'

# params for classifier
load_model_name_e_CNN = 'Classifier_epoch_2000'

load_model = 1

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
device = 'cuda'
ignore_index = 0
experiment_id = 'SLTLAE_CL'
experiment_id_e_CNN = 'Classifier'

# training params
batch_size = 256
lambda_val = 0.01
gamma_val = 1
num_epochs = 3000   # 2500
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
