#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# import torch

# experiment details

experiment_id = 'SLTLAE_CL'
# MODEL_NAME = 'SLTLAE_CL_epoch_2000'

load_model = 1
freeze_layers = 0
not_freeze_list = ['out1.weight','out1.bias','out2.weight','out2.bias']
no_epochs_train = 50
label_dict = {1:0,3:1,4:2,5:3}
batch_size_labelled_only = 64
same_sample_no = 10

# load_experiment_id = 'SLTLAE_CL'
# load_MODEL_NAME = 'SLTLAE_CL_epoch_2000'

load_MODEL_DIR = "../../MODELS/" + str(experiment_id)

#####################

lower_limit = 350
upper_limit = 400
warp_size = 64
no_train_images = 10000
water_index = 2

sub_conti = 1
continent_no = 5

time_steps = 442
channels = 1
patch_size = 64
learning_rate = 0.001 
n_epochs = 2000
batch_size = 256
test_batch_size = 256
ignore_index = 0
code_dim = 256
no_classes = 10
device = 'cpu'
percent_to_include = 0.25

labels_for_cl = [1,3,4,5]
out_classes = 4

num_clusters = 200
num_clusters_model = 50
num_clusters_only_labelled = 20
num_clusters_only_labelled_dataset = 20

DATA_DIR = "/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/"+str(lower_limit) + "_" + str(upper_limit) +"_stage2"
PAD_DIR = "/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/"
PAD_FOLDER = str(lower_limit) + "_" + str(upper_limit) +"_stage2_padded"
WARP_DIR = "/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/"
WARP_FOLDER = str(lower_limit) + "_" + str(upper_limit) +"_stage2_warped_" + str(warp_size) +"x" +str(warp_size)
FRAC_MAP_DIR = "/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/"
FRAC_MAP_FOLDER = str(lower_limit) + "_" + str(upper_limit) +"_stage2_warped_" + str(warp_size) +"x" +str(warp_size) + "_frac_map"
TIME_SERIES_DIR = "/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/"
TIME_SERIES_FOLDER = str(lower_limit) + "_" + str(upper_limit) +"_stage2_padded_time_series"
# DATASET_DIR = "../../DATASET_ARRAYS"
DATASET_DIR = "/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/DATASET_ARRAYS"
DATASET_NAME_TRAIN = "train_350_400_labels_427_78_317_143_288_52_255_12_50percentsplit.npy"
DATASET_NAME_TEST = "test_350_400_labels_427_78_317_143_288_52_255_12_50percentsplit.npy"


MODEL_DIR = "/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/MODELS/" + str(experiment_id)


RESULTS_DIR = "/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/RESULT_DATA"
RESULT_FOLDER = str(experiment_id) 

# VISUALISATION_DIR = "../../new_lakes_visualisation"

# farm_label_array_path = '../farm_labels.npy'
# continent_info_array_path = '../continent_info.npy'
# all_label_array_path = '../all_IDs_labels.npy'
