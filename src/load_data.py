'''
Code for handling data loading, transformations, preprocessing
'''

'''####################################################### Imports #######################################################''' 

import numpy as np
import pandas as pd
import os
import config
import torch
from torch.utils.data.dataset import Dataset

'''####################################################### Get Train/test split #######################################################''' 


def get_data_spatial_temporal_label_ID_load_arrays(ID,label_array):
    
    label_ID = label_array[int(ID)]
    
    frac_map_array = np.load(os.path.join(config.frac_map_data_dir, 'ID_' + str(ID).zfill(6)+'_frac_map.npy'))
    time_series_array = np.load(os.path.join(config.timeseries_data_dir, 'ID_' + str(ID).zfill(6)+'_time_series.npy'))
    
    return frac_map_array, frac_map_array, time_series_array, time_series_array,label_ID,ID


# get training dataset
def get_training_dataset():

    ids_and_labels_full = pd.read_csv('/users/6/uribe055/AI_for_Earth_Water-body_Classification/data/ShapeFiles/NorthAmericaWaterBodies_FirstPoints_WithLabels.csv')

    ids_and_labels = ids_and_labels_full[ids_and_labels_full['LABEL'].isin([1, 3, 4, 5])]
    print('\tfiltered water body labels')
    # Boundary for label 1
    boundary = -92.32695843970778

    # get region 1 and 2
    region_1 = ids_and_labels[ids_and_labels['min_lon'] >= boundary]    # training dataset
    region_2 = ids_and_labels[ids_and_labels['min_lon'] < boundary]     # testing dataset
    print('\tcreated regions')
    # get id numbers in a numpy array
    region_1_IDs = region_1['ID']
    region_2_IDs = region_2['ID']

    image_patches_spatial_list = []
    label_patches_spatial_list = []
    image_patches_temp_list = []
    label_patches_temp_list = []
    label_IDs_list = []
    IDs_list = []

    print('\tMaking lists')
    for ID_no in region_1_IDs:

        image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp, label_ID,ID = get_data_spatial_temporal_label_ID_load_arrays(ID_no,np.load(config.label_info_filepath))

        if(len(image_patches_spatial_list) % 1000 == 0):
                print(len(image_patches_spatial_list))
                
        image_patches_spatial_list.append(image_patches_spatial)
        label_patches_spatial_list.append(label_patches_spatial)    
        image_patches_temp_list.append(image_patches_temp)
        label_patches_temp_list.append(label_patches_temp)
        label_IDs_list.append(label_ID)
        IDs_list.append(ID)

    print(len(image_patches_spatial_list))
    print('LABEL COUNT:',np.bincount(np.array(label_IDs_list)))

    return image_patches_spatial_list, label_patches_spatial_list, image_patches_temp_list, label_patches_temp_list, label_IDs_list, IDs_list

# get training dataset
def get_testing_dataset():

    ids_and_labels_full = pd.read_csv('/users/6/uribe055/AI_for_Earth_Water-body_Classification/data/ShapeFiles/NorthAmericaWaterBodies_FirstPoints_WithLabels.csv')

    ids_and_labels = ids_and_labels_full[ids_and_labels_full['LABEL'].isin([1, 3, 4, 5])]
    print('\tfiltered water body labels')
    # Boundary for label 1
    boundary = -92.32695843970778

    # get region 1 and 2
    region_1 = ids_and_labels[ids_and_labels['min_lon'] >= boundary]    # training dataset
    region_2 = ids_and_labels[ids_and_labels['min_lon'] < boundary]     # testing dataset
    print('\tcreated regions')
    # get id numbers in a numpy array
    region_1_IDs = region_1['ID']
    region_2_IDs = region_2['ID']

    image_patches_spatial_list = []
    label_patches_spatial_list = []
    image_patches_temp_list = []
    label_patches_temp_list = []
    label_IDs_list = []
    IDs_list = []

    print('\tMaking lists')
    for ID_no in region_2_IDs:

        image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp, label_ID,ID = get_data_spatial_temporal_label_ID_load_arrays(ID_no,np.load(config.label_info_filepath))

        if(len(image_patches_spatial_list) % 1000 == 0):
                print(len(image_patches_spatial_list))
                
        image_patches_spatial_list.append(image_patches_spatial)
        label_patches_spatial_list.append(label_patches_spatial)    
        image_patches_temp_list.append(image_patches_temp)
        label_patches_temp_list.append(label_patches_temp)
        label_IDs_list.append(label_ID)
        IDs_list.append(ID)

    print(len(image_patches_spatial_list))
    print('LABEL COUNT:',np.bincount(np.array(label_IDs_list)))

    return image_patches_spatial_list, label_patches_spatial_list, image_patches_temp_list, label_patches_temp_list, label_IDs_list, IDs_list


class SEGMENTATION_SLTL_PRED(Dataset):

    def __init__(self, image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp,label_IDs,IDs):
        self.image_patches_spatial = image_patches_spatial
        self.label_patches_spatial = label_patches_spatial        
        self.image_patches_temp = image_patches_temp
        self.label_patches_temp = label_patches_temp
        self.label_IDs = label_IDs
        self.IDs = IDs

    def __len__(self):
        return len(self.label_patches_spatial)

    def __getitem__(self, index):
        return self.image_patches_spatial[index], self.label_patches_spatial[index],self.image_patches_temp[index], self.label_patches_temp[index],self.label_IDs[index],self.IDs[index]
    