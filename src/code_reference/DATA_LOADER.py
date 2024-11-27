import sys
sys.path.append("../")
import os
import config
import copy
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from collections import Counter
import torch
import glob
from torch.utils.data.dataset import Dataset
# from osgeo import gdal, gdalconst, osr

def create_patches_time_series(path,label_array):
    
    ID = int(path.split('/')[-1].split('_')[-4])
    label_ID = label_array[int(ID)]
    
    image = np.load(path)
    
    water_count = []
    for t in range(image.shape[0]):
        water_count.append(np.sum(image[t,0,:,:] == config.water_index))
        
    image_patches = np.array(water_count).astype(np.float32)
    image_patches = image_patches/np.max(image_patches)
    image_patches = np.expand_dims(image_patches, axis=1)
    
    label_patches = np.array(water_count).astype(np.float32)
    label_patches = label_patches/np.max(label_patches)
    label_patches = np.expand_dims(label_patches, axis=1)
    
    return image_patches, label_patches,label_ID,ID

def create_patches_frac_map(path,label_array):
    
    ID = int(path.split('/')[-1].split('_')[-4])
    label_ID = label_array[int(ID)]
    
    image = np.load(path)
    
    # converting labels to binary, i.e land or water
    image[image == 1] = 0 
    image[image == 2] = 1 
        
    frac_map_image = np.mean(image,axis = 0)

    image_patches = np.array(frac_map_image).astype(np.float32)
    label_patches = np.array(frac_map_image).astype(np.float32)

    return image_patches, label_patches,label_ID,ID

def create_patches_spatial_temporal(path,patch_size):
    
    ID = path.split('/')[-1].split('_')[-4] 
    image = np.load(path)
    pad_image = np.load(os.path.join(config.PAD_DIR,config.PAD_FOLDER, 'ID_' + str(ID)+'_orbit_updated_padded.npy'))
                        
    # time series
    water_count = []
    for t in range(pad_image.shape[0]):
        water_count.append(np.sum(pad_image[t,0,:,:] == config.water_index))
        
    image_patches_temp = np.array(water_count).astype(np.float32)
    image_patches_temp = image_patches_temp/np.max(image_patches_temp)
    image_patches_temp = np.expand_dims(image_patches_temp, axis=1)
    
    label_patches_temp = np.array(water_count).astype(np.float32)
    label_patches_temp = label_patches_temp/np.max(label_patches_temp)
    label_patches_temp = np.expand_dims(label_patches_temp, axis=1)
    
    # fraction map
    # converting labels to binary, i.e land or water
    image[image == 1] = 0 
    image[image == 2] = 1 
        
    frac_map_image = np.mean(image,axis = 0)

    image_patches_spatial = np.array(frac_map_image).astype(np.float32)
    label_patches_spatial = np.array(frac_map_image).astype(np.float32)
    
    return image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp

def create_patches_spatial_temporal_label_ID(path,label_array):
    
    ID = int(path.split('/')[-1].split('_')[-4])
    label_ID = label_array[int(ID)]
    
    image = np.load(path)
    pad_image = np.load(os.path.join(config.PAD_DIR,config.PAD_FOLDER, 'ID_' + str(ID)+'_orbit_updated_padded.npy'))
                        
    # time series
    water_count = []
    for t in range(pad_image.shape[0]):
        water_count.append(np.sum(pad_image[t,0,:,:] == config.water_index))
        
    image_patches_temp = np.array(water_count).astype(np.float32)
    image_patches_temp = image_patches_temp/np.max(image_patches_temp)
    image_patches_temp = np.expand_dims(image_patches_temp, axis=1)
    
    label_patches_temp = np.array(water_count).astype(np.float32)
    label_patches_temp = label_patches_temp/np.max(label_patches_temp)
    label_patches_temp = np.expand_dims(label_patches_temp, axis=1)
    
    # fraction map
    # converting labels to binary, i.e land or water
    image[image == 1] = 0 
    image[image == 2] = 1 
        
    frac_map_image = np.mean(image,axis = 0)

    image_patches_spatial = np.array(frac_map_image).astype(np.float32)
    label_patches_spatial = np.array(frac_map_image).astype(np.float32)
    
    return image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp,label_ID,ID

def get_data_spatial_temporal_label_ID_load_arrays(ID,label_array):
    
#     ID = int(path.split('/')[-1].split('_')[-4])
    label_ID = label_array[int(ID)]
    
    frac_map_array = np.load(os.path.join(config.FRAC_MAP_DIR,config.FRAC_MAP_FOLDER, 'ID_' + str(ID).zfill(6)+'_frac_map.npy'))
    time_series_array = np.load(os.path.join(config.TIME_SERIES_DIR,config.TIME_SERIES_FOLDER, 'ID_' + str(ID).zfill(6)+'_time_series.npy'))
    
    return frac_map_array, frac_map_array, time_series_array, time_series_array,label_ID,ID

    

class SEGMENTATION_SLTL(Dataset):

    def __init__(self, image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp):
        self.image_patches_spatial = image_patches_spatial
        self.label_patches_spatial = label_patches_spatial        
        self.image_patches_temp = image_patches_temp
        self.label_patches_temp = label_patches_temp

    def __len__(self):
        return len(self.label_patches_spatial)

    def __getitem__(self, index):
        return self.image_patches_spatial[index], self.label_patches_spatial[index],self.image_patches_temp[index], self.label_patches_temp[index]
    
class SEGMENTATION_PRED(Dataset):

    def __init__(self, image_patches, label_patches,label_IDs,IDs):
        self.image_patches = image_patches
        self.label_patches = label_patches
        self.label_IDs = label_IDs
        self.IDs = IDs

    def __len__(self):
        return len(self.label_patches)

    def __getitem__(self, index):
        return self.image_patches[index], self.label_patches[index],self.label_IDs[index],self.IDs[index]

    
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
    
# %%
if __name__ == "__main__":
    print('Main')