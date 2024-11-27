'''
Description: Compare data from different continents to choose data that is most similar to North America dataset

Author: Ana Uribe
Date updated: Nov 20 2024
Class: CSCI 8523 AI for Earth
'''
# imports
import numpy as np
import os
import matplotlib.pyplot as plt

from collections import defaultdict

''' ############################################# Load Data #####################################################'''

print('\nloading data...')

ID_no = 657668 # enter a valid ID number

padded_data_dir = '/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/350_400_stage2_padded/'
warped_data_dir = '/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/350_400_stage2_warped_64x64/'
frac_map_data_dir = '/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/350_400_stage2_warped_64x64_frac_map/'
timeseries_data_dir = '/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/350_400_stage2_padded_time_series/'

padded_name = 'ID_'+str(ID_no)+'_orbit_updated_padded.npy'
warped_name = 'ID_'+str(ID_no)+'_orbit_updated_warped.npy'
frac_map_name = 'ID_'+str(ID_no)+'_frac_map.npy'
time_series_name = 'ID_'+str(ID_no)+'_time_series.npy'

continent_info = np.load('continent_info.npy')
label_info = np.load('all_IDs_labels_realsat.npy')

padded_array = np.load(os.path.join(padded_data_dir,padded_name))
warped_array = np.load(os.path.join(warped_data_dir,warped_name))
frac_map_array = np.load(os.path.join(frac_map_data_dir,frac_map_name))
time_series_array = np.load(os.path.join(timeseries_data_dir,time_series_name))

print('\tPadded Array Shape:',padded_array.shape)
print('\tWarped Array Shape:',warped_array.shape)
print('\tFrac Map Array Shape:',frac_map_array.shape)
print('\tTime series Array Shape:',time_series_array.shape)

print('\tSample from Continent: ',continent_info[ID_no])
print('\tSample has label: ',label_info[ID_no])

# print('\tSample from Continent Index: ',np.where(continent_info == continent_info[ID_no]))
# print('\tSample has label Index: ',np.where(label_info == label_info[ID_no]))

''' ############################################# Get Labels #####################################################'''
print('\ngetting labels for each continent...')

# # Assuming label_info and continent_info are numpy arrays or lists of equal length
# label_counts_by_continent = defaultdict(lambda: defaultdict(int))

# for ID in range(len(label_info)):  # Iterate over IDs
#     label = label_info[ID]
#     continent = continent_info[ID]
#     label_counts_by_continent[continent][label] += 1

# # Convert defaultdict to a regular dict for better readability (optional)
# label_counts_by_continent = {continent: dict(label_counts) for continent, label_counts in label_counts_by_continent.items()}

# # Print the results
# for continent, label_counts in label_counts_by_continent.items():
#     print(f"Continent {continent}:")
#     for label, count in label_counts.items():
#         print(f"  Label {label}: {count}")

def plot_label_counts(label_info, continent_info):
    """
    Plots the counts of each label for each continent and saves the plot as an image file.
    
    Args:
        label_info (list or numpy array): List/array of labels corresponding to IDs.
        continent_info (list or numpy array): List/array of continent info corresponding to IDs.
    """
    # Organize the label counts by continent
    label_counts_by_continent = defaultdict(lambda: defaultdict(int))
    for ID in range(len(label_info)):
        label = label_info[ID]
        continent = continent_info[ID]
        label_counts_by_continent[continent][label] += 1

    # Create and save plots for each continent
    for continent, label_counts in label_counts_by_continent.items():
        labels = list(label_counts.keys())
        counts = list(label_counts.values())

        plt.figure(figsize=(8, 6))
        plt.bar(labels, counts, color='skyblue')
        plt.title(f"Label Counts for {continent}")
        plt.xlabel("Labels")
        plt.ylabel("Counts")
        plt.xticks(labels)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save plot to file with the name of the continent
        filename = f"{continent}_label_counts.png"
        plt.savefig(filename)
        plt.close()  # Close the figure to free memory
        print(f"Saved plot for {continent} as {filename}")

