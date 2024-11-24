'''
Description: Explore the given dataset.
             Compare data from different continents to choose data that is most similar to North America dataset.

Author: Ana Uribe
Date updated: Nov 23 2024
Class: CSCI 8523 AI for Earth
'''
# imports
import numpy as np
import os
import matplotlib.pyplot as plt

from collections import defaultdict

''' ############################################# Load Data #####################################################'''

print('\nloading data...')

verbose = True

padded_data_dir = '../../../../panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/350_400_stage2_padded/'
warped_data_dir = '../../../../panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/350_400_stage2_warped_64x64/'
frac_map_data_dir = '../../../../panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/350_400_stage2_warped_64x64_frac_map/'
timeseries_data_dir = '../../../../panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/350_400_stage2_padded_time_series/'

continent_info = np.load('continent_info.npy')
label_info = np.load('all_IDs_labels_realsat.npy')

continent_dict = {0: 'Arctic',
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

if verbose:
    print('\tUnique continent labels:', np.unique(continent_info))
    print('\tUnique water-body labels:', np.unique(label_info))

''' ############################################# Get Labels #####################################################'''
print('\ngetting labels for each continent...')

def plot_label_counts(label_counts_by_continent):
    """
    Plots the counts of each label for each continent and saves the plot as an image file.
    
    Args:
        label_info (list or numpy array): List/array of labels corresponding to IDs.
        continent_info (list or numpy ai8rray): List/array of continent info corresponding to IDs.
    """
    # Create and save plots for each continent
    for continent, label_counts in label_counts_by_continent.items():
        labels = list(label_counts.keys())
        counts = list(label_counts.values())

        labels_names = list(labels_dict[l] for l in labels)

        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, counts, color='skyblue')
        plt.title(f"Label Counts for {continent_dict[continent]}")
        plt.xlabel("Labels")
        plt.ylabel("Counts")
        plt.xticks(labels)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotate counts above each bar
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # X-coordinate of the bar center
                bar.get_height(),  # Y-coordinate (top of the bar)
                str(count),  # Text to display
                ha='center',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                fontsize=10,  # Font size
                color='black'  # Text color
            )

        # Save plot to file with the name of the continent
        filename = f"{continent}_label_counts.png"
        plt.savefig(os.path.join('figures', filename))
        plt.close()  # Close the figure to free memory
        print(f"Saved plot for {continent} as {filename}")

        # # Print the results
        if verbose:
            print(f"Continent {continent}:")
            for label, count in label_counts.items():
                print(f"  Label {label}: {count}")

def plot_all_continents(label_counts_by_continent):
    """
    Plots the counts of each label for all continents in a single grouped bar chart.
    
    Args:
        label_info (list or numpy array): List/array of labels corresponding to IDs.
        continent_info (list or numpy array): List/array of continent info corresponding to IDs.
    """

    # Prepare data for plotting
    continents = list(label_counts_by_continent.keys())
    all_labels = sorted(set(label for counts in label_counts_by_continent.values() for label in counts))

    # Create a matrix of counts for each label and continent
    data = np.zeros((len(continents), len(all_labels)))
    for i, continent in enumerate(continents):
        for j, label in enumerate(all_labels):
            data[i, j] = label_counts_by_continent[continent].get(label, 0)

    # Plot grouped bar chart
    x = np.arange(len(all_labels))  # Label positions
    width = 0.8 / len(continents)  # Bar width for each continent

    plt.figure(figsize=(10, 6))
    for i, continent in enumerate(continents):
        plt.bar(
            x + i * width,  # Adjust position for each continent
            data[i],        # Heights of bars
            width,          # Width of bars
            label=continent_dict[continent]
        )
        # # Add counts above each bar
        # for j, count in enumerate(data[i]):
        #     if count > 0:
        #         plt.text(
        #             x[j] + i * width, count + 0.1, str(int(count)),
        #             ha='center', va='bottom', fontsize=8
        #         )

    # Add labels and legend
    plt.title("Label Counts by Continent")
    plt.xlabel("Labels")
    plt.ylabel("Counts")
    plt.xticks(x + (width * (len(continents) - 1)) / 2, all_labels)  # Adjust x-tick positions
    plt.legend(title="Continent")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save and show plot
    plt.tight_layout()
    plt.savefig("figures/all_continents_label_counts.png")
    # plt.show()
    print("Saved plot as all_continents_label_counts.png")


if __name__ == 'main':
    # Organize the label counts by continent
    label_counts_by_continent = defaultdict(lambda: defaultdict(int))
    for ID in range(len(label_info)):
        label = label_info[ID]
        continent = continent_info[ID]
        label_counts_by_continent[continent][label] += 1

    plot_label_counts(label_counts_by_continent=label_counts_by_continent)
    plot_all_continents(label_counts_by_continent=label_counts_by_continent)