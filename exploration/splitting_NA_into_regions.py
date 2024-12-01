import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
# Assume the CSV has columns: 'min_lat', 'min_lon', 'LABEL'
df = pd.read_csv('/users/5/husse408/AI_EARTH/AI_for_Earth_Water-body_Classification/data/ShapeFiles/NorthAmericaWaterBodies_FirstPoints_WithLabels.csv')
filtered_df = df[df['LABEL'].isin([1, 3, 4, 5])]
df = filtered_df
# Find the spatial division (median longitude in this example)
# boundary = df['min_lon'].median()
boundary = -92.32695843970778

# Assign regions based on longitude
region1 = df[df['min_lon'] <= boundary]
region2 = df[df['min_lon'] > boundary]

# Ensure similar distribution of classes in both regions
def balance_classes(region1, region2):
    for lbl in df['LABEL'].unique():
        count1 = len(region1[region1['LABEL'] == lbl])
        count2 = len(region2[region2['LABEL'] == lbl])
        diff = abs(count1 - count2)
        
        if count1 > count2:
            # Move points from region1 to region2
            to_move = region1[(region1['LABEL'] == lbl)].iloc[:diff]
            region2 = pd.concat([region2, to_move])
            region1 = region1.drop(to_move.index)
        elif count2 > count1:
            # Move points from region2 to region1
            to_move = region2[(region2['LABEL'] == lbl)].iloc[:diff]
            region1 = pd.concat([region1, to_move])
            region2 = region2.drop(to_move.index)
    
    return region1.reset_index(drop=True), region2.reset_index(drop=True)

region1, region2 = balance_classes(region1, region2)

# Plot the two regions
plt.figure(figsize=(10, 6))

# Region 1
plt.scatter(region1['min_lon'], region1['min_lat'], c='blue', label='Region 1', alpha=0.6, s=50, edgecolor='k')

# Region 2
plt.scatter(region2['min_lon'], region2['min_lat'], c='orange', label='Region 2', alpha=0.6, s=50, edgecolor='k')

# Add plot features
plt.axvline(boundary, color='red', linestyle='--', label='Boundary')  # Add a vertical line for the boundary
plt.xlabel("Longitude (min_lon)")
plt.ylabel("Latitude (min_lat)")
plt.title("Non-overlapping Regions with Balanced Farm Class Distribution")
plt.legend()
plt.grid(True)

# Save the plot
output_dir = '/users/5/husse408/AI_EARTH/AI_for_Earth_Water-body_Classification/data/figures'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_file = os.path.join(output_dir, '2_regions_median_of_label_1.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')

plt.show()

# Print the distribution
print("Region 1 Class Distribution:")
print(region1['LABEL'].value_counts())
print("\nRegion 2 Class Distribution:")
print(region2['LABEL'].value_counts())
