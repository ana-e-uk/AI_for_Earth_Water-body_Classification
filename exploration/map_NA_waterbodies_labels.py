import numpy as np
import pandas as pd

# Load the .npy file containing the labels
labels_file_path = '/users/5/husse408/AI_EARTH/AI_for_Earth_Water-body_Classification/all_IDs_labels_realsat.npy'
all_IDs_labels = np.load(labels_file_path, allow_pickle=True)
print(all_IDs_labels[409811])
# # Load the .csv file containing the waterbody IDs and other information
csv_file_path = '/users/5/husse408/AI_EARTH/AI_for_Earth_Water-body_Classification/data/ShapeFiles/NorthAmericaWaterBodies_FirstPoints.csv'  # Replace with the actual path to your CSV
df = pd.read_csv(csv_file_path)

# Create a new column 'LABEL' by looking up the label for each ID
df['LABEL'] = df['ID'].apply(lambda x: all_IDs_labels[x] )

# Display the updated DataFrame
print(df.head())
print(df['LABEL'].value_counts())
# Optionally, save the DataFrame to a new CSV file
output_file_path = '/users/5/husse408/AI_EARTH/AI_for_Earth_Water-body_Classification/data/ShapeFiles/NorthAmericaWaterBodies_FirstPoints_WithLabels.csv'  # Replace with the desired output path
df.to_csv(output_file_path, index=False)