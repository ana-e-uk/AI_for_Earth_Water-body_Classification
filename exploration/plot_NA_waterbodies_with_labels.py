import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
# Load the CSV data
df = pd.read_csv('/users/5/husse408/AI_EARTH/AI_for_Earth_Water-body_Classification/data/ShapeFiles/NorthAmericaWaterBodies_FirstPoints_WithLabels.csv')
# Define the label-to-name mapping
label_to_name = {
    0: "Unknown",
    1: "Farm",
    2: "Reservoir",
    3: "River",
    4: "Stable Lake",
    5: "Seasonal Lake",
    6: "Highly Seasonal",
    7: "Ephemeral",
    8: "River runoff/oxbow"
}
# # Define a colormap for the classes
classes = df['LABEL'].unique()
colors = plt.cm.tab10(range(len(classes)))  # Use a predefined colormap

# Map each class to a specific color
class_color_map = {cls: colors[i] for i, cls in enumerate(classes)}

# Create the plot
plt.figure(figsize=(10, 6))

for cls in classes:
    if cls in [1,3,4,5]:
        subset = df[df['LABEL'] == cls]
        label_name = label_to_name.get(cls, f"Class {cls}")  # Get the name from the mapping
        

        plt.scatter(subset['min_lon'], subset['min_lat'], label=label_name,
                    color=class_color_map[cls], s=50, alpha=0.8, edgecolor='k')

# Add plot features
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographic Locations Colored by Class")
plt.legend(title="Class")
plt.grid(True)

# Save the plot
output_dir = '/users/5/husse408/AI_EARTH/AI_for_Earth_Water-body_Classification/data/figures'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_file = os.path.join(output_dir, 'geographic_plot.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')