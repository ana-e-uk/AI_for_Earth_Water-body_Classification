import geopandas as gpd

# Path to the shapefile
shapefile_path = "/panfs/jay/groups/32/kumarv/pravirat/AI4EARTH/REALSAT/shape_files/ReaLSAT_351_400_only.shp"
gdf = gpd.read_file(shapefile_path)

# Filter for North America water bodies
north_america_gdf = gdf[gdf['CONTINENT'] == 2]  # Assuming '2' corresponds to North America
print(north_america_gdf.geometry.geom_type.value_counts())
# Extract first and last points of each geometry
def get_first_points(geometry):
    coords = list(geometry.coords) if geometry.geom_type == 'LineString' else list(geometry.exterior.coords)
    first_point = coords[0]
    return first_point

north_america_gdf['first_point'] = north_america_gdf.geometry.apply(lambda geom: get_first_points(geom))

# Split first and last points into longitude and latitude
north_america_gdf['min_lon'] = north_america_gdf['first_point'].apply(lambda point: point[0])
north_america_gdf['min_lat'] = north_america_gdf['first_point'].apply(lambda point: point[1])

# Select required columns
selected_columns = ['ID', 'CONTINENT', 'min_lon', 'min_lat']
new_gdf = north_america_gdf[selected_columns]
# print(new_gdf.type)
# Save to a new shapefile
output_shapefile_path = "/users/5/husse408/AI_EARTH/AI_for_Earth_Water-body_Classification/data/ShapeFiles/NorthAmericaWaterBodies_FirstPoints.csv"
new_gdf.to_csv(output_shapefile_path,index=False)
print(f"New shapefile saved to {output_shapefile_path}")