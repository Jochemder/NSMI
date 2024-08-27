import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool, cpu_count

# Define file paths
files = [
    'output/MS_2022/MS_prediction_short.pkl',
    'output/MS_2022/MS_prediction_medium.pkl',
    'output/MS_2022/MS_prediction_long.pkl'
]

# Define multipliers for sustainability score
multipliers = {
    'car': 0.0000,
    'public transport': 0.8121,
    'Bike': 1.0000,
    'Walk': 1.0000
}

# Function to calculate sustainability score
def calculate_sustainability_score(row):
    score = 0
    for col, multiplier in multipliers.items():
        score += row.get(col, 0) * multiplier
    return score

# Function to process each file
def process_file(file):
    print(f"Processing file: {file}")
    df = pd.read_pickle(file)
    df.reset_index(inplace=True)  # Reset index to make the first column a regular column
    print(f"Columns in {file} after loading and resetting index:", df.columns)
    
    df['sustainability_score'] = df.apply(calculate_sustainability_score, axis=1)
    print(f"Columns in {file} after adding sustainability_score:", df.columns)
    return df

# Use multiprocessing to process files
print("Starting multiprocessing to process files...")
with Pool(cpu_count()) as pool:
    dfs = pool.map(process_file, files)
print("Completed multiprocessing.")

# Calculate NSMI for each row
print("Calculating NSMI for each row...")
for df in dfs:
    df['NSMI'] = df['sustainability_score'] / 3
print("NSMI calculation completed.")

# Save updated files
print("Saving updated files...")
for file, df in zip(files, dfs):
    output_file = file.replace('.pkl', '_NSMI.pkl')
    df.to_pickle(output_file)
    print(f"Saved updated file: {output_file}")

# Load geospatial data
print("Loading geospatial data...")
gdf = gpd.read_file('output/7_OMD_pt_2022.gpkg')
print("Geospatial data loaded.")

# Print columns of dfs[0] to debug
print("Columns in dfs[0]:", dfs[0].columns)

# Merge geospatial data with NSMI data
print("Merging geospatial data with NSMI data...")
merged_gdf = gdf.merge(dfs[1][['index', 'NSMI']], left_on='neighborhoodcode', right_on='index')
print("Merge completed.")

# Define color map
cmap = LinearSegmentedColormap.from_list('nsmi_cmap', ['red', 'orange', 'green'])

# Plot the data
print("Plotting the data...")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
merged_gdf.plot(column='NSMI', cmap=cmap, legend=True, ax=ax, legend_kwds={'label': "NSMI Score", 'orientation': "horizontal"})
ax.set_title('NSMI for 2022')
plt.savefig('output/NSMI_2022_colorchange.png')
print("Plot saved as 'output/NSMI_2022_colorchange.png'.")
plt.show()