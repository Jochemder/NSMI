# ---------------------------------------- File 8_merge_mobility_BE_2018 ---------------------------------------- #
# This is file is the eighth step in creating the NSMI.
# This script will locate file 1_ODIN_2018.csv and file 7_OMD_pt_2018.gpkg and merge them.

# Importing necessary libraries for this script
import pandas as pd
import geopandas as gpd
import os
import multiprocessing as mp

# ---------------------------------------- Part 1 ---------------------------------------- #
print("Step 1: Opening the Built Environment file for 2018.")
# This piece of code opens the Built Environment file for 2018 that was created in step 7.
gdf_BE_2018 = gpd.read_file("output/7_OMD_pt_2018.gpkg")

# Display the GeoDataFrame
print(gdf_BE_2018.head(1))

print("Step 2: Locating the file that contains the geometries of the PC4 areas in the Netherlands for 2018.")
# Locate the file that contains the geometries of the PC4 areas in the Netherlands for 2018.
# Set the SHAPE_RESTORE_SHX configuration option to YES
import os
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Read the shapefile
gdf_PC4_2018 = gpd.read_file('output/CBS_PC4_2018_v3.shp')

# Print the column names and the first few rows of gdf_PC4_2018 to see what they are
print("Columns in gdf_PC4_2018:", gdf_PC4_2018.columns)
print(gdf_PC4_2018.head())

print("Step 3: Deleting unnecessary columns.")
# This piece of code deletes all rows except from the columns that contain the geometries of the PC4 areas and the PC4 code.

# Print the column names of gdf_PC4_2018 to see what they are
print(gdf_PC4_2018.columns)

# Copy the DataFrame and select only the 'postcode4' and 'geometry' columns
gdf_PC4_2018_cleaned = gdf_PC4_2018[['PC4', 'geometry']].copy()

# Rename the 'postcode4' column to 'PC4_code'
gdf_PC4_2018_cleaned = gdf_PC4_2018_cleaned.rename(columns={'PC4': 'PC4_code'})
print(gdf_PC4_2018_cleaned)

print("Step 4: Printing the CRS of both dataframes.")
# Print the CRS of both dataframes.
print("CRS of gdf_BE_2018: ", gdf_BE_2018.crs)
print("CRS of gdf_PC4_2018_cleaned: ", gdf_PC4_2018_cleaned.crs)

print("Step 5: Setting the CRS of each GeoDataFrame to EPSG:28992.")
# This code sets the CRS of each GeoDataFrame to EPSG:28992. The EPSG:28992 CRS is a common CRS used for the Netherlands.
# It's important to ensure that all GeoDataFrames use the same CRS when performing spatial operations between them.
gdf_BE_2018 = gdf_BE_2018.to_crs(epsg=28992)
gdf_PC4_2018_cleaned = gdf_PC4_2018_cleaned.to_crs(epsg=28992)

print("Step 6: Performing a spatial join between gdf_BE_2018 and gdf_PC4_2018_cleaned.")
# This code first performs a spatial join between gdf_BE_2018 and gdf_PC4_2018_cleaned to find which neighborhoods intersect with which PC4 areas.
# It then calculates the intersection area between each neighborhood and PC4 area, and uses this to calculate the percentage of each neighborhood's area that falls within each PC4 area.
# The resulting DataFrame gdf_BE_2018 will have a new column 'PC4_code' indicating the PC4 area each neighborhood is in, and a new column 'percent_PC4' indicating the percentage of the neighborhood's area that falls within this PC4 area.
# If a neighborhood is within multiple PC4 areas, it will have multiple rows in gdf_BE_2018, one for each PC4 area it is in.

# Perform a spatial join between gdf_BE_2018 and gdf_PC4_2018_cleaned
joined = gpd.sjoin(gdf_BE_2018, gdf_PC4_2018_cleaned, how='inner', predicate='intersects')

print("Step 7: Calculating the intersection area between each neighborhood and PC4 area.")
# Calculate the intersection area between each neighborhood and PC4 area
def calculate_intersection_area(row):
    if isinstance(row['index_right'], list):
        return sum(row['geometry'].intersection(gdf_PC4_2018_cleaned.loc[idx, 'geometry']).area for idx in row['index_right'])
    else:
        return row['geometry'].intersection(gdf_PC4_2018_cleaned.loc[row['index_right'], 'geometry']).area

joined['intersection_area'] = joined.apply(calculate_intersection_area, axis=1)

print("Step 8: Calculating the percentage of each neighborhood's area that falls within each PC4 area.")
# Calculate the percentage of each neighborhood's area that falls within each PC4 area
joined['percent_PC4'] = joined['intersection_area'] / joined['geometry'].area * 100

print("Step 9: Renaming and dropping unnecessary columns.")
# Rename the 'PC4_code' column
joined = joined.rename(columns={'PC4_code_right': 'PC4_code'})

# Drop the unnecessary columns
joined = joined.drop(columns=['index_right', 'intersection_area'])

# Update gdf_BE_2018
gdf_BE_2018 = joined

# Show whether the code has worked
print(gdf_BE_2018)

# ---------------------------------------- Part 2 ---------------------------------------- #
print("Step 10: Opening the ODiN mobility dataset for 2018.")
# This part opens the file 1_ODIN_2018.csv.

# Locate the ODiN mobility dataset for 2018.
df_ODIN_2018 = pd.read_csv("output/1_ODIN_2018.csv", encoding='ISO-8859-1', sep=';')

# Display the ODiN dataframe
print(df_ODIN_2018)

# ---------------------------------------- Part 3 ---------------------------------------- #
print("Step 11: Displaying datatypes of required columns.")
# This part merges the ODiN (mobility) dataset and the Built Environment (spatial neighborhood data) datasets

# Display which datatypes the columns required for are.
print("Data type of 'Startingpostalcode' in df_ODIN_2018: ", df_ODIN_2018['Starting_postalcode'].dtypes)
print("Data type of 'PC4_code' in gdf_BE_2018: ", gdf_BE_2018['PC4_code'].dtypes)

print("Step 12: Converting columns to object and formatting postal codes.")
# Convert both columns to object. Normally, this would provide problems since the first zeros would be deleted. However, there are no Dutch postal codes starting with 0.
df_ODIN_2018 = df_ODIN_2018.dropna(subset=['Starting_postalcode'])
df_ODIN_2018['Starting_postalcode'] = df_ODIN_2018['Starting_postalcode'].apply(lambda x: f"{int(x):04d}")

gdf_BE_2018 = gdf_BE_2018.dropna(subset=['PC4_code'])
gdf_BE_2018['PC4_code'] = gdf_BE_2018['PC4_code'].apply(lambda x: f"{int(x):04d}")

# Show what the datatypes are
print("Data type of 'Startingpostalcode' in df_ODIN_2018: ", df_ODIN_2018['Starting_postalcode'].dtypes)
print("Data type of 'PC4_code' in gdf_BE_2018: ", gdf_BE_2018['PC4_code'].dtypes)

print(df_ODIN_2018.head(1))
print(gdf_BE_2018.head(1))

print("Step 13: Merging the datasets based on Starting_postalcode and PC4_code.")
# This code will create a new dataframe df_ODIN_2018_merged which is a merged version of df_ODIN_2018 and gdf_BE_2018.
# The how='left' parameter ensures that all rows from df_ODIN_2018 are included in the merged dataframe, and they are duplicated if there are multiple matching rows in gdf_BE_2018.
# Merge the datasets based on Starting_postalcode and PC4_code with inner join to drop rows without a match
# Function to merge a chunk of the DataFrame
def merge_chunk(chunk, gdf_BE_2018):
    return chunk.merge(gdf_BE_2018, left_on='Starting_postalcode', right_on='PC4_code', how='inner')

# Function to split DataFrame into chunks
def split_dataframe(df, chunk_size):
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    return chunks

# Apply the function to merge DataFrames using multiprocessing
if __name__ == '__main__':
    
    # Split the DataFrame into chunks
    chunk_size = len(df_ODIN_2018) // mp.cpu_count()
    df_chunks = split_dataframe(df_ODIN_2018, chunk_size)
    
    # Use multiprocessing to merge chunks
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(merge_chunk, [(chunk, gdf_BE_2018) for chunk in df_chunks])
    
    # Concatenate the results
    df_ODIN_2018_merged = pd.concat(results, ignore_index=True)
    
    # Display the resulting dataframe
    print(df_ODIN_2018_merged)
    print("Step 13: Merging the datasets based on Starting_postalcode and PC4_code finished.")

print("Step 14: Duplicating rows based on percent_PC4.")
# Function to duplicate rows based on percent_PC4
def duplicate_rows(row):
    num_duplicates = int(round(row['percent_PC4'] / 10))
    return pd.DataFrame([row] * num_duplicates)

# Function to apply duplication in parallel
def parallel_duplicate_rows(df):
    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.map(duplicate_rows, [row for _, row in df.iterrows()])
    return pd.concat(result, ignore_index=True)

# Apply the function to duplicate rows using multiprocessing
if __name__ == '__main__':
    df_ODIN_2018_duplicated = parallel_duplicate_rows(df_ODIN_2018_merged)

    # Display the resulting dataframe
    print(df_ODIN_2018_duplicated)

print("Step 15: Calculating the number of columns that could not be merged.")
# This code first calculates the difference between the columns of df_ODIN_2018 and df_ODIN_2018_merged, which gives the columns that are in df_ODIN_2018 but not in df_ODIN_2018_merged.
# Then it calculates the number of these columns and prints it.
unmerged_columns = df_ODIN_2018.columns.difference(df_ODIN_2018_merged.columns)
num_unmerged_columns = len(unmerged_columns)
print(f"Number of columns that could not be merged: {num_unmerged_columns}")

print("Step 16: Checking if the merging was successful.")
# Check if the merging was successful
pd.set_option('display.max_columns', None)

# Now when you display a DataFrame, all columns will be shown
print(df_ODIN_2018_merged)

print("Step 17: Dropping unnecessary columns.")
# Drop unnecessary columns.

# List of columns to drop
columns_to_drop = ['Starting_postalcode', 'neighborhoodcode', 'geometry', 'PC4_code', 'percent_PC4']

# Drop the columns
df_ODIN_2018_merged = df_ODIN_2018_merged.drop(columns=columns_to_drop)

# Now when you display a DataFrame, all columns will be shown
print(df_ODIN_2018_merged)

print("Step 18: Saving the dataframe to a CSV file.")
# As a last step, save the dataframe to a CSV file.
df_ODIN_2018_merged.to_csv("output/8_merge_mobility_BE_2018_new.csv", index=False, encoding='ISO-8859-1', sep=';')
print("CSV file has been saved.")

print("Step 19: Saving the dataframe to a pickle file.")
# As a last step, save the dataframe to a pickle file.
df_ODIN_2018_merged.to_pickle("output/8_merge_mobility_BE_2018_new.pkl")
print("Pickle file has been saved.")

print(f"Number of rows: {df_ODIN_2018_merged.shape[0]}")