# ------------------------------------------- 15_MS_2018 -------------------------------------------
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle

# Define distance intervals
short_distance_range = (0, 5)
medium_distance_range = (5, 20)
long_distance_range = (20, 100)  # Assuming 100 as an upper bound for long distances

# Load the Random Forest model
model_path = 'output/RF_2018_finalrun_top10/random_forest_model.pkl'
with open(model_path, 'rb') as model_file:
    random_forest_model = pickle.load(model_file)

# Function to calculate percentages for Moti_sparetime and Moti_work for each distance range
def calculate_percentages(df, distance_range):
    condition = (df['Trip_distance'] >= distance_range[0]) & (df['Trip_distance'] < distance_range[1])
    subset = df[condition]
    moti_sparetime_percentage = subset['Moti_sparetime'].mean()
    moti_work_percentage = subset['Moti_work'].mean()
    return moti_sparetime_percentage, moti_work_percentage

# Function to modify DataFrame
def modify_df(df, distance_range, moti_sparetime_percentage, moti_work_percentage):
    modified_df = df.copy()
    modified_df['Trip_distance'] = np.random.randint(distance_range[0], distance_range[1], size=len(df))
    
    # Ensure Moti_sparetime and Moti_work are not both 1
    moti_sparetime = np.random.choice([0, 1], size=len(df), p=[1 - moti_sparetime_percentage, moti_sparetime_percentage])
    moti_work = np.random.choice([0, 1], size=len(df), p=[1 - moti_work_percentage, moti_work_percentage])
    
    # Resolve conflicts where both are 1
    conflict_indices = np.where((moti_sparetime == 1) & (moti_work == 1))[0]
    for idx in conflict_indices:
        if np.random.rand() < 0.5:
            moti_sparetime[idx] = 0
        else:
            moti_work[idx] = 0
    
    modified_df['Moti_sparetime'] = moti_sparetime
    modified_df['Moti_work'] = moti_work
    return modified_df

# Function to predict mode choice and return results
def predict_mode_choice(data):
    try:
        # Predict the mode choice using the Random Forest model
        predictions = random_forest_model.predict(data)

        # Add the predictions to the DataFrame
        data['mode_choice'] = predictions

        # Calculate the percentage for each mode choice
        mode_choice_counts = data['mode_choice'].value_counts(normalize=True) * 100
        mode_choice_percentages = mode_choice_counts.rename(index={1: 'Car', 2: 'public transport', 3: 'Bike', 4: 'Walk'})

        return mode_choice_percentages
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Function to process a single pickle file for a specific distance range
def process_pickle_file(args):
    pickle_file, distance_range, moti_sparetime_percentage, moti_work_percentage = args
    try:
        # Extract the neighborhood code from the filename
        neighborhood_code = os.path.basename(pickle_file).replace('.pkl', '')

        # Step 1: Load the list of feature names
        with open("output/RF_2018/metrics/top_6_features_2018.txt", "r") as file:
            top_features = file.read().splitlines()

        # Step 2: Load the DataFrame from the pickle file
        df = pd.read_pickle(pickle_file)

        # Step 3: Identify columns in the feature list that are not in the DataFrame
        missing_columns = [col for col in top_features if col not in df.columns]

        # Step 4: Load the additional data from the GeoPackage file
        gdf = gpd.read_file("output/7_OMD_pt_2018.gpkg")

        # Filter the GeoDataFrame for the specific neighborhood code
        neighborhood_data = gdf[gdf['neighborhoodcode'] == neighborhood_code]

        # Step 5: Add the missing columns to the DataFrame
        for col in missing_columns:
            if col in neighborhood_data.columns:
                df[col] = neighborhood_data[col].values[0]

        # Step 6: Filter the DataFrame to keep only the columns that are in the list of feature names
        filtered_df = df[top_features]

        print(f"Filtered DataFrame for {neighborhood_code}.")

        # Modify DataFrame for the specified distance interval
        modified_df = modify_df(filtered_df, distance_range, moti_sparetime_percentage, moti_work_percentage)

        # Predict mode choice and get percentages
        mode_choice_percentages = predict_mode_choice(modified_df)

        # Add neighborhood code to the result
        mode_choice_percentages.name = neighborhood_code

        return mode_choice_percentages
    except Exception as e:
        print(f"Error processing {pickle_file}: {e}")
        return None

# Main function to process all pickle files in the directory for a specific distance range
def process_all_files(pickle_dir, distance_range):
    pickle_files = [os.path.join(pickle_dir, f) for f in os.listdir(pickle_dir) if f.endswith('.pkl')]

    print(f"Found {len(pickle_files)} pickle files to process in {pickle_dir}.")

    # Load the original data from the pickle file
    df_original = pd.read_pickle("output/13_prepare_synthetic_population_2018.pkl")

    # Calculate percentages for Moti_sparetime and Moti_work for the specified distance range
    moti_sparetime_percentage, moti_work_percentage = calculate_percentages(df_original, distance_range)

    # Prepare arguments for multiprocessing
    args = [(pickle_file, distance_range, moti_sparetime_percentage, moti_work_percentage) for pickle_file in pickle_files]

    # Use multiprocessing to process files in parallel
    num_cores = os.cpu_count()
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_pickle_file, args), total=len(pickle_files)))

    # Filter out None results and combine all percentages into a single DataFrame
    all_percentages = [result for result in results if result is not None]
    result_df = pd.DataFrame(all_percentages)

    return result_df

if __name__ == "__main__":
    # Process short distance files
    short_df = process_all_files('output/synthetic_population_2018/', short_distance_range)

    # Process medium distance files
    medium_df = process_all_files('output/synthetic_population_2018/', medium_distance_range)

    # Process long distance files
    long_df = process_all_files('output/synthetic_population_2018/', long_distance_range)

    # Save the aggregated results to pickle files
    short_df.to_pickle('output/MS_2018/MS_prediction_short.pkl')
    medium_df.to_pickle('output/MS_2018/MS_prediction_medium.pkl')
    long_df.to_pickle('output/MS_2018/MS_prediction_long.pkl')

    print("Aggregated results saved to output/MS_2018/")