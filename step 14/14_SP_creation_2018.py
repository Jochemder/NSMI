# ------------------------------------------- file 14_SP_creation_2018.py -------------------------------------------
import os
import pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import geopandas as gpd

# Define the columns for age groups, household composition, and gender
age_columns = ['0_15_yes', '15_25_yes', '25_45_yes', '45_65_yes', '65_plus_yes']
household_columns = ['onepersonhousehold_yes', 'householdwithchildren_yes', 'householdwithoutchildren_yes']
gender_column = 'Gender_male'

# Load the synthetic population dataset
synthetic_population_path = 'output/13_prepare_synthetic_population_2018.pkl'
synthetic_population = pd.read_pickle(synthetic_population_path)

# Load the neighborhood dataset
neighborhoods = gpd.read_file('output/7_OMD_pt_2018.gpkg')

# Print success message
print('neighborhoods loaded')

# Define the percentage columns in the neighborhood dataset
age_percentage_columns = ['percentage0to15years', 'percentage15to25years', 'percentage25to45years', 'percentage45to65years', 'percentage65yearsorolder']
household_percentage_columns = ['percentageonepersonhouseholds', 'percentagehouseholdswithoutchildren', 'percentagehouseholdswithchildren']
gender_percentage_column = 'percentagemen'

# Create a directory to save the synthetic populations
output_dir = 'output/synthetic_population_2018'
os.makedirs(output_dir, exist_ok=True)

# Define a tolerance level for deviations
tolerance = 0.05  # 5% tolerance

# Function to create a synthetic population for a neighborhood
def create_synthetic_population(neighborhood, synthetic_population, age_columns, household_columns, gender_column, age_percentage_columns, household_percentage_columns, gender_percentage_column):
    # Shuffle the synthetic population dataset
    synthetic_population_shuffled = synthetic_population.sample(frac=1).reset_index(drop=True)
    
    # Get the target percentages for the neighborhood
    target_percentages = {
        'age': neighborhood[age_percentage_columns].values / 100,
        'household': neighborhood[household_percentage_columns].values / 100,
        'gender': neighborhood[gender_percentage_column] / 100
    }
    
    # Initialize an empty DataFrame for the synthetic population
    synthetic_pop = pd.DataFrame()
    
    # Sample individuals to match the target percentages
    for hh_col, hh_pct in zip(household_columns, target_percentages['household']):
        subset_hh = synthetic_population_shuffled[synthetic_population_shuffled[hh_col] == 1]
        if subset_hh.empty:
            continue

        # Print the neighborhood code for debugging
        print(f"Creating synthetic population for neighborhood code: {neighborhood['neighborhoodcode']}")
    
        for age_col, age_pct in zip(age_columns, target_percentages['age']):
            subset_age = subset_hh[subset_hh[age_col] == 1]
            if subset_age.empty:
                continue

            for gender_val in [0, 1]:
                gender_pct = target_percentages['gender'] if gender_val == 1 else 1 - target_percentages['gender']
                subset_gender = subset_age[subset_age[gender_column] == gender_val]
                if subset_gender.empty:
                    continue

                n_samples = int(len(synthetic_population) * hh_pct * age_pct * gender_pct)
                if n_samples > 0:
                    sampled = subset_gender.sample(n=n_samples, replace=True)
                    synthetic_pop = pd.concat([synthetic_pop, sampled])

    # Allow for some deviation by adjusting the sample size within the tolerance level
    actual_counts = synthetic_pop.shape[0]
    target_counts = int(len(synthetic_population) * sum(target_percentages['household']) * sum(target_percentages['age']) * target_percentages['gender'])
    if abs(actual_counts - target_counts) / target_counts > tolerance:
        adjustment_factor = target_counts / actual_counts
        synthetic_pop = synthetic_pop.sample(frac=adjustment_factor, replace=True)

    return synthetic_pop

# Function to process a single neighborhood
def process_neighborhood(neighborhood, synthetic_population, age_columns, household_columns, gender_column, age_percentage_columns, household_percentage_columns, gender_percentage_column, output_dir):
    # Set a unique random seed for each neighborhood
    np.random.seed(hash(f"{neighborhood['neighborhoodcode']}") % (2**32))
    
    print(f"Processing neighborhood: {neighborhood['neighborhoodcode']}")
    synthetic_pop = create_synthetic_population(neighborhood, synthetic_population, age_columns, household_columns, gender_column, age_percentage_columns, household_percentage_columns, gender_percentage_column)
    synthetic_pop['neighborhood_code'] = neighborhood['neighborhoodcode']
    
    # Save the synthetic population to a pickle file
    output_file_path = os.path.join(output_dir, f"{neighborhood['neighborhoodcode']}.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump(synthetic_pop, f)
    print(f'Synthetic population saved to {output_file_path}')

# Function to chunk the neighborhoods list
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

# Use multiprocessing to process neighborhoods in parallel
if __name__ == '__main__':
    num_cores = cpu_count()
    neighborhoods_list = [neighborhood for idx, neighborhood in neighborhoods.iterrows()]
    
    print(f"Starting processing with {num_cores} cores...")
    with Pool(num_cores) as pool:
        pool.starmap(process_neighborhood, [(neighborhood, synthetic_population, age_columns, household_columns, gender_column, age_percentage_columns, household_percentage_columns, gender_percentage_column, output_dir) for neighborhood in neighborhoods_list])
    print("Processing complete.")