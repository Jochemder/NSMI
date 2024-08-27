# ------------------------------------------- file 13_SP_preparation_2021.py -------------------------------------------
# Load the correct libraries
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Function to create new columns based on the "Age" column
def create_age_based_columns(df):
    df['0_15_yes'] = df['Age'].apply(lambda age: 1 if age < 15 else 0)
    df['15_25_yes'] = df['Age'].apply(lambda age: 1 if 15 <= age < 25 else 0)
    df['25_45_yes'] = df['Age'].apply(lambda age: 1 if 25 <= age < 45 else 0)
    df['45_65_yes'] = df['Age'].apply(lambda age: 1 if 45 <= age < 65 else 0)
    df['65_plus_yes'] = df['Age'].apply(lambda age: 1 if age >= 65 else 0)
    return df

# Function to create new columns based on household information
def create_household_based_columns(df):
    df['onepersonhousehold_yes'] = df['HHC_oneperson'].apply(lambda x: 1 if x == 1 else 0)
    df['householdwithchildren_yes'] = df.apply(lambda row: 1 if row['HHC_couple_with_children'] == 1 or row['HHC_oneperson_with_children'] == 1 else 0, axis=1)
    df['householdwithoutchildren_yes'] = df['HHC_couple'].apply(lambda x: 1 if x == 1 else 0)
    return df

# Path to the CSV file
file_path = 'output/1_ODIN_2021.csv'

# Read the CSV file into a pandas DataFrame with the correct parameters
target_percentage_2021 = pd.read_csv(file_path, encoding='ISO-8859-1', sep=';')

# Check if 'Age' column exists
if 'Age' not in target_percentage_2021.columns:
    raise KeyError("The 'Age' column is missing from the DataFrame")

# Split the DataFrame into chunks
num_chunks = 45
chunks = np.array_split(target_percentage_2021, num_chunks)

# Process each chunk in parallel for age-based columns
with ProcessPoolExecutor(max_workers=45) as executor:
    age_results = list(executor.map(create_age_based_columns, chunks))

# Concatenate the results back into a single DataFrame
target_percentage_2021 = pd.concat(age_results)

# Split the DataFrame into chunks again for household-based columns
chunks = np.array_split(target_percentage_2021, num_chunks)

# Process each chunk in parallel for household-based columns
with ProcessPoolExecutor(max_workers=45) as executor:
    household_results = list(executor.map(create_household_based_columns, chunks))

# Concatenate the results back into a single DataFrame
target_percentage_2021 = pd.concat(household_results)

# Save the changed dataset to a new pickle file
output_path = 'output/13_prepare_synthetic_population_2021.pkl'
target_percentage_2021.to_pickle(output_path)

# Display the first few rows to verify the new columns
print(target_percentage_2021.head())