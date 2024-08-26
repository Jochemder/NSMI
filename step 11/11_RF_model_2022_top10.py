# ---------------------------------------- File 11_RF_model_2022_top10 ---------------------------------------- #

# Importing necessary libraries for this script
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pickle

# ---------------------------------------- running the random forest ---------------------------------------- #
script_dir = os.path.dirname(__file__)

# Load the data
print('Loading the data...')
data_2022 = pd.read_pickle("output/8_merge_mobility_BE_2022.pkl")

print('Reading the top 10 features from the text file...')
# Read the top 10 feature names from the text file
with open("output/RF_2022/metrics/top_6_features_2022.txt", "r") as file:
    top_10_features = file.read().splitlines()
print('Top 10 features read successfully!')

# Ensure the features are in the DataFrame columns
top_10_features = [feature for feature in top_10_features if feature in data_2022.columns]

# Select only the top 10 features for X
X = data_2022[top_10_features].values
# Dependent variable remains the same
y = data_2022['Trip_transportation_type'].values

# Instead of mapping column names to index numbers,
# map index numbers to column names for direct access
index_to_feature_mapping = {i: feature for i, feature in enumerate(top_10_features)}

# Save the adjusted mapping to a file
with open("output/RF_2022_finalrun_top10/feature_index_mapping.pkl", "wb") as f:
    pickle.dump(index_to_feature_mapping, f)

# Define class names
CLASS_NAMES = ['Car', 'public transport', 'Bike', 'Walk']

# Step 1: Parameter tuning with 250 trees on 25% of the data
print('Performing parameter tuning with 250 trees on 25% of the data...')

# Split the data into training and test sets (25% of the data) with stratification
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.75, random_state=45, stratify=y)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=45)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# print length of X_train_resampled
print(len(X_train_resampled))
len_x_train_resampled = len(X_train_resampled)

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [2000],
    'max_depth': [10],
    'min_samples_split': [int(0.01*len_x_train_resampled)], 
    'min_samples_leaf': [int(0.01*len_x_train_resampled)], 
    'max_features': ['sqrt'],
    'bootstrap': [True]
}

# Create a RandomForestClassifier object
rf = RandomForestClassifier()

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=45)

# Fit the RandomizedSearchCV object to the resampled data
random_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters
best_params = random_search.best_params_

# Print the best parameters and save them to a txt file
print('Best parameters:')
print(best_params)
loc_txt = os.path.join(script_dir, 'output/RF_2022_finalrun_top10/best_params.txt')
with open(loc_txt, 'w') as f:
    f.write(str(best_params))

# Step 2: Estimate the RF model with 1500 trees on 90% of the data with the best parameters from step 1
# Split the data into training and test sets (90% of the data) with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=45, stratify=y)

# Apply SMOTE to balance the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print('Fitting the RF model with the best parameters to the data...')
# Create a RandomForestClassifier object with the best parameters
rf = RandomForestClassifier(**best_params, n_jobs=-1, verbose=2)

# Fit the RandomForestClassifier object to the resampled training data
rf.fit(X_train_resampled, y_train_resampled)

# Print performance metrics
print('Performance metrics:')
print(f'Training accuracy: {rf.score(X_train_resampled, y_train_resampled)}')
print(f'Test accuracy: {rf.score(X_test, y_test)}')
print(f'F1 score: {f1_score(y_test, rf.predict(X_test), average="weighted")}')
print(f'Precision: {precision_score(y_test, rf.predict(X_test), average="weighted")}')
print(f'Recall: {recall_score(y_test, rf.predict(X_test), average="weighted")}')

# Make confusion matrix and store to file
# Before calculating the confusion matrix, predictions need to be generated using the test data
predictions = rf.predict(X_test)
# Calculate the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Calculate percentages for the confusion matrix
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Define file paths for saving the confusion matrices
loc_confusion_matrix_absolute = os.path.join(script_dir, 'output/RF_2022_finalrun_top10/confusion_matrix_absolute_2022.png')
loc_confusion_matrix_percentage = os.path.join(script_dir, 'output/RF_2022_finalrun_top10/confusion_matrix_percentage_2022.png')

# Define the custom labels
custom_labels = ['car', 'public transport', 'bike', 'walk']

# Plot and save the absolute confusion matrix
fig_abs, ax_abs = plt.subplots()
disp_abs = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=custom_labels)
disp_abs.plot(ax=ax_abs, cmap=plt.cm.Blues)
ax_abs.set_title('Confusion Matrix 2022 (Absolute)')
ax_abs.set_xticklabels(ax_abs.get_xticklabels(), rotation=45, ha="right")
ax_abs.invert_xaxis()  # Reverse the x-axis
fig_abs.tight_layout()
fig_abs.savefig(loc_confusion_matrix_absolute)

# Plot and save the percentage confusion matrix
fig_pct, ax_pct = plt.subplots()
disp_pct = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=custom_labels)
disp_pct.plot(ax=ax_pct, cmap=plt.cm.Blues, values_format='.2%')
ax_pct.set_title('Confusion Matrix 2022 (Percentage)')
ax_pct.set_xticklabels(ax_pct.get_xticklabels(), rotation=45, ha="right")
ax_pct.invert_xaxis()  # Reverse the x-axis
fig_pct.tight_layout()
fig_pct.savefig(loc_confusion_matrix_percentage)

print('Storing the RF model and the training and test data to files...')
# Save the model to a file
with open('output/RF_2022_finalrun_top10/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Save the training and test data to files
with open('output/RF_2022_finalrun_top10/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('output/RF_2022_finalrun_top10/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('output/RF_2022_finalrun_top10/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('output/RF_2022_finalrun_top10/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)