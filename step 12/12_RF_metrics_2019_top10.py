# ---------------------------------------- File 10_RF_metrics_2019 ---------------------------------------- #

# Importing necessary libraries for this script
import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, accuracy_score, classification_report, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import multiprocessing
import threading
import time
from tqdm import tqdm
import shap
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.inspection import (partial_dependence, PartialDependenceDisplay, permutation_importance)

SHAP_RECALCULATE = True
SHAP_APPROXIMATE = True
SHAP_SAMPLE_FRAC = 0.25

RANDOM_STATE = 42

CLASS_NAMES = ['Car', 'public transport', 'Bike', 'Walk']

def load_rf_data(script_dir):
    print('Loading the data...')
    X_train = pickle.load(open(os.path.join(script_dir, 'output/RF_2019_finalrun_top10/X_train.pkl'), 'rb'))
    X_test = pickle.load(open(os.path.join(script_dir, 'output/RF_2019_finalrun_top10/X_test.pkl'), 'rb'))
    y_train = pickle.load(open(os.path.join(script_dir, 'output/RF_2019_finalrun_top10/y_train.pkl'), 'rb'))
    y_test = pickle.load(open(os.path.join(script_dir, 'output/RF_2019_finalrun_top10/y_test.pkl'), 'rb'))
    rf = pickle.load(open(os.path.join(script_dir, 'output/RF_2019_finalrun_top10/random_forest_model.pkl'), 'rb'))

    print('Data loaded successfully!')
    return X_train, X_test, y_train, y_test, rf

def preprocess_rf_data(X_train_pkl, X_test_pkl, y_train_pkl, y_test_pkl):
    print('Preprocessing the data...')
    # Convert the loaded pickle objects into pandas DataFrames
    X_train = pd.DataFrame(X_train_pkl)
    X_test = pd.DataFrame(X_test_pkl)
    y_train = pd.DataFrame(y_train_pkl)
    y_test = pd.DataFrame(y_test_pkl)

    # For all dataframes, convert the column names to strings
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    y_train.columns = y_train.columns.astype(str)
    y_test.columns = y_test.columns.astype(str)

    print('Data preprocessed successfully!')
    return X_train, X_test, y_train, y_test

def generate_report(rf, X_test, y_test, CLASS_NAMES):
    print('Generating the classification report...')
    
    # Predict the test set results
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovr')
    mcc = matthews_corrcoef(y_test, y_pred)

    # Extracting macro and weighted averages
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']


    # Create a DataFrame for the metrics
    metrics = {
        'Metric': [
            'Accuracy', 
            'Precision', 
            'Recall', 
            'F1-Score', 
            'Macro-Averaged Precision', 
            'Macro-Averaged Recall', 
            'Macro-Averaged F1-Score', 
            'Weighted-Averaged Precision', 
            'Weighted-Averaged Recall', 
            'Weighted-Averaged F1-Score',
            'ROC AUC Score',
            'Matthews Correlation Coefficient',
        ],
        'Value': [
            accuracy,
            weighted_avg['precision'],
            weighted_avg['recall'],
            weighted_avg['f1-score'],
            macro_avg['precision'],
            macro_avg['recall'],
            macro_avg['f1-score'],
            weighted_avg['precision'],
            weighted_avg['recall'],
            weighted_avg['f1-score'],
            roc_auc,
            mcc,
        ]
    }
    metrics_df = pd.DataFrame(metrics)

    # Save the metrics to an .xlsx file
    metrics_df.to_excel('output/RF_2019_finalrun_top10/metrics/metrics_report.xlsx', index=False)
    
    print('Classification report generated successfully!')

def calculate_shap_values(args):
    explainer, data = args
    return explainer.shap_values(data, approximate=SHAP_APPROXIMATE)

def SHAP(rf, X_test, shap_values_file, X_test_sample):
    print('Calculating SHAP values...')

    # Create the explainer
    explainer = shap.TreeExplainer(rf)
    del rf
    
    # Split the data into chunks for each processor
    data_chunks = np.array_split(X_test_sample, multiprocessing.cpu_count())
    
    # Create a pool of processes
    with Pool() as pool:
        # Use imap for lazy evaluation with tqdm for progress updates
        results = list(tqdm(pool.imap(calculate_shap_values, [(explainer, data) for data in data_chunks]), total=len(data_chunks)))
    
    # Combine the results from each process
    shap_values = np.concatenate(results, axis=0)

    # Store to .pkl file
    with open(shap_values_file, 'wb') as f:
        pickle.dump(shap_values, f)

    print('SHAP values calculated successfully!')

    # Calculate SHAP values per class once
    num_classes = len(CLASS_NAMES)
    shap_values_per_class = [shap_values[:, :, i] for i in range(num_classes)]

    return shap_values, shap_values_per_class

def generate_beeswarm_plot(shap_values_per_class, X_test_sample, rf, feature_names):
    print('Generating the beeswarm plot...')
    
    num_classes = len(rf.classes_)  # Use the number of classes from the model

    # Generate beeswarm plots for each class
    for i, class_name in enumerate(rf.classes_):
        class_name_verbose = CLASS_NAMES[i]
        print(f"Generating SHAP beeswarm plot for class: {class_name_verbose}")
        shap.summary_plot(shap_values_per_class[i], X_test_sample, show=False, feature_names=feature_names)
        plt.title(f'SHAP Values for Class: {class_name_verbose} ({i})')
        plt.tight_layout()
        plt.savefig(f'output/RF_2019_finalrun_top10/metrics/shap_beeswarm_plot_{class_name_verbose}.png', dpi=300)
        plt.close()

def retrieve_feature_names():
    with open("output/RF_2019_finalrun_top10/feature_index_mapping.pkl", "rb") as f:
        index_to_feature_mapping = pickle.load(f)
    # Convert the mapping to a list of column names, maintaining the order based on index numbers
    feature_names = [index_to_feature_mapping[i] for i in sorted(index_to_feature_mapping)]
    return feature_names

def generate_shap_report(shap_values_per_class, rf, feature_names):
    print('Generating the SHAP report...')
    
    num_classes = len(rf.classes_)  # Use the number of classes from the model

    # Create a DataFrame to store the SHAP values
    shap_report_df = pd.DataFrame(index=feature_names, columns=CLASS_NAMES)

    # Populate the DataFrame with the SHAP values
    for i, class_name in enumerate(rf.classes_):
        class_name_verbose = CLASS_NAMES[i]
        shap_report_df[class_name_verbose] = shap_values_per_class[i].mean(axis=0)

    # Save the DataFrame to an XLSX file
    shap_report_df.to_excel('output/RF_2019_finalrun_top10/metrics/shap_report_2019.xlsx')
    print('SHAP report saved to output/RF_2019_finalrun_top10/metrics/shap_report_2019.xlsx')

def save_top_features(shap_values_per_class, feature_names, top_n):
    print('Calculating and saving top features based on SHAP values for a multiclass problem...')

    top_features_indices_set = set()

    for i, class_name in enumerate(CLASS_NAMES):
        mean_abs_shap_class = np.abs(shap_values_per_class[i]).mean(axis=0)
        
        # Get indices of the top N features for this class
        top_features_indices_class = np.argsort(-mean_abs_shap_class)[:top_n]
        
        # Add these indices to the set
        top_features_indices_set.update(top_features_indices_class)

    # Extract names of the top features
    top_features_names = [feature_names[i] for i in top_features_indices_set]

    # Ensure the output directory exists
    output_dir = 'output/RF_2019_finalrun_top10/metrics'
    os.makedirs(output_dir, exist_ok=True)

    # Save the names of the top features to a text file
    output_file_path = os.path.join(output_dir, f'top_{top_n}_features_2019.txt')
    with open(output_file_path, 'w') as file:
        for feature_name in top_features_names:
            file.write(f"{feature_name}\n")

    print(f"Saved top features to {output_file_path}")
   
def calculate_feature_importances(rf, X_test, y_test, feature_names, output_file_FI='output/RF_2019_finalrun_top10/metrics/feature_importance_report_2019.xlsx'):
    # Calculate feature importances from the model
    feature_importances = rf.feature_importances_
    
    # Calculate permutation feature importances
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    perm_importances = perm_importance.importances_mean
    
    # Create a DataFrame with feature importances and permutation importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances,
        'Permutation Importance': perm_importances
    })
    
    # Sort the DataFrame by feature importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Save the DataFrame to an Excel file
    importance_df.to_excel(output_file_FI, index=False)
    
    print(f'Feature importance report saved to {output_file_FI}')

def generate_partial_dependence_plots(shap_values, X_test_sample, rf, feature_names):
    print('Generating Partial Dependence Plots...')
    
    num_classes = shap_values.shape[2]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Calculate global min for y-axis limits
    global_min = float('inf')

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"Calculating y-axis limits for class: {class_name}")

        # Identify top 14 features for this class based on mean absolute SHAP values
        top_features_indices = np.argsort(-mean_abs_shap[:, i])[:14]
        
        for feature_index in top_features_indices:
            try:
                display = PartialDependenceDisplay.from_estimator(
                    rf,
                    X_test_sample,
                    features=[feature_index],
                    kind='average',
                    grid_resolution=20,
                    feature_names=feature_names,
                    target=rf.classes_[i]
                )
                pdp_values = display.pd_results[0].average
                global_min = min(global_min, pdp_values.min())
            except ValueError as e:
                print(f"Error calculating y-axis limits for {class_name} - {feature_names[feature_index]}: {e}")

    global_max = 0.5  # Set the global max to 0.5 as required
    print(f"Global y-axis limits: min={global_min}, max={global_max}")

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"Generating PDPs for class: {class_name}")

        # Identify top 10 features for this class based on mean absolute SHAP values
        top_features_indices = np.argsort(-mean_abs_shap[:, i])[:10]
        
        for feature_index in top_features_indices:
            feature_name = feature_names[feature_index]
            safe_feature_name = feature_name.replace('/', '_')
            print(f"Generating PDP for feature: {safe_feature_name}")

            # Generate the Partial Dependence Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            try:
                display = PartialDependenceDisplay.from_estimator(
                    rf,
                    X_test_sample,
                    features=[feature_index],
                    kind='average',
                    ax=ax,
                    grid_resolution=20,
                    feature_names=feature_names,
                    target=rf.classes_[i]
                )
                ax.set_ylim(global_min, global_max)
                ax.set_yticks(np.linspace(global_min, global_max, num=6))  # Set y-axis ticks with consistent steps
                ax.set_title(f'PDP for {class_name} - {safe_feature_name}')
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))  # Remove weird dashes
                plt.tight_layout()
                output_dir = f'output/RF_2019_finalrun_top10/metrics/pdp_{class_name}_{safe_feature_name}'
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                plt.savefig(f'{output_dir}.png', dpi=300)
            except ValueError as e:
                print(f"Error generating PDP for {class_name} - {safe_feature_name}: {e}")
            plt.close()

        print(f"Completed PDP generation for class: {class_name}.")

def plot_roc_curves(rf, X_test, y_test, CLASS_NAMES):
    print('Generating ROC curves and calculating AUC...')
    
    # Predict probabilities
    y_pred_proba = rf.predict_proba(X_test)
    
    # Adjust class values to start from 0
    y_test_adjusted = y_test - 1
    
    # Binarize the labels for multiclass classification
    y_test_bin = label_binarize(y_test_adjusted, classes=range(len(CLASS_NAMES)))
    
    # Calculate ROC curves and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = {class_name: np.nan for class_name in CLASS_NAMES}
    
    for i in range(len(CLASS_NAMES)):
        if np.sum(y_test_bin[:, i]) == 0:
            print(f"No positive samples in class {CLASS_NAMES[i]}, skipping ROC curve calculation.")
            continue
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[CLASS_NAMES[i]] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves for each class
    plt.figure()
    for i in range(len(CLASS_NAMES)):
        if not np.isnan(roc_auc[CLASS_NAMES[i]]):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[CLASS_NAMES[i]]:.2f}) for class {CLASS_NAMES[i]}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig('output/RF_2019_finalrun_top10/metrics/roc_curves.png')
    plt.close()
    
    # Save AUC values to an .xlsx file
    auc_df = pd.DataFrame.from_dict(roc_auc, orient='index', columns=['AUC'])
    auc_df.index = CLASS_NAMES
    os.makedirs('output/RF_2019_finalrun_top10/metrics', exist_ok=True)
    auc_df.to_excel('output/RF_2019_finalrun_top10/metrics/auc_values.xlsx')
    
    print('ROC curves and AUC calculated and plotted successfully!')

# Execute functions
if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    
    # Load data and model
    X_train_pkl, X_test_pkl, y_train_pkl, y_test_pkl, rf = load_rf_data(script_dir)
    X_train, X_test, y_train, y_test = preprocess_rf_data(X_train_pkl, X_test_pkl, y_train_pkl, y_test_pkl)
    
    # Generate initial report
    generate_report(rf, X_test, y_test, CLASS_NAMES)
    
    # Retrieve feature names
    feature_names = retrieve_feature_names()  
    
    # Sample the test set for SHAP analysis
    X_test_sample = X_test.sample(frac=SHAP_SAMPLE_FRAC, random_state=RANDOM_STATE)

    # Set file paths
    shap_values_file = os.path.join(script_dir, 'output/RF_2019_finalrun_top10/metrics/shap_values.pkl')
    output_file_FI = os.path.join(script_dir, 'output/RF_2019_finalrun_top10/metrics/feature_importance_report_2019.xlsx')

    # Calculate or load SHAP values
    if SHAP_RECALCULATE:
        shap_values, shap_values_per_class = SHAP(rf, X_test, shap_values_file, X_test_sample)
    else:
        with open(shap_values_file, 'rb') as f:
            shap_values, shap_values_per_class = pickle.load(f)

    # Generate various plots and reports
    generate_beeswarm_plot(shap_values_per_class, X_test_sample, rf, feature_names)
    generate_partial_dependence_plots(shap_values, X_test_sample, rf, feature_names)
    save_top_features(shap_values_per_class, feature_names, top_n=5)
    save_top_features(shap_values_per_class, feature_names, top_n=6)
    plot_roc_curves(rf, X_test, y_test, CLASS_NAMES)
    generate_shap_report(shap_values_per_class, rf, feature_names)
    calculate_feature_importances(rf, X_test, y_test, feature_names, output_file_FI)