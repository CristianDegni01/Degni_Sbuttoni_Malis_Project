import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import LabelEncoder

# Loads data from a CSV file and separates features from the target variable.
def load_data(file_path="../data/survey_lung_cancer.csv", target_column='LUNG_CANCER'):
    try:
        data = pd.read_csv(file_path)
        if target_column:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            return X, y
        else:
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Encodes categorical variables using Label Encoding.
def preprocess_data_label_encoding(df):
    categorical_columns = ['GENDER']
    label_encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    return df, label_encoders

# Splits the dataset into training and testing sets.
def data_split(X, y, test_size=0.2, random_state=45):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Merges two lung cancer datasets, sampling the second one to match a specific target distribution.
def merge_dataset():
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..', 'proposal')))

    warnings.simplefilter("ignore")

    # Dataset paths
    principal_path = "../data/survey_lung_cancer_small.csv"
    lung_cancer_path = "../data/lung_cancer.csv"

    # Load the main dataset (used as the reference distribution)
    X_principal, y_principal = load_data(principal_path)
    X_principal, _ = preprocess_data_label_encoding(X_principal)

    # Load the second dataset (to be adjusted to match the distribution of the first one)
    X_lung, y_lung = load_data(lung_cancer_path)
    X_lung, _ = preprocess_data_label_encoding(X_lung)

    # Define the maximum number of samples to be selected from the second dataset
    num_samples = 1501 - len(y_principal)

    target_distribution = {"YES": 0.871383, "NO": 0.128617}

    # Calculate how many samples are needed for each class
    samples_per_class = {cls: int(num_samples * target_distribution[cls]) for cls in target_distribution}

    # Create the sampled dataset
    sampled_indices = []
    for cls, num_cls_samples in samples_per_class.items():
        cls_indices = y_lung[y_lung == cls].index  # Get indices of rows belonging to this class
        num_available = len(cls_indices)  # Number of available samples for this class

        # If the required number of samples exceeds the available ones, take all
        if num_cls_samples > num_available:
            sampled_cls_indices = cls_indices.to_series()
        else:
            sampled_cls_indices = cls_indices.to_series().sample(n=num_cls_samples, random_state=42)

        sampled_indices.extend(sampled_cls_indices)

    # Create the final sampled datasets
    X_sampled = X_lung.loc[sampled_indices]
    y_sampled = y_lung.loc[sampled_indices]

    # Merge both datasets to obtain a single dataset of approximately 1500 records
    X_final = pd.concat([X_principal, X_sampled], ignore_index=True)
    y_final = pd.concat([y_principal, y_sampled], ignore_index=True)

    return X_final, y_final