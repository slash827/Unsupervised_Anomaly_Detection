import os
import time
import glob

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from umap import UMAP


def load_and_preprocess_data():
    """
    Load, combine and preprocess datasets from CSV files.
    Returns preprocessed features, target variables, and column information.
    """
    # Load and combine datasets
    csv_files = glob.glob(f"data{os.sep}*[!dns].csv")
    data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Display initial information
    print("Dataset shape:", data.shape)
    print("\nFirst few rows:")
    print(data.head())
    print("\nMissing values:")
    print(data.isnull().sum())

    # Separate numeric and categorical columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    print("\nNumeric columns:", numeric_columns.tolist())
    print("Categorical columns:", categorical_columns.tolist())

    # Handle missing values
    if len(numeric_columns) > 0:
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    if len(categorical_columns) > 0:
        data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

    # Encode categorical features
    le_dict = {}
    for col in categorical_columns:
        le_dict[col] = LabelEncoder()
        data[col] = le_dict[col].fit_transform(data[col])

    # Feature-target split (using 'evil' as target)
    X = data.drop(['evil', 'sus'], axis=1)  # Dropping both 'evil' and 'sus' columns
    y = data['evil']
    
    return X, y, numeric_columns, categorical_columns


def sample_data(X, y, sample_size=10000):
    """
    Sample data using stratified sampling to maintain class distribution.
    
    Args:
        X: Feature matrix
        y: Target vector
        sample_size: Number of samples to extract
        
    Returns:
        X_sample: Sampled feature matrix
        y_sample: Sampled target vector
    """
    print(f"Original dataset size: {X.shape[0]} samples")
    sample_size = min(sample_size, X.shape[0])  # Cap at specified sample size
    print(f"Sampling {sample_size} records for UMAP analysis...")

    # Stratified sampling to maintain class distribution
    sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size/X.shape[0], random_state=42)
    for _, sample_idx in sss.split(X, y):
        X_sample = X.iloc[sample_idx].reset_index(drop=True)
        y_sample = y.iloc[sample_idx].reset_index(drop=True)
    
    print(f"Working with sampled dataset of shape: {X_sample.shape}")
    return X_sample, y_sample


def scale_features(X):
    """
    Standardize features.
    
    Args:
        X: Feature matrix
        
    Returns:
        Scaled feature matrix
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def perform_umap(X, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    Perform UMAP dimensionality reduction.
    
    Args:
        X: Feature matrix
        n_neighbors: Size of local neighborhood
        min_dist: Minimum distance between points in the low-dimensional representation
        n_components: Number of dimensions in the output
        
    Returns:
        UMAP-transformed feature matrix
    """
    print("Performing UMAP dimensionality reduction...")
    # Fixed: Using the correct UMAP class
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
        verbose=True
    )
    X_umap = reducer.fit_transform(X)
    return X_umap


def visualize_umap(X_umap, y, title="UMAP Visualization", filename="umap_visualization.png"):
    """
    Visualize UMAP results.
    
    Args:
        X_umap: UMAP-transformed feature matrix
        y: Target vector
        title: Plot title
        filename: Output filename
    """
    # Create a DataFrame for the UMAP results
    umap_df = pd.DataFrame()
    umap_df['umap_1'] = X_umap[:, 0]
    umap_df['umap_2'] = X_umap[:, 1]
    umap_df['label'] = y

    # Visualize UMAP results
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='umap_1', y='umap_2',
        hue='label',
        palette=sns.color_palette("hls", len(np.unique(y))),
        data=umap_df,
        legend="full",
        alpha=0.8
    )

    plt.title(title)
    plt.xlabel('UMAP Feature 1')
    plt.ylabel('UMAP Feature 2')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
    return umap_df


def analyze_feature_importance(X, y):
    """
    Analyze feature importance using Random Forest.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        DataFrame with feature importance
    """
    print("\nTraining Random Forest to analyze feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Visualize feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Features by Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return feature_importance


def compare_umap_parameters(X, y, n_neighbors_list=[5, 15, 30], min_dist_list=[0.0, 0.1, 0.5]):
    """
    Compare UMAP results with different parameter settings.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_neighbors_list: List of n_neighbors values to test
        min_dist_list: List of min_dist values to test
    """
    print(f"\nComparing UMAP with different parameter values...")
    fig, axes = plt.subplots(len(n_neighbors_list), len(min_dist_list), figsize=(15, 15))
    
    for i, n_neighbors in enumerate(n_neighbors_list):
        for j, min_dist in enumerate(min_dist_list):
            print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")
            
            reducer = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=2,
                random_state=42
            )
            
            umap_result = reducer.fit_transform(X)
            
            ax = axes[i, j]
            scatter = ax.scatter(
                umap_result[:, 0], 
                umap_result[:, 1],
                c=y,
                cmap='viridis', 
                alpha=0.8, 
                s=5
            )
            
            ax.set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist}')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('umap_parameter_comparison.png')
    plt.show()


def main():
    # Load and preprocess data
    X, y, numeric_columns, categorical_columns = load_and_preprocess_data()
    
    # Sample data to make UMAP feasible
    X_sample, y_sample = sample_data(X, y, sample_size=10000)
    
    # Scale features
    X_scaled = scale_features(X_sample)
    
    # Perform UMAP
    X_umap = perform_umap(X_scaled, n_neighbors=15, min_dist=0.1)
    
    # Visualize results
    umap_df = visualize_umap(X_umap, y_sample, 
                           title='UMAP Visualization of BETH Dataset',
                           filename='umap_visualization.png')
    
    # Calculate the number of points for each label
    label_counts = umap_df['label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']
    print("Distribution of labels:")
    print(label_counts)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(X_sample, y_sample)
    
    # Compare different UMAP parameter settings
    # Using smaller parameter set for efficiency
    compare_umap_parameters(X_scaled, y_sample, 
                         n_neighbors_list=[5, 15, 30],
                         min_dist_list=[0.0, 0.1, 0.5])


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")