# Pre-process the dataset similar to the preprocessing that was done in the BETH article
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import os, glob
import zipfile
import requests
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')


def check_and_download_beth_data(url):
    """
    Checks if BETH CSV files exist in the 'data' directory.
    If not, downloads them from Google Drive and extracts them.
    
    Returns:
        list: List of paths to the CSV files ending with 'data.csv'
    """
    # Make sure we're in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(script_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Check for CSV files ending with 'data.csv'
    csv_files = glob.glob(f"data{os.sep}*data.csv")
    
    if len(csv_files) >= 3:
        print(f"Found {len(csv_files)} BETH data files: {[os.path.basename(f) for f in csv_files]}")
        return csv_files
    
    print(f"Insufficient BETH data files found (only {len(csv_files)}). Downloading from DropBox...")
    
    # DropBox URL and local path for the zip file
    dropbox_url = url
    local_zip_path = os.path.join(data_dir, "beth_dataset.zip")
    
    try:
        print(f"Downloading BETH dataset from Dropbox url: {dropbox_url}")
        response = requests.get(dropbox_url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Get total file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(local_zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)
        
        print(f"Download complete. Extracting files to {data_dir}...")
        
        # Extract the zip file
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print("Extraction complete.")
        
        # Clean up the zip file
        os.remove(local_zip_path)
        print(f"Removed zip file: {local_zip_path}")
        
        # Check if we now have the files
        csv_files = glob.glob(f"data{os.sep}*data.csv")
        if len(csv_files) >= 3:
            print(f"Successfully downloaded {len(csv_files)} BETH data files: {[os.path.basename(f) for f in csv_files]}")
        else:
            print(f"Warning: Still only found {len(csv_files)} files after download: {[os.path.basename(f) for f in csv_files]}")
            
            # Try searching with a more general pattern in case file naming is different
            all_csv_files = glob.glob(f"data{os.sep}*.csv")
            print(f"Found {len(all_csv_files)} total CSV files: {[os.path.basename(f) for f in all_csv_files]}")
            
            if len(all_csv_files) > len(csv_files):
                print("Using all CSV files found instead of just those ending with 'data.csv'")
                csv_files = all_csv_files
        
        return csv_files
        
    except Exception as e:
        print(f"Error during download: {e}")
    return csv_files  # Return whatever files we have
    

def load_and_preprocess_beth_data(csv_files, file_path):
    """
    Load the BETH kernel process logs and preprocess them for analysis.
    
    Args:
        file_path: Path to the CSV file containing the BETH dataset
        
    Returns:
        Preprocessed DataFrame and the preprocessing pipeline
    """
    print(f"Loading data from {file_path}...")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns in the dataset: {df.columns.tolist()}")
    
    # Display class distribution correctly
    print("\nClass distribution:")
    benign_count = ((df['evil'] == 0) & (df['sus'] == 0)).sum()
    suspicious_count = ((df['evil'] == 0) & (df['sus'] == 1)).sum()
    malicious_count = (df['evil'] == 1).sum()
    
    print(f"Benign samples (evil=0, sus=0): {benign_count} ({benign_count/len(df)*100:.2f}%)")
    print(f"Suspicious samples (evil=0, sus=1): {suspicious_count} ({suspicious_count/len(df)*100:.2f}%)")
    print(f"Malicious samples (evil=1): {malicious_count} ({malicious_count/len(df)*100:.2f}%)")
    
    # Create binary features as described in the paper
    print("\nCreating binary features based on paper recommendations...")
    df['isSystemProcess'] = df['processId'].isin([0, 1, 2]).astype(int)
    df['isSystemParentProcess'] = df['parentProcessId'].isin([0, 1, 2]).astype(int)
    df['isSystemUser'] = (df['userId'] < 1000).astype(int)
    df['isDefaultMountNamespace'] = (df['mountNamespace'] == 4026531840).astype(int)
    
    # Add return value categorization
    df['returnValueCat'] = np.select(
        [df['returnValue'] < 0, df['returnValue'] == 0, df['returnValue'] > 0],
        [-1, 0, 1]
    )
    
    # Features to use for analysis
    features_for_analysis = [
        'isSystemProcess', 'isSystemParentProcess', 'isSystemUser', 
        'isDefaultMountNamespace', 'eventId', 'argsNum', 'returnValueCat'
    ]
    
    # Display feature statistics
    print("\nFeature value ranges before preprocessing:")
    for feature in features_for_analysis:
        print(f"{feature}: Min={df[feature].min()}, Max={df[feature].max()}, Unique values={df[feature].nunique()}")
    
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[features_for_analysis])
    
    # Create a dataframe with scaled features
    df_scaled = pd.DataFrame(
        features_scaled, 
        columns=features_for_analysis
    )
    
    # Add back labels
    df_scaled['sus'] = df['sus'].values
    df_scaled['evil'] = df['evil'].values
    
    print(f"Processed dataset shape: {df_scaled.shape}")
    return df_scaled, features_for_analysis


def properly_balanced_stratified_sample(X, y, n_samples, random_state=None):
    """
    Improved stratified sampling that ensures proper representation of classes.
    
    Args:
        X: Feature matrix
        y: Target labels
        n_samples: Number of samples to return
        random_state: Random seed
        
    Returns:
        X_sampled, y_sampled, indices
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate class distributions
    unique_classes = np.unique(y)
    class_counts = {cls: np.sum(y == cls) for cls in unique_classes}
    total_samples = len(y)
    
    # Log original class distribution
    print(f"Original distribution: ", end="")
    for cls in unique_classes:
        print(f"Class {cls}: {class_counts[cls]} ({class_counts[cls]/total_samples*100:.2f}%), ", end="")
    print()
    
    # Option 1: Maintain original class distribution (true stratified sampling)
    samples_per_class = {
        cls: int(np.round((count / total_samples) * n_samples))
        for cls, count in class_counts.items()
    }
    
    # Ensure we get exactly n_samples (adjust largest class if needed)
    total_allocated = sum(samples_per_class.values())
    if total_allocated != n_samples:
        largest_class = max(class_counts, key=class_counts.get)
        samples_per_class[largest_class] += (n_samples - total_allocated)
    
    # Option 2: Balance classes more evenly (if very imbalanced)
    if max(samples_per_class.values()) / min(samples_per_class.values()) > 10:
        print("Highly imbalanced classes detected - applying more balanced sampling")
        
        # Choose a more balanced allocation, ensuring minimum representation
        min_per_class = max(5, int(n_samples * 0.1))  # At least 5 samples or 10% per class
        remaining = n_samples - (min_per_class * len(unique_classes))
        
        if remaining > 0:
            # Allocate remaining proportionally to original distribution
            total_orig = sum(class_counts.values())
            extra_per_class = {
                cls: int(np.floor((count / total_orig) * remaining))
                for cls, count in class_counts.items()
            }
            # Distribute any remainder
            remainder = remaining - sum(extra_per_class.values())
            for cls in sorted(class_counts, key=class_counts.get, reverse=True):
                if remainder <= 0:
                    break
                extra_per_class[cls] += 1
                remainder -= 1
                
            # Final allocation
            samples_per_class = {
                cls: min_per_class + extra_per_class[cls]
                for cls in unique_classes
            }
    
    # Sample indices for each class
    sampled_indices = []
    for cls, n_samples_cls in samples_per_class.items():
        cls_indices = np.where(y == cls)[0]
        
        # Handle case where requested samples exceeds available
        if n_samples_cls > len(cls_indices):
            print(f"Warning: Requested {n_samples_cls} samples for class {cls} "
                  f"but only {len(cls_indices)} available.")
            n_samples_cls = len(cls_indices)
            
        # Sample with or without replacement as needed
        if n_samples_cls <= len(cls_indices):
            # Without replacement
            cls_sampled = np.random.choice(cls_indices, size=n_samples_cls, replace=False)
        else:
            # With replacement (should not happen with fixed code above)
            cls_sampled = np.random.choice(cls_indices, size=n_samples_cls, replace=True)
            
        sampled_indices.extend(cls_sampled)
    
    # Shuffle the sampled indices
    np.random.shuffle(sampled_indices)
    sampled_indices = np.array(sampled_indices)
    
    # Log resulting distribution
    y_sampled = y[sampled_indices]
    sampled_counts = {cls: np.sum(y_sampled == cls) for cls in unique_classes}
    print(f"Sampled distribution: ", end="")
    for cls in unique_classes:
        print(f"Class {cls}: {sampled_counts[cls]} ({sampled_counts[cls]/len(y_sampled)*100:.2f}%), ", end="")
    print()
    
    return X[sampled_indices], y[sampled_indices], sampled_indices
