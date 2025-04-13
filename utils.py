# Pre-process the dataset similar to the preprocessing that was done in the BETH article
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# Function to load and preprocess the BETH dataset
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
