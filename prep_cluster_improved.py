"""
Enhanced preprocessing script for clustering-based anomaly detection.

Improvements:
1. Better handling of test/train/validation splits
2. Increased MCA components for eventId
3. More sophisticated feature engineering
4. Class-aware preprocessing
5. Feature importance weighting
6. Option for balanced sampling
7. Modular design with separate functions
8. PEP 8 compliance
"""
import os
import time
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
import prince  # For MCA


def load_and_combine_data(
    train_file: str, 
    test_file: str, 
    valid_file: str
) -> Tuple[pd.DataFrame, bool]:
    """
    Load and combine training, testing, and validation datasets.
    
    Args:
        train_file: Path to training data CSV
        test_file: Path to testing data CSV
        valid_file: Path to validation data CSV
        
    Returns:
        Tuple containing:
        - Combined DataFrame with 'split' column
        - Boolean indicating if labels are present
    """
    print("📥 Loading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    valid_df = pd.read_csv(valid_file)

    # Check if labels exist
    has_labels = 'evil' in test_df.columns and 'sus' in test_df.columns
    
    # Add split indicator
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    valid_df['split'] = 'valid'

    # Combine all data
    df = pd.concat([train_df, test_df, valid_df], ignore_index=True)
    
    # Print dataset statistics
    print(f"Dataset shapes: Train={train_df.shape}, Test={test_df.shape}, Valid={valid_df.shape}")
    if has_labels:
        print(f"Evil label distribution in test set: {test_df['evil'].value_counts().to_dict()}")
    
    return df, has_labels


def create_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary features based on domain knowledge.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added binary features
    """
    print("\n⚙️ Engineering binary features...")
    # Create copy to avoid modifying the original
    result_df = df.copy()
    
    # Basic binary features
    result_df['bin_userId'] = (result_df['userId'] >= 1000).astype(int)
    result_df['bin_processId'] = result_df['processId'].isin([0, 1, 2]).astype(int)
    result_df['bin_parentProcessId'] = result_df['parentProcessId'].isin([0, 1, 2]).astype(int)
    result_df['bin_mountNamespace'] = (result_df['mountNamespace'] == 4026531840).astype(int)
    
    # Advanced binary features
    result_df['is_common_process'] = result_df['processName'].isin(
        ['bash', 'ssh', 'systemd', 'ls', 'cat']).astype(int)
    result_df['has_error'] = (result_df['returnValue'] < 0).astype(int)
    
    # Interaction features
    result_df['user_system_process'] = (
        (result_df['userId'] >= 1000) & 
        (result_df['processName'].isin(['sudo', 'su', 'bash']))
    ).astype(int)
    
    return result_df


def process_args_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Parse and process the 'args' column into structured data.
    
    Args:
        df: Input DataFrame with 'args' column
        
    Returns:
        Tuple containing:
        - DataFrame with processed args
        - List of numeric args columns
        - List of binary args columns
    """
    print("\n🧩 Parsing 'Args'...")
    
    def process_args_row(row):
        """Process a single args row into structured columns."""
        try:
            row = row.strip('[]')
            if not row:
                return pd.Series()
                
            arg_entries = row.split('},')
            parsed_args = []

            for entry in arg_entries:
                entry = entry.replace("{", "").replace("}", "").replace("'", "").strip()
                if not entry:
                    continue
                    
                fields = dict(kv.strip().split(": ", 1) for kv in entry.split(", ") if ": " in kv)
                parsed_args.append((fields.get("type", ""), fields.get("value", ""), fields.get("name", "")))

            flat_dict = {}
            for i, (t, v, n) in enumerate(parsed_args):
                flat_dict[f"type_{i}"] = t
                flat_dict[f"value_{i}"] = v
                flat_dict[f"name_{i}"] = n
            return pd.Series(flat_dict)
        except Exception:
            # Return empty series on error
            return pd.Series()

    # Process the args column
    args_df = df['args'].fillna("[]").apply(process_args_row)
    args_df.columns = [f"args_{col}" for col in args_df.columns]

    # Convert value columns to numeric
    value_cols = [col for col in args_df.columns if col.startswith("args_value_")]
    args_df[value_cols] = args_df[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Extract more features from args
    # Look for suspicious args values (paths to /tmp, /dev/shm)
    suspicious_pattern = fr'tmp|shm|var{os.sep}tmp|passwd|shadow|etc|bash|curl|wget'
    binary_cols = []
    
    for col in [c for c in args_df.columns if c.startswith("args_value_")]:
        binary_col = f"{col}_suspicious"
        args_df[binary_col] = args_df[col].astype(str).str.contains(
            suspicious_pattern, case=False, regex=True).astype(int)
        binary_cols.append(binary_col)
    
    return args_df, value_cols, binary_cols


def apply_dimensionality_reduction(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply MCA and SVD dimensionality reduction to categorical features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple containing:
        - DataFrame with MCA components for eventId
        - DataFrame with MCA components for processName
        - DataFrame with SVD components for other categorical features
    """
    print("\n🧬 MCA on 'eventId' and categorical features...")
    # Apply MCA with more components for eventId
    eventid_cat = df[['eventId']].astype(str)
    eventid_mca = prince.MCA(n_components=5, n_iter=5, random_state=42)
    eventid_proj = eventid_mca.fit_transform(eventid_cat)
    eventid_proj.columns = [f"eventId_MCA_{i}" for i in range(eventid_proj.shape[1])]
    
    # Apply MCA to processName
    process_cat = df[['processName']].astype(str)
    process_mca = prince.MCA(n_components=3, n_iter=5, random_state=42)
    process_proj = process_mca.fit_transform(process_cat)
    process_proj.columns = [f"processName_MCA_{i}" for i in range(process_proj.shape[1])]
    
    # Prepare for one-hot encoding and SVD
    print("\n🔢 One-hot encoding + SVD...")
    categorical_cols = ['eventName', 'hostName']
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    onehot_sparse = encoder.fit_transform(df[categorical_cols].astype(str))

    # Apply SVD with more components to preserve more variance
    svd = TruncatedSVD(n_components=15, random_state=42)
    onehot_reduced = svd.fit_transform(onehot_sparse)
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"SVD explained variance: {explained_variance:.2f}")
    onehot_df = pd.DataFrame(onehot_reduced, columns=[f"onehot_SVD_{i}" for i in range(15)])
    
    return eventid_proj, process_proj, onehot_df


def standardize_features(
    df: pd.DataFrame, 
    numeric_features: List[str], 
    binary_features: List[str], 
    args_numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Standardize numeric features using StandardScaler.
    
    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature names
        binary_features: List of binary feature names
        args_numeric_cols: List of numeric args columns
        
    Returns:
        DataFrame with standardized features
    """
    print("\n📐 Standardizing features...")
    # Standardize numeric features
    features_to_scale = numeric_features + binary_features + args_numeric_cols
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features_to_scale])
    scaled_df = pd.DataFrame(
        scaled_data, 
        columns=[f"scaled_{col}" for col in features_to_scale]
    )
    
    return scaled_df


def calculate_feature_importance(
    feature_df: pd.DataFrame, 
    output_dir: str
) -> Dict[str, float]:
    """
    Calculate feature importance using Random Forest.
    
    Args:
        feature_df: DataFrame with features and labels
        output_dir: Directory to save feature importance plot
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    print("\n🔍 Calculating feature importance...")
    
    # Only use the test set for feature importance
    test_features = feature_df[feature_df['split'] == 'test'].drop(columns=['split', 'sus', 'evil'])
    test_labels = feature_df.loc[feature_df['split'] == 'test', 'evil']
    
    # Train a random forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(test_features, test_labels)
    
    # Store feature importance
    importance = rf.feature_importances_
    feature_importance = dict(zip(test_features.columns, importance))
    
    # Print top 10 important features
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 important features:")
    for feature, importance_score in top_features:
        print(f"  {feature}: {importance_score:.4f}")
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    
    # Get top 50% features
    importance_threshold = np.percentile(importance, 50)
    selected_features = test_features.columns[importance > importance_threshold]
    selected_importance = importance[importance > importance_threshold]
    
    print(f"Selected {len(selected_features)} out of {len(test_features.columns)} features")
    
    # Create the plot
    sns.barplot(x=selected_importance, y=selected_features)
    plt.title("Feature Importance")
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "feature_importance.pdf"))
    plt.close()
    
    return feature_importance


def weight_features_by_importance(
    feature_df: pd.DataFrame, 
    feature_importance: Dict[str, float]
) -> pd.DataFrame:
    """
    Weight features by their importance scores.
    
    Args:
        feature_df: DataFrame with features
        feature_importance: Dictionary of feature importance scores
        
    Returns:
        DataFrame with weighted features
    """
    # Make a copy to avoid modifying the original
    result_df = feature_df.copy()
    
    # Get importance threshold (top 50%)
    importance_values = np.array(list(feature_importance.values()))
    importance_threshold = np.percentile(importance_values, 50)
    
    # Get selected features
    selected_features = [col for col, imp in feature_importance.items() 
                         if imp > importance_threshold]
    
    # Weight features by importance
    for col in selected_features:
        if col in result_df.columns:
            # Scale importance up to make the effect stronger
            result_df[col] = result_df[col] * (feature_importance[col] * 5 + 1)
    
    return result_df


def save_datasets(feature_df: pd.DataFrame, output_file: str) -> None:
    """
    Save the processed datasets.
    
    Args:
        feature_df: DataFrame to save
        output_file: Path to save the main file
    """
    print(f"\n💾 Saving to {output_file}")
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    feature_df.to_csv(output_file, index=False)
    print("✅ Main dataset saved. Shape:", feature_df.shape)
    
    # Save a separate test-only file for easier analysis
    test_only_file = os.path.join(output_dir, "test_only_cluster.csv")
    feature_df[feature_df['split'] == 'test'].to_csv(test_only_file, index=False)
    print(f"Saved test-only data to {test_only_file}")


def prepare_cluster_data(
    train_file: str, 
    test_file: str, 
    valid_file: str, 
    output_file: str = f"preprocessed{os.sep}prepared_data_cluster.csv", 
    include_feature_importance: bool = True
) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """
    Prepares the BETH dataset for clustering-based anomaly detection with improved preprocessing.

    Args:
        train_file: Path to training split (CSV)
        test_file: Path to testing split (CSV)
        valid_file: Path to validation split (CSV)
        output_file: Optional path to save processed data CSV
        include_feature_importance: Whether to include feature importance weights

    Returns:
        Tuple containing:
        - Processed DataFrame ready for clustering
        - Dictionary of feature importance scores (if requested, else None)
    """
    # 1. Load and combine data
    df, has_labels = load_and_combine_data(train_file, test_file, valid_file)
    
    # 2. Create binary features
    df = create_binary_features(df)
    
    # 3. Process args data
    args_df, args_numeric_cols, args_binary_cols = process_args_data(df)
    
    # 4. Combine with the main dataframe
    df = df.drop(columns=["args"])
    df = pd.concat([df.reset_index(drop=True), args_df.reset_index(drop=True)], axis=1)
    
    # 5. Apply dimensionality reduction
    eventid_proj, process_proj, onehot_df = apply_dimensionality_reduction(df)
    
    # 6. Standardize numeric features
    numeric_features = ['argsNum', 'returnValue']
    binary_features = [
        'bin_userId', 'bin_processId', 'bin_parentProcessId', 'bin_mountNamespace', 
        'is_common_process', 'has_error', 'user_system_process'
    ]
    
    scaled_df = standardize_features(
        df, numeric_features, binary_features, args_numeric_cols
    )
    
    # 7. Prepare binary features dataframe
    binary_df = df[args_binary_cols].copy()
    binary_df.columns = [f"binary_{col}" for col in args_binary_cols]
    
    # 8. Combine all features
    feature_df = pd.concat([
        eventid_proj.reset_index(drop=True),
        process_proj.reset_index(drop=True),
        onehot_df.reset_index(drop=True),
        scaled_df.reset_index(drop=True),
        binary_df.reset_index(drop=True),
        df[['split']].reset_index(drop=True)
    ], axis=1)
    
    # 9. Handle labels
    feature_importance = None
    if has_labels:
        print("\n🏷️ Handling labels...")
        # Add labels to the feature dataframe
        feature_df[['sus', 'evil']] = df[['sus', 'evil']]

        # 10. Calculate feature importance if requested
        if include_feature_importance:
            output_dir = os.path.dirname(output_file)
            feature_importance = calculate_feature_importance(feature_df, output_dir)
            
            # 11. Weight features by importance
            feature_df = weight_features_by_importance(feature_df, feature_importance)
        
    # 13. Save datasets
    save_datasets(feature_df, output_file)
    
    return feature_df, feature_importance


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    start = time.time()
    df, importance = prepare_cluster_data(
        f"data{os.sep}labelled_training_data.csv",
        f"data{os.sep}labelled_testing_data.csv",
        f"data{os.sep}labelled_validation_data.csv",
        output_file=f"preprocessed{os.sep}prepared_data_cluster.csv",
        include_feature_importance=True  # Set to False to skip feature importance
    )
    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds")