'''
Merges training, validation, and testing splits.

Adds binary features based on expert rules:  
1) userId >=1000 ->  Is Sytem or user as user would havea a value >=1000 
2) bin_processId -> 0 if idle task, 2 if init/systemd , 3 if kthreadd
3) bin_mountNamespace =  is mountNamespace == 4026531840 This namespace is commonly used for user-space processes with mnt/ access
Parses args column → flattens to structured numeric fields.
MCA on eventId and SVD on one-hot of other categoricals.
Scales numeric + binary + args fields.
'''
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import prince  # For MCA
from tqdm import tqdm
import time

def prepare_cluster_data(train_file, test_file, valid_file, output_file="preprocessed/prepared_data_cluster.csv"):
    """
    Prepares the BETH dataset for clustering-based anomaly detection.

    Parameters:
    - train_file: path to training split (CSV)
    - test_file: path to testing split (CSV)
    - valid_file: path to validation split (CSV)
    - output_file: optional path to save processed data CSV

    Returns:
    - final_df: processed DataFrame ready for clustering
    """

    print("📥 Loading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    valid_df = pd.read_csv(valid_file)

    train_df['split'] = 'train'
    test_df['split'] = 'test'
    valid_df['split'] = 'valid'

    df = pd.concat([train_df, test_df, valid_df], ignore_index=True)

    print("⚙️ Engineering binary features...")
    df['bin_userId'] = (df['userId'] >= 1000).astype(int)
    df['bin_processId'] = df['processId'].isin([0, 1, 2]).astype(int)
    df['bin_parentProcessId'] = df['parentProcessId'].isin([0, 1, 2]).astype(int)
    df['bin_mountNamespace'] = (df['mountNamespace'] == 4026531840).astype(int)

    def process_args_row(row):
        try:
            row = row.strip('[]')
            arg_entries = row.split('},')
            parsed_args = []

            for entry in arg_entries:
                entry = entry.replace("{", "").replace("}", "").replace("'", "").strip()
                fields = dict(kv.strip().split(": ", 1) for kv in entry.split(", ") if ": " in kv)
                parsed_args.append((fields.get("type", ""), fields.get("value", ""), fields.get("name", "")))

            flat_dict = {}
            for i, (t, v, n) in enumerate(parsed_args):
                flat_dict[f"type_{i}"] = t
                flat_dict[f"value_{i}"] = v
                flat_dict[f"name_{i}"] = n
            return pd.Series(flat_dict)
        except Exception:
            return pd.Series()

    print("🧩 Parsing 'Args'...")
    args_df = df['args'].fillna("[]").apply(process_args_row)
    args_df.columns = [f"args_{col}" for col in args_df.columns]

    value_cols = [col for col in args_df.columns if col.startswith("args_value_")]
    args_df[value_cols] = args_df[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    df = df.drop(columns=["args"])
    df = pd.concat([df.reset_index(drop=True), args_df.reset_index(drop=True)], axis=1)

    print("🧬 MCA on 'eventId'...")
    eventid_cat = df[['eventId']].astype(str)
    eventid_mca = prince.MCA(n_components=1, n_iter=5, random_state=42)
    eventid_proj = eventid_mca.fit_transform(eventid_cat)
    eventid_proj.columns = [f"eventId_MCA_{i}" for i in range(eventid_proj.shape[1])]

    print("🔢 One-hot encoding + SVD...")
    categorical_cols = ['processName', 'eventName', 'hostName']
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    onehot_sparse = encoder.fit_transform(df[categorical_cols].astype(str))

    svd = TruncatedSVD(n_components=10, random_state=42)
    onehot_reduced = svd.fit_transform(onehot_sparse)
    onehot_df = pd.DataFrame(onehot_reduced, columns=[f"onehot_SVD_{i}" for i in range(10)])

    print("📐 Standardizing features...")
    numeric_features = ['argsNum', 'returnValue']
    binary_features = ['bin_userId', 'bin_processId', 'bin_parentProcessId', 'bin_mountNamespace']
    args_numeric_cols = value_cols

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_features + binary_features + args_numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"scaled_{col}" for col in numeric_features + binary_features + args_numeric_cols])

    print("🔗 Merging all features...")
    feature_df = pd.concat([
        eventid_proj.reset_index(drop=True),
        onehot_df.reset_index(drop=True),
        scaled_df.reset_index(drop=True),
        df[['split']].reset_index(drop=True)
    ], axis=1)

    # Add sus + evil only for test rows
    if 'sus' in df.columns and 'evil' in df.columns:
        feature_df[['sus', 'evil']] = df[['sus', 'evil']]

        # Zero out for non-test rows
        feature_df.loc[feature_df['split'] != 'test', ['sus', 'evil']] = np.nan

    print(f"💾 Saving to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    feature_df.drop(columns=['split']).to_csv(output_file, index=False)

    print("✅ Done. Shape:", feature_df.shape)
    return feature_df

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    start = time.time()
    df = prepare_cluster_data("data/labelled_training_data.csv",
                              "data/labelled_testing_data.csv",
                              "data/labelled_validation_data.csv")
    end = time.time()
    print(f"total time took is: {end - start}")
