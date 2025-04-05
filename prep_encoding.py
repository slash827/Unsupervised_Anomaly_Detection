import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import prince  # For MCA
from tqdm import tqdm

# === Config ===
train_file = "labelled_training_data.csv"
test_file = "labelled_testing_data.csv"
valid_file = "labelled_validation_data.csv"
output_file = "preprocessed/prepared_data_encoding.csv"

# === Load and Merge ===
print("📥 Loading data...")
df = pd.concat([
    pd.read_csv(train_file),
    pd.read_csv(test_file),
    pd.read_csv(valid_file)
], ignore_index=True)

# === Binary Features ===
print("⚙️ Engineering binary features...")
df['bin_userId'] = (df['userId'] >= 1000).astype(int)
df['bin_processId'] = df['processId'].isin([0, 1, 2]).astype(int)
df['bin_parentProcessId'] = df['parentProcessId'].isin([0, 1, 2]).astype(int)
df['bin_mountNamespace'] = (df['mountNamespace'] == 4026531840).astype(int)

# === Args Parsing ===
def process_args_row(row):
    try:
        row = row.strip('[]')
        arg_entries = row.split('},')
        parsed_args = []

        for entry in arg_entries:
            entry = entry.replace("{", "").replace("}", "").replace("'", "").strip()
            fields = dict(kv.strip().split(": ", 1) for kv in entry.split(", ") if ": " in kv)
            parsed_args.append((
                fields.get("type", ""),
                fields.get("value", ""),
                fields.get("name", "")
            ))

        flat_dict = {}
        for i, (t, v, n) in enumerate(parsed_args):
            flat_dict[f"type_{i}"] = t
            flat_dict[f"value_{i}"] = v
            flat_dict[f"name_{i}"] = n
        return pd.Series(flat_dict)
    except Exception:
        return pd.Series()

print("🧩 Parsing 'Args' column...")
args_df = df['args'].fillna("[]").apply(process_args_row)
args_df.columns = [f"args_{col}" for col in args_df.columns]

# Numeric conversion for values
value_cols = [col for col in args_df.columns if col.startswith("args_value_")]
args_df[value_cols] = args_df[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# Drop original Args and merge new features
df = df.drop(columns=["args"])
df = pd.concat([df.reset_index(drop=True), args_df.reset_index(drop=True)], axis=1)

# === MCA on 'eventId' (low cardinality) ===
print("🧬 Applying MCA to 'eventId'...")
eventid_cat = df[['eventId']].astype(str)
eventid_mca = prince.MCA(n_components=1, n_iter=5, random_state=42)
eventid_proj = eventid_mca.fit_transform(eventid_cat)
eventid_proj.columns = [f"eventId_MCA_{i}" for i in range(eventid_proj.shape[1])]

# === One-Hot + SVD on moderate categoricals ===
categorical_cols = ['processName', 'eventName', 'hostName']
print("🔢 One-hot encoding + SVD on:", categorical_cols)
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
onehot_sparse = encoder.fit_transform(df[categorical_cols].astype(str))

svd = TruncatedSVD(n_components=10, random_state=42)
onehot_reduced = svd.fit_transform(onehot_sparse)
onehot_df = pd.DataFrame(onehot_reduced, columns=[f"onehot_SVD_{i}" for i in range(10)])

# === Standardize numeric and binary features ===
print("📐 Scaling numeric, binary, and parsed value fields...")
numeric_features = ['argsNum', 'returnValue']
binary_features = ['bin_userId', 'bin_processId', 'bin_parentProcessId', 'bin_mountNamespace']
args_numeric_cols = value_cols

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_features + binary_features + args_numeric_cols])
scaled_df = pd.DataFrame(scaled_data, columns=[f"scaled_{col}" for col in numeric_features + binary_features + args_numeric_cols])

# === Final Merge ===
print("🔗 Merging all processed features + labels...")
final_df = pd.concat([
    eventid_proj.reset_index(drop=True),
    onehot_df.reset_index(drop=True),
    scaled_df.reset_index(drop=True),
    df[['sus', 'evil']].reset_index(drop=True)
], axis=1)

# === Save Final Output ===
print(f"💾 Saving to {output_file}")
os.makedirs("preprocessed", exist_ok=True)
final_df.to_csv(output_file, index=False)

print("✅ Done. Final shape:", final_df.shape)
print("📊 Columns:", list(final_df.columns))
