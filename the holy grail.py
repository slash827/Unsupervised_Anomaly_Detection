import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === Step 1: Load and Merge Dataset ===
train_file = "labelled_training_data.csv"
test_file = "labelled_testing_data.csv"
valid_file = "labelled_validation_data.csv"

# Concatenate all splits
df = pd.concat([
    pd.read_csv(train_file),
    pd.read_csv(test_file),
    pd.read_csv(valid_file)
], ignore_index=True)
df =df.sample(frac=1)

# === Step 2: Feature Engineering ===
df['bin_userId'] = (df['userId'] >= 1000).astype(int)
df['bin_processId'] = df['processId'].isin([0, 1, 2]).astype(int)
df['bin_parentProcessId'] = df['parentProcessId'].isin([0, 1, 2]).astype(int)
df['bin_mountNamespace'] = (df['mountNamespace'] == 4026531840).astype(int)

categorical_features = ['processName', 'eventName', 'hostName']
numeric_features = ['eventId', 'argsNum', 'returnValue']
binary_features = ['bin_userId', 'bin_processId', 'bin_parentProcessId', 'bin_mountNamespace']

X_cat = df[categorical_features].astype(str).copy()
X_num = df[numeric_features].copy()
X_bin = df[binary_features].copy()
y = df[['sus', 'evil']].copy()

# Label: 1 if sus or evil
labels = np.logical_or(y['sus'], y['evil']).astype(int).values

# Encode categorical
label_encoders = {}
cat_dims = {}
for col in X_cat.columns:
    le = LabelEncoder()
    X_cat[col] = le.fit_transform(X_cat[col])
    label_encoders[col] = le
    cat_dims[col] = len(le.classes_)

# Scale numeric + binary
scaler = StandardScaler()
X_num_bin = pd.concat([X_num, X_bin], axis=1)
X_scaled = scaler.fit_transform(X_num_bin)

X_cat_np = X_cat.values
X_num_np = X_scaled

# === Step 3: Deep SVDD Model Definition ===
class DeepSVDDWithEmbedding(nn.Module):
    def __init__(self, cat_dims, num_dim, rep_dim=32):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, min(50, (cat_dim + 1) // 2)) for cat_dim in cat_dims
        ])
        emb_dim = sum([emb.embedding_dim for emb in self.embeddings])
        input_dim = emb_dim + num_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, rep_dim)
        )

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.net(x)

# === Step 4: Prepare Training Data ===
mask_normal = (y['sus'] == 0) & (y['evil'] == 0)
X_cat_train = X_cat_np[mask_normal.values]
X_num_train = X_num_np[mask_normal.values]

X_cat_tensor = torch.tensor(X_cat_np, dtype=torch.long)
X_num_tensor = torch.tensor(X_num_np, dtype=torch.float32)
X_cat_train_tensor = torch.tensor(X_cat_train, dtype=torch.long)
X_num_train_tensor = torch.tensor(X_num_train, dtype=torch.float32)

# === Step 5: Train Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepSVDDWithEmbedding(list(cat_dims.values()), X_num_np.shape[1]).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    output = model(X_cat_train_tensor.to(device), X_num_train_tensor.to(device))
    center = output.mean(dim=0)
    loss = ((output - center) ** 2).mean()
    loss.backward()
    optimizer.step()

# === Step 6: Score Anomalies ===
model.eval()
with torch.no_grad():
    z = model(X_cat_tensor.to(device), X_num_tensor.to(device))
    anomaly_scores = ((z - center) ** 2).sum(dim=1).cpu().numpy()

# Save scores and labels for later use
os.makedirs("test/deep_svdd", exist_ok=True)
scores_df = pd.DataFrame({
    "anomaly_score": anomaly_scores,
    "label": labels,
    "sus": y['sus'].values,
    "evil": y['evil'].values
})


# === Step 7: Evaluate by Percentile ===
percentile = 10

threshold = np.percentile(anomaly_scores, 100 - percentile)
preds = (anomaly_scores >= threshold).astype(int)
num_detected = preds.sum()
evil_total = y['evil'].sum()
sus_total = y['sus'].sum()
evil_detected = ((preds == 1) & (y['evil'] == 1)).sum()
sus_detected = ((preds == 1) & (y['sus'] == 1)).sum()

print(f"Total points flagged as anomalies: {num_detected}")
print(f"Detected {evil_detected} out of {evil_total} evil points ({(evil_detected / evil_total) * 100:.2f}%)")
print(f"Detected {sus_detected} out of {sus_total} sus points ({(sus_detected / sus_total) * 100:.2f}%)")
