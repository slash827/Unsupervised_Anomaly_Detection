import os
import time
import glob

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


start = time.time()
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

# ================ T-SNE IMPLEMENTATION ================

# 1. Standardize the features (important for T-SNE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Sample the dataset to make T-SNE feasible
# With 4M samples, we need to subsample significantly
print(f"Original dataset size: {X.shape[0]} samples")
sample_size = min(10000, X.shape[0])  # Cap at 10,000 samples
print(f"Sampling {sample_size} records for T-SNE analysis...")

# Stratified sampling to maintain class distribution
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size/X.shape[0], random_state=42)
for _, sample_idx in sss.split(X_scaled, y):
    X_sample = X_scaled[sample_idx]
    y_sample = y.iloc[sample_idx].reset_index(drop=True)

print(f"Working with sampled dataset of shape: {X_sample.shape}")

# 3. Apply T-SNE for dimensionality reduction
print("Performing T-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, perplexity=min(30, sample_size//5), n_iter=1000,
            random_state=42, verbose=1, n_jobs=-1)  # Use all cores
X_tsne = tsne.fit_transform(X_sample)

# 3. Create a DataFrame for the T-SNE results
tsne_df = pd.DataFrame()
tsne_df['tsne_1'] = X_tsne[:, 0]
tsne_df['tsne_2'] = X_tsne[:, 1]
tsne_df['label'] = y_sample

# 4. Visualize T-SNE results
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='tsne_1', y='tsne_2',
    hue='label',
    palette=sns.color_palette("hls", len(np.unique(y))),
    data=tsne_df,
    legend="full",
    alpha=0.8
)

plt.title('T-SNE Visualization of BETH Dataset')
plt.xlabel('T-SNE Feature 1')
plt.ylabel('T-SNE Feature 2')
plt.tight_layout()
plt.savefig('tsne_visualization.png')
plt.show()

# 5. Additional analysis: Feature importance in original space
print("\nTraining Random Forest to analyze feature importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# 6. Visualize feature importance
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

# 7. Cluster analysis based on T-SNE
print("\nExamining clusters in T-SNE space...")
# Calculate the number of points for each label
label_counts = tsne_df['label'].value_counts().reset_index()
label_counts.columns = ['Label', 'Count']
print("Distribution of labels:")
print(label_counts)

# 8. T-SNE with different perplexity values (optional)
# Choose smaller perplexity values for smaller sample size
max_perp = min(sample_size // 5, 50)  # Perplexity shouldn't exceed n/5
perplexities = [max(5, int(max_perp/4)), max(15, int(max_perp/2)), max_perp]
print(f"Testing perplexity values: {perplexities}")

plt.figure(figsize=(18, 6))

for i, perp in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000, random_state=42, n_jobs=-1)
    tsne_result = tsne.fit_transform(X_sample)

    plt.subplot(1, 3, i+1)
    sns.scatterplot(
        x=tsne_result[:, 0], y=tsne_result[:, 1],
        hue=y_sample,
        palette=sns.color_palette("hls", len(np.unique(y_sample))),
        legend="full" if i == 2 else None,
        alpha=0.8
    )
    plt.title(f'T-SNE with Perplexity = {perp}')

plt.tight_layout()
plt.savefig('tsne_perplexity_comparison.png')
plt.show()

end = time.time()
print (f"total time took is: {end - start}")