import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from scipy.stats import f_oneway, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Load Data ===
data_path = "preprocessed/prepared_data_cluster.csv"
print(f"📥 Loading data from {data_path}")
df = pd.read_csv(data_path)

# === Labels and Features ===
X = df.drop(columns=['sus', 'evil']).values
y_evil = df['evil'].values
y_sus = df['sus'].values
y_true = ((y_evil == 1) | (y_sus == 1)).astype(int)

# === Run KMeans ===
n_clusters = 5
print(f"🔹 Running KMeans with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# === Find smallest clusters as anomalies ===
cluster_sizes = Counter(int(label) for label in labels)
print("📊 Cluster sizes:", dict(cluster_sizes))

# === Choose smallest clusters to treat as anomalies ===
num_anomaly_clusters = 2
sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1])
anomaly_cluster_ids = [int(cid) for cid, _ in sorted_clusters[:num_anomaly_clusters]]
print(f"🔎 Treating clusters {anomaly_cluster_ids} as anomalies")

# === Assign Predictions ===
preds = np.isin(labels, anomaly_cluster_ids).astype(int)

# === ANOVA and T-Test Per Column ===
feature_cols = df.drop(columns=['sus', 'evil']).columns
df_features = df[feature_cols]
df['cluster'] = labels
df['is_anomaly'] = preds

print("\n📈 Statistical Tests per feature (Anomalies vs. Normals):")
print(f"{'Feature':35s} {'ANOVA p-value':>15s} {'T-test p-value':>15s}")

for col in feature_cols:
    try:
        cluster_groups = [df[col][df['cluster'] == i].values for i in range(n_clusters)]
        anova_p = f_oneway(*cluster_groups).pvalue

        anomaly_values = df[col][df['is_anomaly'] == 1]
        normal_values = df[col][df['is_anomaly'] == 0]
        ttest_p = ttest_ind(anomaly_values, normal_values, equal_var=False).pvalue

        print(f"{col:35s} {anova_p:15.3e} {ttest_p:15.3e}")

    except Exception as e:
        print(f"{col:35s} FAILED: {e}")

# === Multi-panel KDE Plot (Normal vs Anomaly for each feature) ===
normal_df = df[df["is_anomaly"] == 0]
anomaly_df = df[df["is_anomaly"] == 1]

n_cols = 3
n_rows = int(np.ceil(len(feature_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    sns.kdeplot(data=normal_df, x=col, label="Normal", ax=axes[idx], fill=True, alpha=0.5, common_norm=False)
    sns.kdeplot(data=anomaly_df, x=col, label="Anomaly", ax=axes[idx], fill=True, alpha=0.5, common_norm=False)
    axes[idx].set_title(col)
    axes[idx].legend()

for j in range(len(feature_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/anova_ttest_features.pdf", dpi=600)
plt.close()

print("PDF with KDE plots saved to plots/anova_ttest_features.pdf")

print(f"\nSuccess Rates:")
print(f"{percent_of_anomalies_that_are_evil:.2f}% of predicted anomalies were evil")
print(f"{percent_of_evil_found:.2f}% of all evil events were successfully detected")


