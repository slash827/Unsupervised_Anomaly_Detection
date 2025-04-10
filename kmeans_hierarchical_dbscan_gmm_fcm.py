import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from scipy.stats import f_oneway, ttest_rel

# === CONFIG ===
start = time.time()
DATA_PATH = "preprocessed/prepared_data_cluster.csv"

# === Load the dataset ===
df = pd.read_csv(DATA_PATH)
X_full = df.drop(columns=["sus", "evil"]).values

# All results will be saved in "final_tests"
RESULTS_DIR = "final_tests"
os.makedirs(RESULTS_DIR, exist_ok=True)
HEATMAP_DIR = RESULTS_DIR  # Heatmaps will be saved here as well

REPEATS = 5
SAMPLE_SIZE = 10000
dims_range = list(range(2, df.shape[1], 2))  # PCA dimensions
cluster_range = list(range(2, 11))  # Cluster counts
algorithms = ['kmeans', 'hierarchical', 'dbscan', 'gmm', 'fcm']
# == ground truth tests ==
EXT_REPEATS = 10
EXT_SAMPLE_SIZE = 10000


# === Initialize average accumulator for internal silhouette scores ===
average_scores = {alg: np.zeros((len(dims_range), len(cluster_range))) for alg in algorithms}


# === Helper function: compute silhouette safely ===
def safe_silhouette(X, labels):
    try:
        if len(np.unique(labels)) > 1:
            return silhouette_score(X, labels)
        else:
            return -1.0
    except Exception:
        return -1.0


# === Internal Evaluation: Compute silhouette scores for different PCA dims and cluster counts ===
print(f"Running {REPEATS} silhouette rounds with {SAMPLE_SIZE} samples each (internal evaluation)")
for repeat in range(REPEATS):
    print(f"\nRound {repeat + 1}/{REPEATS}")
    np.random.seed(42 + repeat)
    idx = np.random.choice(len(X_full), SAMPLE_SIZE, replace=False)
    X_sample = X_full[idx]

    for i, dim in enumerate(dims_range):
        print(f"  PCA to {dim} dimensions")
        X_pca = PCA(n_components=dim, random_state=42).fit_transform(X_sample)

        for j, k in enumerate(cluster_range):
            print(f"    Clusters: {k}")
            # KMeans
            try:
                labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_pca)
                average_scores['kmeans'][i, j] += safe_silhouette(X_pca, labels)
            except Exception as e:
                print("KMeans error:", e)
            # Hierarchical Clustering
            try:
                labels = AgglomerativeClustering(n_clusters=k).fit_predict(X_pca)
                average_scores['hierarchical'][i, j] += safe_silhouette(X_pca, labels)
            except Exception as e:
                print("Hierarchical error:", e)
            # DBSCAN (heuristic eps)
            try:
                eps = 0.2 + 0.1 * (k - 2)
                labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_pca)
                average_scores['dbscan'][i, j] += safe_silhouette(X_pca, labels)
            except Exception as e:
                print("DBSCAN error:", e)
            # Gaussian Mixture Model (GMM)
            try:
                gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
                labels = gmm.fit(X_pca).predict(X_pca)
                average_scores['gmm'][i, j] += safe_silhouette(X_pca, labels)
            except Exception as e:
                print("GMM error:", e)
            # Fuzzy C-means (FCM)
            try:
                cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_pca.T, c=k, m=2, error=0.005, maxiter=1000, seed=42)
                labels = np.argmax(u, axis=0)
                average_scores['fcm'][i, j] += safe_silhouette(X_pca, labels)
            except Exception as e:
                print("FCM error:", e)

# Average the scores across repeats
for alg in algorithms:
    average_scores[alg] /= REPEATS


# === Plotting Function: Save Heatmaps for Internal Evaluation ===
def save_heatmap(matrix, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(cluster_range)))
    ax.set_xticklabels(cluster_range)
    ax.set_yticks(np.arange(len(dims_range)))
    ax.set_yticklabels(dims_range)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("PCA Dimensions")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Silhouette Score")
    plt.tight_layout()
    path = os.path.join(HEATMAP_DIR, filename)
    plt.savefig(path, format="pdf", dpi=600)
    plt.close()
    print(f"Saved -> {path}")


print("\nSaving final average silhouette heatmaps (internal evaluation)...")
for alg in algorithms:
    save_heatmap(average_scores[alg], f"Avg Silhouette - {alg.upper()}", f"{alg}_avg_silhouette_heatmap.pdf")

# =============================================================================
# External Evaluation:
#
# New steps:
# 1. For each algorithm, use the internal evaluation (silhouette only) to select the top 5
#    (PCA dimension, k) parameter combinations.
# 2. For each candidate combination, run EXT_REPEATS rounds external evaluation on the full data,
#    computing:
#       - Silhouette score
#       - Mutual Information (MI) between clusters and "evil" ground truth
#       - Combined score = Silhouette + MI
#       - Ground truth measures: for KMeans, GMM and FCM, use worst 15% (anomaly scores) instead of best cluster.
# 3. For each algorithm, perform one-way ANOVA across the 5 candidate groups and a paired T-test between
#    the best and second best candidate (based on average combined score).
# =============================================================================

# Ground truth for "evil" (binary: 1 = evil, 0 = not evil)
evil_truth = df['evil'].values

# --- Step 1: Select Top 5 Candidate (dim, k) Combos per Algorithm using internal silhouette ---
candidate_params = {}
for alg in algorithms:
    scores_flat = average_scores[alg].flatten()
    sorted_indices = np.argsort(scores_flat)[::-1]  # descending order
    candidates = []
    for idx in sorted_indices:
        i, j = np.unravel_index(idx, average_scores[alg].shape)
        dim_val = dims_range[i]
        k_val = cluster_range[j]
        candidates.append((dim_val, k_val, average_scores[alg][i, j]))
        if len(candidates) >= 5:
            break
    candidate_params[alg] = candidates
    print(f"Top 5 candidate combos for {alg}:")
    for (d, k, score) in candidates:
        print(f"  PCA {d}D, k = {k} (Internal Silhouette = {score:.3f})")

# --- Step 2: External Evaluation for Each Candidate Combo ---
# We will store external evaluation results per algorithm and per candidate combo.
external_results = {alg: {} for alg in algorithms}
for alg in algorithms:
    for (dim_val, k_val, _) in candidate_params[alg]:
        external_results[alg][(dim_val, k_val)] = {
            'silhouette': [],
            'mi': [],
            'score': [],  # silhouette + mi
            'evil_recall': [],
            'evil_precision': []
        }

print(f"\nRunning external evaluation with {EXT_REPEATS} rounds and {EXT_SAMPLE_SIZE} samples each")
for rep in range(EXT_REPEATS):
    print(f"  External round {rep + 1}/{EXT_REPEATS}")
    np.random.seed(1000 + rep)
    idx = np.random.choice(len(X_full), EXT_SAMPLE_SIZE, replace=False)
    X_ext_sample = X_full[idx]
    evil_sample = evil_truth[idx]  # Ground truth for this round

    # Evaluate each algorithm and each candidate combo
    for alg in algorithms:
        for (dim_val, k_val, _) in candidate_params[alg]:
            # Apply PCA with the candidate dimension
            X_pca_ext = PCA(n_components=dim_val, random_state=42).fit_transform(X_ext_sample)
            labels = None
            sil_score = -1.0  # default value
            # First, get clustering labels and compute silhouette and MI as before
            if alg == 'kmeans':
                model = KMeans(n_clusters=k_val, random_state=42).fit(X_pca_ext)
                labels = model.labels_
                sil_score = safe_silhouette(X_pca_ext, labels)
            elif alg == 'hierarchical':
                labels = AgglomerativeClustering(n_clusters=k_val).fit_predict(X_pca_ext)
                sil_score = safe_silhouette(X_pca_ext, labels)
            elif alg == 'dbscan':
                eps = 0.2 + 0.1 * (k_val - 2)
                labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_pca_ext)
                sil_score = safe_silhouette(X_pca_ext, labels)
            elif alg == 'gmm':
                try:
                    model = GaussianMixture(n_components=k_val, covariance_type='full', random_state=42)
                    model.fit(X_pca_ext)
                    labels = model.predict(X_pca_ext)
                    sil_score = safe_silhouette(X_pca_ext, labels)
                except Exception:
                    labels = np.zeros(len(X_pca_ext))
                    sil_score = -1.0
            elif alg == 'fcm':
                try:
                    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_pca_ext.T, c=k_val, m=2, error=0.005, maxiter=1000, seed=42)
                    labels = np.argmax(u, axis=0)
                    sil_score = safe_silhouette(X_pca_ext, labels)
                except Exception:
                    labels = np.zeros(len(X_pca_ext))
                    sil_score = -1.0

            try:
                mi = mutual_info_score(evil_sample, labels)
            except Exception:
                mi = 0
            combined_score = sil_score + mi
            external_results[alg][(dim_val, k_val)]['silhouette'].append(sil_score)
            external_results[alg][(dim_val, k_val)]['mi'].append(mi)
            external_results[alg][(dim_val, k_val)]['score'].append(combined_score)

            # --- Ground Truth Measures Based on Anomaly Detection ---
            # For kmeans, gmm, and fcm: compute an anomaly score for each point and select worst 15%
            if alg in ['kmeans', 'gmm', 'fcm']:
                if alg == 'kmeans':
                    # Use Euclidean distance from each point to its cluster centroid
                    centroids = model.cluster_centers_
                    distances = np.linalg.norm(X_pca_ext - centroids[labels], axis=1)
                    anomaly_scores = distances
                elif alg == 'gmm':
                    # Use negative log likelihood as anomaly score (higher value = more anomalous)
                    log_probs = model.score_samples(X_pca_ext)
                    anomaly_scores = -log_probs
                elif alg == 'fcm':
                    # Use 1 - (max membership) as anomaly score (higher value = more anomalous)
                    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_pca_ext.T, c=k_val, m=2, error=0.005, maxiter=1000, seed=42)
                    labels = np.argmax(u, axis=0)  # update labels (if necessary)
                    max_memberships = np.max(u, axis=0)
                    anomaly_scores = 1.0 - max_memberships

                # Select the worst 15% as anomalies
                threshold = np.percentile(anomaly_scores, 85)  # 85th percentile cutoff
                anomaly_indices = np.where(anomaly_scores >= threshold)[0]

                # Compute ground truth measures for anomalies only
                if len(anomaly_indices) == 0:
                    evil_recall = 0.0
                    evil_precision = 0.0
                else:
                    count_evil_anom = np.sum(evil_sample[anomaly_indices] == 1)
                    total_evil = np.sum(evil_sample == 1)
                    evil_recall = (count_evil_anom / total_evil * 100) if total_evil > 0 else 0.0
                    evil_precision = (count_evil_anom / len(anomaly_indices) * 100)
            else:
                # For hierarchical and DBSCAN, use the best cluster (as before)
                unique_labels = np.unique(labels)
                valid_labels = [lab for lab in unique_labels if lab != -1]
                best_cluster = None
                best_ratio = -1
                for lab in valid_labels:
                    cluster_idx = np.where(labels == lab)[0]
                    if len(cluster_idx) == 0:
                        continue
                    count_evil = np.sum(evil_sample[cluster_idx] == 1)
                    ratio = count_evil / len(cluster_idx)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_cluster = lab
                if best_cluster is None:
                    evil_recall = 0.0
                    evil_precision = 0.0
                else:
                    cluster_idx = np.where(labels == best_cluster)[0]
                    count_evil_cluster = np.sum(evil_sample[cluster_idx] == 1)
                    total_evil = np.sum(evil_sample == 1)
                    evil_recall = (count_evil_cluster / total_evil * 100) if total_evil > 0 else 0.0
                    evil_precision = (count_evil_cluster / len(cluster_idx) * 100) if len(cluster_idx) > 0 else 0.0

            external_results[alg][(dim_val, k_val)]['evil_recall'].append(evil_recall)
            external_results[alg][(dim_val, k_val)]['evil_precision'].append(evil_precision)

# --- Step 3: For each algorithm, analyze the external evaluation results ---
summary_lines = []
summary_lines.append("External Evaluation Statistical Summary (Score = Silhouette + MI)\n")
summary_lines.append("============================================================\n\n")

for alg in algorithms:
    summary_lines.append(f"Algorithm: {alg.upper()}\n")
    summary_lines.append("Candidate Parameter Combos (based on internal evaluation ranking):\n")
    candidate_stats = {}  # To store mean score for each candidate
    for (dim_val, k_val, internal_score) in candidate_params[alg]:
        res = external_results[alg][(dim_val, k_val)]
        sil_mean = np.mean(res['silhouette'])
        sil_std = np.std(res['silhouette'])
        mi_mean = np.mean(res['mi'])
        mi_std = np.std(res['mi'])
        score_mean = np.mean(res['score'])
        score_std = np.std(res['score'])
        recall_mean = np.mean(res['evil_recall'])
        recall_std = np.std(res['evil_recall'])
        prec_mean = np.mean(res['evil_precision'])
        prec_std = np.std(res['evil_precision'])
        candidate_stats[(dim_val, k_val)] = score_mean
        summary_lines.append(f"  PCA {dim_val}D, k={k_val} (Internal Silhouette: {internal_score:.3f})\n"
                             f"    External -> Silhouette: {sil_mean:.3f} ± {sil_std:.3f}, MI: {mi_mean:.3f} ± {mi_std:.3f}\n"
                             f"                 Score: {score_mean:.3f} ± {score_std:.3f}\n"
                             f"                 % Evil Found: {recall_mean:.1f} ± {recall_std:.1f}, "
                             f"% Points Actually Evil: {prec_mean:.1f} ± {prec_std:.1f}\n")
    # Identify best and second best candidates (based on external average score)
    sorted_candidates = sorted(candidate_stats.items(), key=lambda x: x[1], reverse=True)
    best_candidate, best_score = sorted_candidates[0]
    second_best_candidate, second_best_score = sorted_candidates[1]
    summary_lines.append(
        f"Best Candidate for {alg}: PCA {best_candidate[0]}D, k={best_candidate[1]} (Score: {best_score:.3f})\n")

    # Collect score arrays for ANOVA over the 5 candidates (each candidate has a list of EXT_REPEATS values)
    candidate_score_arrays = [np.array(external_results[alg][cand]['score']) for cand in candidate_stats.keys()]
    try:
        f_stat, anova_p = f_oneway(*candidate_score_arrays)
    except Exception:
        f_stat, anova_p = np.nan, np.nan

    summary_lines.append(f"   ANOVA test across candidate combos: F = {f_stat:.4f}, p-value = {anova_p:.2e}\n")

    # Paired T-test between best and second best candidate (if possible)
    best_scores = np.array(external_results[alg][best_candidate]['score'])
    second_best_scores = np.array(external_results[alg][second_best_candidate]['score'])
    try:
        t_stat, ttest_p = ttest_rel(best_scores, second_best_scores)
    except Exception:
        t_stat, ttest_p = np.nan, np.nan
    summary_lines.append(f"   Paired T-test (Best vs. 2nd best): T = {t_stat:.4f}, p-value = {ttest_p:.2e}\n\n")

# --- Save summary results ---
stats_filepath = os.path.join(RESULTS_DIR, "statistics_results.txt")
with open(stats_filepath, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"Saved statistical results to {stats_filepath}")

print("\nDone in", round(time.time() - start, 2), "seconds.")
