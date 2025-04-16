# clustering_pipeline_parallel.py
import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from joblib import Parallel, delayed
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from matplotlib import cm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Utility Functions ---------------- #
def safe_silhouette(X, labels):
    try:
        if np.any(np.isnan(labels)):
            return -1.0
        if len(np.unique(labels)) > 1:
            if -1 in np.unique(labels) and len(np.unique(labels)) > 2:
                mask = labels != -1
                return silhouette_score(X[mask], labels[mask], ensure_all_finite=True)
            else:
                return silhouette_score(X, labels, ensure_all_finite=True)
        else:
            return -1.0
    except Exception:
        return -1.0

def compute_cluster_centers(X, labels):
    centers = {}
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        if lab == -1:
            continue
        cluster_points = X[labels == lab]
        if len(cluster_points) > 0:
            centers[lab] = np.mean(cluster_points, axis=0)
    return centers

def assign_to_clusters(X, centers):
    center_values = np.array(list(centers.values()))
    center_labels = np.array(list(centers.keys()))
    dists = np.linalg.norm(X[:, None, :] - center_values[None, :, :], axis=2)
    min_indices = np.argmin(dists, axis=1)
    return center_labels[min_indices]

# ---------------- Internal Evaluation ---------------- #
def run_internal_trial(X_full, dims_range, cluster_range, algorithms, trial, sample_size):
    np.random.seed(42 + trial)
    idx = np.random.choice(len(X_full), sample_size, replace=False)
    X_sample = X_full[idx]
    scores = {alg: np.zeros((len(dims_range), len(cluster_range))) for alg in algorithms}

    for i, dim in enumerate(dims_range):
        X_pca = PCA(n_components=dim, random_state=42).fit_transform(X_sample)
        for j, k in enumerate(cluster_range):
            for alg in algorithms:
                try:
                    if alg == 'kmeans':
                        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_pca)
                    elif alg == 'hierarchical':
                        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X_pca)
                    elif alg == 'dbscan':
                        eps = 0.2 + 0.1 * (k - 2)
                        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_pca)
                    elif alg == 'gmm':
                        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
                        labels = gmm.fit(X_pca).predict(X_pca)
                    elif alg == 'fcm':
                        try:
                            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_pca.T, c=k, m=2, error=0.005, maxiter=1000, seed=42)
                        except TypeError:
                            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_pca.T, c=k, m=2, error=0.005, maxiter=1000)
                        labels = np.argmax(u, axis=0)
                    else:
                        continue
                    scores[alg][i, j] = safe_silhouette(X_pca, labels)
                except Exception as e:
                    print(f"❌ {alg} failed on PCA={dim}, k={k}: {e}")
                    scores[alg][i, j] = -1.0
    return scores

def internal_evaluation(X_full, dims_range, cluster_range, algorithms, n_trials, sample_size, n_jobs=-1):
    print(f"🔁 Running {n_trials} internal silhouette trials with sample size = {sample_size}")
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_internal_trial)(X_full, dims_range, cluster_range, algorithms, t, sample_size) for t in range(n_trials)
    )
    avg_scores = {alg: np.mean([r[alg] for r in results], axis=0) for alg in algorithms}
    return avg_scores

# ---------------- External Evaluation ---------------- #
def run_external_trial(X_full, evil_truth, alg, dim_val, clusters, ext_seed,
                        cluster_sample_size, train_sample_size, test_sample_size):
    try:
        np.random.seed(ext_seed)
        idx_cluster = np.random.choice(len(X_full), cluster_sample_size, replace=False)
        X_cluster = X_full[idx_cluster]
        pca = PCA(n_components=dim_val, random_state=42)
        X_cluster_pca = pca.fit_transform(X_cluster)

        if alg == 'kmeans':
            labels = KMeans(n_clusters=clusters, random_state=42).fit_predict(X_cluster_pca)
        elif alg == 'hierarchical':
            labels = AgglomerativeClustering(n_clusters=clusters).fit_predict(X_cluster_pca)
        elif alg == 'dbscan':
            eps = 0.2 + 0.1 * (clusters - 2)
            labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_cluster_pca)
        elif alg == 'gmm':
            gmm = GaussianMixture(n_components=clusters, covariance_type='full', random_state=42)
            labels = gmm.fit(X_cluster_pca).predict(X_cluster_pca)
        elif alg == 'fcm':
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_cluster_pca.T, c=clusters, m=2, error=0.005, maxiter=1000, seed=42)
            labels = np.argmax(u, axis=0)
        else:
            return None

        centers = compute_cluster_centers(X_cluster_pca, labels)
        if not centers:
            return None

        idx_train = np.random.choice(len(X_full), train_sample_size, replace=False)
        X_train = X_full[idx_train]
        y_train = evil_truth[idx_train]
        X_train_pca = pca.transform(X_train)
        train_labels = assign_to_clusters(X_train_pca, centers)

        cluster_flags = {}
        for cl in centers:
            idxs = train_labels == cl
            cluster_flags[cl] = (np.mean(y_train[idxs] == 1) >= 0.5) if np.any(idxs) else False

        idx_test = np.random.choice(len(X_full), test_sample_size, replace=False)
        X_test = X_full[idx_test]
        y_test = evil_truth[idx_test]
        X_test_pca = pca.transform(X_test)
        test_labels = assign_to_clusters(X_test_pca, centers)
        pred = np.array([1 if cluster_flags.get(cl, False) else 0 for cl in test_labels])

        tp = np.sum((y_test == 1) & (pred == 1))
        fp = np.sum((y_test == 0) & (pred == 1))
        fn = np.sum((y_test == 1) & (pred == 0))
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mi = mutual_info_score(y_test, pred)

        return {"mi": mi, "precision": precision, "recall": recall, "f1": f1, "dim": dim_val, "clusters": clusters, "alg": alg}
    except Exception as e:
        print(f"❌ External trial failed for {alg} - dim={dim_val}, k={clusters}: {e}")
        return None

def external_evaluation_parallel(X_full, evil_truth, candidate_params, algorithms, n_trials, cluster_sample_size, train_sample_size, test_sample_size, n_jobs=-1):
    all_jobs = []
    seed = 10000
    for alg in algorithms:
        for dim_val, clusters, _ in candidate_params[alg]:
            for t in range(n_trials):
                job_seed = seed + t
                all_jobs.append((X_full, evil_truth, alg, dim_val, clusters, job_seed, cluster_sample_size, train_sample_size, test_sample_size))

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_external_trial)(*args) for args in all_jobs
    )
    return pd.DataFrame([r for r in results if r is not None])

# ---------------- Visualization Functions ---------------- #
def save_heatmaps(average_scores, dims_range, cluster_range, output_dir):
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    for alg, matrix in average_scores.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(np.arange(len(cluster_range)))
        ax.set_xticklabels(cluster_range)
        ax.set_yticks(np.arange(len(dims_range)))
        ax.set_yticklabels(dims_range)
        ax.set_xlabel("Clusters")
        ax.set_ylabel("PCA Dimensions")
        ax.set_title(f"Silhouette Heatmap - {alg.upper()}")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(heatmap_dir, f"{alg}_heatmap.pdf"))
        plt.close()

def plot_external_results(df_results, output_dir):
    plots_dir = os.path.join(output_dir, "external_plots")
    os.makedirs(plots_dir, exist_ok=True)
    metrics = ["f1", "precision", "recall", "mi"]
    for metric in metrics:
        for group in ["clusters", "dim"]:
            plt.figure(figsize=(10, 6))
            for alg in df_results["alg"].unique():
                sub_df = df_results[df_results["alg"] == alg]
                grouped = sub_df.groupby(group)[metric].mean()
                plt.plot(grouped.index, grouped.values, label=alg)
            plt.xlabel(group.capitalize())
            plt.ylabel(metric.upper())
            plt.title(f"{metric.upper()} vs {group.capitalize()}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{metric}_vs_{group}.pdf"))
            plt.close()

# ---------------- Top Feature Extraction ---------------- #
def extract_top_features(X_full, df_results, feature_names, output_dir, top_n=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    feature_dir = os.path.join(output_dir, "top_features")
    os.makedirs(feature_dir, exist_ok=True)

    for alg in df_results["alg"].unique():
        top = df_results[df_results["alg"] == alg].sort_values(by="f1", ascending=False).head(top_n)
        rows = []
        for _, row in top.iterrows():
            dim = int(row["dim"])
            clusters = int(row["clusters"])
            pca = PCA(n_components=dim).fit(X_scaled)
            X_pca = pca.transform(X_scaled[:10000])
            if alg == 'kmeans':
                centers = KMeans(n_clusters=clusters, random_state=42).fit(X_pca).cluster_centers_
            elif alg == 'hierarchical':
                labels = AgglomerativeClustering(n_clusters=clusters).fit_predict(X_pca)
                centers = np.vstack([X_pca[labels == i].mean(axis=0) for i in range(clusters)])
            elif alg == 'dbscan':
                eps = 0.2 + 0.1 * (clusters - 2)
                labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_pca)
                unique = [u for u in np.unique(labels) if u != -1]
                centers = np.vstack([X_pca[labels == u].mean(axis=0) for u in unique]) if unique else np.array([])
            elif alg == 'gmm':
                model = GaussianMixture(n_components=clusters, covariance_type='full', random_state=42).fit(X_pca)
                labels = model.predict(X_pca)
                centers = np.vstack([X_pca[labels == i].mean(axis=0) for i in range(clusters)])
            elif alg == 'fcm':
                cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_pca.T, c=clusters, m=2, error=0.005, maxiter=1000, seed=42)
                centers = cntr
            else:
                continue
            if centers.size == 0:
                continue
            orig_centers = pca.inverse_transform(centers)
            feature_std = np.std(orig_centers, axis=0)
            top_indices = np.argsort(feature_std)[::-1][:top_n]
            top_feats = [(feature_names[i], feature_std[i]) for i in top_indices]
            rows.append({"dim": dim, "clusters": clusters, "f1": row["f1"], "features": "; ".join([f"{f}:{v:.3f}" for f, v in top_feats])})
        pd.DataFrame(rows).to_csv(os.path.join(feature_dir, f"{alg}_top_features.csv"), index=False)

# ---------------- Main Pipeline ---------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="preprocessed/prepared_data_cluster.csv")
    parser.add_argument("--out", type=str, default="final_tests")
    args = parser.parse_args()

    DATA_PATH = args.data
    RESULTS_DIR = args.out

    # Hard-code the number of top candidate configurations to use for external evaluation
    X_TOP = 5
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X_full = df.drop(columns=["sus", "evil", "split"]).values
    evil_truth = df['evil'].values
    feature_names = list(df.drop(columns=["sus", "evil", "split"]).columns)
    dims_range = list(range(3, X_full.shape[1], 2))
    CLUSTER_RANGE = list(range(11, 30))
    ALGORITHMS = ['kmeans', 'hierarchical', 'dbscan', 'gmm', 'fcm']

    # Run internal evaluation to get silhouette scores
    avg_scores = internal_evaluation(X_full, dims_range, CLUSTER_RANGE, ALGORITHMS, n_trials=10, sample_size=5000)

    # Filter candidate parameters to the top X for each algorithm based on internal score (X_TOP is hard-coded)
    candidate_params = {}
    for alg in ALGORITHMS:
        candidates = [(dims_range[i], CLUSTER_RANGE[j], avg_scores[alg][i, j])
                      for i in range(len(dims_range))
                      for j in range(len(CLUSTER_RANGE))
                      if avg_scores[alg][i, j] > -1]
        candidates_sorted = sorted(candidates, key=lambda x: x[2], reverse=True)
        candidate_params[alg] = candidates_sorted[:X_TOP]

    # Run external evaluation on the filtered candidate parameters
    df_results = external_evaluation_parallel(
        X_full, evil_truth, candidate_params, ALGORITHMS, n_trials=5,
        cluster_sample_size=5000, train_sample_size=1000, test_sample_size=500
    )
    df_results.to_csv(os.path.join(RESULTS_DIR, "external_eval_results.csv"), index=False)
    print(f"📈 Saved external evaluation results to: {os.path.join(RESULTS_DIR, 'external_eval_results.csv')}")

    save_heatmaps(avg_scores, dims_range, CLUSTER_RANGE, RESULTS_DIR)
    plot_external_results(df_results, RESULTS_DIR)
    extract_top_features(X_full, df_results, feature_names, RESULTS_DIR)
    print("🏁 All visualizations and top features saved!")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"🏁 Pipeline complete in {round(time.time() - start, 2)}s")
