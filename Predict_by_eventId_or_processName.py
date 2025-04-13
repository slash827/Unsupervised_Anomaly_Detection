import os
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from scipy.stats import ttest_rel, friedmanchisquare, chi2


def load_and_split_data(df, feature_prefix, cluster_size, label_size, test_size):
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    if len(feature_cols) == 0:
        raise ValueError(f"No columns start with '{feature_prefix}'")

    cluster_df = df.iloc[:cluster_size].copy()
    label_df = df.iloc[cluster_size:cluster_size + label_size].copy()
    test_df = df.iloc[cluster_size + label_size:].copy()

    return cluster_df, label_df, test_df, feature_cols


def cluster_and_label(X_cluster, X_label, y_label, eps=0.1, min_samples=10):
    pca = PCA(n_components=2, random_state=42)
    X_cluster_pca = pca.fit_transform(X_cluster)
    X_label_pca = pca.transform(X_label)

    dbscan = DBSCAN(eps=eps, min_samples=10)
    cluster_labels = dbscan.fit_predict(X_cluster_pca)

    label_nbrs = NearestNeighbors(n_neighbors=1).fit(X_cluster_pca)
    _, indices = label_nbrs.kneighbors(X_label_pca)
    matched_labels = cluster_labels[indices.flatten()]

    cluster_pred = {}
    for label in np.unique(cluster_labels):
        mask = matched_labels == label
        cluster_pred[label] = 1 if np.mean(y_label[mask]) >= 0.5 else 0 if np.sum(mask) > 0 else 0

    return pca, cluster_labels, cluster_pred, X_cluster_pca


def evaluate_on_test(X_test, y_test, pca, cluster_labels, cluster_pred, X_cluster_pca):
    X_test_pca = pca.transform(X_test)
    test_nbrs = NearestNeighbors(n_neighbors=1).fit(X_cluster_pca)
    _, test_idx = test_nbrs.kneighbors(X_test_pca)
    test_cluster_labels = cluster_labels[test_idx.flatten()]
    test_preds = np.array([cluster_pred.get(label, 0) for label in test_cluster_labels])

    report = classification_report(y_test, test_preds, output_dict=True)
    print(classification_report(y_test, test_preds))

    return test_preds, y_test, report


def run_dbscan_pipeline(df, feature_prefix, cluster_size, label_size, test_df, test_size, eps=0.1, min_samples=10):
    cluster_df, label_df, _, feature_cols = load_and_split_data(df, feature_prefix, cluster_size, label_size, test_size)
    X_cluster = cluster_df[feature_cols].values
    X_label = label_df[feature_cols].values
    y_label = label_df['evil'].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df['evil'].values.astype(int)

    pca, cluster_labels, cluster_pred, X_cluster_pca = cluster_and_label(X_cluster, X_label, y_label, eps, min_samples)
    test_preds, y_test, report = evaluate_on_test(X_test, y_test, pca, cluster_labels, cluster_pred, X_cluster_pca)

    return test_preds, y_test, report


if __name__ == "__main__":
    start_time = time.time()
    data_file = "preprocessed/prepared_data_cluster.csv"
    df = pd.read_csv(data_file).sample(frac=1, random_state=42).reset_index(drop=True)

    # Parameters
    cluster_size = 25000
    label_size = 10000
    test_size = 10000
    n_test_runs = 10

    test_pool_df = df[cluster_size + label_size:].copy().reset_index(drop=True)
    results_log = []

    # Create base clustering once
    for i in range(n_test_runs):
        test_df = test_pool_df.sample(n=test_size, random_state=100 + i).reset_index(drop=True)

        event_pred, y_test, event_report = run_dbscan_pipeline(
            df, "eventId_MCA", cluster_size, label_size, test_df, test_size)
        proc_pred, _, proc_report = run_dbscan_pipeline(
            df, "processName_MCA", cluster_size, label_size, test_df, test_size)

        combined_pred = np.logical_or(event_pred, proc_pred).astype(int)

        combined_precision = precision_score(y_test, combined_pred)
        combined_recall = recall_score(y_test, combined_pred)
        combined_f1 = f1_score(y_test, combined_pred)

        fn_event = np.sum((event_pred == 0) & (y_test == 1)) / np.sum(y_test == 1)
        fn_process = np.sum((proc_pred == 0) & (y_test == 1)) / np.sum(y_test == 1)
        fn_combined = np.sum((combined_pred == 0) & (y_test == 1)) / np.sum(y_test == 1)

        results_log.append({
            'run': i + 1,
            'event_precision': event_report['1']['precision'],
            'event_recall': event_report['1']['recall'],
            'event_f1': event_report['1']['f1-score'],
            'process_precision': proc_report['1']['precision'],
            'process_recall': proc_report['1']['recall'],
            'process_f1': proc_report['1']['f1-score'],
            'combined_precision': combined_precision,
            'combined_recall': combined_recall,
            'combined_f1': combined_f1,
            'fnr_event': fn_event,
            'fnr_process': fn_process,
            'fnr_combined': fn_combined
        })

        print(f"\nTest Run {i + 1}: Combined Precision: {combined_precision:.4f}, Recall: {combined_recall:.4f}, F1: {combined_f1:.4f}")
        print(f"FNRs - Event: {fn_event:.4f}, Process: {fn_process:.4f}, Combined: {fn_combined:.4f}")

    # Save to CSV
    results_df = pd.DataFrame(results_log)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/dbscan_combined_results.csv", index=False)

    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
