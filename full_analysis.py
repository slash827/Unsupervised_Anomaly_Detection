#!/usr/bin/env python3
"""
combined_clustering_pipeline.py

Unified script: performs internal clustering evaluation, determines optimal hyperparameters
via elbow (inertia) and silhouette analysis, runs external evaluations for various algorithms
with those hyperparameters, runs DBSCAN pipelines (full‐feature and eventId_MCA),
and computes feature importance via PCA loadings.
"""
import os
import warnings
import time

import numpy as np
import random
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    precision_score,
    recall_score,
    f1_score,
    mutual_info_score,
    confusion_matrix
)
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
from kneed import KneeLocator
import skfuzzy as fuzz
from scipy.stats import f_oneway, ttest_rel

plt.rcParams.update({'axes.labelsize': 16,})
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Utility ---------------- #

def safe_silhouette(X, labels):
    """Compute silhouette score ignoring noise points (label == -1)."""
    mask = labels != -1
    labels_clean = labels[mask]
    if len(np.unique(labels_clean)) < 2:
        return -1.0
    return silhouette_score(X[mask], labels_clean)

def compute_true_f1(y_true, y_pred):
    """Compute micro-averaged F1 from total TP, FP, FN."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    if (2 * TP + FP + FN) == 0:
        return 0.0
    return (2 * TP) / (2 * TP + FP + FN)


def save_full_metrics(metrics_dict, out_dir, file_prefix, title):
    """
    Saves metrics:
    - Pretty printed report with mean, std, avg
    - Full per-run metrics to CSV

    Args:
    - metrics_dict: dict of metric name -> list of values
    - out_dir: folder to save files in
    - file_prefix: filename prefix (no extension)
    - title: title for the .txt report
    """
    # Compute averages
    avg_metrics = {k: np.mean(v) for k, v in metrics_dict.items()}

    # Print to screen
    print(f"\n📊 {title} (Mean ± Std):")
    for k in metrics_dict:
        vals = metrics_dict[k]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"{k:>15}: mean={mean_val:.4f}, std={std_val:.4f}, avg={mean_val:.4f}")

    # Save report txt
    with open(os.path.join(out_dir, f'{file_prefix}_report.txt'), 'w') as f:
        f.write(f"{title} (Mean ± Std)\n")
        for k in metrics_dict:
            vals = metrics_dict[k]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            f.write(f"{k:>15}: mean={mean_val:.4f}, std={std_val:.4f}, avg={mean_val:.4f}\n")

    # Save full per-run CSV
    pd.DataFrame(metrics_dict).to_csv(
        os.path.join(out_dir, f'{file_prefix}_metrics.csv'), index=False
    )


# ---------------- Feature Importance ---------------- #

def investigate_cluster_statistically(df, labels, cluster_id, cluster_is_evil, feature_types=None):
    import scipy.stats as stats
    import warnings
    import pandas as pd

    if not cluster_is_evil:
        return []  # Skip non-evil clusters, no printing!

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    feature_cols = [c for c in df.columns if c not in {'evil', 'sus', 'split'}]
    cluster_mask = (labels == cluster_id)
    results = []

    for col in feature_cols:
        x_in = df.loc[cluster_mask, col]
        x_out = df.loc[~cluster_mask, col]

        # Guess feature type
        if feature_types:
            ftype = feature_types.get(col, 'numeric')
        else:
            ftype = 'categorical' if df[col].nunique() <= 10 else 'numeric'

        try:
            if ftype == 'numeric':
                stat, p = stats.ttest_ind(x_in, x_out, equal_var=False)
                d = (x_in.mean() - x_out.mean()) / (x_in.std() + 1e-6)
                results.append({
                    'cluster': cluster_id, 'is_evil': cluster_is_evil,
                    'feature': col, 'group': col.split('_')[0],
                    'type': 'numeric', 'p': p, 'effect_size': abs(d)
                })
            else:
                tbl = pd.crosstab(labels == cluster_id, df[col])
                chi2, p, _, _ = stats.chi2_contingency(tbl)
                results.append({
                    'cluster': cluster_id, 'is_evil': cluster_is_evil,
                    'feature': col, 'group': col.split('_')[0],
                    'type': 'categorical', 'p': p, 'effect_size': chi2
                })
        except Exception:
            continue

    results.sort(key=lambda x: (x['p'], -x['effect_size']))

    # ✅ Only if evil and there are significant features
    df_res = pd.DataFrame(results)
    df_res = df_res[df_res['p'] < 0.05]
    if not df_res.empty:
        grouped = df_res.groupby('group').agg(
            total_effect=('effect_size', 'sum'),
            count=('feature', 'count')
        )
        grouped['normalized'] = grouped['total_effect'] / grouped['count']
        top = grouped['normalized'].sort_values(ascending=False).head(3)
        print(f"\n🏆 Top normalized feature groups for evil cluster {cluster_id}:")
        for i, (name, score) in enumerate(top.items(), 1):
            print(f"  {i}. {name} (score = {score:.2f})")

    return results

# ---------------- Internal Evaluation ---------------- #

def run_internal_trial(X, dims, clusters, algs, seed, sample_size):
    np.random.seed(seed)
    idx = np.random.choice(len(X), sample_size, replace=False)
    Xs = X[idx]
    inertia_scores = {'kmeans': np.zeros((len(dims), len(clusters)))}
    sil_scores     = {a: np.zeros((len(dims), len(clusters))) for a in algs}

    for i, d in enumerate(dims):
        Xp = PCA(n_components=d, random_state=seed).fit_transform(Xs)
        for j, k in enumerate(clusters):
            for a in algs:
                try:
                    if a == 'kmeans':
                        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xp)
                        inertia_scores[a][i, j] = model.inertia_
                        sil_scores[a][i, j]     = safe_silhouette(Xp, model.labels_)
                    else:
                        if a == 'hierarchical':
                            labels = AgglomerativeClustering(n_clusters=k).fit_predict(Xp)
                        elif a == 'dbscan':
                            eps_val = 0.1 + 0.1 * (k - 2)
                            labels = DBSCAN(eps=eps_val, min_samples=5).fit_predict(Xp)
                        elif a == 'gmm':
                            labels = GaussianMixture(n_components=k, random_state=seed).fit_predict(Xp)
                        elif a == 'fcm':
                            cntr, u, *_ = fuzz.cluster.cmeans(
                                Xp.T, c=k, m=2.0, error=0.005, maxiter=1000
                            )
                            labels = np.argmax(u, axis=0)
                        else:
                            continue
                        sil_scores[a][i, j] = safe_silhouette(Xp, labels)
                except Exception:
                    # on error, record as bad score
                    if a == 'kmeans':
                        inertia_scores[a][i, j] = np.nan
                    sil_scores[a][i, j] = -1.0

    return inertia_scores, sil_scores

def internal_evaluation(X, dims, clusters, algs, n_trials, sample_size, n_jobs=-1):
    print(f"🔁 Running {n_trials} internal trials ({sample_size} samples)")
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_internal_trial)(X, dims, clusters, algs, 42 + t, sample_size)
        for t in range(n_trials)
    )
    # average over trials
    avg_inertia = {'kmeans': np.nanmean([r[0]['kmeans'] for r in results], axis=0)}
    avg_sil     = {a: np.nanmean([r[1][a] for r in results], axis=0) for a in algs}
    # gather raw silhouette records
    sil_records = []
    for t, (_, sil_mat) in enumerate(results):
        for a in algs:
            M = sil_mat[a]
            for i, d in enumerate(dims):
                for j, k in enumerate(clusters):
                    sil_records.append({
                        'trial': t, 'alg': a,
                        'dim': d, 'clusters': k,
                        'sil': M[i, j]
                    })
    df_sil = pd.DataFrame(sil_records)
    return avg_inertia, avg_sil, df_sil

# ---------------- Plot Helpers ---------------- #

def _save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved {path}")

def plot_internal_box(df_sil, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_sil, x='alg', y='sil', ax=ax)
    _save_fig(fig, out_dir, 'internal_silhouette_boxplot.pdf')

def plot_trends(avg_sil, dims, clusters, out_dir):
    # Silhouette vs Clusters
    fig, ax = plt.subplots(figsize=(8, 6))
    for alg, mat in avg_sil.items():
        if alg == 'kmeans': continue
        best_i = np.nanargmax(mat) // mat.shape[1]
        ax.plot(clusters, mat[best_i, :], label=alg)
    ax.set(xlabel='Clusters', ylabel='Silhouette')
    ax.legend()
    _save_fig(fig, out_dir, 'trend_sil_vs_clusters.pdf')
    # Silhouette vs Dimensions
    fig, ax = plt.subplots(figsize=(8, 6))
    for alg, mat in avg_sil.items():
        if alg == 'kmeans': continue
        best_j = np.nanargmax(mat) % mat.shape[1]
        ax.plot(dims, mat[:, best_j], label=alg)
    ax.set(xlabel='Dimensions', ylabel='Silhouette')
    ax.legend()
    _save_fig(fig, out_dir, 'trend_sil_vs_dims.pdf')

def plot_kmeans_inertia_heatmap(avg_inertia, dims, clusters, out_dir):
    mat = avg_inertia['kmeans']
    df_mat = pd.DataFrame(mat, index=dims, columns=clusters)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_mat, ax=ax, cmap='viridis_r',
                xticklabels=clusters, yticklabels=dims)
    ax.set( xlabel='Clusters (k)', ylabel='Dimensions')
    _save_fig(fig, out_dir, 'kmeans_inertia_heatmap.pdf')

def plot_kmeans_elbow(avg_inertia, dims, clusters, out_dir):
    for i, d in enumerate(dims):
        inertia_vals = avg_inertia['kmeans'][i, :]
        ks = clusters
        try:
            kl = KneeLocator(ks, inertia_vals, curve='convex', direction='decreasing')
            elbow_k = kl.elbow
        except Exception:
            elbow_k = None
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ks, inertia_vals, marker='o')
        if elbow_k:
            ax.axvline(elbow_k, linestyle='--', label=f'Elbow at k={elbow_k}')
            ax.legend()
        ax.set( xlabel='Clusters', ylabel='Inertia')
        _save_fig(fig, out_dir, f'kmeans_elbow_dim{d}.pdf')

def plot_silhouette_heatmaps(avg_sil, dims, clusters, out_dir):
    for alg, mat in avg_sil.items():
        df_mat = pd.DataFrame(mat, index=dims, columns=clusters)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_mat, ax=ax, cmap='viridis',
                    xticklabels=clusters, yticklabels=dims)
        ax.set(
               xlabel='Clusters', ylabel='Dimensions')
        _save_fig(fig, out_dir, f'{alg}_silhouette_heatmap.pdf')

# ---------------- External Evaluation ---------------- #

def load_and_split_data(df, cluster_size, label_size, test_size, seed=None):
    non_feats = {'evil', 'sus', 'split'}
    feature_cols = [c for c in df.columns if c not in non_feats]
    if seed is not None:
        cluster_df = df.sample(n=cluster_size, random_state=seed)
        rem        = df.drop(cluster_df.index)
        label_df   = rem.sample(n=label_size,   random_state=seed+1)
        rem2       = rem.drop(label_df.index)
        test_df    = rem2.sample(n=test_size,   random_state=seed+2)
    else:
        cluster_df = df.sample(n=cluster_size)
        rem        = df.drop(cluster_df.index)
        label_df   = rem.sample(n=label_size)
        rem2       = rem.drop(label_df.index)
        test_df    = rem2.sample(n=test_size)
    return cluster_df, label_df, test_df, feature_cols

def external_eval_single(X, y_true, cutoffs_frac,
                         method, n_clusters, pca_dims,
                         sample_size=10000, seed=0):
    """
    Evaluate anomaly detection performance for various clustering/statistical methods.

    Parameters:
    - X: array-like of shape (n_samples, n_features)
    - y_true: 1D array of true binary labels (0 = normal, 1 = anomaly)
    - cutoffs_frac: iterable of fractions (0 < frac < 1) to threshold anomalies
    - method: one of 'kmeans', 'hierarchical', 'dbscan', 'gmm', 'fcm'
    - n_clusters: number of clusters/components for clustering methods
    - pca_dims: number of PCA dimensions to reduce to before clustering/statistics
    - sample_size: number of points to subsample for each run
    - seed: RNG seed for reproducibility

    Returns:
    - metrics: dict mapping metric names to lists of values over cutoffs_frac
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), sample_size, replace=False)
    Xs, ys = X[idx], y_true[idx]

    pca = PCA(n_components=pca_dims, random_state=seed)
    Xp = pca.fit_transform(Xs)

    metrics = {
        'precision_pos': [], 'recall_pos': [], 'f1_pos': [],
        'precision_neg': [], 'recall_neg': [], 'f1_neg': [],
        'f1': [], 'mutual_info': []
    }

    if method in ('kmeans', 'hierarchical', 'dbscan'):
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(Xp)
            centers = model.cluster_centers_
        elif method == 'hierarchical':
            labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Xp)
            centers = np.vstack([Xp[labels == i].mean(axis=0) for i in range(n_clusters)])
        else:
            eps_val = 0.1 + 0.1 * (n_clusters - 2)
            labels = DBSCAN(eps=eps_val, min_samples=5).fit_predict(Xp)
            centers = np.vstack([Xp[labels == c].mean(axis=0) for c in np.unique(labels) if c != -1])

        for frac in cutoffs_frac:
            N = max(1, int(frac * len(Xp)))
            dists_matrix = np.stack([np.linalg.norm(Xp - ctr, axis=1) for ctr in centers], axis=1)
            dmin = dists_matrix.min(axis=1)
            idx_anom = np.argsort(dmin)[-N:]
            pred = np.zeros(len(Xp), dtype=int)
            pred[idx_anom] = 1

            metrics['precision_pos'].append(precision_score(ys, pred, zero_division=0))
            metrics['recall_pos'].append(recall_score(ys, pred, zero_division=0))
            metrics['f1_pos'].append(f1_score(ys, pred, zero_division=0))
            metrics['precision_neg'].append(precision_score(ys, pred, pos_label=0, zero_division=0))
            metrics['recall_neg'].append(recall_score(ys, pred, pos_label=0, zero_division=0))
            metrics['f1_neg'].append(f1_score(ys, pred, pos_label=0, zero_division=0))
            metrics['f1'].append(compute_true_f1(ys, pred))
            metrics['mutual_info'].append(mutual_info_score(ys, pred))

        return metrics

    if method == 'gmm':
        gmm = GaussianMixture(n_components=n_clusters, random_state=seed).fit(Xp)
        log_probs = gmm.score_samples(Xp)

        for frac in cutoffs_frac:
            cutoff = np.percentile(log_probs, frac * 100)
            pred = (log_probs < cutoff).astype(int)

            metrics['precision_pos'].append(precision_score(ys, pred, zero_division=0))
            metrics['recall_pos'].append(recall_score(ys, pred, zero_division=0))
            metrics['f1_pos'].append(f1_score(ys, pred, zero_division=0))
            metrics['precision_neg'].append(precision_score(ys, pred, pos_label=0, zero_division=0))
            metrics['recall_neg'].append(recall_score(ys, pred, pos_label=0, zero_division=0))
            metrics['f1_neg'].append(f1_score(ys, pred, pos_label=0, zero_division=0))
            metrics['f1'].append(compute_true_f1(ys, pred))
            metrics['mutual_info'].append(mutual_info_score(ys, pred))

        return metrics

    if method == 'fcm':
        cntr, u, *_ = fuzz.cluster.cmeans(Xp.T, c=n_clusters, m=2.0, error=0.005, maxiter=1000)
        max_u = np.max(u, axis=0)

        for frac in cutoffs_frac:
            pred = (max_u < frac).astype(int)

            metrics['precision_pos'].append(precision_score(ys, pred, zero_division=0))
            metrics['recall_pos'].append(recall_score(ys, pred, zero_division=0))
            metrics['f1_pos'].append(f1_score(ys, pred, zero_division=0))
            metrics['precision_neg'].append(precision_score(ys, pred, pos_label=0, zero_division=0))
            metrics['recall_neg'].append(recall_score(ys, pred, pos_label=0, zero_division=0))
            metrics['f1_neg'].append(f1_score(ys, pred, pos_label=0, zero_division=0))
            metrics['f1'].append(compute_true_f1(ys, pred))
            metrics['mutual_info'].append(mutual_info_score(ys, pred))

        return metrics

    raise ValueError(f"Unknown method '{method}'")


def external_eval_kmeans_split(df, cluster_size, label_size, test_size,
                               n_clusters, pca_dims, n_repeats=30, seed_base=42):
    """
    Evaluate anomaly detection with KMeans, using DBSCAN-split style:
    Label clusters as evil/good based on labeled set.
    """

    cutoffs_frac = np.arange(0.01, 1.0, 0.01)
    threshold = 0.2  # Evil threshold (same as dbscan_split)

    metrics = {metric: np.zeros((n_repeats, len(cutoffs_frac))) for metric in (
        'precision_pos', 'recall_pos', 'f1_pos',
        'precision_neg', 'recall_neg', 'f1_neg',
        'f1', 'mutual_info'
    )}

    feats = [c for c in df.columns if c not in {'evil', 'sus', 'split'}]

    for rep in range(n_repeats):
        rng = np.random.RandomState(seed_base + rep)

        # --- Splitting
        cluster_df = df.sample(n=cluster_size, random_state=seed_base)
        remaining = df.drop(cluster_df.index)
        label_df = remaining.sample(n=label_size, random_state=seed_base + 1)
        test_df = remaining.drop(label_df.index).sample(n=test_size, random_state=seed_base + 2 + rep)

        Xc = cluster_df[feats].values
        Xl = label_df[feats].values
        Xt = test_df[feats].values

        pca = PCA(n_components=pca_dims, random_state=seed_base)
        Xc_pca = pca.fit_transform(Xc)
        Xl_pca = pca.transform(Xl)
        Xt_pca = pca.transform(Xt)

        # --- Train KMeans
        model = KMeans(n_clusters=n_clusters, random_state=seed_base, n_init=10)
        model.fit(Xc_pca)
        centers = model.cluster_centers_

        # --- Label clusters
        cluster_assignments_label = model.predict(Xl_pca)

        yl = label_df['evil'].astype(int).values
        cluster_pred = {
            cl: int((cluster_assignments_label == cl).sum() > 0 and yl[cluster_assignments_label == cl].mean() > threshold)
            for cl in range(n_clusters)
        }

        # --- Predict test points
        cluster_assignments_test = model.predict(Xt_pca)
        preds_raw = np.array([cluster_pred.get(cl, 0) for cl in cluster_assignments_test])
        y_true = test_df['evil'].astype(int).values

        # --- Distance matrix for refining by cutoff
        dists_matrix = np.linalg.norm(Xt_pca[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        dmin = dists_matrix.min(axis=1)

        for i, frac in enumerate(cutoffs_frac):
            N = max(1, int(frac * len(Xt_pca)))
            idx_anom = np.argsort(dmin)[-N:]
            preds = np.zeros(len(Xt_pca), dtype=int)
            preds[idx_anom] = 1

            metrics['precision_pos'][rep, i] = precision_score(y_true, preds, zero_division=0)
            metrics['recall_pos'][rep, i] = recall_score(y_true, preds, zero_division=0)
            metrics['f1_pos'][rep, i] = f1_score(y_true, preds, zero_division=0)
            metrics['precision_neg'][rep, i] = precision_score(y_true, preds, pos_label=0, zero_division=0)
            metrics['recall_neg'][rep, i] = recall_score(y_true, preds, pos_label=0, zero_division=0)
            metrics['f1_neg'][rep, i] = f1_score(y_true, preds, pos_label=0, zero_division=0)
            metrics['f1'][rep, i] = compute_true_f1(y_true, preds)
            metrics['mutual_info'][rep, i] = mutual_info_score(y_true, preds)

    return {k: np.nanmean(v, axis=0).tolist() for k, v in metrics.items()}



def external_eval_dbscan_split(
    df,
    cluster_size, label_size, test_size,
    eps, pca_dims, n_repeats=30, seed_base=42
):
    """
    DBSCAN evaluation using cluster-assignment logic (like eventID_MCA version).

    Each cluster is labeled as evil if > threshold of its matched labeled points are evil.
    Then, each test point is assigned to its nearest cluster, and inherits the cluster's label.

    Parameters:
        df : pd.DataFrame - full dataset
        cluster_size : int - number of clustering samples
        label_size : int - number of labeled points
        test_size : int - number of test points per repeat
        eps : float - DBSCAN epsilon radius
        pca_dims : int - PCA dimensions
        n_repeats : int - number of test runs
        seed_base : int - base seed

    Returns:
        dict of average precision, recall, f1, and MI
    """
    metrics_acc = {k: [] for k in (
        'precision_pos', 'recall_pos', 'f1_pos',
        'precision_neg', 'recall_neg', 'f1_neg',
        'f1', 'mutual_info'
    )}

    feats = [c for c in df.columns if c not in {'evil', 'sus', 'split'}]
    threshold = 0.2  # evil threshold

    for rep in range(n_repeats):
        rng = np.random.RandomState(seed_base + rep)

        # Split data
        cluster_df = df.sample(n=cluster_size, random_state=seed_base)
        remaining = df.drop(cluster_df.index)
        label_df = remaining.sample(n=label_size, random_state=seed_base + 1)
        test_df = remaining.drop(label_df.index).sample(n=test_size, random_state=seed_base + 2 + rep)

        # Fit PCA and DBSCAN
        Xc = cluster_df[feats].values
        Xl = label_df[feats].values
        Xt = test_df[feats].values

        pca = PCA(n_components=pca_dims, random_state=seed_base)
        Xc_pca = pca.fit_transform(Xc)
        Xl_pca = pca.transform(Xl)
        Xt_pca = pca.transform(Xt)

        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(Xc_pca)
        if len(set(labels)) <= 1 or (set(labels) == {-1}):
            continue  # skip if all noise

        # Match labeled points to clusters
        nbrs = NearestNeighbors(n_neighbors=1).fit(Xc_pca)
        _, idx_lbl = nbrs.kneighbors(Xl_pca)
        matched_lbls = labels[idx_lbl.flatten()]
        yl = label_df['evil'].astype(int).values

        # Label clusters
        cluster_pred = {
            cl: int((matched_lbls == cl).sum() > 0 and yl[matched_lbls == cl].mean() > threshold)
            for cl in np.unique(labels)
        }

        # Predict test set
        _, idx_test = nbrs.kneighbors(Xt_pca)
        test_lbls = labels[idx_test.flatten()]
        preds = np.array([cluster_pred.get(lbl, 0) for lbl in test_lbls])
        y_true = test_df['evil'].astype(int).values

        # Evaluate
        metrics_acc['precision_pos'].append(precision_score(y_true, preds, zero_division=0))
        metrics_acc['recall_pos'].append(recall_score(y_true, preds, zero_division=0))
        metrics_acc['f1_pos'].append(f1_score(y_true, preds, zero_division=0))
        metrics_acc['precision_neg'].append(precision_score(y_true, preds, pos_label=0, zero_division=0))
        metrics_acc['recall_neg'].append(recall_score(y_true, preds, pos_label=0, zero_division=0))
        metrics_acc['f1_neg'].append(f1_score(y_true, preds, pos_label=0, zero_division=0))
        metrics_acc['f1'].append(compute_true_f1(y_true, preds))
        metrics_acc['mutual_info'].append(mutual_info_score(y_true, preds))

    # Average results
    return {k: float(np.mean(v)) for k, v in metrics_acc.items()}


def average_metrics(runs):
    avg = {}
    for k in runs[0].keys():
        avg[k] = np.mean([r[k] for r in runs], axis=0).tolist()
    return avg

# ---------------- DBSCAN Pipelines ---------------- #

def cluster_and_label_dbscan(X_cluster, X_label, y_label,
                             eps, pca_dims, min_samples=20, thr=0.2):
    """DBSCAN pipeline on arbitrary features with PCA."""
    pca    = PCA(n_components=pca_dims, random_state=42)
    Xc_pca = pca.fit_transform(X_cluster)
    Xl_pca = pca.transform(X_label)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xc_pca)
    nbrs   = NearestNeighbors(n_neighbors=1).fit(Xc_pca)
    _, idx = nbrs.kneighbors(Xl_pca)
    matched = labels[idx.flatten()]
    cluster_pred = {
        c: int((matched==c).sum()>0 and y_label[matched==c].mean()>thr)
        for c in np.unique(labels)
    }
    return pca, labels, cluster_pred, Xc_pca

def run_dbscan_eventid_mcat(df, cutoffs_frac, cluster_size, label_size, test_size,
                                    mca_cols, eps, pca_dims, min_samples=20, thr=0.2, n_runs=30):
    """
    DBSCAN pipeline (eventId_MCA columns), but selects the best cutoff
    based on F1 score averaged over 30 runs.
    """
    f1_matrix = []

    for run in range(n_runs):
        cluster_df, label_df, test_df, _ = load_and_split_data(
            df, cluster_size, label_size, test_size, seed=42 + run
        )

        Xc = cluster_df[mca_cols].values
        Xl = label_df[mca_cols].values
        Xt = test_df[mca_cols].values

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xc)
        nbrs = NearestNeighbors(n_neighbors=1).fit(Xc)

        _, idx_lbl = nbrs.kneighbors(Xl)
        matched_lbls = labels[idx_lbl.flatten()]
        yl = label_df['evil'].astype(int).values

        cluster_pred = {
            cl: int((matched_lbls == cl).sum() > 0 and yl[matched_lbls == cl].mean() > thr)
            for cl in np.unique(labels)
        }

        _, idx_test = nbrs.kneighbors(Xt)
        test_lbls = labels[idx_test.flatten()]
        y_true = test_df['evil'].astype(int).values

        # Score by cutoff
        dists_matrix = np.stack([np.linalg.norm(Xt - ctr, axis=1)
                                 for ctr in [Xc[labels == c].mean(axis=0) for c in cluster_pred if c != -1]])
        dmin = dists_matrix.min(axis=0)

        f1_scores = []
        for frac in cutoffs_frac:
            N = max(1, int(frac * len(Xt)))
            idx_anom = np.argsort(dmin)[-N:]
            preds = np.zeros(len(Xt), dtype=int)
            preds[idx_anom] = 1
            f1 = f1_score(y_true, preds, zero_division=0)
            f1_scores.append(f1)

        f1_matrix.append(f1_scores)

    # Select best cutoff (highest average F1)
    f1_matrix = np.array(f1_matrix)
    avg_f1_by_cutoff = f1_matrix.mean(axis=0)
    best_cutoff_index = np.argmax(avg_f1_by_cutoff)
    best_frac = cutoffs_frac[best_cutoff_index]

    print(f"\n🎯 Best cutoff for eventId_MCA DBSCAN: {best_frac:.2%} (avg F1 = {avg_f1_by_cutoff[best_cutoff_index]:.4f})")

    # Re-run to compute metrics at that best cutoff
    metrics_final = {
        'precision_pos': [], 'recall_pos': [], 'f1_pos': [],
        'precision_neg': [], 'recall_neg': [], 'f1_neg': [],
        'f1':[], 'mutual_info': []
    }

    for run in range(n_runs):
        cluster_df, label_df, test_df, _ = load_and_split_data(
            df, cluster_size, label_size, test_size, seed=42 + run
        )

        Xc = cluster_df[mca_cols].values
        Xl = label_df[mca_cols].values
        Xt = test_df[mca_cols].values

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xc)
        nbrs = NearestNeighbors(n_neighbors=1).fit(Xc)

        _, idx_lbl = nbrs.kneighbors(Xl)
        matched_lbls = labels[idx_lbl.flatten()]
        yl = label_df['evil'].astype(int).values

        cluster_pred = {
            cl: int((matched_lbls == cl).sum() > 0 and yl[matched_lbls == cl].mean() > thr)
            for cl in np.unique(labels)
        }

        _, idx_test = nbrs.kneighbors(Xt)
        test_lbls = labels[idx_test.flatten()]
        y_true = test_df['evil'].astype(int).values

        dists_matrix = np.stack([np.linalg.norm(Xt - ctr, axis=1)
                                 for ctr in [Xc[labels == c].mean(axis=0) for c in cluster_pred if c != -1]])
        dmin = dists_matrix.min(axis=0)

        N = max(1, int(best_frac * len(Xt)))
        idx_anom = np.argsort(dmin)[-N:]
        preds = np.zeros(len(Xt), dtype=int)
        preds[idx_anom] = 1

        metrics_final['precision_pos'].append(precision_score(y_true, preds, zero_division=0))
        metrics_final['recall_pos'].append(recall_score(y_true, preds, zero_division=0))
        metrics_final['f1_pos'].append(f1_score(y_true, preds, zero_division=0))
        metrics_final['precision_neg'].append(precision_score(y_true, preds, pos_label=0, zero_division=0))
        metrics_final['recall_neg'].append(recall_score(y_true, preds, pos_label=0, zero_division=0))
        metrics_final['f1_neg'].append(f1_score(y_true, preds, pos_label=0, zero_division=0))
        metrics_final['mutual_info'].append(mutual_info_score(y_true, preds))
    return {k: np.mean(v) for k, v in metrics_final.items()}


# ---------------- Feature Importance ---------------- #

def compute_pca_importance(df, n_components=30):
    feature_cols = [c for c in df.columns if c not in {'evil','sus','split'}]
    X = df[feature_cols].values
    pca = PCA(n_components=n_components, random_state=42)
    _ = pca.fit_transform(X)
    loadings = np.abs(pca.components_)
    importance = loadings.sum(axis=0)
    idx = np.argsort(importance)[::-1]
    return [(feature_cols[j], float(importance[j])) for j in idx]

# ---------------- Main ---------------- #

def main():
    # --- Params ---
    df = pd.read_csv('preprocessed/prepared_data_cluster.csv')
    cluster_size = 20000
    label_size = 20000
    test_size = 10000
    n_runs = 30
    int_sample = 10000
    ext_fracs = np.arange(0.01, 1, 0.01)
    threshold_k = 5
    cutoffs_frac = np.arange(0.01, 1.0, 0.01)
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    report_dir = os.path.join(out_dir, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(os.path.join(report_dir, 'dbscan_mca'), exist_ok=True)
    os.makedirs(os.path.join(report_dir, 'dbscan_processname'), exist_ok=True)
    os.makedirs(os.path.join(report_dir, 'statistical_tests'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'metrics'), exist_ok=True)
    # Prepare data
    df_feat = df.drop(columns=['sus', 'split'])
    y_full = df_feat.pop('evil').astype(int).values
    X_full = df_feat.values

    dims = list(range(5, 41, 5))
    clusters = list(range(2, 31, 3))
    algs = ['kmeans', 'hierarchical', 'dbscan', 'gmm', 'fcm']

    # 1) Internal evaluation
    avg_inertia, avg_sil, df_sil = internal_evaluation(
        X_full, dims, clusters, algs, n_runs, int_sample, n_jobs=-1
    )
    # Find the PCA dimension with lowest total inertia across all k
    total_inertia_per_dim = np.nansum(avg_inertia['kmeans'], axis=1)
    best_dim_inertia = dims[np.argmin(total_inertia_per_dim)]
    df_sil.to_csv(os.path.join(out_dir, 'internal_silhouettes.csv'), index=False)

    # 1b) Plot internal results
    plot_internal_box(df_sil, os.path.join(out_dir, 'internal_plots'))
    plot_trends(avg_sil, dims, clusters, os.path.join(out_dir, 'internal_plots'))
    plot_kmeans_inertia_heatmap(avg_inertia, dims, clusters, os.path.join(out_dir, 'kmeans_plots'))
    plot_kmeans_elbow(avg_inertia, dims, clusters, os.path.join(out_dir, 'kmeans_plots'))
    plot_silhouette_heatmaps(avg_sil, dims, clusters, os.path.join(out_dir, 'heatmaps'))

    # Determine optimal hyperparameters
    ks = clusters
    kl = KneeLocator(ks, avg_inertia['kmeans'][0], curve='convex', direction='decreasing')
    best_k_inertia = kl.elbow or ks[int(np.argmin(np.gradient(avg_inertia['kmeans'][0])))]
    # silhouette-based best params
    best_params_sil = {}
    for a in algs:
        mat = avg_sil[a]
        idx_max = np.unravel_index(np.nanargmax(mat), mat.shape)
        best_params_sil[a] = (clusters[idx_max[1]], dims[idx_max[0]])
    best_k_sil = {a: best_params_sil[a][0] for a in algs}
    best_dim_sil = {a: best_params_sil[a][1] for a in algs}
    best_eps_dbscan = 0.1 + 0.1 * (best_k_sil['dbscan'] - 2)

    # 2) External evaluation for clustering-based cuts
    ext_methods = [
        ('kmeans_inertia', 'kmeans', best_k_inertia, best_dim_inertia),
        ('hierarchical', 'hierarchical', best_k_sil['hierarchical'], best_dim_sil['hierarchical']),
        ('gmm', 'gmm', best_k_sil['gmm'], best_dim_sil['gmm']),
        ('fcm', 'fcm', best_k_sil['fcm'], best_dim_sil['fcm']),
    ]

    print("\n📌 External Evaluation Configuration:")
    for mid, method, k, d in ext_methods:
        print(f"{mid:>16}: method='{method}', clusters={k}, PCA dims={d}")
    print(f"{'dbscan_split':>16}: method='dbscan', eps={best_eps_dbscan:.2f}, PCA dims={best_dim_sil['dbscan']}")

    results = {}
    ext_raw = {}
    for mid, method, k, d in ext_methods:
        if method == 'kmeans':
            # Run KMeans split labeling
            kmeans_result = external_eval_kmeans_split(
                df,
                cluster_size=cluster_size,
                label_size=label_size,
                test_size=test_size,
                n_clusters=k,
                pca_dims=d,
                n_repeats=n_runs
            )
            results[mid] = kmeans_result
            # Save KMeans result as CSV
            # After getting the result
            df_kmeans = pd.DataFrame(kmeans_result, index=(cutoffs_frac * 100))
            df_kmeans.index.name = '% cutoff'
            df_kmeans.to_csv(os.path.join(out_dir, 'metrics', f'{mid}_metrics.csv'))

        else:
            # Handle other algorithms
            method_flag = 'dbscan' if (method != 'kmeans' and k >= threshold_k) else method
            if method_flag == 'dbscan':
                print(f"⚠️ {method!r} has k={k} ≥ {threshold_k}; using DBSCAN anomaly‐labeling logic")

            # Run multiple trials
            runs = [
                external_eval_single(
                    X_full, y_full, ext_fracs,
                    method=method_flag,
                    n_clusters=k,
                    pca_dims=d,
                    sample_size=int_sample,
                    seed=seed
                )
                for seed in range(n_runs)
            ]

            ext_raw[mid] = runs
            results[mid] = average_metrics(runs)

            # Save multi-cutoff result
            df_mid = pd.DataFrame(results[mid], index=(ext_fracs * 100))
            df_mid.index.name = '% cutoff'
            df_mid.to_csv(os.path.join(out_dir, 'metrics', f'{mid}_metrics.csv'))

    # --- 2b) Save DBSCAN split result separately
    dbsplit_result = external_eval_dbscan_split(
        df,
        cluster_size=cluster_size,
        label_size=label_size,
        test_size=test_size,
        eps=best_eps_dbscan,
        pca_dims=best_dim_sil['dbscan']
    )
    results['dbscan_split'] = {k: [v] for k, v in dbsplit_result.items()}

    # Save DBSCAN split results
    df_dbscan = pd.DataFrame(results['dbscan_split'], index=[1])
    df_dbscan.index.name = '% cutoff'
    df_dbscan.to_csv(os.path.join(out_dir, 'metrics', 'dbscan_split_metrics.csv'))
    stat_reports = []
    # ANOVA
    f_stat, p_anova = f_oneway(*(results[mid]['f1_pos'] for mid in results))
    stat_reports.append(f"ANOVA across methods: F={f_stat:.3f}, p={p_anova:.3e}")
    # MI ANOVA
    f_stat_mi, p_anova_mi = f_oneway(*(results[mid]['mutual_info'] for mid in results))
    stat_reports.append(f"MI ANOVA across methods: F={f_stat_mi:.3f}, p={p_anova_mi:.3e}")

    # paired t-tests
    mids = list(results.keys())
    for i in range(len(mids)):
        for j in range(i + 1, len(mids)):
            m1, m2 = mids[i], mids[j]
            try:
                t_stat, p_val = ttest_rel(results[m1]['mutual_info'], results[m2]['mutual_info'])
                stat_reports.append(f"MI {m1} vs {m2}: t={t_stat:.3f}, p={p_val:.3e}")
            except Exception as e:
                stat_reports.append(f"MI {m1} vs {m2}: ERROR — {e}")
    # paired t-tests for overall F1
    for i in range(len(mids)):
        for j in range(i + 1, len(mids)):
            m1, m2 = mids[i], mids[j]
            try:
                t_stat, p_val = ttest_rel(results[m1]['f1'], results[m2]['f1'])
                stat_reports.append(f"Overall F1 {m1} vs {m2}: t={t_stat:.3f}, p={p_val:.3e}")
            except Exception as e:
                stat_reports.append(f"Overall F1 {m1} vs {m2}: ERROR — {e}")

    with open(os.path.join(report_dir, 'statistical_tests', 'stat_tests.txt'), 'w') as f:
        f.write("\n".join(stat_reports))

    # 3) DBSCAN pipeline (full-feature)
    cluster_df, label_df, test_df, feats = load_and_split_data(
        df, cluster_size, label_size, test_size, seed=42
    )
    pca_full, labels_full, pred_map_full, Xc_full_pca = cluster_and_label_dbscan(
        cluster_df[feats].values,
        label_df[feats].values,
        label_df['evil'].astype(int).values,
        eps=best_eps_dbscan, pca_dims=best_dim_sil['dbscan']
    )
    nbrs_full = NearestNeighbors(n_neighbors=1).fit(Xc_full_pca)
    Xt_full = pca_full.transform(test_df[feats].values)
    _, idx_f = nbrs_full.kneighbors(Xt_full)

    evil_clusters = [cl for cl, is_evil in pred_map_full.items() if is_evil == 1]
    print(f"\n🔥 Found {len(evil_clusters)} evil clusters out of {len(set(labels_full)) - (1 if -1 in labels_full else 0)} total clusters")

    important_features = {}

    for cl in evil_clusters:
        # Get all points in that cluster
        mask = (labels_full == cl)
        if not np.any(mask):
            continue
        cluster_pca_points = Xc_full_pca[mask]

        # Compute mean point in PCA space (i.e., the cluster "center")
        pca_center = cluster_pca_points.mean(axis=0)

        # Inverse transform to original space
        orig_center = pca_full.inverse_transform(pca_center)

        # Save the inverse cluster center
        important_features[cl] = orig_center

    # 4) statistical proof of feature importance
    print("\n🔎 Statistical investigation of all clusters...")
    feats_df = cluster_df[feats].copy()
    all_stats = []

    for cl in np.unique(labels_full):
        if cl == -1:
            continue  # skip noise
        cluster_is_evil = pred_map_full.get(cl, 0) == 1
        stats_result = investigate_cluster_statistically(feats_df, labels_full, cl, cluster_is_evil)
        all_stats.extend(stats_result)

    pd.DataFrame(all_stats).to_csv(os.path.join(out_dir, 'all_clusters_stats.csv'), index=False)

    # ---------------- KMeans-based Feature Importance ---------------- #

    print("\n🔍 Now running feature importance using KMeans clustering (5 dims, 8 clusters)...")

    # Prepare data
    X_cluster = cluster_df[feats].values

    # PCA to 5 dims
    pca_kmeans = PCA(n_components=5, random_state=42)
    Xc_pca = pca_kmeans.fit_transform(X_cluster)

    # KMeans with 8 clusters
    kmeans_model = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans_labels = kmeans_model.fit_predict(Xc_pca)

    # Investigate each cluster
    feats_df = cluster_df[feats].copy()
    all_stats_kmeans = []

    for cl in np.unique(kmeans_labels):
        # (here we *pretend* a cluster is "evil" just to get feature separation)
        stats_result = investigate_cluster_statistically(feats_df, kmeans_labels, cl, cluster_is_evil=True)
        all_stats_kmeans.extend(stats_result)

    # Save KMeans-based feature importance to CSV
    pd.DataFrame(all_stats_kmeans).to_csv(os.path.join(out_dir, 'all_clusters_stats_kmeans.csv'), index=False)

    # -------------------- 5) DBSCAN and KMeans on eventId_MCA -------------------- #

    print("\n🔍 Running DBSCAN and KMeans on eventId_MCA columns using nearest-cluster evil label assignment")

    mca_cols = [c for c in df.columns if c.startswith('eventId_MCA_')]

    # Initialize metric collectors
    metrics_mca_dbscan = {
        'precision_pos': [], 'recall_pos': [], 'f1_pos': [],
        'precision_neg': [], 'recall_neg': [], 'f1_neg': [],
        'f1': [], 'mutual_info': []
    }

    for run in range(n_runs):
        cluster_df, label_df, test_df, _ = load_and_split_data(
            df, cluster_size, label_size, test_size, seed=42 + run
        )

        # Prepare features
        Xc = cluster_df[mca_cols].values
        Xl = label_df[mca_cols].values
        Xt = test_df[mca_cols].values
        yl = label_df['evil'].astype(int).values
        y_true = test_df['evil'].astype(int).values

        # ------------------ DBSCAN ------------------ #
        dbscan_labels = DBSCAN(eps=best_eps_dbscan, min_samples=20).fit_predict(Xc)
        dbscan_nbrs = NearestNeighbors(n_neighbors=1).fit(Xc)
        _, idx_lbl_dbscan = dbscan_nbrs.kneighbors(Xl)
        matched_dbscan = dbscan_labels[idx_lbl_dbscan.flatten()]

        cluster_pred_dbscan = {
            cl: int((matched_dbscan == cl).sum() > 0 and yl[matched_dbscan == cl].mean() > 0.2)
            for cl in np.unique(dbscan_labels)
        }

        _, idx_test_dbscan = dbscan_nbrs.kneighbors(Xt)
        test_lbls_dbscan = dbscan_labels[idx_test_dbscan.flatten()]
        preds_dbscan = np.array([cluster_pred_dbscan.get(lbl, 0) for lbl in test_lbls_dbscan])

        # Save DBSCAN metrics
        metrics_mca_dbscan['precision_pos'].append(precision_score(y_true, preds_dbscan, zero_division=0))
        metrics_mca_dbscan['recall_pos'].append(recall_score(y_true, preds_dbscan, zero_division=0))
        metrics_mca_dbscan['f1_pos'].append(f1_score(y_true, preds_dbscan, zero_division=0))
        metrics_mca_dbscan['precision_neg'].append(precision_score(y_true, preds_dbscan, pos_label=0, zero_division=0))
        metrics_mca_dbscan['recall_neg'].append(recall_score(y_true, preds_dbscan, pos_label=0, zero_division=0))
        metrics_mca_dbscan['f1_neg'].append(f1_score(y_true, preds_dbscan, pos_label=0, zero_division=0))
        metrics_mca_dbscan['f1'].append(compute_true_f1(y_true, preds_dbscan))
        metrics_mca_dbscan['mutual_info'].append(mutual_info_score(y_true, preds_dbscan))


    # ------------------ Save Results ------------------ #
    #DBSCAN metrics
    save_full_metrics(metrics_mca_dbscan, out_dir, 'dbscan_mca', "EventId_MCA DBSCAN Metrics (Cluster-Based Labeling)")
    # -------------------- 6) DBSCAN on processName_MCA -------------------- #

    print("\n🔍 Running DBSCAN and KMeans on processName_MCA columns using nearest-cluster evil label assignment")

    mca_cols = [c for c in df.columns if c.startswith('processName_MCA')]

    # Initialize metric collectors
    metrics_mca_dbscan = {
        'precision_pos': [], 'recall_pos': [], 'f1_pos': [],
        'precision_neg': [], 'recall_neg': [], 'f1_neg': [],
        'f1': [], 'mutual_info': []
    }

    for run in range(n_runs):
        cluster_df, label_df, test_df, _ = load_and_split_data(
            df, cluster_size, label_size, test_size, seed=42 + run
        )

        # Prepare features
        Xc = cluster_df[mca_cols].values
        Xl = label_df[mca_cols].values
        Xt = test_df[mca_cols].values
        yl = label_df['evil'].astype(int).values
        y_true = test_df['evil'].astype(int).values

        # ------------------ DBSCAN ------------------ #
        dbscan_labels = DBSCAN(eps=best_eps_dbscan, min_samples=20).fit_predict(Xc)
        dbscan_nbrs = NearestNeighbors(n_neighbors=1).fit(Xc)
        _, idx_lbl_dbscan = dbscan_nbrs.kneighbors(Xl)
        matched_dbscan = dbscan_labels[idx_lbl_dbscan.flatten()]

        cluster_pred_dbscan = {
            cl: int((matched_dbscan == cl).sum() > 0 and yl[matched_dbscan == cl].mean() > 0.2)
            for cl in np.unique(dbscan_labels)
        }

        _, idx_test_dbscan = dbscan_nbrs.kneighbors(Xt)
        test_lbls_dbscan = dbscan_labels[idx_test_dbscan.flatten()]
        preds_dbscan = np.array([cluster_pred_dbscan.get(lbl, 0) for lbl in test_lbls_dbscan])

        # Save DBSCAN metrics
        metrics_mca_dbscan['precision_pos'].append(precision_score(y_true, preds_dbscan, zero_division=0))
        metrics_mca_dbscan['recall_pos'].append(recall_score(y_true, preds_dbscan, zero_division=0))
        metrics_mca_dbscan['f1_pos'].append(f1_score(y_true, preds_dbscan, zero_division=0))
        metrics_mca_dbscan['precision_neg'].append(precision_score(y_true, preds_dbscan, pos_label=0, zero_division=0))
        metrics_mca_dbscan['recall_neg'].append(recall_score(y_true, preds_dbscan, pos_label=0, zero_division=0))
        metrics_mca_dbscan['f1_neg'].append(f1_score(y_true, preds_dbscan, pos_label=0, zero_division=0))
        metrics_mca_dbscan['f1'].append(compute_true_f1(y_true, preds_dbscan))
        metrics_mca_dbscan['mutual_info'].append(mutual_info_score(y_true, preds_dbscan))


    # ------------------ Save Results ------------------ #

    # DBSCAN metrics
    save_full_metrics(metrics_mca_dbscan, out_dir, 'processName_mca_dbscan', "processName_MCA DBSCAN Metrics (Cluster-Based Labeling)")

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"\n✅ Pipeline completed in {time.time()- start} seconds. Check the 'results/' folder for all outputs.")
