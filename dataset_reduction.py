import os
import glob
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from scipy.stats import entropy
import seaborn as sns

from utils import load_and_preprocess_beth_data

# Set plot style and figure size for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Constants
TARGET_SIZE = 20000  # Number of samples to keep in the reduced dataset
RANDOM_SAMPLES = 5   # Number of random samples to generate for comparison


def evaluate_multiple_random_samples(features, labels, n_samples=5, target_size=20000):
    """
    Evaluate multiple random samples and average their information loss metrics.
    
    Args:
        features: Feature matrix (numpy array)
        labels: Target labels (numpy array)
        n_samples: Number of random samples to evaluate
        target_size: Size of each reduced dataset
    
    Returns:
        Average metrics across all samples
    """
    all_metrics = []
    
    print(f"\nEvaluating {n_samples} random stratified samples for comparison:")
    for i in range(n_samples):
        print(f"\n  Sample {i+1}/{n_samples}:")
        
        # Perform stratified random sampling
        sampled_indices = stratified_random_sampling(features, labels, target_size)
        reduced_features = features[sampled_indices]
        reduced_labels = labels[sampled_indices]
        
        print(f"  Measuring information loss for random sample {i+1}...")
        metrics = measure_information_loss(features, reduced_features, labels, reduced_labels)
        all_metrics.append(metrics)
        
        # Print key metrics for this sample
        print(f"  Sample {i+1} key metrics:")
        print(f"    KL divergence: {metrics.get('avg_kl_divergence', 0):.4f}")
        print(f"    JS distance: {metrics.get('avg_js_distance', 0):.4f}")
        print(f"    Class ratio diff: {metrics.get('class_ratio_diff', 0):.4f}")
        print(f"    ROC-AUC diff: {metrics.get('auc_diff', 0):.4f} " 
              f"({metrics.get('auc_orig', 0):.4f} -> {metrics.get('auc_red', 0):.4f})")
        print(f"    PR-AUC diff: {metrics.get('pr_auc_diff', 0):.4f} " 
              f"({metrics.get('pr_auc_orig', 0):.4f} -> {metrics.get('pr_auc_red', 0):.4f})")
        print(f"    Anomaly AUC diff: {metrics.get('anomaly_auc_diff', 0):.4f} " 
              f"({metrics.get('anomaly_auc_orig', 0):.4f} -> {metrics.get('anomaly_auc_red', 0):.4f})")
    
    # Average the metrics
    avg_metrics = {}
    # Get all unique keys from all metrics dictionaries
    all_keys = set()
    for metrics in all_metrics:
        all_keys.update(metrics.keys())
    
    # Average each metric across all samples
    for key in all_keys:
        # Use only metrics that have this key (some might be missing if errors occurred)
        values = [m[key] for m in all_metrics if key in m]
        if values:
            avg_metrics[key] = np.mean(values)
        else:
            avg_metrics[key] = 0.0  # Default if no valid values
    
    # Print average metrics
    print("\nAverage metrics across all random samples:")
    for key, value in sorted(avg_metrics.items()):
        print(f"  {key}: {value:.4f}")
    
    # Calculate standard deviation for key metrics
    std_metrics = {}
    key_metrics = ['avg_kl_divergence', 'avg_js_distance', 'class_ratio_diff', 
                  'pca_variance_diff', 'auc_diff', 'pr_auc_diff', 'anomaly_auc_diff']
    
    for key in key_metrics:
        values = [m.get(key, 0) for m in all_metrics if key in m]
        if len(values) > 1:  # Need at least 2 values for std
            std_metrics[key] = np.std(values)
        else:
            std_metrics[key] = 0.0
    
    print("\nStandard deviation across random samples:")
    for key, value in sorted(std_metrics.items()):
        print(f"  {key}: {value:.4f}")
    
    return avg_metrics


def stratified_random_sampling(features, labels, target_size=20000):
    """
    Perform stratified random sampling to preserve class distribution.
    
    Args:
        features: Feature matrix (numpy array)
        labels: Target labels (numpy array)
        target_size: Number of samples to keep
    
    Returns:
        Indices of selected samples
    """
    # Calculate the original class distribution
    n_total = len(labels)
    n_evil = np.sum(labels)
    evil_ratio = n_evil / n_total
    
    # Calculate how many samples of each class to keep
    n_evil_to_keep = int(target_size * evil_ratio)
    n_benign_to_keep = target_size - n_evil_to_keep
    
    # Get indices for each class
    evil_indices = np.where(labels == 1)[0]
    benign_indices = np.where(labels == 0)[0]
    
    # Make sure we don't try to sample more than available
    n_evil_to_keep = min(n_evil_to_keep, len(evil_indices))
    n_benign_to_keep = min(n_benign_to_keep, len(benign_indices))
    
    # Sample from each class
    np.random.seed(42)  # For reproducibility
    sampled_evil_indices = np.random.choice(evil_indices, size=n_evil_to_keep, replace=False)
    sampled_benign_indices = np.random.choice(benign_indices, size=n_benign_to_keep, replace=False)
    
    # Combine the indices
    sampled_indices = np.concatenate([sampled_evil_indices, sampled_benign_indices])
    np.random.shuffle(sampled_indices)
    
    print(f"  Stratified random sampling:")
    print(f"    Original evil ratio: {evil_ratio:.4f}")
    print(f"    Keeping {n_evil_to_keep} evil and {n_benign_to_keep} benign samples")
    print(f"    Total samples selected: {len(sampled_indices)}")
    
    return sampled_indices


def kmeans_stratified_sampling(features, labels, target_size=TARGET_SIZE):
    """
    Perform a hybrid sampling using K-means clustering within each class.
    
    Args:
        features: Feature matrix
        labels: Target labels (evil)
        target_size: Number of samples to keep
    
    Returns:
        Indices of selected samples
    """
    from tqdm import tqdm
    
    # Calculate the original class distribution
    n_total = len(labels)
    n_evil = np.sum(labels)
    evil_ratio = n_evil / n_total
    
    # Calculate how many samples of each class to keep
    n_evil_to_keep = int(target_size * evil_ratio)
    n_benign_to_keep = target_size - n_evil_to_keep
    
    # Get indices and features for each class
    evil_indices = np.where(labels == 1)[0]
    benign_indices = np.where(labels == 0)[0]
    evil_features = features[evil_indices]
    benign_features = features[benign_indices]
    
    # Number of clusters for each class
    # We use slightly more clusters than needed to have options
    n_clusters_evil = min(n_evil_to_keep, 750)  # Cap at a reasonable number
    n_clusters_benign = min(n_benign_to_keep, 1500)
    
    print("\nHybrid K-means + Stratified Sampling:")
    print(f"  Original distribution: {len(evil_indices)} evil, {len(benign_indices)} benign")
    print(f"  Original evil ratio: {evil_ratio:.4f}")
    print(f"  Target: {n_evil_to_keep} evil samples, {n_benign_to_keep} benign samples")
    print(f"  Using {n_clusters_evil} clusters for evil class")
    print(f"  Using {n_clusters_benign} clusters for benign class")
    
    # Apply K-means to evil class
    print("  Running K-means for evil samples...")
    # Let's use regular KMeans for simplicity but with progress output
    kmeans_evil = KMeans(n_clusters=n_clusters_evil, random_state=42, n_init=10, verbose=0)
    
    # Since we had issues with custom initialization, let's use the standard KMeans
    with tqdm(total=100, desc="Evil clustering") as pbar:
        # We can't easily track progress with standard KMeans, so we'll simulate it
        pbar.update(10)  # Show initial progress
        evil_cluster_labels = kmeans_evil.fit_predict(evil_features)
        pbar.update(90)  # Complete the progress bar
    
    # Count points per cluster for evil class
    evil_cluster_counts = np.bincount(evil_cluster_labels, minlength=n_clusters_evil)
    print(f"  Evil cluster sizes - min: {np.min(evil_cluster_counts)}, "
          f"max: {np.max(evil_cluster_counts)}, "
          f"mean: {np.mean(evil_cluster_counts):.1f}")
    
    # Apply K-means to benign class
    print("  Running K-means for benign samples...")
    kmeans_benign = KMeans(n_clusters=n_clusters_benign, random_state=42, n_init=10, verbose=0)
    
    with tqdm(total=100, desc="Benign clustering") as pbar:
        # We can't easily track progress with standard KMeans, so we'll simulate it
        pbar.update(10)  # Show initial progress
        benign_cluster_labels = kmeans_benign.fit_predict(benign_features)
        pbar.update(90)  # Complete the progress bar
    
    # Count points per cluster for benign class
    benign_cluster_counts = np.bincount(benign_cluster_labels, minlength=n_clusters_benign)
    print(f"  Benign cluster sizes - min: {np.min(benign_cluster_counts)}, "
          f"max: {np.max(benign_cluster_counts)}, "
          f"mean: {np.mean(benign_cluster_counts):.1f}")
    
    # Find the closest point to each cluster center for evil samples
    print("  Selecting representative points from evil clusters...")
    selected_evil_indices = []
    for i in tqdm(range(n_clusters_evil), desc="Processing evil clusters"):
        cluster_points = np.where(evil_cluster_labels == i)[0]
        if len(cluster_points) > 0:
            cluster_center = kmeans_evil.cluster_centers_[i]
            distances = np.linalg.norm(evil_features[cluster_points] - cluster_center, axis=1)
            closest_point_idx = cluster_points[np.argmin(distances)]
            selected_evil_indices.append(evil_indices[closest_point_idx])
    
    # Find the closest point to each cluster center for benign samples
    print("  Selecting representative points from benign clusters...")
    selected_benign_indices = []
    for i in tqdm(range(n_clusters_benign), desc="Processing benign clusters"):
        cluster_points = np.where(benign_cluster_labels == i)[0]
        if len(cluster_points) > 0:
            cluster_center = kmeans_benign.cluster_centers_[i]
            distances = np.linalg.norm(benign_features[cluster_points] - cluster_center, axis=1)
            closest_point_idx = cluster_points[np.argmin(distances)]
            selected_benign_indices.append(benign_indices[closest_point_idx])
    
    # If we have more selected points than needed, take the ones closest to their centroids
    if len(selected_evil_indices) > n_evil_to_keep:
        selected_evil_indices = selected_evil_indices[:n_evil_to_keep]
        print(f"  Trimmed evil samples to {n_evil_to_keep}")
    
    if len(selected_benign_indices) > n_benign_to_keep:
        selected_benign_indices = selected_benign_indices[:n_benign_to_keep]
        print(f"  Trimmed benign samples to {n_benign_to_keep}")
    
    # After clustering and finding representatives from each cluster
    # Ensure we get exactly the right number of samples per class
    
    # Store original counts before adding random samples
    n_from_clustering_evil = len(selected_evil_indices)
    n_from_clustering_benign = len(selected_benign_indices)

    # If we have too few samples
    if len(selected_evil_indices) < n_evil_to_keep:
        # Add more evil samples using random selection
        remaining_evil = set(evil_indices) - set(selected_evil_indices)
        additional_needed = n_evil_to_keep - len(selected_evil_indices)
        
        if additional_needed > 0 and len(remaining_evil) > 0:
            additional_indices = np.random.choice(
                list(remaining_evil), 
                size=min(additional_needed, len(remaining_evil)),
                replace=False
            )
            selected_evil_indices = np.append(selected_evil_indices, additional_indices)
    
    # Same for benign samples
    if len(selected_benign_indices) < n_benign_to_keep:
        # Add more benign samples using random selection
        remaining_benign = set(benign_indices) - set(selected_benign_indices)
        additional_needed = n_benign_to_keep - len(selected_benign_indices)
        
        if additional_needed > 0 and len(remaining_benign) > 0:
            additional_indices = np.random.choice(
                list(remaining_benign), 
                size=min(additional_needed, len(remaining_benign)),
                replace=False
            )
            selected_benign_indices = np.append(selected_benign_indices, additional_indices)
    
    # Combine and return exactly target_size samples
    sampled_indices = np.concatenate([
        selected_evil_indices[:n_evil_to_keep], 
        selected_benign_indices[:n_benign_to_keep]
    ])
    
    # Report statistics
    total_from_clustering = n_from_clustering_evil + n_from_clustering_benign
    print(f"Samples from clustering: {total_from_clustering}")
    print(f"Randomly added samples: {target_size - total_from_clustering}")
    print(f"Percentage from clustering: {total_from_clustering/target_size*100:.2f}%")

    print(f"  Final selection: {len(selected_evil_indices)} evil, {len(selected_benign_indices)} benign")
    print(f"  Total samples selected: {len(sampled_indices)}")
    print(f"  Final evil ratio: {len(selected_evil_indices)/len(sampled_indices):.4f}")
    
    return sampled_indices

def calculate_kl_divergence(hist1, hist2):
    """
    Calculate KL divergence between two histograms.
    Add small epsilon to avoid division by zero.
    
    Args:
        hist1: First histogram
        hist2: Second histogram
        
    Returns:
        KL divergence value
    """
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # Normalize
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    return entropy(hist1, hist2)


def calculate_js_distance(p, q):
    """
    Calculate Jensen-Shannon distance between distributions p and q.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        JS distance value
    """
    p = p + 1e-10
    q = q + 1e-10
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def measure_information_loss(original_features, reduced_features, original_labels, reduced_labels):
    """
    Measure information loss between original and reduced datasets.
    
    Args:
        original_features: Features of the original dataset (numpy array)
        reduced_features: Features of the reduced dataset (numpy array)
        original_labels: Labels of the original dataset (numpy array)
        reduced_labels: Labels of the reduced dataset (numpy array)
    
    Returns:
        Dictionary of information loss metrics
    """
    metrics = {}
    
    print("\nMeasuring information loss:")
    
    # 1. Feature distribution similarity (KL divergence)
    kl_divs = []
    js_dists = []
    
    print("  Calculating distribution similarity metrics...")
    # Calculate histograms for each feature
    n_features = original_features.shape[1]
    for i in range(n_features):
        # Create histograms with the same bins
        try:
            # Get min and max to determine range
            min_val = min(original_features[:, i].min(), reduced_features[:, i].min())
            max_val = max(original_features[:, i].max(), reduced_features[:, i].max())
            
            # Create bins with the same range for both datasets
            bins = np.linspace(min_val, max_val, 50)
            
            # Calculate histograms
            hist_orig, _ = np.histogram(original_features[:, i], bins=bins, density=True)
            hist_red, _ = np.histogram(reduced_features[:, i], bins=bins, density=True)
            
            # Calculate KL divergence and JS distance
            # Add small epsilon to avoid division by zero or log(0)
            hist_orig = hist_orig + 1e-10
            hist_red = hist_red + 1e-10
            
            # Normalize
            hist_orig = hist_orig / np.sum(hist_orig)
            hist_red = hist_red / np.sum(hist_red)
            
            # Calculate KL divergence
            kl_div = entropy(hist_orig, hist_red)
            
            # Calculate JS distance
            m = 0.5 * (hist_orig + hist_red)
            js_dist = 0.5 * (entropy(hist_orig, m) + entropy(hist_red, m))
            
            kl_divs.append(kl_div)
            js_dists.append(js_dist)
        except Exception as e:
            print(f"    Warning: Error calculating distribution metrics for feature {i}: {e}")
            kl_divs.append(0.0)
            js_dists.append(0.0)
    
    # Average KL divergence and JS distance
    metrics['avg_kl_divergence'] = np.mean(kl_divs)
    metrics['avg_js_distance'] = np.mean(js_dists)
    
    # Print feature-level divergence information
    print(f"  Feature divergence summary:")
    print(f"    KL divergence - min: {np.min(kl_divs):.4f}, max: {np.max(kl_divs):.4f}, mean: {np.mean(kl_divs):.4f}")
    print(f"    JS distance - min: {np.min(js_dists):.4f}, max: {np.max(js_dists):.4f}, mean: {np.mean(js_dists):.4f}")
    
    # 2. Class distribution preservation
    orig_class_ratio = np.mean(original_labels)
    red_class_ratio = np.mean(reduced_labels)
    metrics['class_ratio_diff'] = abs(orig_class_ratio - red_class_ratio)
    
    print(f"  Class distribution:")
    print(f"    Original evil ratio: {orig_class_ratio:.4f}")
    print(f"    Reduced evil ratio: {red_class_ratio:.4f}")
    print(f"    Absolute difference: {metrics['class_ratio_diff']:.4f}")
    
    # 3. PCA variance preservation
    print("  Calculating PCA variance preservation...")
    try:
        from sklearn.decomposition import PCA
        
        # Choose fewer components to avoid issues with small datasets
        n_components = min(5, n_features, reduced_features.shape[0] - 1)
        
        pca_orig = PCA(n_components=n_components)
        pca_red = PCA(n_components=n_components)
        
        pca_orig.fit(original_features)
        pca_red.fit(reduced_features)
        
        # Compare explained variance ratios
        orig_var_ratio = pca_orig.explained_variance_ratio_
        red_var_ratio = pca_red.explained_variance_ratio_
        
        # Ensure both have the same length
        min_len = min(len(orig_var_ratio), len(red_var_ratio))
        orig_var_ratio = orig_var_ratio[:min_len]
        red_var_ratio = red_var_ratio[:min_len]
        
        variance_diff = np.sum(np.abs(orig_var_ratio - red_var_ratio))
        metrics['pca_variance_diff'] = variance_diff
        
        print(f"    Original explained variance ratios: {orig_var_ratio}")
        print(f"    Reduced explained variance ratios: {red_var_ratio}")
        print(f"    Total variance difference: {variance_diff:.4f}")
    except Exception as e:
        print(f"    Warning: Error calculating PCA variance: {e}")
        metrics['pca_variance_diff'] = 0.0
    
    # 4. Model performance comparison (using a test set)
    print("  Evaluating model performance on test set...")
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        
        # Split original data into train and test
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            original_features, original_labels, test_size=0.2, random_state=42, stratify=original_labels
        )
        
        # Train classifiers
        print("    Training classifier on original data...")
        clf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_orig.fit(X_train_orig, y_train_orig)
        
        print("    Training classifier on reduced data...")
        clf_red = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_red.fit(reduced_features, reduced_labels)
        
        # Evaluate on the same test set
        y_pred_orig = clf_orig.predict_proba(X_test_orig)[:, 1]
        y_pred_red = clf_red.predict_proba(X_test_orig)[:, 1]
        
        # Calculate ROC-AUC
        auc_orig = roc_auc_score(y_test_orig, y_pred_orig)
        auc_red = roc_auc_score(y_test_orig, y_pred_red)
        metrics['auc_diff'] = auc_orig - auc_red
        metrics['auc_orig'] = auc_orig
        metrics['auc_red'] = auc_red
        
        print(f"    ROC-AUC scores:")
        print(f"      Original model: {auc_orig:.4f}")
        print(f"      Reduced model: {auc_red:.4f}")
        print(f"      Difference: {metrics['auc_diff']:.4f}")
        
        # Calculate Precision-Recall AUC
        precision_orig, recall_orig, _ = precision_recall_curve(y_test_orig, y_pred_orig)
        precision_red, recall_red, _ = precision_recall_curve(y_test_orig, y_pred_red)
        
        pr_auc_orig = auc(recall_orig, precision_orig)
        pr_auc_red = auc(recall_red, precision_red)
        
        metrics['pr_auc_diff'] = pr_auc_orig - pr_auc_red
        metrics['pr_auc_orig'] = pr_auc_orig
        metrics['pr_auc_red'] = pr_auc_red
        
        print(f"    Precision-Recall AUC scores:")
        print(f"      Original model: {pr_auc_orig:.4f}")
        print(f"      Reduced model: {pr_auc_red:.4f}")
        print(f"      Difference: {metrics['pr_auc_diff']:.4f}")
    except Exception as e:
        print(f"    Warning: Error calculating model performance metrics: {e}")
        # Set default values if calculation fails
        metrics['auc_diff'] = 0.0
        metrics['auc_orig'] = 0.0
        metrics['auc_red'] = 0.0
        metrics['pr_auc_diff'] = 0.0
        metrics['pr_auc_orig'] = 0.0
        metrics['pr_auc_red'] = 0.0
    
    # 5. Anomaly detection performance using Isolation Forest
    print("  Evaluating anomaly detection performance...")
    try:
        from sklearn.ensemble import IsolationForest
        
        # Train Isolation Forest models
        contamination = min(0.2, max(0.001, float(np.mean(original_labels))))
        iso_orig = IsolationForest(random_state=42, contamination=contamination)
        iso_red = IsolationForest(random_state=42, contamination=contamination)
        
        iso_orig.fit(original_features)
        iso_red.fit(reduced_features)
        
        # Use the test set from the previous split
        scores_orig = -iso_orig.decision_function(X_test_orig)  # Negative so higher = more anomalous
        scores_red = -iso_red.decision_function(X_test_orig)
        
        # Calculate anomaly detection AUC
        anomaly_auc_orig = roc_auc_score(y_test_orig, scores_orig)
        anomaly_auc_red = roc_auc_score(y_test_orig, scores_red)
        
        metrics['anomaly_auc_diff'] = anomaly_auc_orig - anomaly_auc_red
        metrics['anomaly_auc_orig'] = anomaly_auc_orig
        metrics['anomaly_auc_red'] = anomaly_auc_red
        
        print(f"    Anomaly detection AUC scores:")
        print(f"      Original model: {anomaly_auc_orig:.4f}")
        print(f"      Reduced model: {anomaly_auc_red:.4f}")
        print(f"      Difference: {metrics['anomaly_auc_diff']:.4f}")
    except Exception as e:
        print(f"    Warning: Error calculating anomaly detection metrics: {e}")
        metrics['anomaly_auc_diff'] = 0.0
        metrics['anomaly_auc_orig'] = 0.0
        metrics['anomaly_auc_red'] = 0.0
    
    return metrics


def plot_comparison(hybrid_metrics, random_metrics):
    """
    Plot comparison between hybrid approach and random sampling.
    
    Args:
        hybrid_metrics: Metrics from hybrid sampling
        random_metrics: Metrics from random sampling
    """
    # Select keys for comparison (ignore raw AUC values, just look at differences)
    compare_keys = [
        'avg_kl_divergence', 'avg_js_distance', 'class_ratio_diff', 
        'pca_variance_diff', 'auc_diff', 'pr_auc_diff', 'anomaly_auc_diff'
    ]
    
    # Better labels for plot
    label_map = {
        'avg_kl_divergence': 'KL Divergence',
        'avg_js_distance': 'JS Distance',
        'class_ratio_diff': 'Class Ratio Difference',
        'pca_variance_diff': 'PCA Variance Difference',
        'auc_diff': 'ROC-AUC Loss',
        'pr_auc_diff': 'PR-AUC Loss',
        'anomaly_auc_diff': 'Anomaly AUC Loss'
    }
    
    # Prepare data for plotting
    metrics_df = pd.DataFrame({
        'Metric': [label_map[k] for k in compare_keys],
        'Hybrid': [hybrid_metrics[k] for k in compare_keys],
        'Random': [random_metrics[k] for k in compare_keys]
    })
    
    print("\nInformation Loss Metrics Comparison (lower is better):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Melt the dataframe for easier plotting
    melted_df = pd.melt(metrics_df, id_vars=['Metric'], var_name='Method', value_name='Value')
    
    # Create a grouped bar plot
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(x='Metric', y='Value', hue='Method', data=melted_df)
    plt.title('Information Loss Comparison: Hybrid vs Random Sampling', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Metric Value (lower is better)', fontsize=14)
    plt.xlabel('', fontsize=14)
    
    # Add value labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.4f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=10, rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('information_loss_comparison.png')
    print("\nSaved information loss comparison plot to 'information_loss_comparison.png'")
    plt.close()
    
    # Plot performance metrics separately
    performance_keys = ['auc_orig', 'auc_red', 'pr_auc_orig', 'pr_auc_red', 
                        'anomaly_auc_orig', 'anomaly_auc_red']
    
    performance_df = pd.DataFrame({
        'Metric': ['AUC', 'AUC', 'PR-AUC', 'PR-AUC', 'Anomaly AUC', 'Anomaly AUC'],
        'Type': ['Original', 'Reduced', 'Original', 'Reduced', 'Original', 'Reduced'],
        'Hybrid': [hybrid_metrics[k] for k in performance_keys],
        'Random': [random_metrics[k] for k in performance_keys]
    })
    
    print("\nPerformance Metrics Comparison (higher is better):")
    print(performance_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Plot performance comparison
    plt.figure(figsize=(14, 8))
    
    metrics_to_plot = ['AUC', 'PR-AUC', 'Anomaly AUC']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i+1)
        
        # Filter data for this metric
        metric_data = performance_df[performance_df['Metric'] == metric]
        
        # Plot grouped bars
        ax = sns.barplot(x='Type', y='Hybrid', data=metric_data, color='skyblue', label='Hybrid')
        sns.barplot(x='Type', y='Random', data=metric_data, color='lightcoral', label='Random', alpha=0.6)
        
        # Add value labels on top of bars
        for j, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_height():.4f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=9, rotation=0)
        
        plt.title(f'{metric} Comparison', fontsize=14)
        plt.xlabel('')
        plt.ylim(0, 1.0)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("Saved performance comparison plot to 'performance_comparison.png'")
    plt.close()
    
    # Create a detailed comparison table
    print("\nDetailed metrics comparison:")
    detailed_metrics = {
        'Metric': list(hybrid_metrics.keys()),
        'Hybrid': [hybrid_metrics[k] for k in hybrid_metrics.keys()],
        'Random': [random_metrics[k] for k in hybrid_metrics.keys()],
        'Difference (Hybrid - Random)': [hybrid_metrics[k] - random_metrics[k] for k in hybrid_metrics.keys()]
    }
    
    metrics_table = pd.DataFrame(detailed_metrics)
    print(metrics_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Save as CSV
    metrics_table.to_csv('metrics_comparison.csv', index=False)
    print("Saved detailed metrics to 'metrics_comparison.csv'")
    
    return metrics_table


def main():
    """Main function to run data reduction and information loss measurement."""
    # File path to the BETH csv files
    data_path = os.getcwd() + os.sep + "data"
    csv_files = glob.glob(f"data{os.sep}*data.csv")

    print("=" * 80)
    print("BETH Dataset Reduction with Information Loss Measurement")
    print("=" * 80)
    
    # Load and preprocess data
    print(f"\nLoading data from {len(csv_files)} files in {data_path}...")
    df_scaled, feature_names = load_and_preprocess_beth_data(csv_files, data_path)
    df_scaled = df_scaled[:100000]
    TARGET_SIZE = 5000

    # Separate features and labels
    features = df_scaled[feature_names].values
    evil_labels = df_scaled['evil'].values
    
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(features):,}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Evil samples: {np.sum(evil_labels):,} ({np.mean(evil_labels)*100:.2f}%)")
    print(f"  Target size for reduced dataset: {TARGET_SIZE:,} samples " 
          f"({TARGET_SIZE/len(features)*100:.2f}% of original)")
    
    # Create test set for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        features, evil_labels, test_size=0.2, random_state=42, stratify=evil_labels
    )
    
    print(f"\nSplit dataset into train and test sets:")
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # 1. Hybrid approach (KMeans + Stratified)
    print("\n" + "=" * 50)
    print("Running hybrid sampling (KMeans + Stratified)...")
    print("=" * 50)
    hybrid_indices = kmeans_stratified_sampling(X_train, y_train, target_size=TARGET_SIZE)
    hybrid_features = X_train[hybrid_indices]
    hybrid_labels = y_train[hybrid_indices]
    
    # 2. Random stratified sampling (average of multiple runs)
    print("\n" + "=" * 50)
    print(f"Running {RANDOM_SAMPLES} random stratified samples for comparison...")
    print("=" * 50)
    random_metrics = evaluate_multiple_random_samples(X_train, y_train, 
                                                    n_samples=RANDOM_SAMPLES, 
                                                    target_size=TARGET_SIZE)
    
    # 3. Measure information loss for hybrid approach
    print("\n" + "=" * 50)
    print("Measuring information loss for hybrid approach...")
    print("=" * 50)
    hybrid_metrics = measure_information_loss(X_train, hybrid_features, y_train, hybrid_labels)
    
    # 4. Compare and visualize results
    print("\n" + "=" * 50)
    print("Comparing approaches and generating visualizations:")
    print("=" * 50)
    comparison_table = plot_comparison(hybrid_metrics, random_metrics)
    
    # 5. Save the reduced dataset from the hybrid approach
    # Convert back to DataFrame
    reduced_df = pd.DataFrame(hybrid_features, columns=feature_names)
    reduced_df['evil'] = hybrid_labels
    
    # Save to CSV
    output_file = 'reduced_beth_dataset.csv'
    reduced_df.to_csv(output_file, index=False)
    print(f"\nReduced dataset of {len(reduced_df):,} samples saved to '{output_file}'")
    
    # Calculate reduction ratio
    reduction_ratio = len(reduced_df) / len(df_scaled)
    print(f"Reduction ratio: {reduction_ratio:.4f} " 
          f"({reduction_ratio*100:.2f}% of original size)")
    
    # Compare evil ratio
    original_evil_ratio = df_scaled['evil'].mean()
    reduced_evil_ratio = reduced_df['evil'].mean()
    print(f"Evil ratio - Original: {original_evil_ratio:.4f}, " 
          f"Reduced: {reduced_evil_ratio:.4f}, "
          f"Difference: {abs(original_evil_ratio - reduced_evil_ratio):.4f}")
    
    # Summary of findings
    print("\n" + "=" * 50)
    print("Summary of findings:")
    print("=" * 50)
    
    # Determine if hybrid is better than random
    better_count = sum(1 for k in ['avg_kl_divergence', 'avg_js_distance', 'class_ratio_diff', 
                                   'pca_variance_diff', 'auc_diff', 'pr_auc_diff', 'anomaly_auc_diff'] 
                      if hybrid_metrics[k] < random_metrics[k])
    
    print(f"Hybrid approach performed better than random in {better_count}/7 key metrics")
    
    # Show the biggest improvement
    improvements = {k: random_metrics[k] - hybrid_metrics[k] 
                   for k in ['avg_kl_divergence', 'avg_js_distance', 'class_ratio_diff', 
                            'pca_variance_diff', 'auc_diff', 'pr_auc_diff', 'anomaly_auc_diff']}
    
    best_metric = max(improvements.items(), key=lambda x: x[1])
    if best_metric[1] > 0:
        print(f"Biggest improvement: {best_metric[0]} reduced by {best_metric[1]:.4f}")
    
    # Model performance impact
    print(f"Model performance impact:")
    print(f"  ROC-AUC: {hybrid_metrics['auc_diff']:.4f} absolute difference from original")
    print(f"  PR-AUC: {hybrid_metrics['pr_auc_diff']:.4f} absolute difference from original")
    print(f"  Anomaly detection: {hybrid_metrics['anomaly_auc_diff']:.4f} absolute difference from original")
    
    # Final recommendation
    print("\nFinal recommendation:")
    if better_count >= 4:
        print("The hybrid K-means + stratified approach is recommended for reducing the BETH dataset.")
    else:
        print("Simple stratified random sampling is sufficient for reducing the BETH dataset.")
    
    print("=" * 80)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time took: {end - start} seconds")