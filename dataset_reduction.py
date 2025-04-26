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
from scipy.stats import entropy

from utils import load_and_preprocess_beth_data

# Set plot style and figure size for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Constants
TARGET_SIZE = 20000  # Number of samples to keep in the reduced dataset
RANDOM_SAMPLES = 5  # Number of random samples to generate for comparison


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


def multi_feature_stratified_sampling(features, labels, target_size=20000):
    """
    Stratified sampling based on multiple features.
    """
    print("Running multi-feature stratified sampling...")
    
    # Create bins based on multiple features
    isSystemProcess = features[:, 0]  # First feature
    eventId = features[:, 4]  # Fifth feature or another important feature
    
    # Create compound strata
    strata = []
    for sys_val in [0, 1]:
        for event_range in range(0, 50, 10):  # Create ranges of event IDs
            mask = (isSystemProcess == sys_val) & (eventId >= event_range) & (eventId < event_range + 10)
            if np.sum(mask) > 0:
                strata.append(np.where(mask)[0])
    
    # Also stratify by evil/benign
    evil_strata = []
    for s in strata:
        # Split each stratum into evil/benign
        evil_s = s[labels[s] == 1]
        benign_s = s[labels[s] == 0]
        if len(evil_s) > 0:
            evil_strata.append((evil_s, 1))  # (indices, label)
        if len(benign_s) > 0:
            evil_strata.append((benign_s, 0))  # (indices, label)
    
    # Sample proportionally from each stratum
    sampled_indices = []
    evil_count = 0
    benign_count = 0
    
    for s, is_evil in evil_strata:
        # Calculate proportional sample size
        size = max(1, int(len(s) / len(features) * target_size))
        if len(s) <= size:
            sampled_indices.extend(s)
            if is_evil == 1:
                evil_count += len(s)
            else:
                benign_count += len(s)
        else:
            selected = np.random.choice(s, size=size, replace=False)
            sampled_indices.extend(selected)
            if is_evil == 1:
                evil_count += size
            else:
                benign_count += size
    
    # Ensure we get exactly target_size samples
    if len(sampled_indices) > target_size:
        sampled_indices = np.random.choice(sampled_indices, size=target_size, replace=False)
    elif len(sampled_indices) < target_size:
        # Add more samples randomly from unrepresented indices
        remaining = list(set(range(len(features))) - set(sampled_indices))
        additional = np.random.choice(remaining, size=target_size-len(sampled_indices), replace=False)
        sampled_indices.extend(additional)
    
    final_indices = np.array(sampled_indices)
    final_evil_count = np.sum(labels[final_indices])
    final_benign_count = len(final_indices) - final_evil_count
    
    print(f"  Total samples selected: {len(final_indices)}")
    print(f"  Evil samples: {final_evil_count}")
    print(f"  Benign samples: {final_benign_count}")
    print(f"  Evil ratio: {final_evil_count/len(final_indices):.4f}")
    
    return final_indices


def optimized_density_based_sampling(features, labels, target_size=20000):
    """
    Optimized density-based sampling with subsampling for speed.
    """
    from sklearn.neighbors import KernelDensity
    from sklearn.preprocessing import StandardScaler
    
    print("Running optimized density-based sampling...")
    
    # Subsample for density estimation (max 50k points)
    max_subsample = 50000
    if len(features) > max_subsample:
        subsample_idx = np.random.choice(len(features), max_subsample, replace=False)
        density_features = features[subsample_idx]
    else:
        density_features = features
        subsample_idx = np.arange(len(features))
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(density_features)
    
    # Calculate density for subsample
    print("  Calculating density estimates on subsample...")
    kde = KernelDensity(bandwidth=0.5).fit(features_scaled)
    log_density = kde.score_samples(features_scaled)
    
    # Create 10 percentile bins
    n_bins = 10
    density_bins = np.percentile(log_density, np.linspace(0, 100, n_bins+1))
    
    # Sample from each bin
    print("  Sampling from density bins...")
    sampled_indices = []
    
    for i in range(n_bins):
        # Find points in this density bin
        if i < n_bins-1:
            bin_indices = np.where((log_density >= density_bins[i]) & 
                                  (log_density < density_bins[i+1]))[0]
        else:
            bin_indices = np.where(log_density >= density_bins[i])[0]
        
        # Map back to original indices
        original_bin_indices = subsample_idx[bin_indices]
        
        # Sample proportionally
        bin_size = int(len(bin_indices) / len(density_features) * target_size)
        if bin_size > 0:
            bin_sample = np.random.choice(original_bin_indices, 
                                          size=min(bin_size, len(original_bin_indices)), 
                                          replace=False)
            sampled_indices.extend(bin_sample)
    
    # Ensure class ratio is preserved
    sampled_indices = np.array(sampled_indices)
    evil_ratio = np.mean(labels)
    sampled_evil_ratio = np.mean(labels[sampled_indices])
    
    # Adjust to maintain class ratio
    final_indices = stratify_and_adjust(sampled_indices, labels, target_size, evil_ratio)
    return final_indices


def stratify_and_adjust(indices, labels, target_size, target_evil_ratio):
    """Helper function to adjust samples to match target size and class ratio"""
    evil_indices = indices[labels[indices] == 1]
    benign_indices = indices[labels[indices] == 0]
    
    n_evil = int(target_size * target_evil_ratio)
    n_benign = target_size - n_evil
    
    # Sample with replacement only if necessary
    if len(evil_indices) < n_evil:
        final_evil = np.random.choice(evil_indices, size=n_evil, replace=True)
    else:
        final_evil = np.random.choice(evil_indices, size=n_evil, replace=False)
        
    if len(benign_indices) < n_benign:
        final_benign = np.random.choice(benign_indices, size=n_benign, replace=True)
    else:
        final_benign = np.random.choice(benign_indices, size=n_benign, replace=False)
    
    return np.concatenate([final_evil, final_benign])


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
    # df_scaled = df_scaled[:100000]
    # TARGET_SIZE = 5000

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
    
    # Compare different sampling approaches
    print("\n" + "=" * 50)
    print("COMPARING DIFFERENT SAMPLING APPROACHES")
    print("=" * 50)
    
    # 1. Run the hybrid approach (which yields ~260 samples + random)
    print("\n" + "=" * 30)
    print("Running original hybrid approach...")
    print("=" * 30)
    hybrid_indices = kmeans_stratified_sampling(X_train, y_train, target_size=TARGET_SIZE)
    hybrid_features = X_train[hybrid_indices]
    hybrid_labels = y_train[hybrid_indices]
    
    # 2. Run multi-feature stratified approach
    print("\n" + "=" * 30)
    print("Running multi-feature stratified approach...")
    print("=" * 30)
    mf_indices = multi_feature_stratified_sampling(X_train, y_train, target_size=TARGET_SIZE)
    mf_features = X_train[mf_indices]
    mf_labels = y_train[mf_indices]
    
    # 3. Run density-based approach
    print("\n" + "=" * 30)
    print("Running density-based approach...")
    print("=" * 30)
    density_indices = optimized_density_based_sampling(X_train, y_train, target_size=TARGET_SIZE)
    density_features = X_train[density_indices]
    density_labels = y_train[density_indices]
    
    # 4. Run standard random stratified approach for comparison
    print("\n" + "=" * 30)
    print("Running random stratified approach...")
    print("=" * 30)
    random_metrics = evaluate_multiple_random_samples(X_train, y_train, n_samples=1, target_size=TARGET_SIZE)
    
    # Measure and compare information loss for each approach
    print("\n" + "=" * 50)
    print("Measuring information loss for different approaches...")
    print("=" * 50)
    
    print("\nMeasuring for multi-feature stratified approach...")
    mf_metrics = measure_information_loss(X_train, mf_features, y_train, mf_labels)
    
    print("\nMeasuring for density-based approach...")
    density_metrics = measure_information_loss(X_train, density_features, y_train, density_labels)
    
    print("\nMeasuring for hybrid approach...")
    hybrid_metrics = measure_information_loss(X_train, hybrid_features, y_train, hybrid_labels)
    
    # Combine metrics for comparison
    compare_metrics = {
        'Multi-Feature': mf_metrics,
        'Density-Based': density_metrics,
        'Hybrid': hybrid_metrics,
        'Random': random_metrics
    }
    
    # Display comparison
    print("\n" + "=" * 50)
    print("COMPARISON OF SAMPLING APPROACHES")
    print("=" * 50)
    
    # Create comparison table
    metrics_to_compare = ['avg_kl_divergence', 'avg_js_distance', 'class_ratio_diff', 
                         'pca_variance_diff', 'auc_diff', 'pr_auc_diff', 'anomaly_auc_diff']
    
    comparison_df = pd.DataFrame({
        method: [metrics.get(key, 0) for key in metrics_to_compare]
        for method, metrics in compare_metrics.items()
    }, index=metrics_to_compare)
    
    print("\nInformation Loss Metrics Comparison (lower is better):")
    print(comparison_df.to_string(float_format=lambda x: f"{x:.4f}"))
    
    # Save the best performing method's dataset
    best_method = comparison_df.sum().idxmin()  # Method with lowest total information loss
    print(f"\nBest performing method: {best_method}")
    
    if best_method == 'Multi-Feature':
        best_features = mf_features
        best_labels = mf_labels
    elif best_method == 'Density-Based':
        best_features = density_features
        best_labels = density_labels
    elif best_method == 'Hybrid':
        best_features = hybrid_features
        best_labels = hybrid_labels
    else:
        # Use the original random approach result
        temp_indices = stratified_random_sampling(X_train, y_train, target_size=TARGET_SIZE)
        best_features = X_train[temp_indices]
        best_labels = y_train[temp_indices]
    
    # Save the reduced dataset from the best approach
    reduced_df = pd.DataFrame(best_features, columns=feature_names)
    reduced_df['evil'] = best_labels
    
    output_file = f'reduced_beth_dataset_{best_method.lower()}.csv'
    reduced_df.to_csv(output_file, index=False)
    print(f"\nBest reduced dataset ({len(reduced_df)} samples) saved to '{output_file}'")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time took: {end - start} seconds")