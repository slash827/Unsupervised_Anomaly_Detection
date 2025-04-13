"""
Improved sampling and clustering approach for BETH dataset analysis.
This script implements the recommended changes for better sampling and evaluation.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, mutual_info_score, classification_report
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from scipy.stats import f_oneway, ttest_rel

from utils import properly_balanced_stratified_sample
# Try to use Intel acceleration
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("🚀 Intel acceleration enabled for scikit-learn!")
except ImportError:
    print("⚠️ Intel acceleration not available, using standard scikit-learn")


def load_data_proper(data_path, use_test_only=True):
    """
    Load dataset with improved handling of test vs. train splits.
    
    Args:
        data_path: Path to the preprocessed CSV file
        use_test_only: If True, only use test set data which has reliable labels
        
    Returns:
        X_full, evil_truth, original_df
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully: {df.shape[0]} samples with {df.shape[1]} features")
    
    # Filter data based on split
    if use_test_only:
        print("Using test set data only for more reliable labels")
        df = df[df['split'] == 'test'].copy()
        
    print(f"Dataset with: {df.shape[0]} samples")

    # Check if evil/sus columns exist and have valid values
    has_labels = 'evil' in df.columns and not df['evil'].isna().all()
    if has_labels:
        # Remove rows with NaN labels if any remain
        df = df.dropna(subset=['evil'])
        print(f"Class distribution after filtering: {df['evil'].value_counts().to_dict()}")
        print(f"Evil percentage: {df['evil'].mean() * 100:.2f}%")
    else:
        print("Warning: No valid labels found in the dataset")

    print(f"Dataset with: {df.shape[0]} samples")

    # Drop non-feature columns for clustering
    columns_to_drop = ['split', 'sus', 'evil']
    feature_cols = [col for col in df.columns if col not in columns_to_drop]
    
    # Convert all feature columns to numeric, errors='coerce' will convert non-numeric to NaN
    X_numeric = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    print(f"Dataset with: {df.shape[0]} samples")


    # Fill NaN values with column means
    X_full = X_numeric.fillna(X_numeric.mean())
    
    # Get the labels
    evil_truth = df['evil'].values if 'evil' in df.columns else np.zeros(len(df))
    
    print(f"Final feature matrix shape: {X_full.shape}")
    
    return X_full.values, evil_truth, df


def safe_silhouette(X, labels):
    """Compute silhouette score safely with proper error handling."""
    try:
        # Check for NaN values in labels
        if np.any(np.isnan(labels)):
            print("Warning: NaN values found in labels")
            return -1.0
            
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            return silhouette_score(X, labels)
        elif len(unique_labels) > 1 and -1 in unique_labels and len(unique_labels) > 2:
            # Filter out noise points (label -1) for silhouette calculation
            mask = labels != -1
            return silhouette_score(X[mask], labels[mask])
        else:
            return -1.0
    except Exception as e:
        print(f"Silhouette computation error: {e}")
        return -1.0


def apply_clustering_with_metrics(X, y, algorithm, n_clusters, pca_dim=None, random_state=42):
    """
    Apply clustering algorithm and compute comprehensive evaluation metrics.
    
    Args:
        X: Feature matrix
        y: Target labels (for evaluation)
        algorithm: Clustering algorithm name
        n_clusters: Number of clusters or equivalent parameter
        pca_dim: Dimensionality for PCA (optional)
        random_state: Random seed
        
    Returns:
        dict of metrics, labels, transformed data, model
    """
    # Apply PCA if specified
    if pca_dim is not None and pca_dim < X.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X_transformed = pca.fit_transform(X)
        variance_explained = np.sum(pca.explained_variance_ratio_)
    else:
        X_transformed = X
        variance_explained = 1.0
    
    # Apply clustering
    try:
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = model.fit_predict(X_transformed)
            
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X_transformed)
            
        elif algorithm == 'dbscan':
            # Adaptive eps based on PCA dimension
            eps = 0.2 + 0.1 * (n_clusters - 2) * (1 + 0.1 * pca_dim)
            min_samples = max(5, int(len(X) * 0.01))  # At least 1% of data
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_transformed)
            
        elif algorithm == 'gmm':
            model = GaussianMixture(n_components=n_clusters, 
                                   covariance_type='full', 
                                   random_state=random_state)
            model.fit(X_transformed)
            labels = model.predict(X_transformed)
            
        elif algorithm == 'fcm':
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                X_transformed.T, c=n_clusters, m=2, error=0.005, maxiter=1000, seed=random_state
            )
            labels = np.argmax(u, axis=0)
            # Store fuzzy c-means results in a dictionary
            model = {
                'centers': cntr,
                'membership': u,
                'fpc': fpc
            }
    
    except Exception as e:
        print(f"Error applying {algorithm} clustering: {e}")
        labels = np.zeros(len(X))
        model = None
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    cluster_counts = np.bincount(labels[labels >= 0]) if len(labels) > 0 else []
    metrics['n_clusters_found'] = len(cluster_counts)
    metrics['variance_explained'] = variance_explained
    
    # Silhouette score (internal metric)
    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            metrics['silhouette'] = silhouette_score(X_transformed, labels)
        elif len(unique_labels) > 1 and -1 in unique_labels and len(unique_labels) > 2:
            # Filter out noise points (label -1) for silhouette calculation
            mask = labels != -1
            metrics['silhouette'] = silhouette_score(X_transformed[mask], labels[mask])
        else:
            metrics['silhouette'] = -1.0
    except Exception as e:
        print(f"Silhouette computation error: {e}")
        metrics['silhouette'] = -1.0
    
    # External metrics (using ground truth)
    if y is not None:
        # Mutual information
        try:
            metrics['mi'] = mutual_info_score(y, labels)
        except Exception as e:
            print(f"MI error: {e}")
            metrics['mi'] = 0.0
            
        # Compute anomaly scores and evaluate detection performance
        if algorithm in ['kmeans', 'gmm', 'fcm']:
            if algorithm == 'kmeans':
                # Distance to closest centroid
                centroids = model.cluster_centers_
                anomaly_scores = np.linalg.norm(X_transformed - centroids[labels], axis=1)
            elif algorithm == 'gmm':
                # Negative log likelihood
                anomaly_scores = -model.score_samples(X_transformed)
            elif algorithm == 'fcm':
                # Use stored membership matrix to avoid recomputation
                u = model['membership']
                # Use 1 - (max membership) as anomaly score
                anomaly_scores = 1.0 - np.max(u, axis=0)
        else:
            # For hierarchical and DBSCAN, use a simple approach
            anomaly_scores = np.zeros(len(X))
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels >= 0]  # Remove noise if present
            
            # For each cluster, calculate its size and set anomaly score accordingly
            # Small clusters are more likely to be anomalies
            if len(unique_labels) > 0:
                for label in unique_labels:
                    cluster_size = np.sum(labels == label)
                    cluster_size_ratio = cluster_size / len(labels)
                    anomaly_scores[labels == label] = 1.0 - cluster_size_ratio
                # Set noise points (if any) to highest anomaly score
                if -1 in np.unique(labels):
                    anomaly_scores[labels == -1] = 1.0
        
        # Select top 15% as anomalies
        threshold = np.percentile(anomaly_scores, 85)
        predicted_anomalies = anomaly_scores >= threshold
        
        # Calculate anomaly detection metrics
        true_positives = np.sum((y == 1) & predicted_anomalies)
        total_predicted = np.sum(predicted_anomalies)
        total_actual = np.sum(y == 1)
        
        metrics['precision'] = (true_positives / total_predicted * 100) if total_predicted > 0 else 0
        metrics['recall'] = (true_positives / total_actual * 100) if total_actual > 0 else 0
        
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0
    
    # Combined score
    metrics['combined_score'] = metrics.get('silhouette', 0) + metrics.get('mi', 0)
    
    return metrics, labels, X_transformed, model


def evaluate_multiple_configs(X, y, algorithms, pca_dims, cluster_counts, n_trials=3):
    """
    Evaluate multiple clustering configurations with comprehensive metrics.
    
    Args:
        X: Feature matrix
        y: Target labels
        algorithms: List of algorithms to test
        pca_dims: List of PCA dimensions to test
        cluster_counts: List of cluster counts to test
        n_trials: Number of trials per configuration (for stability)
        
    Returns:
        DataFrame with results
    """
    results = []
    
    total_configs = len(algorithms) * len(pca_dims) * len(cluster_counts)
    config_counter = 0
    
    print(f"Evaluating {total_configs} configurations with {n_trials} trials each")
    start_time = time.time()
    
    for algorithm in algorithms:
        for pca_dim in pca_dims:
            for n_clusters in cluster_counts:
                config_counter += 1
                print(f"Config {config_counter}/{total_configs}: {algorithm}, PCA {pca_dim}D, k={n_clusters}")
                
                # Run multiple trials for stability
                trial_metrics = []
                for trial in range(n_trials):
                    metrics, labels, _, _ = apply_clustering_with_metrics(
                        X, y, algorithm, n_clusters, pca_dim, random_state=42+trial
                    )
                    trial_metrics.append(metrics)
                
                # Average metrics across trials
                avg_metrics = {}
                std_metrics = {}
                
                if trial_metrics:
                    for key in trial_metrics[0].keys():
                        values = [m.get(key, 0) for m in trial_metrics]
                        avg_metrics[key] = np.mean(values)
                        std_metrics[key] = np.std(values)
                
                # Store results
                result = {
                    'algorithm': algorithm,
                    'pca_dim': pca_dim,
                    'n_clusters': n_clusters,
                }
                result.update({f"{k}_mean": v for k, v in avg_metrics.items()})
                result.update({f"{k}_std": v for k, v in std_metrics.items()})
                
                results.append(result)
                
                # Print summary
                print(f"  Silhouette: {avg_metrics.get('silhouette', 0):.3f} ± {std_metrics.get('silhouette', 0):.3f}")
                print(f"  MI: {avg_metrics.get('mi', 0):.3f} ± {std_metrics.get('mi', 0):.3f}")
                print(f"  Combined: {avg_metrics.get('combined_score', 0):.3f} ± {std_metrics.get('combined_score', 0):.3f}")
                print(f"  Precision: {avg_metrics.get('precision', 0):.1f}% ± {std_metrics.get('precision', 0):.1f}%")
                print(f"  Recall: {avg_metrics.get('recall', 0):.1f}% ± {std_metrics.get('recall', 0):.1f}%")
                print(f"  F1: {avg_metrics.get('f1', 0):.1f} ± {std_metrics.get('f1', 0):.1f}")
    
    elapsed_time = time.time() - start_time
    print(f"Evaluation complete in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def analyze_and_visualize_results(results_df, output_dir='clustering_results'):
    """
    Analyze results and create visualizations.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find best configurations
    print("\n--- Best Configurations by Combined Score ---")
    for algorithm in results_df['algorithm'].unique():
        alg_results = results_df[results_df['algorithm'] == algorithm]
        best_row = alg_results.loc[alg_results['combined_score_mean'].idxmax()]
        print(f"{algorithm.upper()}: PCA {best_row['pca_dim']}D, k={best_row['n_clusters']}, "
              f"Score={best_row['combined_score_mean']:.3f}")
        print(f"  Silhouette={best_row['silhouette_mean']:.3f}, MI={best_row['mi_mean']:.3f}")
        print(f"  Precision={best_row['precision_mean']:.1f}%, Recall={best_row['recall_mean']:.1f}%")
    
    # 2. Create heatmaps for each algorithm
    for algorithm in results_df['algorithm'].unique():
        alg_results = results_df[results_df['algorithm'] == algorithm]
        
        # Create heatmaps for combined score
        pca_dims = sorted(alg_results['pca_dim'].unique())
        cluster_counts = sorted(alg_results['n_clusters'].unique())
        
        # Initialize the matrix
        score_matrix = np.zeros((len(pca_dims), len(cluster_counts)))
        
        # Fill the matrix
        for i, dim in enumerate(pca_dims):
            for j, k in enumerate(cluster_counts):
                mask = (alg_results['pca_dim'] == dim) & (alg_results['n_clusters'] == k)
                if mask.any():
                    score_matrix[i, j] = alg_results.loc[mask, 'combined_score_mean'].values[0]
        
        # Create and save the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(score_matrix, cmap='viridis', aspect='auto')
        
        # Customize the plot
        ax.set_xticks(np.arange(len(cluster_counts)))
        ax.set_yticks(np.arange(len(pca_dims)))
        ax.set_xticklabels(cluster_counts)
        ax.set_yticklabels(pca_dims)
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("PCA Dimensions")
        ax.set_title(f"{algorithm.upper()}: Combined Score (Silhouette + MI)")
        
        # Add text annotations
        for i in range(len(pca_dims)):
            for j in range(len(cluster_counts)):
                text = ax.text(j, i, f"{score_matrix[i, j]:.2f}", 
                              ha="center", va="center", color="w")
        
        # Add colorbar
        plt.colorbar(im)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"{algorithm}_heatmap.pdf"))
        plt.close()
    
    # 3. Create dimension vs performance plots
    for metric in ['combined_score_mean', 'mi_mean', 'silhouette_mean', 'precision_mean', 'recall_mean']:
        plt.figure(figsize=(12, 6))
        
        for algorithm in results_df['algorithm'].unique():
            # Get best configuration for each dimension
            dim_performance = []
            
            for dim in sorted(results_df['pca_dim'].unique()):
                dim_data = results_df[(results_df['algorithm'] == algorithm) & 
                                     (results_df['pca_dim'] == dim)]
                if not dim_data.empty:
                    best_value = dim_data[metric].max()
                    dim_performance.append((dim, best_value))
            
            # Plot line
            if dim_performance:
                dims, perf = zip(*dim_performance)
                plt.plot(dims, perf, 'o-', label=algorithm)
        
        plt.xlabel('PCA Dimensions')
        metric_name = metric.replace('_mean', '').replace('_', ' ').title()
        plt.ylabel(metric_name)
        plt.title(f'Best {metric_name} vs. PCA Dimensions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        clean_name = metric.replace('_mean', '')
        plt.savefig(os.path.join(output_dir, f"dimension_vs_{clean_name}.pdf"))
        plt.close()
    
    # 4. Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
    print(f"Results saved to {output_dir}")


def focused_on_dbscan(X, y, sample_size=5000, n_trials=3):
    """
    Run focused optimization for DBSCAN parameters.
    
    Args:
        X: Feature matrix
        y: Ground truth labels
        sample_size: Sample size for evaluations
        n_trials: Number of trials per configuration
    
    Returns:
        DataFrame with results
    """
    print("\n=== FOCUSED DBSCAN PARAMETER OPTIMIZATION ===")
    
    # Sample data
    if len(X) > sample_size:
        X_sample, y_sample, _ = properly_balanced_stratified_sample(X, y, sample_size, random_state=42)
    else:
        X_sample, y_sample = X, y
    
    # Parameters to test
    pca_dims = [3, 4, 5, 6]
    eps_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    min_samples_values = [5, 10, 15, 20, int(sample_size * 0.01)]
    
    results = []
    
    # Total configs to evaluate
    total_configs = len(pca_dims) * len(eps_values) * len(min_samples_values)
    config_counter = 0
    
    for pca_dim in pca_dims:
        # Apply PCA once per dimension
        pca = PCA(n_components=pca_dim, random_state=42)
        X_transformed = pca.fit_transform(X_sample)
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                config_counter += 1
                print(f"Config {config_counter}/{total_configs}: "
                      f"PCA {pca_dim}D, eps={eps}, min_samples={min_samples}")
                
                # Run multiple trials
                trial_metrics = []
                for trial in range(n_trials):
                    try:
                        # Apply DBSCAN directly with custom parameters
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = model.fit_predict(X_transformed)
                        
                        # Calculate metrics
                        metrics = {}
                        
                        # Basic metrics
                        cluster_counts = np.bincount(labels[labels >= 0]) if len(labels) > 0 else []
                        metrics['n_clusters_found'] = len(cluster_counts)
                        
                        # Silhouette score
                        try:
                            unique_labels = np.unique(labels)
                            if len(unique_labels) > 1 and -1 not in unique_labels:
                                metrics['silhouette'] = silhouette_score(X_transformed, labels)
                            elif len(unique_labels) > 1 and -1 in unique_labels and len(unique_labels) > 2:
                                mask = labels != -1
                                metrics['silhouette'] = silhouette_score(X_transformed[mask], labels[mask])
                            else:
                                metrics['silhouette'] = -1.0
                        except Exception as e:
                            metrics['silhouette'] = -1.0
                        
                        # MI score
                        try:
                            metrics['mi'] = mutual_info_score(y_sample, labels)
                        except Exception as e:
                            metrics['mi'] = 0.0
                        
                        # Combined score
                        metrics['combined_score'] = metrics.get('silhouette', 0) + metrics.get('mi', 0)
                        
                        # Evaluate cluster quality for evil detection
                        # Find the cluster with highest concentration of evil
                        best_evil_ratio = 0
                        best_evil_cluster = None
                        
                        for label in np.unique(labels):
                            if label == -1:  # Skip noise
                                continue
                                
                            cluster_mask = labels == label
                            cluster_size = np.sum(cluster_mask)
                            
                            if cluster_size > 0:
                                evil_ratio = np.sum(y_sample[cluster_mask] == 1) / cluster_size
                                if evil_ratio > best_evil_ratio:
                                    best_evil_ratio = evil_ratio
                                    best_evil_cluster = label
                        
                        # Calculate precision/recall using best cluster
                        if best_evil_cluster is not None:
                            predicted_evil = labels == best_evil_cluster
                            
                            true_positives = np.sum((y_sample == 1) & predicted_evil)
                            total_predicted = np.sum(predicted_evil)
                            total_actual = np.sum(y_sample == 1)
                            
                            metrics['precision'] = (true_positives / total_predicted * 100) if total_predicted > 0 else 0
                            metrics['recall'] = (true_positives / total_actual * 100) if total_actual > 0 else 0
                            
                            if metrics['precision'] > 0 and metrics['recall'] > 0:
                                metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                            else:
                                metrics['f1'] = 0
                        else:
                            metrics['precision'] = 0
                            metrics['recall'] = 0
                            metrics['f1'] = 0
                        
                        # Add percentage of points classified as noise
                        metrics['noise_percentage'] = np.sum(labels == -1) / len(labels) * 100
                        
                        trial_metrics.append(metrics)
                    
                    except Exception as e:
                        print(f"  Error: {e}")
                
                # Average metrics across trials
                if trial_metrics:
                    avg_metrics = {k: np.mean([m.get(k, 0) for m in trial_metrics]) 
                                   for k in trial_metrics[0].keys()}
                    std_metrics = {k: np.std([m.get(k, 0) for m in trial_metrics]) 
                                   for k in trial_metrics[0].keys()}
                    
                    # Store results
                    result = {
                        'pca_dim': pca_dim,
                        'eps': eps,
                        'min_samples': min_samples,
                    }
                    result.update(avg_metrics)
                    result.update({f"{k}_std": v for k, v in std_metrics.items()})
                    
                    results.append(result)
                    
                    # Print key metrics
                    print(f"  Clusters found: {avg_metrics.get('n_clusters_found', 0):.1f}")
                    print(f"  Silhouette: {avg_metrics.get('silhouette', 0):.3f}")
                    print(f"  MI: {avg_metrics.get('mi', 0):.3f}")
                    print(f"  Combined: {avg_metrics.get('combined_score', 0):.3f}")
                    print(f"  Precision: {avg_metrics.get('precision', 0):.1f}%")
                    print(f"  Recall: {avg_metrics.get('recall', 0):.1f}%")
                    print(f"  Noise: {avg_metrics.get('noise_percentage', 0):.1f}%")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print best configurations
    if not results_df.empty:
        print("\n--- Best DBSCAN Configurations ---")
        
        # By combined score
        best_combined = results_df.loc[results_df['combined_score'].idxmax()]
        print(f"Best by combined score: PCA {best_combined['pca_dim']}D, "
              f"eps={best_combined['eps']}, min_samples={best_combined['min_samples']}")
        print(f"  Score: {best_combined['combined_score']:.3f} "
              f"(Silhouette={best_combined['silhouette']:.3f}, MI={best_combined['mi']:.3f})")
        
        # By MI score
        best_mi = results_df.loc[results_df['mi'].idxmax()]
        print(f"Best by MI score: PCA {best_mi['pca_dim']}D, "
              f"eps={best_mi['eps']}, min_samples={best_mi['min_samples']}")
        print(f"  MI: {best_mi['mi']:.3f} (Silhouette={best_mi['silhouette']:.3f})")
        
        # By precision
        best_precision = results_df.loc[results_df['precision'].idxmax()]
        print(f"Best by precision: PCA {best_precision['pca_dim']}D, "
              f"eps={best_precision['eps']}, min_samples={best_precision['min_samples']}")
        print(f"  Precision: {best_precision['precision']:.1f}% (Recall={best_precision['recall']:.1f}%)")
        
        # Save results
        results_df.to_csv('dbscan_optimization_results.csv', index=False)
    
    return results_df


def main():
    """
    Main function to run the improved clustering analysis.
    """
    # Configuration
    DATA_PATH = "preprocessed/prepared_data_cluster.csv"
    OUTPUT_DIR = "improved_clustering_results"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load data properly (test set only)
    X_full, evil_truth, df = load_data_proper(DATA_PATH, use_test_only=False)
    
    # Step 2: Sample a small dataset for initial evaluation with proper balancing
    initial_sample_size = 5000  # Start with a moderate sample size
    X_sample, y_sample, _ = properly_balanced_stratified_sample(
        X_full, evil_truth, initial_sample_size, random_state=42
    )
    
    # Basic approach: Evaluate traditional configurations
    print("\n=== STEP 1: Basic Algorithm Comparison ===")
    
    # Define parameters to test
    algorithms = ['kmeans', 'hierarchical', 'dbscan', 'gmm', 'fcm']
    pca_dims = [2, 4, 6, 8]  # Focusing on lower dimensions based on previous results
    cluster_counts = [2, 3, 4, 5, 7]  # Testing few clusters based on previous results
    
    # Run the evaluation
    results_df = evaluate_multiple_configs(
        X_sample, y_sample, algorithms, pca_dims, cluster_counts, n_trials=3
    )
    
    # Analyze and visualize results
    analyze_and_visualize_results(results_df, output_dir=OUTPUT_DIR)
    
    # Step 3: Focused optimization on DBSCAN (since it showed most promise)
    print("\n=== STEP 2: Focused DBSCAN Optimization ===")
    dbscan_results = focused_on_dbscan(X_full, evil_truth, sample_size=5000, n_trials=3)
    
    # Step 4: Final evaluation with best configuration on larger sample
    print("\n=== STEP 3: Final Evaluation ===")
    
    # Get best DBSCAN parameters (if available)
    if 'dbscan_results' in locals() and not dbscan_results.empty:
        best_config = dbscan_results.loc[dbscan_results['combined_score'].idxmax()]
        best_pca_dim = int(best_config['pca_dim'])
        best_eps = float(best_config['eps'])
        best_min_samples = int(best_config['min_samples'])
        
        print(f"Best DBSCAN configuration: PCA {best_pca_dim}D, eps={best_eps}, min_samples={best_min_samples}")
    else:
        # Default values based on previous findings
        best_pca_dim = 4
        best_eps = 0.5
        best_min_samples = 15
        print(f"Using default DBSCAN configuration: PCA {best_pca_dim}D, eps={best_eps}, min_samples={best_min_samples}")
    
    # Sample a larger dataset for final evaluation
    final_sample_size = min(10000, len(X_full))  # Up to 10K samples
    X_final, y_final, _ = properly_balanced_stratified_sample(
        X_full, evil_truth, final_sample_size, random_state=100
    )
    
    # Apply PCA
    pca = PCA(n_components=best_pca_dim, random_state=42)
    X_transformed = pca.fit_transform(X_final)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    labels = dbscan.fit_predict(X_transformed)
    
    # Calculate metrics
    silhouette = safe_silhouette(X_transformed, labels)
    mi = mutual_info_score(y_final, labels) if len(np.unique(labels)) > 1 else 0
    
    # Find the cluster with highest evil concentration
    best_evil_ratio = 0
    best_evil_cluster = None
    
    for label in np.unique(labels):
        if label == -1:  # Skip noise
            continue
            
        cluster_mask = labels == label
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size > 0:
            evil_ratio = np.sum(y_final[cluster_mask] == 1) / cluster_size
            if evil_ratio > best_evil_ratio:
                best_evil_ratio = evil_ratio
                best_evil_cluster = label
    
    # Calculate precision/recall using best cluster
    if best_evil_cluster is not None:
        predicted_evil = labels == best_evil_cluster
        
        # Detailed classification report
        from sklearn.metrics import classification_report
        print("\nClassification Report (using best evil cluster):")
        print(classification_report(y_final, predicted_evil))
        
        # Calculate metrics manually for verification
        true_positives = np.sum((y_final == 1) & predicted_evil)
        false_positives = np.sum((y_final == 0) & predicted_evil)
        false_negatives = np.sum((y_final == 1) & ~predicted_evil)
        
        precision = (true_positives / (true_positives + false_positives) * 100) if (true_positives + false_positives) > 0 else 0
        recall = (true_positives / (true_positives + false_negatives) * 100) if (true_positives + false_negatives) > 0 else 0
        
        print(f"\nDetailed metrics:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Mutual Information: {mi:.3f}")
        print(f"Evil Precision: {precision:.1f}%")
        print(f"Evil Recall: {recall:.1f}%")
        
        # Cluster details
        print(f"\nCluster statistics:")
        for label in np.unique(labels):
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            cluster_evil = np.sum(y_final[cluster_mask] == 1)
            cluster_evil_pct = (cluster_evil / cluster_size * 100) if cluster_size > 0 else 0
            
            label_name = "Noise" if label == -1 else f"Cluster {label}"
            print(f"{label_name}: {cluster_size} points, {cluster_evil} evil ({cluster_evil_pct:.1f}%)")
    
    print(f"\nAnalysis complete with {initial_sample_size} samples!")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds")