"""
DBSCAN Cluster Visualization and Investigation
This script visualizes DBSCAN clustering results and investigates specific clusters
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import umap  # For dimensionality reduction (optional, install with: pip install umap-learn)

import warnings
warnings.filterwarnings('ignore')


def visualize_clusters(X, labels, evil_truth=None, method='pca', n_components=2, 
                      title='Cluster Visualization', highlight_clusters=None,
                      save_path=None, figsize=(12, 10), should_show=False):
    """
    Visualize clusters using dimensionality reduction
    
    Parameters:
    -----------
    X : array-like
        Features matrix (if already reduced, set method=None)
    labels : array-like
        Cluster labels
    evil_truth : array-like, optional
        Ground truth labels (1=evil, 0=benign)
    method : str
        Dimensionality reduction method: 'pca', 'tsne', 'umap' or None
    n_components : int
        Number of components for visualization (2 or 3)
    title : str
        Plot title
    highlight_clusters : list
        List of cluster IDs to highlight (others will be grey)
    save_path : str, optional
        Path to save the visualization
    figsize : tuple
        Figure size
    """
    # Apply dimensionality reduction if needed
    if method is not None and X.shape[1] > n_components:
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_reduced = reducer.fit_transform(X)
    else:
        # No reduction needed or requested
        X_reduced = X[:, :n_components]
    
    # Set up colors
    n_clusters = len(np.unique(labels))
    
    if highlight_clusters is not None:
        # Create a colormap with highlighted clusters in color, others in grey
        colors = ['lightgrey'] * n_clusters
        cmap = plt.cm.get_cmap('tab20', len(highlight_clusters))
        
        for i, cluster_id in enumerate(highlight_clusters):
            if cluster_id == -1:  # Special case for noise
                colors[0] = 'black'
            else:
                cluster_position = np.where(np.unique(labels) == cluster_id)[0][0]
                colors[cluster_position] = cmap(i)
    else:
        # Use a colorful palette for all clusters
        if n_clusters <= 20:
            cmap = plt.cm.get_cmap('tab20', n_clusters)
            colors = [cmap(i) for i in range(n_clusters)]
            colors[0] = 'black' if -1 in np.unique(labels) else colors[0]  # Noise in black
        else:
            # For many clusters, use a continuous colormap
            cmap = plt.cm.viridis
            colors = [cmap(i/n_clusters) for i in range(n_clusters)]
            colors[0] = 'black' if -1 in np.unique(labels) else colors[0]  # Noise in black
    
    # Create the plot
    if n_components == 2:
        plt.figure(figsize=figsize)
        
        # Default: plot by cluster
        unique_labels = np.unique(labels)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = 'Noise' if label == -1 else f'Cluster {label}'
            
            if evil_truth is not None:
                # When ground truth is available, shape represents truth, color represents cluster
                evil_mask = (labels == label) & (evil_truth == 1)
                benign_mask = (labels == label) & (evil_truth == 0)
                
                plt.scatter(X_reduced[evil_mask, 0], X_reduced[evil_mask, 1], 
                           s=80, marker='^', c=[colors[i]], alpha=0.8,
                           label=f"{label_name} (Evil: {np.sum(evil_mask)})")
                
                plt.scatter(X_reduced[benign_mask, 0], X_reduced[benign_mask, 1], 
                           s=40, marker='o', c=[colors[i]], alpha=0.5,
                           label=f"{label_name} (Benign: {np.sum(benign_mask)})")
            else:
                plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                           s=50, c=[colors[i]], alpha=0.6, label=label_name)
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Put legend outside plot if many clusters
        if n_clusters > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend()
        
        plt.tight_layout()
        
    elif n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        unique_labels = np.unique(labels)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = 'Noise' if label == -1 else f'Cluster {label}'
            
            if evil_truth is not None:
                # Shape represents truth, color represents cluster
                evil_mask = (labels == label) & (evil_truth == 1)
                benign_mask = (labels == label) & (evil_truth == 0)
                
                ax.scatter(X_reduced[evil_mask, 0], X_reduced[evil_mask, 1], X_reduced[evil_mask, 2],
                          s=80, marker='^', c=[colors[i]], alpha=0.8,
                          label=f"{label_name} (Evil: {np.sum(evil_mask)})")
                
                ax.scatter(X_reduced[benign_mask, 0], X_reduced[benign_mask, 1], X_reduced[benign_mask, 2],
                          s=40, marker='o', c=[colors[i]], alpha=0.5,
                          label=f"{label_name} (Benign: {np.sum(benign_mask)})")
            else:
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                          s=50, c=[colors[i]], alpha=0.6, label=label_name)
        
        ax.set_title(title)
        ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if should_show:
        plt.show()


def investigate_cluster(X, labels, evil_truth, cluster_id, feature_names=None, 
                       pca_components=None, output_dir='cluster_analysis', should_show=False):
    """
    Investigate a specific cluster to understand its characteristics
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    labels : array-like
        Cluster labels
    evil_truth : array-like
        Ground truth labels (1=evil, 0=benign)
    cluster_id : int
        ID of the cluster to investigate
    feature_names : list
        List of feature names
    pca_components : array-like
        PCA components if X is PCA-transformed
    output_dir : str
        Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the cluster
    cluster_mask = labels == cluster_id
    cluster_points = X[cluster_mask]
    cluster_labels = evil_truth[cluster_mask]
    
    # Basic statistics
    cluster_size = len(cluster_points)
    evil_count = np.sum(cluster_labels == 1)
    evil_percent = evil_count / cluster_size * 100 if cluster_size > 0 else 0
    
    print(f"\n==== INVESTIGATION OF CLUSTER {cluster_id} ====")
    print(f"Size: {cluster_size} points")
    print(f"Evil points: {evil_count} ({evil_percent:.1f}%)")
    
    # If no points, stop
    if cluster_size == 0:
        print("No points in this cluster to analyze.")
        return
    
    # Determine feature names
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # 1. Feature Distribution Analysis
    print("\n--- Feature Distributions ---")
    
    # If PCA transformed, analyze component contributions
    if pca_components is not None:
        # Get the mean point of the cluster
        cluster_center = np.mean(cluster_points, axis=0)
        
        # Reconstruct what this means in original feature space
        feature_contributions = np.abs(pca_components[:, :len(cluster_center)] @ cluster_center)
        
        # Get top contributing features
        top_indices = np.argsort(feature_contributions)[::-1]
        
        print("Top contributing features to this cluster (via PCA reconstruction):")
        for i, idx in enumerate(top_indices[:10]):  # Top 10
            print(f"  {i+1}. {feature_names[idx]}: {feature_contributions[idx]:.3f}")
            
        # Visualize feature contributions
        plt.figure(figsize=(12, 6))
        plt.bar(range(min(20, len(feature_contributions))), 
               feature_contributions[top_indices[:20]])
        plt.xticks(range(min(20, len(feature_contributions))), 
                  [feature_names[i] for i in top_indices[:20]], rotation=90)
        plt.title(f"Top 20 Feature Contributions to Cluster {cluster_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cluster_{cluster_id}_feature_contributions.png"))
        if should_show:
            plt.show()    
    
    # Otherwise analyze the mean and variance of each feature
    else:
        # Calculate statistics for each feature
        feature_stats = []
        for i, feature in enumerate(feature_names):
            cluster_values = cluster_points[:, i]
            non_cluster_values = X[~cluster_mask, i]
            
            if len(non_cluster_values) > 0:
                # Calculate how different this feature is inside vs outside the cluster
                mean_diff = np.mean(cluster_values) - np.mean(non_cluster_values)
                std_ratio = np.std(cluster_values) / (np.std(non_cluster_values) + 1e-10)
                
                feature_stats.append({
                    'feature': feature,
                    'mean_diff': mean_diff,
                    'std_ratio': std_ratio,
                    'importance': abs(mean_diff) * (1 + abs(1 - std_ratio))
                })
        
        # Sort by importance
        feature_stats.sort(key=lambda x: x['importance'], reverse=True)
        
        print("Most distinctive features for this cluster:")
        for i, stat in enumerate(feature_stats[:10]):  # Top 10
            print(f"  {i+1}. {stat['feature']}: "
                 f"Mean diff={stat['mean_diff']:.3f}, Std ratio={stat['std_ratio']:.3f}")
        
        # Visualize feature importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(min(20, len(feature_stats))), 
               [x['importance'] for x in feature_stats[:20]])
        plt.xticks(range(min(20, len(feature_stats))), 
                  [x['feature'] for x in feature_stats[:20]], rotation=90)
        plt.title(f"Top 20 Distinctive Features for Cluster {cluster_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cluster_{cluster_id}_feature_importance.png"))
        if should_show:
            plt.show()    
            
    # 2. Visualize this cluster in 2D/3D space
    print("\n--- Cluster Visualization ---")
    
    # T-SNE for better separation
    if cluster_size > 3:  # Need at least 3 points for t-SNE
        # Project to 2D
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        # Show all points with the cluster highlighted
        plt.figure(figsize=(12, 10))
        
        # Plot non-cluster points
        plt.scatter(X_tsne[~cluster_mask, 0], X_tsne[~cluster_mask, 1], 
                   c='lightgrey', s=10, alpha=0.3, label='Other Clusters')
        
        # Plot the cluster points
        if np.any(cluster_labels == 1) and np.any(cluster_labels == 0):
            # If mixed, show evil and benign differently
            evil_mask = cluster_mask & (evil_truth == 1)
            benign_mask = cluster_mask & (evil_truth == 0)
            
            plt.scatter(X_tsne[evil_mask, 0], X_tsne[evil_mask, 1], 
                       c='red', s=50, alpha=0.8, label=f'Cluster {cluster_id} (Evil)')
            
            plt.scatter(X_tsne[benign_mask, 0], X_tsne[benign_mask, 1], 
                       c='blue', s=30, alpha=0.6, label=f'Cluster {cluster_id} (Benign)')
        else:
            # If all evil or all benign, use one color
            color = 'red' if evil_percent > 50 else 'blue'
            plt.scatter(X_tsne[cluster_mask, 0], X_tsne[cluster_mask, 1], 
                       c=color, s=50, alpha=0.8, label=f'Cluster {cluster_id}')
        
        plt.title(f"T-SNE Visualization of Cluster {cluster_id} (Evil: {evil_percent:.1f}%)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"cluster_{cluster_id}_tsne.png"))
        if should_show:
            plt.show()
        
        # If sufficient data, also show 3D visualization
        if cluster_size >= 10:
            try:
                tsne3d = TSNE(n_components=3, random_state=42)
                X_tsne3d = tsne3d.fit_transform(X)
                
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot non-cluster points
                ax.scatter(X_tsne3d[~cluster_mask, 0], X_tsne3d[~cluster_mask, 1], X_tsne3d[~cluster_mask, 2],
                          c='lightgrey', s=10, alpha=0.1, label='Other Clusters')
                
                # Plot the cluster points
                if np.any(cluster_labels == 1) and np.any(cluster_labels == 0):
                    # If mixed, show evil and benign differently
                    evil_mask = cluster_mask & (evil_truth == 1)
                    benign_mask = cluster_mask & (evil_truth == 0)
                    
                    ax.scatter(X_tsne3d[evil_mask, 0], X_tsne3d[evil_mask, 1], X_tsne3d[evil_mask, 2],
                              c='red', s=50, alpha=0.8, label=f'Cluster {cluster_id} (Evil)')
                    
                    ax.scatter(X_tsne3d[benign_mask, 0], X_tsne3d[benign_mask, 1], X_tsne3d[benign_mask, 2],
                              c='blue', s=30, alpha=0.6, label=f'Cluster {cluster_id} (Benign)')
                else:
                    # If all evil or all benign, use one color
                    color = 'red' if evil_percent > 50 else 'blue'
                    ax.scatter(X_tsne3d[cluster_mask, 0], X_tsne3d[cluster_mask, 1], X_tsne3d[cluster_mask, 2],
                              c=color, s=50, alpha=0.8, label=f'Cluster {cluster_id}')
                
                ax.set_title(f"3D T-SNE Visualization of Cluster {cluster_id}")
                ax.legend()
                plt.savefig(os.path.join(output_dir, f"cluster_{cluster_id}_tsne3d.png"))
                if should_show:
                    plt.show()
            except Exception as e:
                print(f"Couldn't create 3D visualization: {e}")
    
    # 3. Compare to other clusters
    print("\n--- Comparison with Other Clusters ---")
    
    # Calculate average distance to other clusters
    unique_labels = np.unique(labels)
    cluster_centers = {}
    
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
        mask = labels == label
        if np.sum(mask) > 0:
            cluster_centers[label] = np.mean(X[mask], axis=0)
    
    if cluster_id in cluster_centers:
        distances = {}
        for label, center in cluster_centers.items():
            if label != cluster_id:
                distance = np.linalg.norm(cluster_centers[cluster_id] - center)
                distances[label] = distance
        
        # Sort by distance
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        
        print(f"Distance to other cluster centers:")
        for label, distance in sorted_distances[:5]:  # Top 5 closest
            print(f"  Cluster {label}: {distance:.3f}")
        
        # Visualize distances
        plt.figure(figsize=(10, 6))
        labels_list, distances_list = zip(*sorted_distances)
        plt.bar(range(len(distances_list)), distances_list)
        plt.xticks(range(len(distances_list)), [f"Cluster {l}" for l in labels_list], rotation=90)
        plt.title(f"Distance from Cluster {cluster_id} to Other Clusters")
        plt.ylabel("Euclidean Distance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cluster_{cluster_id}_distances.png"))
        if should_show:
            plt.show()
    
    print(f"\nCluster investigation complete. Results saved to {output_dir}.")


def run_dbscan_visualization_demo(data_file, output_dir='visualization_results'):
    """
    Run a demonstration of DBSCAN visualization and analysis
    
    Parameters:
    -----------
    data_file : str
        Path to preprocessed data file
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from {data_file}...")
    
    # Load the data
    df = pd.read_csv(data_file)
    
    # # Extract test set which has reliable labels
    # df_test = df[df['split'] == 'test'].copy()
    
    # Drop non-feature columns
    feature_cols = [col for col in df.columns if col not in ['split', 'sus', 'evil']]
    
    # Sample a subset for visualization (use balanced stratified sampling)
    print("Sampling data for visualization...")
    np.random.seed(42)
    
    # Count class frequencies
    evil_count = np.sum(df['evil'] == 1)
    benign_count = np.sum(df['evil'] == 0)
    
    # Determine sample size and distribution
    sample_size = min(10000, len(df))
    evil_ratio = evil_count / len(df)
    
    # Sample from each class
    evil_indices = df[df['evil'] == 1].index.values
    benign_indices = df[df['evil'] == 0].index.values
    
    evil_sample_size = int(sample_size * evil_ratio)
    benign_sample_size = sample_size - evil_sample_size
    
    # Ensure we don't request more than available
    evil_sample_size = min(evil_sample_size, len(evil_indices))
    benign_sample_size = min(benign_sample_size, len(benign_indices))
    
    # Sample indices
    evil_sampled = np.random.choice(evil_indices, size=evil_sample_size, replace=False)
    benign_sampled = np.random.choice(benign_indices, size=benign_sample_size, replace=False)
    
    # Combine and shuffle
    sampled_indices = np.concatenate([evil_sampled, benign_sampled])
    np.random.shuffle(sampled_indices)
    
    # Extract the sampled data
    df_sample = df.loc[sampled_indices].copy()
    
    # Apply preprocessing
    X = df_sample[feature_cols].values
    y = df_sample['evil'].values
    
    print(f"Sampled {len(df_sample)} points: {np.sum(y==1)} evil, {np.sum(y==0)} benign")
    
    # Apply PCA for dimensionality reduction
    print("Applying PCA...")
    pca = PCA(n_components=6)  # As per best results
    X_pca = pca.fit_transform(X)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")
    
    # Run DBSCAN with the optimal parameters
    print("Running DBSCAN...")
    dbscan = DBSCAN(eps=0.2, min_samples=20)
    labels = dbscan.fit_predict(X_pca)
    
    # Calculate cluster statistics
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    
    # Calculate percentage of evil points in each cluster
    cluster_stats = []
    evil_clusters = []
    
    for label in np.unique(labels):
        mask = labels == label
        cluster_size = np.sum(mask)
        if cluster_size > 0:
            evil_count = np.sum(y[mask] == 1)
            evil_percent = evil_count / cluster_size * 100
            
            label_name = "Noise" if label == -1 else f"Cluster {label}"
            cluster_stats.append({
                'label': label,
                'size': cluster_size,
                'evil_count': evil_count,
                'evil_percent': evil_percent
            })
            
            # Track clusters with high evil percentage
            if evil_percent >= 90 and cluster_size >= 20:
                evil_clusters.append(label)
    
    # Sort by evil percentage
    cluster_stats.sort(key=lambda x: x['evil_percent'], reverse=True)
    
    # Print top clusters by evil percentage
    print("\n--- Clusters sorted by evil percentage ---")
    for stat in cluster_stats:
        label_name = "Noise" if stat['label'] == -1 else f"Cluster {stat['label']}"
        print(f"{label_name}: {stat['size']} points, {stat['evil_count']} evil ({stat['evil_percent']:.1f}%)")
    
    # Visualize all clusters using PCA
    print("\nCreating visualizations...")
    visualize_clusters(X_pca, labels, evil_truth=y, method=None, n_components=2,
                     title="DBSCAN Clusters (PCA)", 
                     save_path=os.path.join(output_dir, "dbscan_clusters_pca.png"))
    
    # Visualize all clusters using t-SNE for better separation
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    visualize_clusters(X_tsne, labels, evil_truth=y, method=None, n_components=2,
                     title="DBSCAN Clusters (t-SNE)", 
                     save_path=os.path.join(output_dir, "dbscan_clusters_tsne.png"))
    
    # Highlight evil clusters
    if evil_clusters:
        visualize_clusters(X_tsne, labels, evil_truth=y, method=None, n_components=2,
                         title=f"Evil Clusters Highlighted (t-SNE)", 
                         highlight_clusters=evil_clusters,
                         save_path=os.path.join(output_dir, "evil_clusters_highlighted.png"))
    
    # Investigate the top evil cluster
    if evil_clusters:
        print(f"\nInvestigating top evil cluster...")
        investigate_cluster(X_pca, labels, y, evil_clusters[0], 
                          feature_names=feature_cols,
                          pca_components=pca.components_,
                          output_dir=os.path.join(output_dir, f"cluster_{evil_clusters[0]}_analysis"))
    
    print(f"\nVisualization complete! Results saved to {output_dir}.")


if __name__ == "__main__":
    start = time.time()
    # Replace with your data file path
    data_file = "preprocessed/prepared_data_cluster.csv"
    run_dbscan_visualization_demo(data_file)
    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds")