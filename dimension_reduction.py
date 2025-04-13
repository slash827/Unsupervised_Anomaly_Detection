import os
import time
import glob
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from umap import UMAP

from utils import load_and_preprocess_beth_data, properly_balanced_stratified_sample


class DataProcessor:
    """Class for loading, preprocessing, and sampling data."""
    
    def __init__(self):
        self.X = None
        self.y = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.X_sample = None
        self.y_sample = None
        self.X_scaled = None
        self.features_for_analysis = None
        self.original_data = None
        self.sample_indices = None
    
    def load_and_preprocess(self):
        """Load, preprocess and prepare data using utils.py functions."""
        print("\n" + "="*80)
        print(f"DATA LOADING AND PREPROCESSING - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        # Load and combine datasets
        csv_files = glob.glob(f"data{os.sep}*data.csv")
        print(f"Found {len(csv_files)} CSV files: {csv_files}")
        
        df_scaled, self.features_for_analysis = load_and_preprocess_beth_data(csv_files, "data/")
        
        # Extract features and target
        self.X = df_scaled.drop(['sus', 'evil'], axis=1)
        self.y = df_scaled['evil']
        self.original_data = df_scaled
        
        # Display additional information about processed data
        print("\nProcessed feature statistics:")
        for col in self.X.columns:
            print(f"{col}: Mean={self.X[col].mean():.4f}, Std={self.X[col].std():.4f}")
        
        print("\nCorrelation with target:")
        correlations = df_scaled.corr()['evil'].sort_values(ascending=False)
        print(correlations)
        
        print(f"\nPreprocessing completed. Final dataset shape: {self.X.shape}")
        print("="*80)
        
        return self
    
    def sample_data(self, sample_size=10000):
        """
        Sample data using properly balanced stratified sampling from utils.py.
        
        Args:
            sample_size: Number of samples to extract
        """
        print("\n" + "="*80)
        print(f"DATA SAMPLING - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        print(f"Original dataset size: {self.X.shape[0]} samples with {self.X.shape[1]} features")
        sample_size = min(sample_size, self.X.shape[0])  # Cap at specified sample size
        print(f"Sampling {sample_size} records using balanced stratified sampling")
        
        # Convert to numpy arrays for sampling function
        X_array = self.X.values
        y_array = self.y.values
        
        # Perform balanced stratified sampling
        X_sampled, y_sampled, sample_indices = properly_balanced_stratified_sample(
            X_array, y_array, sample_size, random_state=42
        )
        
        self.sample_indices = sample_indices
        
        # Convert back to pandas DataFrame/Series with original column names
        self.X_sample = pd.DataFrame(X_sampled, columns=self.X.columns)
        self.y_sample = pd.Series(y_sampled, name='evil')
        
        print(f"\nSampled dataset shape: {self.X_sample.shape[0]} rows × {self.X_sample.shape[1]} columns")
        
        # Calculate class distribution similarity
        original_class_dist = np.bincount(self.y.astype(int)) / len(self.y)
        sampled_class_dist = np.bincount(self.y_sample.astype(int)) / len(self.y_sample)
        
        # Create comparison dataframe for reporting
        classes = np.unique(np.concatenate([self.y.unique(), self.y_sample.unique()]))
        comparison_df = pd.DataFrame({
            'Class': classes,
            'Original Count': [np.sum(self.y == c) for c in classes],
            'Original (%)': [np.sum(self.y == c)/len(self.y)*100 for c in classes],
            'Sampled Count': [np.sum(self.y_sample == c) for c in classes],
            'Sampled (%)': [np.sum(self.y_sample == c)/len(self.y_sample)*100 for c in classes],
        })
        
        print("\nClass distribution comparison (original vs. sampled):")
        print(comparison_df)
        
        # Check if original suspicious samples are present in the sample
        if 'sus' in self.original_data.columns:
            suspicious_mask = (self.original_data['evil'] == 0) & (self.original_data['sus'] == 1)
            suspicious_indices = suspicious_mask[suspicious_mask].index
            
            suspicious_in_sample = np.intersect1d(suspicious_indices, sample_indices)
            print(f"\nSuspicious samples (evil=0, sus=1) in the sampled data: {len(suspicious_in_sample)}")
            print(f"Percentage of sampled data that is suspicious: {len(suspicious_in_sample)/len(sample_indices)*100:.2f}%")
        
        print("="*80)
        return self
    
    def scale_features(self):
        """
        Scale features if needed and handle any NaN values.
        """
        print("\n" + "="*80)
        print(f"FEATURE SCALING AND NAN HANDLING - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        # Check for NaN values before scaling
        nan_count_before = np.isnan(self.X_sample.values).sum()
        if nan_count_before > 0:
            print(f"WARNING: Found {nan_count_before} NaN values in the data before scaling")
            print("Filling NaN values with column medians...")
            
            # Check which columns have NaNs
            nan_cols = self.X_sample.columns[self.X_sample.isna().any()].tolist()
            print(f"Columns with NaN values: {nan_cols}")
            
            # Fill NaN values by column
            for col in nan_cols:
                median_val = self.X_sample[col].median()
                nan_count_in_col = self.X_sample[col].isna().sum()
                print(f"  - Column '{col}': {nan_count_in_col} NaN values, filling with median {median_val}")
                self.X_sample[col] = self.X_sample[col].fillna(median_val)
        else:
            print("No NaN values found in the data before scaling")
        
        # Data might already be scaled by the preprocessing function, but we'll check
        # and standardize if needed
        if np.abs(self.X_sample.mean().mean()) < 0.01 and np.abs(self.X_sample.std().mean() - 1.0) < 0.01:
            print("Data appears to be already standardized, skipping StandardScaler")
            self.X_scaled = self.X_sample.values
        else:
            print("Standardizing features using StandardScaler (mean=0, std=1)")
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(self.X_sample)
        
        # Verify no NaN values after scaling
        nan_count_after = np.isnan(self.X_scaled).sum()
        if nan_count_after > 0:
            print(f"WARNING: Found {nan_count_after} NaN values after scaling!")
            print("Replacing remaining NaN values with zeros...")
            self.X_scaled = np.nan_to_num(self.X_scaled)
            
            # Double-check
            final_nan_count = np.isnan(self.X_scaled).sum()
            if final_nan_count > 0:
                raise ValueError(f"Could not remove all NaN values! {final_nan_count} NaNs remain.")
            else:
                print("Successfully removed all NaN values")
        else:
            print("No NaN values found after scaling")
        
        # Display scaling statistics
        scale_mean = np.mean(self.X_scaled, axis=0)
        scale_std = np.std(self.X_scaled, axis=0)
        
        print("\nScaling verification (checking all columns):")
        for i, col in enumerate(self.X_sample.columns):
            print(f"{col}: Mean = {scale_mean[i]:.6f}, Std = {scale_std[i]:.6f}")
            
        print(f"\nOverall mean of scaled data: {np.mean(scale_mean):.6f}")
        print(f"Overall std of scaled data: {np.mean(scale_std):.6f}")
        
        # Check for any outliers in the scaled data
        print("\nChecking for outliers in scaled data:")
        for i, col in enumerate(self.X_sample.columns):
            q1 = np.percentile(self.X_scaled[:, i], 25)
            q3 = np.percentile(self.X_scaled[:, i], 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = np.sum((self.X_scaled[:, i] < lower_bound) | (self.X_scaled[:, i] > upper_bound))
            print(f"{col}: {outliers} outliers ({outliers/len(self.X_scaled)*100:.2f}% of samples)")
        
        print("="*80)
        return self


class DimensionalityReduction:
    """Base class for dimensionality reduction methods."""
    
    def __init__(self, should_show=True, plot_dir=""):
        self.should_show = should_show
        self.results_df = None
        self.name = "Generic"
        self.plot_dir = plot_dir
    
    def visualize_reduction(self, X_reduced, y, title, filename, x_label, y_label):
        """
        Visualize dimensionality reduction results.
        
        Args:
            X_reduced: Reduced feature matrix
            y: Target vector
            title: Plot title
            filename: Output filename
            x_label: X-axis label
            y_label: Y-axis label
        """
        print(f"\nVisualizing {self.name} results...")
        
        # Create a DataFrame for the results
        self.results_df = pd.DataFrame()
        self.results_df['x'] = X_reduced[:, 0]
        self.results_df['y'] = X_reduced[:, 1]
        self.results_df['label'] = y

        # Visualize results
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x='x', y='y',
            hue='label',
            palette=sns.color_palette("hls", len(np.unique(y))),
            data=self.results_df,
            legend="full",
            alpha=0.8
        )

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        save_path = self.plot_dir + os.sep + filename if self.plot_dir != "" else filename 
        plt.savefig(save_path)
        
        print(f"Visualization saved to {save_path}")
        
        if self.should_show:
            plt.show()
        else:
            plt.close()
        
        # Calculate basic statistics about the embedding
        print("\nEmbedding statistics:")
        print(f"Range of x-coordinates: [{self.results_df['x'].min():.4f}, {self.results_df['x'].max():.4f}]")
        print(f"Range of y-coordinates: [{self.results_df['y'].min():.4f}, {self.results_df['y'].max():.4f}]")
        
        # Check for clustering in the embedding
        if len(np.unique(y)) > 1:
            try:
                silhouette = silhouette_score(X_reduced, y)
                print(f"Silhouette score: {silhouette:.4f}")
                print(f"Interpretation: Values closer to 1 indicate better-defined clusters (range: -1 to 1)")
                
                if silhouette > 0.5:
                    print("The classes are well-separated in the embedding space.")
                elif silhouette > 0.25:
                    print("The classes show moderate separation in the embedding space.")
                elif silhouette > 0:
                    print("The classes show weak separation in the embedding space.")
                else:
                    print("The classes are not well-separated in the embedding space.")
            except Exception as e:
                print(f"Could not calculate silhouette score: {e}")
        
        return self
    
    def analyze_label_distribution(self):
        """
        Analyze the distribution of labels in the data.
        """
        if self.results_df is None:
            print("No results available for analysis. Run dimensionality reduction first.")
            return None
            
        print(f"\nAnalyzing label distribution in {self.name} space...")
        
        # Calculate the distribution of labels
        label_counts = self.results_df['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        label_counts['Percentage'] = (label_counts['Count'] / label_counts['Count'].sum()) * 100
        
        print("Distribution of labels:")
        print(label_counts)
        
        # Calculate cluster-specific statistics
        print("\nCluster-specific statistics:")
        for label in self.results_df['label'].unique():
            mask = self.results_df['label'] == label
            cluster_data = self.results_df[mask]
            
            print(f"\nCluster {label}:")
            print(f"  Number of points: {len(cluster_data)} ({len(cluster_data)/len(self.results_df)*100:.2f}%)")
            print(f"  Centroid: ({cluster_data['x'].mean():.4f}, {cluster_data['y'].mean():.4f})")
            print(f"  Spread: x-std={cluster_data['x'].std():.4f}, y-std={cluster_data['y'].std():.4f}")
            
            # Calculate average distance between points with the same label
            if len(cluster_data) > 1:
                from sklearn.metrics.pairwise import euclidean_distances
                points = cluster_data[['x', 'y']].values
                distances = euclidean_distances(points)
                avg_dist = np.sum(distances) / (len(distances) * len(distances) - len(distances))
                print(f"  Average distance between points: {avg_dist:.4f}")
        
        return label_counts


class TSNEAnalyzer(DimensionalityReduction):
    """Class for t-SNE dimensionality reduction and analysis."""
    
    def __init__(self, should_show=True, plot_dir=""):
        super().__init__(should_show, plot_dir)
        self.X_tsne = None
        self.name = "t-SNE"
        self.perplexity_used = None
    
    def perform_reduction(self, X, perplexity=30, n_iter=1000, n_components=2):
        """
        Perform t-SNE dimensionality reduction.
        
        Args:
            X: Feature matrix
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
            n_components: Number of dimensions in output
        """
        print("\n" + "="*80)
        print(f"t-SNE DIMENSIONALITY REDUCTION - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        # Adjust perplexity if needed - shouldn't exceed n/5
        original_perplexity = perplexity
        perplexity = min(perplexity, len(X) // 5)
        self.perplexity_used = perplexity
        
        if original_perplexity != perplexity:
            print(f"Adjusted perplexity from {original_perplexity} to {perplexity} (max = n/5)")
        
        print(f"Running t-SNE with:")
        print(f"  - Perplexity: {perplexity}")
        print(f"  - Iterations: {n_iter}")
        print(f"  - Components: {n_components}")
        print(f"  - Input dimensions: {X.shape[1]}")
        
        start_time = time.time()
        tsne = TSNE(
            n_components=n_components, 
            perplexity=perplexity, 
            n_iter=n_iter,
            random_state=42, 
            verbose=1, 
            n_jobs=-1  # Use all cores
        )
        self.X_tsne = tsne.fit_transform(X)
        end_time = time.time()
        
        print(f"\nt-SNE completed in {end_time - start_time:.2f} seconds")
        print(f"Output shape: {self.X_tsne.shape}")
        print(f"Memory usage of t-SNE output: {self.X_tsne.nbytes / (1024*1024):.2f} MB")
        
        # Calculate statistics about the output
        x_range = np.max(self.X_tsne[:, 0]) - np.min(self.X_tsne[:, 0])
        y_range = np.max(self.X_tsne[:, 1]) - np.min(self.X_tsne[:, 1])
        print(f"X-axis range: {x_range:.4f}")
        print(f"Y-axis range: {y_range:.4f}")
        print(f"Aspect ratio (x/y): {x_range/y_range:.4f}")
        
        # Calculate crowding analysis - how spread out are the points?
        from scipy.spatial.distance import pdist
        distances = pdist(self.X_tsne)
        print(f"\nPoint distribution statistics:")
        print(f"  Min distance between points: {np.min(distances):.4f}")
        print(f"  Max distance between points: {np.max(distances):.4f}")
        print(f"  Mean distance between points: {np.mean(distances):.4f}")
        print(f"  Median distance between points: {np.median(distances):.4f}")
        print("="*80)
        
        return self
    
    def visualize(self, y):
        """
        Visualize t-SNE results.
        
        Args:
            y: Target vector
        """
        if self.X_tsne is None:
            print("No t-SNE results available. Run perform_reduction first.")
            return self
            
        return self.visualize_reduction(
            self.X_tsne, 
            y,
            title=f't-SNE Visualization (perplexity={self.perplexity_used})',
            filename=f'tsne_visualization_{len(self.X_tsne)}.png',
            x_label='t-SNE Feature 1',
            y_label='t-SNE Feature 2'
        )
    
    def compare_perplexities(self, X, y, perplexities=None):
        """
        Compare t-SNE results with different perplexity values.
        
        Args:
            X: Feature matrix
            y: Target vector
            perplexities: List of perplexity values to test
        """
        print("\n" + "="*80)
        print(f"t-SNE PERPLEXITY COMPARISON - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        if perplexities is None:
            # Choose appropriate perplexity values based on dataset size
            max_perp = min(len(X) // 5, 50)  # Perplexity shouldn't exceed n/5
            perplexities = [max(5, int(max_perp/4)), max(15, int(max_perp/2)), max_perp]
        
        print(f"Comparing t-SNE with perplexity values: {perplexities}")
        
        plt.figure(figsize=(18, 6))
        
        # For collecting metrics
        metric_data = []
        
        for i, perp in enumerate(perplexities):
            print(f"\nRunning t-SNE with perplexity={perp}")
            start_time = time.time()
            tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000, random_state=42, n_jobs=-1)
            tsne_result = tsne.fit_transform(X)
            end_time = time.time()
            
            print(f"  Completed in {end_time - start_time:.2f} seconds")

            plt.subplot(1, 3, i+1)
            sns.scatterplot(
                x=tsne_result[:, 0], y=tsne_result[:, 1],
                hue=y,
                palette=sns.color_palette("hls", len(np.unique(y))),
                legend="full" if i == 2 else None,
                alpha=0.8
            )
            plt.title(f't-SNE with Perplexity = {perp}')
            
            # Calculate metrics for this perplexity
            if len(np.unique(y)) > 1:
                try:
                    silhouette = silhouette_score(tsne_result, y)
                    print(f"  Silhouette score: {silhouette:.4f}")
                    
                    # Calculate class separations
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=10)
                    nn.fit(tsne_result)
                    distances, indices = nn.kneighbors(tsne_result)
                    same_class = np.array([y.iloc[i] == y.iloc[indices[i][1:]] for i in range(len(y))])
                    same_class_ratio = np.mean(same_class)
                    print(f"  Same-class ratio in 10 nearest neighbors: {same_class_ratio:.4f}")
                    
                    # Add to metrics collection
                    metric_data.append({
                        'Perplexity': perp,
                        'Silhouette': silhouette,
                        'Same-Class Ratio': same_class_ratio,
                        'Runtime (s)': end_time - start_time
                    })
                    
                except Exception as e:
                    print(f"  Could not calculate clustering metrics: {e}")
        
        plt.tight_layout()
        plt.savefig('tsne_perplexity_comparison.png')
        
        if self.should_show:
            plt.show()
        else:
            plt.close()
        
        # Print the metric comparison
        if metric_data:
            print("\nMetrics comparison across perplexity values:")
            metrics_df = pd.DataFrame(metric_data)
            print(metrics_df)
            
            best_silhouette = metrics_df.loc[metrics_df['Silhouette'].idxmax()]
            best_same_class = metrics_df.loc[metrics_df['Same-Class Ratio'].idxmax()]
            
            print(f"\nBest perplexity based on silhouette score: {best_silhouette['Perplexity']}")
            print(f"Best perplexity based on same-class ratio: {best_same_class['Perplexity']}")
        
        print("="*80)
        return self


class UMAPAnalyzer(DimensionalityReduction):
    """Class for UMAP dimensionality reduction and analysis."""
    
    def __init__(self, should_show=True, plot_dir=""):
        super().__init__(should_show, plot_dir=plot_dir)
        self.X_umap = None
        self.name = "UMAP"
        self.n_neighbors_used = None
        self.min_dist_used = None
    
    def perform_reduction(self, X, n_neighbors=15, min_dist=0.1, n_components=2):
        """
        Perform UMAP dimensionality reduction.
        
        Args:
            X: Feature matrix
            n_neighbors: Size of local neighborhood
            min_dist: Minimum distance between points in the low-dimensional representation
            n_components: Number of dimensions in the output
        """
        print("\n" + "="*80)
        print(f"UMAP DIMENSIONALITY REDUCTION - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        self.n_neighbors_used = n_neighbors
        self.min_dist_used = min_dist
        
        print(f"Running UMAP with:")
        print(f"  - n_neighbors: {n_neighbors}")
        print(f"  - min_dist: {min_dist}")
        print(f"  - Components: {n_components}")
        print(f"  - Input dimensions: {X.shape[1]}")
        
        start_time = time.time()
        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=42,
            verbose=True
        )
        self.X_umap = reducer.fit_transform(X)
        end_time = time.time()
        
        print(f"\nUMAP completed in {end_time - start_time:.2f} seconds")
        print(f"Output shape: {self.X_umap.shape}")
        print(f"Memory usage of UMAP output: {self.X_umap.nbytes / (1024*1024):.2f} MB")
        
        # Calculate statistics about the output
        x_range = np.max(self.X_umap[:, 0]) - np.min(self.X_umap[:, 0])
        y_range = np.max(self.X_umap[:, 1]) - np.min(self.X_umap[:, 1])
        print(f"X-axis range: {x_range:.4f}")
        print(f"Y-axis range: {y_range:.4f}")
        print(f"Aspect ratio (x/y): {x_range/y_range:.4f}")
        
        # Calculate crowding analysis - how spread out are the points?
        from scipy.spatial.distance import pdist
        distances = pdist(self.X_umap)
        print(f"\nPoint distribution statistics:")
        print(f"  Min distance between points: {np.min(distances):.4f}")
        print(f"  Max distance between points: {np.max(distances):.4f}")
        print(f"  Mean distance between points: {np.mean(distances):.4f}")
        print(f"  Median distance between points: {np.median(distances):.4f}")
        print("="*80)
        
        return self
    
    def visualize(self, y):
        """
        Visualize UMAP results.
        
        Args:
            y: Target vector
        """
        if self.X_umap is None:
            print("No UMAP results available. Run perform_reduction first.")
            return self
            
        return self.visualize_reduction(
            self.X_umap, 
            y,
            title=f'UMAP Visualization (n_neighbors={self.n_neighbors_used}, min_dist={self.min_dist_used})',
            filename=f'umap_visualization_{len(self.X_umap)}.png',
            x_label='UMAP Feature 1',
            y_label='UMAP Feature 2'
        )
    
    def compare_parameters(self, X, y, n_neighbors_list=None, min_dist_list=None):
        """
        Compare UMAP results with different parameter settings.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_neighbors_list: List of n_neighbors values to test
            min_dist_list: List of min_dist values to test
        """
        print("\n" + "="*80)
        print(f"UMAP PARAMETER COMPARISON - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        if n_neighbors_list is None:
            n_neighbors_list = [5, 15, 30]
        
        if min_dist_list is None:
            min_dist_list = [0.0, 0.1, 0.5]
        
        print(f"Comparing UMAP with different parameter combinations:")
        print(f"  - n_neighbors values: {n_neighbors_list}")
        print(f"  - min_dist values: {min_dist_list}")
        print(f"  - Total combinations: {len(n_neighbors_list) * len(min_dist_list)}")
        
        fig, axes = plt.subplots(len(n_neighbors_list), len(min_dist_list), figsize=(15, 15))
        
        # For collecting metrics
        metric_data = []
        
        for i, n_neighbors in enumerate(n_neighbors_list):
            for j, min_dist in enumerate(min_dist_list):
                print(f"\nRunning UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")
                
                start_time = time.time()
                reducer = UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=2,
                    random_state=42
                )
                
                umap_result = reducer.fit_transform(X)
                end_time = time.time()
                
                print(f"  Completed in {end_time - start_time:.2f} seconds")
                
                ax = axes[i, j]
                scatter = ax.scatter(
                    umap_result[:, 0], 
                    umap_result[:, 1],
                    c=y,
                    cmap='viridis', 
                    alpha=0.8, 
                    s=5
                )
                
                ax.set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist}')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Calculate metrics for this parameter combination
                if len(np.unique(y)) > 1:
                    try:
                        silhouette = silhouette_score(umap_result, y)
                        print(f"  Silhouette score: {silhouette:.4f}")
                        
                        # Calculate class separations
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=10)
                        nn.fit(umap_result)
                        distances, indices = nn.kneighbors(umap_result)
                        same_class = np.array([y.iloc[i] == y.iloc[indices[i][1:]] for i in range(len(y))])
                        same_class_ratio = np.mean(same_class)
                        print(f"  Same-class ratio in 10 nearest neighbors: {same_class_ratio:.4f}")
                        
                        # Calculate average distance between clusters
                        from scipy.spatial.distance import cdist
                        class_centroids = {}
                        for label in np.unique(y):
                            mask = y == label
                            class_centroids[label] = np.mean(umap_result[mask], axis=0)
                        
                        if len(class_centroids) > 1:
                            centroids = np.array(list(class_centroids.values()))
                            inter_class_distances = cdist(centroids, centroids)
                            np.fill_diagonal(inter_class_distances, np.inf)  # Ignore distance to self
                            avg_inter_class_dist = np.mean(np.min(inter_class_distances, axis=1))
                            print(f"  Average minimum distance between cluster centroids: {avg_inter_class_dist:.4f}")
                        
                            # Add to metrics collection
                            metric_data.append({
                                'n_neighbors': n_neighbors,
                                'min_dist': min_dist,
                                'Silhouette': silhouette,
                                'Same-Class Ratio': same_class_ratio,
                                'Inter-Class Distance': avg_inter_class_dist,
                                'Runtime (s)': end_time - start_time
                            })
                        
                    except Exception as e:
                        print(f"  Could not calculate clustering metrics: {e}")
        
        plt.tight_layout()
        plt.savefig('umap_parameter_comparison.png')
        
        if self.should_show:
            plt.show()
        else:
            plt.close()
            
        # Print the metric comparison
        if metric_data:
            print("\nMetrics comparison across parameter combinations:")
            metrics_df = pd.DataFrame(metric_data)
            print(metrics_df.to_string())
            
            # Find best parameters for different metrics
            best_silhouette = metrics_df.loc[metrics_df['Silhouette'].idxmax()]
            best_same_class = metrics_df.loc[metrics_df['Same-Class Ratio'].idxmax()]
            best_inter_class = metrics_df.loc[metrics_df['Inter-Class Distance'].idxmax()]
            
            print("\nBest parameter combinations:")
            print(f"For silhouette score: n_neighbors={best_silhouette['n_neighbors']}, min_dist={best_silhouette['min_dist']}")
            print(f"For same-class ratio: n_neighbors={best_same_class['n_neighbors']}, min_dist={best_same_class['min_dist']}")
            print(f"For inter-class distance: n_neighbors={best_inter_class['n_neighbors']}, min_dist={best_inter_class['min_dist']}")
            
            # Recommendations based on metrics
            print("\nParameter selection recommendations:")
            print("  - Lower n_neighbors values (5-15) tend to preserve local structure")
            print("  - Higher n_neighbors values (30+) tend to preserve global structure")
            print("  - Lower min_dist values (0.0-0.1) create tighter clusters")
            print("  - Higher min_dist values (0.5+) create more spread out visualizations")
            
            best_overall = metrics_df.loc[(metrics_df['Silhouette'] + 
                                         metrics_df['Same-Class Ratio'] + 
                                         metrics_df['Inter-Class Distance']/np.max(metrics_df['Inter-Class Distance'])).idxmax()]
            
            print(f"\nBest overall parameter combination: n_neighbors={best_overall['n_neighbors']}, min_dist={best_overall['min_dist']}")
            print(f"  Silhouette: {best_overall['Silhouette']:.4f}")
            print(f"  Same-Class Ratio: {best_overall['Same-Class Ratio']:.4f}")
            print(f"  Inter-Class Distance: {best_overall['Inter-Class Distance']:.4f}")
        
        print("="*80)
        return self


def main(should_show=True):
    """
    Run the complete analysis pipeline.
    
    Args:
        should_show: Whether to display plots
    """
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load, preprocess, sample, and scale data
    SAMPLE_SIZE = 25000
    data_processor.load_and_preprocess().sample_data(sample_size=SAMPLE_SIZE).scale_features()
    
    if not os.path.exists("plots"):
        os.mkdir("plots")

    if not os.path.exists(f"plots{os.sep}Dimension Reduction"):
        os.mkdir(f"plots{os.sep}Dimension Reduction")

    # Run t-SNE analysis first
    print("\n========== Starting t-SNE Analysis ==========")
    tsne = TSNEAnalyzer(should_show=should_show, plot_dir=f"plots{os.sep}Dimension Reduction")
    tsne.perform_reduction(data_processor.X_scaled)
    tsne.visualize(data_processor.y_sample)
    tsne.analyze_label_distribution()
    tsne.compare_perplexities(data_processor.X_scaled, data_processor.y_sample)
    
    # Run UMAP analysis second
    print("\n========== Starting UMAP Analysis ==========")
    umap = UMAPAnalyzer(should_show=should_show, plot_dir=f"plots{os.sep}Dimension Reduction")
    umap.perform_reduction(data_processor.X_scaled)
    umap.visualize(data_processor.y_sample)
    umap.analyze_label_distribution()
    umap.compare_parameters(data_processor.X_scaled, data_processor.y_sample)


if __name__ == "__main__":
    start = time.time()
    # Set should_show=False if you don't want plots to be displayed (useful for batch processing)
    main(should_show=False)
    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")