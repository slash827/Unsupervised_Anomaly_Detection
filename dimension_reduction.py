import os
import time
import glob
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from umap import UMAP
from sklearn.metrics import silhouette_score
from scipy.stats import entropy


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
        self.original_class_distribution = None
        self.sampled_class_distribution = None
    
    def load_and_preprocess(self):
        """Load, combine and preprocess datasets from CSV files."""
        print("\n" + "="*80)
        print(f"DATA LOADING AND PREPROCESSING - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        # Load and combine datasets
        csv_files = glob.glob(f"data{os.sep}*[!dns].csv")
        print(f"Found {len(csv_files)} CSV files: {csv_files}")
        
        data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        # Display initial information
        print(f"\nDataset shape: {data.shape} - {data.shape[0]} rows × {data.shape[1]} columns")
        print("\nFirst few rows:")
        print(data.head())
        
        # Check for missing values
        missing_values = data.isnull().sum()
        missing_pct = (missing_values / len(data)) * 100
        missing_info = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_pct
        })
        print("\nMissing values:")
        print(missing_info[missing_info['Missing Values'] > 0])
        
        if missing_info['Missing Values'].sum() == 0:
            print("No missing values found in the dataset.")

        # Separate numeric and categorical columns
        self.numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_columns = data.select_dtypes(include=['object']).columns

        print(f"\nNumeric columns ({len(self.numeric_columns)}): {self.numeric_columns.tolist()}")
        print(f"Categorical columns ({len(self.categorical_columns)}): {self.categorical_columns.tolist()}")

        # Handle missing values
        if len(self.numeric_columns) > 0:
            data[self.numeric_columns] = data[self.numeric_columns].fillna(data[self.numeric_columns].median())
            print("\nNumeric missing values filled with column medians")
        if len(self.categorical_columns) > 0:
            data[self.categorical_columns] = data[self.categorical_columns].fillna(data[self.categorical_columns].mode().iloc[0])
            print("Categorical missing values filled with column modes")

        # Display data type information
        print("\nData types:")
        print(data.dtypes.value_counts())
        
        # Encode categorical features
        le_dict = {}
        for col in self.categorical_columns:
            le_dict[col] = LabelEncoder()
            data[col] = le_dict[col].fit_transform(data[col])
            unique_values = data[col].nunique()
            print(f"Encoded column '{col}' with {unique_values} unique values")

        # Feature-target split (using 'evil' as target)
        self.X = data.drop(['evil', 'sus'], axis=1)  # Dropping both 'evil' and 'sus' columns
        self.y = data['evil']
        
        # Display target distribution
        self.original_class_distribution = self.y.value_counts(normalize=True) * 100
        print("\nTarget class distribution:")
        class_dist_df = pd.DataFrame({
            'Class': self.original_class_distribution.index,
            'Count': self.y.value_counts().values,
            'Percentage (%)': self.original_class_distribution.values
        })
        print(class_dist_df)
        
        # Check for class imbalance
        if len(class_dist_df) > 1:
            imbalance_ratio = class_dist_df['Count'].max() / class_dist_df['Count'].min()
            print(f"\nClass imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 10:
                print("WARNING: Severe class imbalance detected. Consider using class weights or balancing techniques.")
        
        print(f"\nPreprocessing completed. Final dataset shape: {self.X.shape}")
        print("="*80)
        
        return self
    
    def sample_data(self, sample_size=10000):
        """
        Sample data using stratified sampling to maintain class distribution.
        
        Args:
            sample_size: Number of samples to extract
        """
        print("\n" + "="*80)
        print(f"DATA SAMPLING - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        print(f"Original dataset size: {self.X.shape[0]} samples with {self.X.shape[1]} features")
        sample_size = min(sample_size, self.X.shape[0])  # Cap at specified sample size
        print(f"Sampling {sample_size} records using stratified sampling to maintain class distribution")

        # Calculate sampling ratio
        sampling_ratio = sample_size / self.X.shape[0]
        print(f"Sampling ratio: {sampling_ratio:.4f} ({sampling_ratio*100:.2f}% of the original data)")

        # Stratified sampling to maintain class distribution
        sss = StratifiedShuffleSplit(n_splits=1, test_size=sampling_ratio, random_state=42)
        for _, sample_idx in sss.split(self.X, self.y):
            self.X_sample = self.X.iloc[sample_idx].reset_index(drop=True)
            self.y_sample = self.y.iloc[sample_idx].reset_index(drop=True)
        
        print(f"Sampled dataset shape: {self.X_sample.shape[0]} rows × {self.X_sample.shape[1]} columns")
        
        # Verify class distribution is maintained
        self.sampled_class_distribution = self.y_sample.value_counts(normalize=True) * 100
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Class': self.original_class_distribution.index,
            'Original (%)': self.original_class_distribution.values,
            'Sampled (%)': [self.sampled_class_distribution.get(c, 0) for c in self.original_class_distribution.index],
            'Difference (%)': [self.sampled_class_distribution.get(c, 0) - self.original_class_distribution.get(c, 0) 
                            for c in self.original_class_distribution.index]
        })
        
        print("\nClass distribution comparison (original vs. sampled):")
        print(comparison_df)
        
        # Calculate distribution similarity measures
        jsd = self._jensen_shannon_divergence(
            self.original_class_distribution.values/100, 
            np.array([self.sampled_class_distribution.get(c, 0) for c in self.original_class_distribution.index])/100
        )
        
        print(f"\nJensen-Shannon divergence between original and sampled distributions: {jsd:.8f}")
        print(f"Interpretation: Values close to 0 indicate similar distributions (ideal: 0)")
        print("="*80)
        
        return self
    
    def _jensen_shannon_divergence(self, p, q):
        """Calculate Jensen-Shannon divergence between two distributions."""
        m = (p + q) / 2
        return (entropy(p, m) + entropy(q, m)) / 2
    
    def scale_features(self):
        """Standardize features."""
        print("\n" + "="*80)
        print(f"FEATURE SCALING - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        print("Standardizing features using StandardScaler (mean=0, std=1)")
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X_sample)
        
        # Display scaling statistics
        scale_mean = np.mean(self.X_scaled, axis=0)
        scale_std = np.std(self.X_scaled, axis=0)
        
        print("\nScaling verification (checking a few columns):")
        for i in range(min(5, self.X_scaled.shape[1])):
            print(f"Column {i}: Mean = {scale_mean[i]:.6f}, Std = {scale_std[i]:.6f}")
            
        print(f"\nOverall mean of scaled data: {np.mean(scale_mean):.6f}")
        print(f"Overall std of scaled data: {np.mean(scale_std):.6f}")
        print("="*80)
        
        return self
    
    def analyze_feature_importance(self, should_show=True):
        """
        Analyze feature importance using Random Forest.
        
        Args:
            should_show: Whether to display the plot
            
        Returns:
            DataFrame with feature importance
        """
        print("\n" + "="*80)
        print(f"FEATURE IMPORTANCE ANALYSIS - Started at {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        print("Training Random Forest to analyze feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_scaled, self.y_sample)

        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'Feature': self.X_sample.columns,
            'Importance': rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Display top and bottom features
        print("\nTop 15 most important features:")
        print(feature_importance.head(15))
        
        print("\nBottom 5 least important features:")
        print(feature_importance.tail(5))
        
        # Calculate cumulative importance
        feature_importance['Cumulative_Importance'] = feature_importance['Importance'].cumsum()
        
        # Find how many features are needed for X% of importance
        thresholds = [0.5, 0.75, 0.9, 0.95]
        for threshold in thresholds:
            n_features = len(feature_importance[feature_importance['Cumulative_Importance'] <= threshold])
            print(f"\nNumber of features needed for {threshold*100}% of total importance: {n_features+1}")
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Top 15 Features by Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        if should_show:
            plt.show()
        else:
            plt.close()
            
        print("\nFeature importance analysis completed")
        print("="*80)
        
        return feature_importance


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
            filename='tsne_visualization.png',
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
            filename=f'umap_visualization_{len(self.y)}_samples.png',
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
    SAMPLE_SIZE = 10000
    data_processor.load_and_preprocess().sample_data(sample_size=SAMPLE_SIZE).scale_features()
    
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
    return tsne, umap


if __name__ == "__main__":
    start = time.time()
    # Set should_show=False if you don't want plots to be displayed (useful for batch processing)
    main(should_show=False)
    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")