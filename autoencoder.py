import os, glob
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.base import BaseEstimator, TransformerMixin

from utils import load_and_preprocess_beth_data

# Set plot style and figure size for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


class Autoencoder(BaseEstimator, TransformerMixin):
    """
    Autoencoder implementation using scikit-learn's MLPRegressor.
    Acts as an encoder and decoder combined.
    """
    def __init__(self, encoding_dim=8, hidden_layers=(128, 64, 32), alpha=0.0001, 
                 max_iter=200, random_state=42, activation='relu', 
                 solver='adam', learning_rate_init=0.001):
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        
        # Define hidden layer architecture for the autoencoder
        # Going from input -> ... -> encoding_dim -> ... -> input
        hidden_layer_sizes = list(hidden_layers) + [encoding_dim] + list(reversed(hidden_layers))
        
        # Initialize the MLP 
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=learning_rate_init,
            verbose=True
        )
        
    def fit(self, X, y=None):
        """
        Fit the autoencoder. In an autoencoder, X is both the input and target.
        """
        self.model.fit(X, X)
        return self
        
    def transform(self, X):
        """
        Extract the encoded representation (latent space)
        This requires getting the output after the bottleneck layer
        """
        # This is an approximation as MLPRegressor doesn't easily expose intermediate layers
        # We compute forward pass until the bottleneck layer
        
        # Get activations from first layer to bottleneck
        activations = []
        activation = X
        
        # Forward through encoder part
        for i in range(len(self.hidden_layers) + 1):
            # Get weights and bias for this layer
            W = self.model.coefs_[i]
            b = self.model.intercepts_[i]
            
            # Linear transformation
            activation = np.dot(activation, W) + b
            
            # Apply activation function (assuming relu)
            if self.activation == 'relu':
                activation = np.maximum(0, activation)
            elif self.activation == 'tanh':
                activation = np.tanh(activation)
            elif self.activation == 'logistic':
                activation = 1 / (1 + np.exp(-activation))
            
            # Store activation
            activations.append(activation)
        
        # Return the bottleneck activation (latent representation)
        return activations[-1]  
    
    def inverse_transform(self, encoded):
        """
        Approximately transform encoded data back to original space.
        This is a rough approximation as MLP doesn't have a built-in decoder path.
        """
        # Since MLPRegressor doesn't allow partial forward pass, this is an approximation
        # We use the trained model to reconstruct from the full pipeline
        # This is not mathematically exact but serves as a demonstration
        
        # For a true encoder/decoder setup, we would train separate encoder and decoder models
        return self.reconstruct(encoded)
    
    def reconstruct(self, X):
        """
        Reconstruct input from either original data or encoded data.
        For simplicity, we'll just use the whole model to reconstruct from original.
        """
        return self.model.predict(X)
    
    def get_reconstruction_error(self, X):
        """
        Calculate reconstruction error for each sample.
        """
        X_reconstructed = self.reconstruct(X)
        # Mean squared error per sample
        reconstruction_error = np.mean(np.square(X - X_reconstructed), axis=1)
        return reconstruction_error


def train_autoencoder(df, feature_names, encoding_dim=8, max_iter=200):
    """
    Train an autoencoder for anomaly detection on BETH dataset.
    
    Args:
        df: Preprocessed dataframe
        feature_names: List of feature names to use
        encoding_dim: Dimension of the latent space
        max_iter: Maximum number of training iterations
        
    Returns:
        trained autoencoder, latent representations, and reconstruction errors
    """
    # Separate features and labels
    features = df[feature_names].values
    labels = df[['sus', 'evil']]
    
    # Split into train/validation sets (only using benign data for training)
    benign_mask = (df['evil'] == 0) & (df['sus'] == 0)
    train_mask = benign_mask  # Only train on benign data
    
    # Split benign data into train and validation sets
    X_train, X_val, _, _ = train_test_split(
        features[benign_mask], 
        np.zeros(np.sum(benign_mask)),  # dummy labels
        test_size=0.2, 
        random_state=42
    )
    
    # Create the autoencoder
    print(f"\nCreating autoencoder with encoding dimension {encoding_dim}...")
    autoencoder = Autoencoder(
        encoding_dim=encoding_dim,
        hidden_layers=(128, 64, 32),
        max_iter=max_iter,
        alpha=0.0001,
        random_state=42
    )
    
    # Train the autoencoder
    print(f"\nTraining autoencoder on {X_train.shape[0]} benign samples...")
    start_time = time.time()
    autoencoder.fit(X_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Calculate reconstruction error for all samples (including anomalies)
    print("\nCalculating reconstruction error for all samples...")
    reconstruction_error = autoencoder.get_reconstruction_error(features)
    
    # Create a latent space representation for all samples
    latent_representation = autoencoder.transform(features)
    latent_df = pd.DataFrame(
        latent_representation, 
        columns=[f'latent_{i+1}' for i in range(encoding_dim)]
    )
    
    # Calculate validation set performance
    val_reconstruction_error = autoencoder.get_reconstruction_error(X_val)
    val_mse = np.mean(val_reconstruction_error)
    print(f"Validation MSE: {val_mse:.6f}")
    
    # Create results dataframe
    results_df = df.copy()
    results_df['reconstruction_error'] = reconstruction_error
    
    return autoencoder, results_df, latent_df


def analyze_autoencoder_results(results_df, feature_names):
    """
    Analyze and visualize autoencoder results.
    """
    # Basic statistics on reconstruction error
    print("\nReconstruction error statistics:")
    print(f"Min: {results_df['reconstruction_error'].min():.6f}")
    print(f"Max: {results_df['reconstruction_error'].max():.6f}")
    print(f"Mean: {results_df['reconstruction_error'].mean():.6f}")
    print(f"Median: {results_df['reconstruction_error'].median():.6f}")
    
    # Calculate average reconstruction error by class
    benign = results_df[(results_df['evil'] == 0) & (results_df['sus'] == 0)]
    suspicious = results_df[(results_df['evil'] == 0) & (results_df['sus'] == 1)]
    malicious = results_df[results_df['evil'] == 1]
    
    print("\nAverage reconstruction error by class:")
    print(f"Benign (n={len(benign)}): {benign['reconstruction_error'].mean():.6f}")
    if len(suspicious) > 0:
        print(f"Suspicious (n={len(suspicious)}): {suspicious['reconstruction_error'].mean():.6f}")
    if len(malicious) > 0:
        print(f"Malicious (n={len(malicious)}): {malicious['reconstruction_error'].mean():.6f}")
    
    # Plot reconstruction error distribution
    plt.figure(figsize=(14, 10))
    
    # Determine optimal bin settings
    max_error = np.percentile(results_df['reconstruction_error'], 99)  # Using 99th percentile for better visualization
    bins = np.linspace(0, max_error, 50)
    
    # Plot histograms for each class
    plt.hist(benign['reconstruction_error'], bins=bins, alpha=0.5, density=True, 
             label=f'Benign (n={len(benign)})', color='green')
    
    if len(suspicious) > 0:
        plt.hist(suspicious['reconstruction_error'], bins=bins, alpha=0.5, density=True,
                 label=f'Suspicious (n={len(suspicious)})', color='orange')
    
    if len(malicious) > 0:
        plt.hist(malicious['reconstruction_error'], bins=bins, alpha=0.5, density=True,
                 label=f'Malicious (n={len(malicious)})', color='red')
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Autoencoder Reconstruction Error Distribution by Class')
    plt.legend()
    plt.grid(True)
    plt.savefig('sklearn_autoencoder_reconstruction_error_dist.png', dpi=300)
    
    # Calculate ROC curve and AUC for evil detection
    if len(malicious) > 0:
        fpr, tpr, thresholds = roc_curve(results_df['evil'], results_df['reconstruction_error'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Malicious Sample Detection')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('sklearn_autoencoder_roc_curve.png', dpi=300)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(results_df['evil'], results_df['reconstruction_error'])
        avg_precision = average_precision_score(results_df['evil'], results_df['reconstruction_error'])
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for Malicious Sample Detection')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig('sklearn_autoencoder_precision_recall_curve.png', dpi=300)
    
    # Find optimal threshold based on F1 score
    if len(malicious) > 0:
        # Calculate F1 score for different thresholds
        thresholds = np.linspace(results_df['reconstruction_error'].min(), 
                               results_df['reconstruction_error'].max(), 100)
        f1_scores = []
        
        for threshold in thresholds:
            predicted = (results_df['reconstruction_error'] > threshold).astype(int)
            true_positives = np.sum((predicted == 1) & (results_df['evil'] == 1))
            false_positives = np.sum((predicted == 1) & (results_df['evil'] == 0))
            false_negatives = np.sum((predicted == 0) & (results_df['evil'] == 1))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        # Find threshold with highest F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"\nOptimal threshold based on F1 score: {best_threshold:.6f} (F1 = {best_f1:.3f})")
        
        # Calculate performance at optimal threshold
        predicted = (results_df['reconstruction_error'] > best_threshold).astype(int)
        true_positives = np.sum((predicted == 1) & (results_df['evil'] == 1))
        false_positives = np.sum((predicted == 1) & (results_df['evil'] == 0))
        true_negatives = np.sum((predicted == 0) & (results_df['evil'] == 0))
        false_negatives = np.sum((predicted == 0) & (results_df['evil'] == 1))
        
        accuracy = (true_positives + true_negatives) / len(results_df)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print("\nPerformance at optimal threshold:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {best_f1:.3f}")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Negatives: {false_negatives}")
        
        # Add prediction to results dataframe
        results_df['predicted_evil'] = predicted


def visualize_latent_space(latent_df, results_df, encoding_dim):
    """
    Visualize the latent space representation.
    """
    # Add class information
    latent_df['sus'] = results_df['sus'].values
    latent_df['evil'] = results_df['evil'].values
    latent_df['reconstruction_error'] = results_df['reconstruction_error'].values
    
    # Create colors based on class
    colors = []
    for s, e in zip(latent_df['sus'], latent_df['evil']):
        if e == 1:  # Malicious
            colors.append('red')
        elif s == 1:  # Suspicious
            colors.append('orange')
        else:  # Benign
            colors.append('green')
    
    # If we have a lot of points, sample for better visualization
    if len(latent_df) > 10000:
        sample_indices = np.random.choice(len(latent_df), 10000, replace=False)
        plot_df = latent_df.iloc[sample_indices]
        plot_colors = [colors[i] for i in sample_indices]
    else:
        plot_df = latent_df
        plot_colors = colors
    
    # Plot first two dimensions of latent space colored by class
    plt.figure(figsize=(12, 10))
    plt.scatter(plot_df['latent_1'], plot_df['latent_2'], c=plot_colors, alpha=0.6, s=10)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Autoencoder Latent Space Projection')
    plt.grid(True)
    
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Benign'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Suspicious'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Malicious'),
    ]
    plt.legend(handles=legend_elements)
    
    plt.savefig('sklearn_autoencoder_latent_space.png', dpi=300)
    
    # Plot the latent space colored by reconstruction error
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        plot_df['latent_1'], 
        plot_df['latent_2'], 
        c=plot_df['reconstruction_error'],
        cmap='viridis',
        alpha=0.6, 
        s=10
    )
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Autoencoder Latent Space Colored by Reconstruction Error')
    plt.colorbar(scatter, label='Reconstruction Error')
    plt.grid(True)
    plt.savefig('sklearn_autoencoder_latent_space_recon_error.png', dpi=300)
    
    # Plot pairs of latent dimensions if we have more than 2 dimensions
    if encoding_dim > 2:
        # Create a pairplot-like visualization for the first 4 dimensions
        dim_to_use = min(4, encoding_dim)
        fig, axes = plt.subplots(dim_to_use, dim_to_use, figsize=(18, 18))
        
        for i in range(dim_to_use):
            for j in range(dim_to_use):
                if i != j:  # Skip diagonal
                    axes[i, j].scatter(
                        plot_df[f'latent_{j+1}'], 
                        plot_df[f'latent_{i+1}'], 
                        c=plot_colors, 
                        alpha=0.6, 
                        s=5
                    )
                    axes[i, j].set_xlabel(f'Latent {j+1}')
                    axes[i, j].set_ylabel(f'Latent {i+1}')
                    axes[i, j].grid(True)
                else:
                    # Histogram on diagonal
                    axes[i, j].hist(
                        [
                            plot_df[plot_df['evil'] == 0][f'latent_{i+1}'],
                            plot_df[plot_df['evil'] == 1][f'latent_{i+1}']
                        ],
                        bins=20,
                        color=['green', 'red'],
                        alpha=0.6,
                        label=['Benign', 'Malicious']
                    )
                    axes[i, j].set_xlabel(f'Latent {i+1}')
                    axes[i, j].legend()
        
        plt.tight_layout()
        plt.savefig('sklearn_autoencoder_latent_pairs.png', dpi=300)
    
    # Plot 3D visualization if we have at least 3 dimensions
    if encoding_dim >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            plot_df['latent_1'],
            plot_df['latent_2'],
            plot_df['latent_3'],
            c=plot_colors,
            alpha=0.7,
            s=10
        )
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_zlabel('Latent Dimension 3')
        ax.set_title('3D Latent Space Colored by Class')
        
        # Add a legend manually
        ax.legend(handles=legend_elements)
        
        plt.savefig('sklearn_autoencoder_latent_3d.png', dpi=300)
        
        # 3D plot colored by reconstruction error
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            plot_df['latent_1'],
            plot_df['latent_2'],
            plot_df['latent_3'],
            c=plot_df['reconstruction_error'],
            cmap='viridis',
            alpha=0.7,
            s=10
        )
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_zlabel('Latent Dimension 3')
        ax.set_title('3D Latent Space Colored by Reconstruction Error')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Reconstruction Error')
        
        plt.savefig('sklearn_autoencoder_latent_3d_error.png', dpi=300)


def main():
    """
    Main function to run the sklearn-based Autoencoder analysis on the BETH dataset.
    """
    # File path to the BETH csv files
    data_path = os.getcwd() + os.sep + "data"
    csv_files = glob.glob(f"data{os.sep}*data.csv")

    # Load and preprocess data
    df_scaled, feature_names = load_and_preprocess_beth_data(csv_files, data_path)
    
    # Configure autoencoder parameters
    encoding_dim = 10  # Dimension of the latent space
    max_iter = 20    # Maximum training iterations
    
    # Train autoencoder
    autoencoder, results_df, latent_df = train_autoencoder(
        df_scaled, 
        feature_names, 
        encoding_dim=encoding_dim,
        max_iter=max_iter
    )
    
    # Analyze results
    analyze_autoencoder_results(results_df, feature_names)
    
    # Visualize latent space
    visualize_latent_space(latent_df, results_df, encoding_dim)
    
    # Save results
    results_df.to_csv('beth_sklearn_autoencoder_anomalies.csv', index=False)
    latent_df.to_csv('beth_sklearn_autoencoder_latent.csv', index=False)
    
    print("\nSklearn-based Autoencoder analysis complete!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    start = time.time()
    main()
    end = time.time()
    print(f"Total time took: {end - start} seconds")