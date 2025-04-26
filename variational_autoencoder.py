import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

import os, glob
import time
import warnings
warnings.filterwarnings('ignore')

from utils import load_and_preprocess_beth_data


# Set plot style and figure size for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


class VAE(BaseEstimator, TransformerMixin):
    """
    Variational Autoencoder implementation using scikit-learn's MLPRegressor.
    This simulates a VAE by training separate encoder and decoder networks and
    implementing the sampling and KL divergence manually.
    """
    def __init__(self, encoding_dim=8, hidden_layers=(128, 64, 32), alpha=0.0001, 
                 max_iter=200, random_state=42, activation='relu', 
                 solver='adam', learning_rate_init=0.001, kl_weight=0.5):
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.kl_weight = kl_weight
        
        # Define encoder architecture
        encoder_layers = list(hidden_layers) + [encoding_dim * 2]  # *2 for mean and log_var
        
        # Define decoder architecture
        decoder_layers = list(hidden_layers) + [0]  # The last layer will be set in fit
        
        # Initialize the encoder
        self.encoder = MLPRegressor(
            hidden_layer_sizes=encoder_layers,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=learning_rate_init,
            verbose=True
        )
        
        # Initialize the decoder
        self.decoder = MLPRegressor(
            hidden_layer_sizes=decoder_layers,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=learning_rate_init,
            verbose=True
        )
        
        # Training history
        self.training_losses = {'encoder': [], 'decoder': []}
        self.validation_losses = {'encoder': [], 'decoder': []}
        
    def _split_encoder_output(self, encoder_output):
        """Split encoder output into mean and log_var."""
        # Get the actual dimensions from the encoder output
        output_dim = encoder_output.shape[1]
        latent_dim = output_dim // 2  # Divide by 2 to get the actual latent dimension
        
        # Update the encoding_dim to match the actual dimensions
        self.encoding_dim = latent_dim
        
        # Split the output
        mean = encoder_output[:, :latent_dim]
        log_var = encoder_output[:, latent_dim:]
        
        return mean, log_var
    
    def _sample_latent(self, mean, log_var):
        """Sample from the latent space using the reparameterization trick."""
        eps = np.random.normal(0, 1, size=(mean.shape[0], self.encoding_dim))
        return mean + np.exp(0.5 * log_var) * eps
    
    def _kl_divergence(self, mean, log_var):
        """Compute KL divergence between the encoder distribution and a standard normal."""
        return -0.5 * np.sum(1 + log_var - np.square(mean) - np.exp(log_var), axis=1)
    
    def fit(self, X, y=None):
        """
        Fit the VAE by training encoder and decoder separately.
        """
        print("Fitting encoder...")
        
        # First, train the encoder
        self.encoder.fit(X, np.hstack([X, X]))  # Arbitrary target to pre-train
        
        # Fix the decoder's output layer size
        input_dim = X.shape[1]
        decoder_layers = list(self.hidden_layers) + [input_dim]
        self.decoder = MLPRegressor(
            hidden_layer_sizes=decoder_layers,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            learning_rate_init=self.learning_rate_init,
            verbose=True
        )
        
        # Training loop with custom loss
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=self.random_state)
        
        # Initial latent space generation
        encoder_output = self.encoder.predict(X_train)
        mean, log_var = self._split_encoder_output(encoder_output)
        z = self._sample_latent(mean, log_var)
        
        print("Fitting decoder...")
        # Train decoder
        self.decoder.fit(z, X_train)
        
        # Now perform iterative training with custom loss
        print("Fine-tuning VAE...")
        for epoch in range(5):  # Additional fine-tuning iterations
            # Generate latent space
            encoder_output = self.encoder.predict(X_train)
            mean, log_var = self._split_encoder_output(encoder_output)
            z = self._sample_latent(mean, log_var)
            
            # Generate reconstructions
            reconstructions = self.decoder.predict(z)
            
            # Calculate reconstruction error
            reconstruction_loss = np.mean(np.square(X_train - reconstructions), axis=1)
            
            # Calculate KL divergence
            kl_loss = self._kl_divergence(mean, log_var)
            
            # Calculate total loss
            total_loss = np.mean(reconstruction_loss + self.kl_weight * kl_loss)
            
            # Store training loss
            self.training_losses['decoder'].append(np.mean(reconstruction_loss))
            self.training_losses['encoder'].append(np.mean(kl_loss))
            
            # Calculate validation loss
            encoder_output_val = self.encoder.predict(X_val)
            mean_val, log_var_val = self._split_encoder_output(encoder_output_val)
            z_val = self._sample_latent(mean_val, log_var_val)
            reconstructions_val = self.decoder.predict(z_val)
            
            reconstruction_loss_val = np.mean(np.square(X_val - reconstructions_val), axis=1)
            kl_loss_val = self._kl_divergence(mean_val, log_var_val)
            
            # Store validation loss
            self.validation_losses['decoder'].append(np.mean(reconstruction_loss_val))
            self.validation_losses['encoder'].append(np.mean(kl_loss_val))
            
            print(f"Epoch {epoch+1}, Train Loss: {total_loss:.6f}, "
                  f"Recon Loss: {np.mean(reconstruction_loss):.6f}, "
                  f"KL Loss: {np.mean(kl_loss):.6f}")
        
        return self
        
    def transform(self, X):
        """
        Transform data to latent space (z_mean).
        """
        encoder_output = self.encoder.predict(X)
        mean, _ = self._split_encoder_output(encoder_output)
        return mean
    
    def fit_transform(self, X, y=None):
        """
        Fit the model and transform the data to latent space.
        """
        self.fit(X)
        return self.transform(X)
    
    def reconstruct(self, X):
        """
        Reconstruct data from original inputs.
        """
        encoder_output = self.encoder.predict(X)
        mean, log_var = self._split_encoder_output(encoder_output)
        z = self._sample_latent(mean, log_var)
        return self.decoder.predict(z)
    
    def generate(self, n_samples=1):
        """
        Generate new samples from the latent space.
        """
        z = np.random.normal(0, 1, size=(n_samples, self.encoding_dim))
        return self.decoder.predict(z)
    
    def get_reconstruction_metrics(self, X):
        """
        Calculate reconstruction error and KL divergence for each sample.
        """
        # Get latent representation
        encoder_output = self.encoder.predict(X)
        mean, log_var = self._split_encoder_output(encoder_output)
        
        # Sample from latent space
        z = self._sample_latent(mean, log_var)
        
        # Reconstruct
        reconstructions = self.decoder.predict(z)
        
        # Calculate reconstruction error
        reconstruction_error = np.mean(np.square(X - reconstructions), axis=1)
        
        # Calculate KL divergence
        kl_divergence = self._kl_divergence(mean, log_var)
        
        # Calculate total loss
        total_loss = reconstruction_error + self.kl_weight * kl_divergence
        
        return {
            'reconstruction_error': reconstruction_error,
            'kl_divergence': kl_divergence,
            'total_loss': total_loss,
            'latent_mean': mean,
            'latent_log_var': log_var
        }


def train_vae(df, feature_names, encoding_dim=8, max_iter=100, kl_weight=0.5):
    """
    Train a variational autoencoder for anomaly detection on BETH dataset.
    
    Args:
        df: Preprocessed dataframe
        feature_names: List of feature names to use
        encoding_dim: Dimension of the latent space
        max_iter: Maximum number of training iterations
        kl_weight: Weight for KL divergence loss
        
    Returns:
        trained VAE, results with anomaly metrics, and latent representations
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
    
    # Create the VAE
    print(f"\nCreating VAE with encoding dimension {encoding_dim} and KL weight {kl_weight}...")
    vae = VAE(
        encoding_dim=encoding_dim,
        hidden_layers=(128, 64, 32),
        max_iter=max_iter,
        alpha=0.0001,
        random_state=42,
        kl_weight=kl_weight
    )
    
    # Train the VAE
    print(f"\nTraining VAE on {X_train.shape[0]} benign samples...")
    start_time = time.time()
    vae.fit(X_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Calculate metrics for all samples
    print("\nCalculating metrics for all samples...")
    metrics = vae.get_reconstruction_metrics(features)
    
    # Create a latent space representation for all samples
    latent_representation = metrics['latent_mean']
    latent_df = pd.DataFrame(
        latent_representation, 
        columns=[f'latent_{i+1}' for i in range(latent_representation.shape[1])]  # Dynamic columns
    )
    
    # Add variance information
    latent_variance = np.exp(metrics['latent_log_var'])
    latent_df['latent_variance_mean'] = np.mean(latent_variance, axis=1)
    
    # Calculate validation set performance
    val_metrics = vae.get_reconstruction_metrics(X_val)
    val_recon_error = np.mean(val_metrics['reconstruction_error'])
    val_kl = np.mean(val_metrics['kl_divergence'])
    val_total = np.mean(val_metrics['total_loss'])
    
    print(f"Validation metrics:")
    print(f"Reconstruction MSE: {val_recon_error:.6f}")
    print(f"KL Divergence: {val_kl:.6f}")
    print(f"Total Loss: {val_total:.6f}")
    
    # Create results dataframe
    results_df = df.copy()
    results_df['reconstruction_error'] = metrics['reconstruction_error']
    results_df['kl_divergence'] = metrics['kl_divergence']
    results_df['total_loss'] = metrics['total_loss']
    
    # Save the training history
    training_history = {
        'training_losses': vae.training_losses,
        'validation_losses': vae.validation_losses
    }
    
    return vae, results_df, latent_df, training_history


def analyze_vae_results(results_df, latent_df, encoding_dim, training_history, feature_names):
    """
    Analyze and visualize VAE results with focus on explaining performance.
    """
    # Plot training history
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_history['training_losses']['decoder'], label='Train Reconstruction Loss')
    plt.plot(training_history['validation_losses']['decoder'], label='Val Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Reconstruction Loss During Training')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_history['training_losses']['encoder'], label='Train KL Divergence')
    plt.plot(training_history['validation_losses']['encoder'], label='Val KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence During Training')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sklearn_vae_training_history.png', dpi=300)
    
    # Basic statistics on reconstruction error, KL divergence, and total loss
    print("\nReconstruction error statistics:")
    print(f"Min: {results_df['reconstruction_error'].min():.6f}")
    print(f"Max: {results_df['reconstruction_error'].max():.6f}")
    print(f"Mean: {results_df['reconstruction_error'].mean():.6f}")
    print(f"Median: {results_df['reconstruction_error'].median():.6f}")
    
    print("\nKL divergence statistics:")
    print(f"Min: {results_df['kl_divergence'].min():.6f}")
    print(f"Max: {results_df['kl_divergence'].max():.6f}")
    print(f"Mean: {results_df['kl_divergence'].mean():.6f}")
    print(f"Median: {results_df['kl_divergence'].median():.6f}")
    
    print("\nTotal loss statistics:")
    print(f"Min: {results_df['total_loss'].min():.6f}")
    print(f"Max: {results_df['total_loss'].max():.6f}")
    print(f"Mean: {results_df['total_loss'].mean():.6f}")
    print(f"Median: {results_df['total_loss'].median():.6f}")
    
    # Calculate average metrics by class
    benign = results_df[(results_df['evil'] == 0) & (results_df['sus'] == 0)]
    suspicious = results_df[(results_df['evil'] == 0) & (results_df['sus'] == 1)]
    malicious = results_df[results_df['evil'] == 1]
    
    print("\nAverage metrics by class:")
    print(f"Benign (n={len(benign)}):")
    print(f"  Reconstruction Error: {benign['reconstruction_error'].mean():.6f}")
    print(f"  KL Divergence: {benign['kl_divergence'].mean():.6f}")
    print(f"  Total Loss: {benign['total_loss'].mean():.6f}")
    
    if len(suspicious) > 0:
        print(f"Suspicious (n={len(suspicious)}):")
        print(f"  Reconstruction Error: {suspicious['reconstruction_error'].mean():.6f}")
        print(f"  KL Divergence: {suspicious['kl_divergence'].mean():.6f}")
        print(f"  Total Loss: {suspicious['total_loss'].mean():.6f}")
    
    if len(malicious) > 0:
        print(f"Malicious (n={len(malicious)}):")
        print(f"  Reconstruction Error: {malicious['reconstruction_error'].mean():.6f}")
        print(f"  KL Divergence: {malicious['kl_divergence'].mean():.6f}")
        print(f"  Total Loss: {malicious['total_loss'].mean():.6f}")
    
    # Create visualization of class separation
    plt.figure(figsize=(15, 6))
    
    # Plot reconstruction error by class with violin plots
    plt.subplot(1, 3, 1)
    violin_parts = plt.violinplot(
        [
            np.log1p(benign['reconstruction_error'].values), 
            np.log1p(suspicious['reconstruction_error'].values) if len(suspicious) > 0 else [],
            np.log1p(malicious['reconstruction_error'].values)
        ],
        showmeans=True
    )
    # Color the violin plots
    colors = ['green', 'orange', 'red']
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    plt.xticks([1, 2, 3], ['Benign', 'Suspicious', 'Malicious'])
    plt.ylabel('Log(Reconstruction Error + 1)')
    plt.title('Reconstruction Error by Class')
    plt.grid(True, axis='y')
    
    # Plot KL divergence by class
    plt.subplot(1, 3, 2)
    violin_parts = plt.violinplot(
        [
            np.log1p(benign['kl_divergence'].values), 
            np.log1p(suspicious['kl_divergence'].values) if len(suspicious) > 0 else [],
            np.log1p(malicious['kl_divergence'].values)
        ],
        showmeans=True
    )
    # Color the violin plots
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    plt.xticks([1, 2, 3], ['Benign', 'Suspicious', 'Malicious'])
    plt.ylabel('Log(KL Divergence + 1)')
    plt.title('KL Divergence by Class')
    plt.grid(True, axis='y')
    
    # Plot total loss by class
    plt.subplot(1, 3, 3)
    violin_parts = plt.violinplot(
        [
            np.log1p(benign['total_loss'].values), 
            np.log1p(suspicious['total_loss'].values) if len(suspicious) > 0 else [],
            np.log1p(malicious['total_loss'].values)
        ],
        showmeans=True
    )
    # Color the violin plots
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    plt.xticks([1, 2, 3], ['Benign', 'Suspicious', 'Malicious'])
    plt.ylabel('Log(Total Loss + 1)')
    plt.title('Total Loss by Class')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('sklearn_vae_class_separation.png', dpi=300)
    
    # Reconstruction error distribution (detailed)
    plt.figure(figsize=(15, 10))
    
    # Determine optimal bin settings with log scale
    log_recon_errors = np.log1p(results_df['reconstruction_error'])
    max_log_error = np.percentile(log_recon_errors, 99)  # Use 99th percentile for better visualization
    bins = np.linspace(0, max_log_error, 100)
    
    # Plot log-scale histograms for each class
    plt.hist(np.log1p(benign['reconstruction_error']), bins=bins, alpha=0.5, density=True, 
             label=f'Benign (n={len(benign)})', color='green')
    
    if len(suspicious) > 0:
        plt.hist(np.log1p(suspicious['reconstruction_error']), bins=bins, alpha=0.5, density=True,
                 label=f'Suspicious (n={len(suspicious)})', color='orange')
    
    if len(malicious) > 0:
        plt.hist(np.log1p(malicious['reconstruction_error']), bins=bins, alpha=0.5, density=True,
                 label=f'Malicious (n={len(malicious)})', color='red')
    
    plt.xlabel('Log(Reconstruction Error + 1)')
    plt.ylabel('Density')
    plt.title('VAE Reconstruction Error Distribution by Class (Log Scale)')
    plt.legend()
    plt.grid(True)
    
    # Add threshold visualization
    if len(malicious) > 0:
        # Find optimal threshold
        thresholds = np.linspace(results_df['reconstruction_error'].min(), 
                               np.percentile(results_df['reconstruction_error'], 99.9), 1000)
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
        
        # Add a vertical line for the threshold
        plt.axvline(np.log1p(best_threshold), color='black', linestyle='--', 
                    label=f'Optimal Threshold (F1={best_f1:.3f})')
        plt.legend()
    
    plt.savefig('sklearn_vae_reconstruction_error_dist.png', dpi=300)
    
    # Plot feature contribution to reconstruction error
    if len(malicious) > 0:
        # Sample data for efficiency
        sample_size = min(10000, len(results_df))
        sample_indices = np.random.choice(len(results_df), sample_size, replace=False)
        sample_df = results_df.iloc[sample_indices].copy()
        
        # Analyze correlations between features and reconstruction error
        feature_corrs = []
        for feature in feature_names:
            corr = np.corrcoef(sample_df[feature], sample_df['reconstruction_error'])[0, 1]
            feature_corrs.append((feature, abs(corr)))
        
        # Sort by correlation magnitude
        feature_corrs.sort(key=lambda x: x[1], reverse=True)
        
        # Plot top features correlation with reconstruction error
        plt.figure(figsize=(12, 8))
        features = [f[0] for f in feature_corrs]
        correlations = [f[1] for f in feature_corrs]
        
        plt.bar(features, correlations, color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('Absolute Correlation with Reconstruction Error')
        plt.title('Feature Importance for Anomaly Detection')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('sklearn_vae_feature_importance.png', dpi=300)
        
        # Calculate ROC and PR curves for the three metrics
        metrics = ['reconstruction_error', 'kl_divergence', 'total_loss']
        plt.figure(figsize=(15, 6))
        
        # ROC curves
        plt.subplot(1, 2, 1)
        for metric in metrics:
            fpr, tpr, _ = roc_curve(results_df['evil'], results_df[metric])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # PR curves
        plt.subplot(1, 2, 2)
        for metric in metrics:
            precision, recall, _ = precision_recall_curve(results_df['evil'], results_df[metric])
            ap_score = average_precision_score(results_df['evil'], results_df[metric])
            plt.plot(recall, precision, lw=2, label=f'{metric} (AP = {ap_score:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('sklearn_vae_roc_pr_curves.png', dpi=300)
        
        # Find optimal thresholds and performance for all metrics
        best_thresholds = {}
        best_performances = {}
        
        for metric in metrics:
            thresholds = np.linspace(results_df[metric].min(), 
                                   np.percentile(results_df[metric], 99.9), 1000)
            f1_scores = []
            
            for threshold in thresholds:
                predicted = (results_df[metric] > threshold).astype(int)
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
            
            best_thresholds[metric] = best_threshold
            
            # Calculate performance metrics
            predicted = (results_df[metric] > best_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(results_df['evil'], predicted).ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            best_performances[metric] = {
                'threshold': best_threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }
            
            # Add predicted column to results_df
            results_df[f'predicted_evil_{metric}'] = predicted
        
        # Print performance metrics
        print("\nOptimal thresholds and performance:")
        for metric, perf in best_performances.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"Threshold: {perf['threshold']:.6f}")
            print(f"Accuracy: {perf['accuracy']:.3f}")
            print(f"Precision: {perf['precision']:.3f}")
            print(f"Recall: {perf['recall']:.3f}")
            print(f"F1 Score: {perf['f1']:.3f}")
            print(f"True Positives: {perf['tp']}")
            print(f"False Positives: {perf['fp']}")
            print(f"True Negatives: {perf['tn']}")
            print(f"False Negatives: {perf['fn']}")
        
        # Create confusion matrix visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            perf = best_performances[metric]
            cm = np.array([[perf['tn'], perf['fp']], 
                           [perf['fn'], perf['tp']]])
            
            # Normalize the confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            im = axes[i].imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            
            # Add text annotations
            thresh = cm_norm.max() / 2.
            for j in range(cm.shape[0]):
                for k in range(cm.shape[1]):
                    axes[i].text(k, j, f'{cm[j, k]}\n({cm_norm[j, k]:.2f})',
                             ha="center", va="center",
                             color="white" if cm_norm[j, k] > thresh else "black")
            
            # Add labels
            axes[i].set_xticks([0, 1])
            axes[i].set_yticks([0, 1])
            axes[i].set_xticklabels(['Benign', 'Malicious'])
            axes[i].set_yticklabels(['Benign', 'Malicious'])
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('sklearn_vae_confusion_matrices.png', dpi=300)
        
        # Calculate and visualize the error distribution for correctly and incorrectly classified samples
        plt.figure(figsize=(15, 8))
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i+1)
            
            # Select samples
            pred_col = f'predicted_evil_{metric}'
            
            tp_samples = results_df[(results_df['evil'] == 1) & (results_df[pred_col] == 1)][metric]
            fp_samples = results_df[(results_df['evil'] == 0) & (results_df[pred_col] == 1)][metric]
            tn_samples = results_df[(results_df['evil'] == 0) & (results_df[pred_col] == 0)][metric]
            fn_samples = results_df[(results_df['evil'] == 1) & (results_df[pred_col] == 0)][metric]
            
            # Plot histogram for each group
            bins = np.linspace(0, np.percentile(results_df[metric], 99), 50)
            plt.hist(tp_samples, bins=bins, alpha=0.5, label='True Positive', color='green')
            plt.hist(fp_samples, bins=bins, alpha=0.5, label='False Positive', color='red')
            plt.hist(tn_samples, bins=bins, alpha=0.5, label='True Negative', color='blue')
            plt.hist(fn_samples, bins=bins, alpha=0.5, label='False Negative', color='orange')
            
            # Add threshold line
            plt.axvline(best_thresholds[metric], color='black', linestyle='--', 
                        label=f'Threshold: {best_thresholds[metric]:.4f}')
            
            plt.xlabel(metric.replace('_', ' ').title())
            plt.ylabel('Count')
            plt.title(f'{metric.replace("_", " ").title()} Distribution by Classification Result')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('sklearn_vae_classification_error_dist.png', dpi=300)
        
        # Create latent space visualization showing feature importance
        # Find which latent dimensions are most important for anomaly detection
        latent_correlation = {}
        for i in range(1, encoding_dim + 1):
            col_name = f'latent_{i}'
            if col_name in latent_df.columns:
                corr = np.corrcoef(latent_df[col_name], results_df['reconstruction_error'])[0, 1]
                latent_correlation[col_name] = abs(corr)
        
        # Sort by correlation magnitude
        sorted_latent = sorted(latent_correlation.items(), key=lambda x: x[1], reverse=True)
        
        # Plot latent dimension correlations
        plt.figure(figsize=(10, 6))
        plt.bar([item[0] for item in sorted_latent], [item[1] for item in sorted_latent], color='skyblue')
        plt.xlabel('Latent Dimension')
        plt.ylabel('Absolute Correlation with Reconstruction Error')
        plt.title('Latent Dimension Importance for Anomaly Detection')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('sklearn_vae_latent_importance.png', dpi=300)


def visualize_latent_space(latent_df, results_df, encoding_dim):
    """
    Visualize the latent space representation with focus on anomaly detection.
    """
    # Add class information
    latent_df['sus'] = results_df['sus'].values
    latent_df['evil'] = results_df['evil'].values
    latent_df['reconstruction_error'] = results_df['reconstruction_error'].values
    latent_df['kl_divergence'] = results_df['kl_divergence'].values
    latent_df['total_loss'] = results_df['total_loss'].values
    
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
        plot_df = latent_df.iloc[sample_indices].copy()
        plot_colors = [colors[i] for i in sample_indices]
    else:
        plot_df = latent_df.copy()
        plot_colors = colors
    
    # Find the two most significant latent dimensions based on variance
    latent_cols = [col for col in latent_df.columns if col.startswith('latent_') and col != 'latent_variance_mean']
    
    # Calculate variance for each latent dimension
    latent_variances = {}
    for col in latent_cols:
        latent_variances[col] = plot_df[col].var()
    
    # Get the two dimensions with highest variance
    top_dims = sorted(latent_variances.items(), key=lambda x: x[1], reverse=True)[:2]
    dim1, dim2 = top_dims[0][0], top_dims[1][0]
    
    # Plot latent space colored by class
    plt.figure(figsize=(12, 10))
    plt.scatter(plot_df[dim1], plot_df[dim2], c=plot_colors, alpha=0.6, s=10)
    plt.xlabel(dim1)
    plt.ylabel(dim2)
    plt.title('VAE Latent Space Projection by Class')
    plt.grid(True)
    
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Benign'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Suspicious'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Malicious'),
    ]
    plt.legend(handles=legend_elements)
    plt.savefig('sklearn_vae_latent_space_class.png', dpi=300)
    
    # Plot the latent space colored by reconstruction error
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        plot_df[dim1], 
        plot_df[dim2], 
        c=plot_df['reconstruction_error'],
        cmap='viridis',
        alpha=0.6, 
        s=10
    )
    plt.xlabel(dim1)
    plt.ylabel(dim2)
    plt.title('VAE Latent Space Colored by Reconstruction Error')
    plt.colorbar(scatter, label='Reconstruction Error')
    plt.grid(True)
    plt.savefig('sklearn_vae_latent_space_recon.png', dpi=300)
    
    # Plot the latent space colored by KL divergence
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        plot_df[dim1], 
        plot_df[dim2], 
        c=plot_df['kl_divergence'],
        cmap='plasma',
        alpha=0.6, 
        s=10
    )
    plt.xlabel(dim1)
    plt.ylabel(dim2)
    plt.title('VAE Latent Space Colored by KL Divergence')
    plt.colorbar(scatter, label='KL Divergence')
    plt.grid(True)
    plt.savefig('sklearn_vae_latent_space_kl.png', dpi=300)


def main():
    """
    Main function to run the sklearn-based VAE analysis on the BETH dataset.
    """
    # File path to the BETH csv files
    data_path = os.getcwd() + os.sep + "data"
    csv_files = glob.glob(f"data{os.sep}*data.csv")

    # Load and preprocess data
    df_scaled, feature_names = load_and_preprocess_beth_data(csv_files, data_path)
    # chosen_idx = np.random.choice(5000, replace=False, size=50)
    # df_scaled = df_scaled.iloc[chosen_idx]    
    
    # Configure VAE parameters
    encoding_dim = 10     # Dimension of the latent space
    max_iter = 30         # Maximum training iterations
    kl_weight = 0.05      # Weight for KL divergence loss
    
    # Train VAE
    vae, results_df, latent_df, training_history = train_vae(
        df_scaled, 
        feature_names, 
        encoding_dim=encoding_dim,
        max_iter=max_iter,
        kl_weight=kl_weight
    )
    
    # Analyze results
    analyze_vae_results(results_df, latent_df, encoding_dim, training_history, feature_names)
    
    # Visualize latent space
    visualize_latent_space(latent_df, results_df, encoding_dim)
    
    # Save results
    results_df.to_csv('beth_sklearn_vae_anomalies.csv', index=False)
    latent_df.to_csv('beth_sklearn_vae_latent.csv', index=False)
    
    # Save model
    joblib.dump(vae, 'beth_sklearn_vae_model.joblib')
    
    print("\nSklearn-based VAE analysis complete!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    start = time.time()
    main()
    end = time.time()
    print(f"Total time took: {end - start} seconds")