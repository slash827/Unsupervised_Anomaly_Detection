import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import os, glob
import warnings
import time
warnings.filterwarnings('ignore')

from utils import load_and_preprocess_beth_data

# Set plot style and figure size for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


def pca_anomaly_detection(df, feature_names, n_components=3):
    """
    Perform PCA-based anomaly detection by using fewer components than features.
    
    This forces the PCA to lose information, resulting in meaningful reconstruction errors.
    """
    # Separate features and labels
    features = df[feature_names].values
    labels = df[['sus', 'evil']]
    
    print(f"\nPerforming PCA with {n_components} components (out of {len(feature_names)} features)...")
    
    # Fit PCA with fewer components than features
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(features)
    
    # Reconstruct the data (will be imperfect due to reduced dimensionality)
    reconstructed_data = pca.inverse_transform(transformed_data)
    
    # Calculate reconstruction error for each sample
    reconstruction_error = np.sum(np.square(features - reconstructed_data), axis=1)
    
    # Print explained variance
    print("\nExplained variance by component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    
    print(f"\nCumulative explained variance: {np.sum(pca.explained_variance_ratio_):.4f} ({np.sum(pca.explained_variance_ratio_)*100:.2f}%)")
    
    # Create results dataframe
    results_df = df.copy()
    results_df['reconstruction_error'] = reconstruction_error
    
    return pca, results_df, transformed_data


def analyze_anomalies(results_df, pca, feature_names):
    """
    Analyze and visualize anomalies based on reconstruction error.
    """
    # Basic statistics on reconstruction error
    print("\nReconstruction error statistics:")
    print(f"Min: {results_df['reconstruction_error'].min():.6f}")
    print(f"Max: {results_df['reconstruction_error'].max():.6f}")
    print(f"Mean: {results_df['reconstruction_error'].mean():.6f}")
    print(f"Median: {results_df['reconstruction_error'].median():.6f}")
    
    # Identify top anomalies
    top_anomalies = results_df.sort_values('reconstruction_error', ascending=False).head(20)
    
    print("\nTop 20 anomalies based on reconstruction error:")
    print(top_anomalies[['sus', 'evil', 'reconstruction_error']])
    
    # Calculate average reconstruction error by class
    benign = results_df[(results_df['evil'] == 0) & (results_df['sus'] == 0)]
    suspicious = results_df[(results_df['evil'] == 0) & (results_df['sus'] == 1)]
    malicious = results_df[results_df['evil'] == 1]
    
    print("\nAverage reconstruction error by class:")
    print(f"Benign: {benign['reconstruction_error'].mean():.6f}")
    print(f"Suspicious: {suspicious['reconstruction_error'].mean():.6f}")
    print(f"Malicious: {malicious['reconstruction_error'].mean():.6f}")
    
    # Plot reconstruction error distribution
    plt.figure(figsize=(14, 10))
    
    # Determine optimal bin settings for each class
    max_error = results_df['reconstruction_error'].max()
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
    plt.title('PCA Reconstruction Error Distribution by Class')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_reconstruction_error_dist.png', dpi=300)
    # plt.show()
    
    # Plot ROC-like curve: TPR vs percentile threshold
    plt.figure(figsize=(10, 8))
    
    # Calculate TPR at different percentile thresholds
    percentiles = np.arange(0, 100, 1)
    thresholds = np.percentile(results_df['reconstruction_error'], percentiles)
    
    tpr_malicious = []
    tpr_suspicious = []
    fpr_benign = []
    
    for threshold in thresholds:
        # True positives: malicious samples above threshold
        tp_malicious = (malicious['reconstruction_error'] > threshold).sum() / len(malicious) if len(malicious) > 0 else 0
        tpr_malicious.append(tp_malicious)
        
        # True positives: suspicious samples above threshold
        tp_suspicious = (suspicious['reconstruction_error'] > threshold).sum() / len(suspicious) if len(suspicious) > 0 else 0
        tpr_suspicious.append(tp_suspicious)
        
        # False positives: benign samples above threshold (false alarm rate)
        fp_benign = (benign['reconstruction_error'] > threshold).sum() / len(benign)
        fpr_benign.append(fp_benign)
    
    plt.plot(percentiles, tpr_malicious, 'r-', label='Malicious Detection Rate')
    plt.plot(percentiles, tpr_suspicious, 'orange', label='Suspicious Detection Rate')
    plt.plot(percentiles, fpr_benign, 'g-', label='False Alarm Rate (Benign)')
    
    plt.xlabel('Percentile Threshold')
    plt.ylabel('Rate')
    plt.title('Detection Rates at Different Percentile Thresholds')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_detection_rates.png', dpi=300)
    # plt.show()
    
    # Plot reconstruction error vs original features
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    categorical_features = ['isSystemProcess', 'isSystemParentProcess', 'isSystemUser', 
                           'isDefaultMountNamespace', 'returnValueCat']
    
    # Sample data if it's too large
    if len(results_df) > 10000:
        sample_indices = np.random.choice(len(results_df), 10000, replace=False)
        plot_df = results_df.iloc[sample_indices]
    else:
        plot_df = results_df
    
    # Create colors based on class
    colors = []
    for s, e in zip(plot_df['sus'], plot_df['evil']):
        if e == 1:  # Malicious
            colors.append('red')
        elif s == 1:  # Suspicious
            colors.append('orange')
        else:  # Benign
            colors.append('green')
    
    # Plot categorical features
    for i, feature in enumerate(categorical_features[:5]):
        axes[i].scatter(plot_df[feature], plot_df['reconstruction_error'], 
                       alpha=0.5, c=colors, s=10)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Reconstruction Error')
        axes[i].set_title(f'{feature} vs Reconstruction Error')
        axes[i].grid(True)
    
    # Plot eventId as a special case (numeric with many values)
    # Use jittering to avoid overplotting
    jitter = np.random.normal(0, 0.1, size=len(plot_df))
    axes[5].scatter(plot_df['eventId'] + jitter, plot_df['reconstruction_error'], 
                   alpha=0.5, c=colors, s=10)
    axes[5].set_xlabel('eventId (with jitter)')
    axes[5].set_ylabel('Reconstruction Error')
    axes[5].set_title('eventId vs Reconstruction Error')
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.savefig('pca_features_vs_error.png', dpi=300)
    # plt.show()
    

def interpret_pca_components(pca, feature_names):
    """
    Interpret what each principal component represents.
    """
    print("\n=== Interpretation of Principal Components ===")
    
    # Get the components (loadings)
    components = pca.components_
    
    # For each component, show the top features
    for i, component in enumerate(components):
        # Sort by absolute value
        sorted_indices = np.argsort(np.abs(component))[::-1]
        
        print(f"\nPrincipal Component {i+1} (explains {pca.explained_variance_ratio_[i]*100:.2f}% of variance):")
        print("This component captures a pattern where:")
        
        # Show top 5 features or all if less than 5
        for j in range(min(5, len(feature_names))):
            idx = sorted_indices[j]
            if component[idx] > 0:
                print(f"  + Higher values of '{feature_names[idx]}' (loading: {component[idx]:.4f})")
            else:
                print(f"  - Lower values of '{feature_names[idx]}' (loading: {component[idx]:.4f})")
        
        # Cybersecurity interpretation
        print("\nCybersecurity interpretation:")
        if i == 0:
            if 'isSystemUser' in [feature_names[idx] for idx in sorted_indices[:3]]:
                print("  This component likely differentiates between system processes and user-initiated activities.")
                print("  It may be useful for detecting privilege escalation attempts or unauthorized user actions.")
            elif 'eventId' in [feature_names[idx] for idx in sorted_indices[:3]]:
                print("  This component captures variations in event types being executed on the system.")
                print("  It helps distinguish between normal system operations and potentially malicious command patterns.")
        elif i == 1:
            if 'isDefaultMountNamespace' in [feature_names[idx] for idx in sorted_indices[:3]]:
                print("  This component appears to be sensitive to filesystem access patterns.")
                print("  It may help identify attempts to access or modify sensitive system directories.")
            else:
                print("  This component represents a secondary pattern of system behavior variation.")
                print("  It captures information orthogonal to the first component.")
        elif i == 2:
            if 'isSystemProcess' in [feature_names[idx] for idx in sorted_indices[:3]] and 'isSystemParentProcess' in [feature_names[idx] for idx in sorted_indices[:3]]:
                print("  This component tracks relationships between parent and child processes.")
                print("  It may help identify unusual process trees that could indicate malicious behavior.")
            else:
                print("  This component captures more subtle variations in system activity.")
                print("  It may represent specific command patterns or resource usage behaviors.")


def main():
    """
    Main function to run the PCA analysis on the BETH dataset.
    """
    # File path to the BETH csv files
    data_path = os.getcwd() + os.sep + "data"
    csv_files = glob.glob(f"data{os.sep}*data.csv")

    # Load and preprocess data
    df_scaled, feature_names = load_and_preprocess_beth_data(csv_files, data_path)
    
    # Perform PCA-based anomaly detection with 3 components (key change from original code)
    n_components = 3  # Using fewer components than features forces information loss
    pca, results_df, transformed_data = pca_anomaly_detection(df_scaled, feature_names, n_components)
    
    # Analyze anomalies
    analyze_anomalies(results_df, pca, feature_names)
    
    # Interpret PCA components
    interpret_pca_components(pca, feature_names)
    
    # Save results
    results_df.to_csv('beth_anomalies.csv', index=False)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print (f"total time took is: {end - start}")