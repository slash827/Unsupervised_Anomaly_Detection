import os, glob
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from utils import load_and_preprocess_beth_data

# Set plot style and figure size for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

def train_isolation_forest(X_train, n_estimators=100, contamination='auto', random_state=42):
    """Train an Isolation Forest model on benign data."""
    print(f"\nTraining Isolation Forest with {n_estimators} estimators...")
    start_time = time.time()
    
    # Train the model
    model = IsolationForest(
        n_estimators=n_estimators, 
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model

def train_ocsvm(X_train, kernel='rbf', nu=0.01, gamma='auto'):
    """Train a One-Class SVM model on benign data."""
    print(f"\nTraining One-Class SVM with {kernel} kernel and nu={nu}...")
    start_time = time.time()
    
    # Train the model
    model = OneClassSVM(
        kernel=kernel,
        nu=nu,
        gamma=gamma
    )
    model.fit(X_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model

def evaluate_model(model, features, labels, model_name, results_df):
    """Evaluate the model and produce visualizations."""
    print(f"\nEvaluating {model_name}...")
    
    # Get anomaly scores
    if model_name == "Isolation Forest":
        # For Isolation Forest, more negative = more anomalous
        anomaly_scores = -model.score_samples(features)
    else:  # One-Class SVM
        # For OCSVM, more positive = more anomalous
        anomaly_scores = -model.score_samples(features)
    
    # Add scores to results_df
    results_df[f'{model_name.lower().replace(" ", "_")}_score'] = anomaly_scores
    
    # Create class-specific dataframes
    benign = results_df[(results_df['evil'] == 0) & (results_df['sus'] == 0)]
    suspicious = results_df[(results_df['evil'] == 0) & (results_df['sus'] == 1)]
    malicious = results_df[results_df['evil'] == 1]
    
    # Calculate basic statistics
    print(f"\nAnomaly score statistics for {model_name}:")
    print(f"Overall - Min: {anomaly_scores.min():.6f}, Max: {anomaly_scores.max():.6f}, Mean: {anomaly_scores.mean():.6f}")
    
    model_name_rep = model_name.lower().replace(' ', '_')
    benign_mean = benign[model_name_rep + '_score'].mean()
    print(f"Benign (n={len(benign)}) - Mean: {benign_mean:.6f}")
    if len(suspicious) > 0:
        suspicious_mean = suspicious[model_name_rep + '_score'].mean()
        print(f"Suspicious (n={len(suspicious)}) - Mean: {suspicious_mean:.6f}")
    
    malicious_mean = malicious[model_name_rep + '_score'].mean()
    print(f"Malicious (n={len(malicious)}) - Mean: {malicious_mean:.6f}")
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Calculate precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(labels, anomaly_scores)
    avg_precision = average_precision_score(labels, anomaly_scores)
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Find optimal threshold using F1 score
    f1_scores = []
    for threshold in thresholds:
        predictions = (anomaly_scores >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Optimal threshold: {best_threshold:.6f} (F1 = {best_f1:.4f})")
    
    # Calculate performance at optimal threshold
    predictions = (anomaly_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Performance at optimal threshold:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision_at_threshold:.4f}")
    print(f"Recall: {recall_at_threshold:.4f}")
    print(f"True Positives: {tp}, False Positives: {fp}")
    print(f"True Negatives: {tn}, False Negatives: {fn}")
    
    # Add predictions to results_df
    results_df[f'{model_name.lower().replace(" ", "_")}_predictions'] = predictions
    
    # Create visualizations
    
    # 1. Score distributions
    plt.figure(figsize=(12, 6))
    bins = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores), 100)
    
    plt.hist(benign[f'{model_name.lower().replace(" ", "_")}_score'], bins=bins, alpha=0.5, 
             label=f'Benign (n={len(benign)})', color='green', density=True)
    
    if len(suspicious) > 0:
        plt.hist(suspicious[f'{model_name.lower().replace(" ", "_")}_score'], bins=bins, alpha=0.5, 
                 label=f'Suspicious (n={len(suspicious)})', color='orange', density=True)
    
    plt.hist(malicious[f'{model_name.lower().replace(" ", "_")}_score'], bins=bins, alpha=0.5, 
             label=f'Malicious (n={len(malicious)})', color='red', density=True)
    
    plt.axvline(best_threshold, color='black', linestyle='--', 
                label=f'Threshold (F1={best_f1:.3f})')
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(f'{model_name} Anomaly Score Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_score_distribution.png', dpi=300)
    
    # 2. ROC and PR curves
    plt.figure(figsize=(14, 6))
    
    # ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # PR curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_roc_pr_curves.png', dpi=300)
    
    # 3. Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = np.array([[tn, fp], [fn, tp]])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]}', 
                     ha="center", va="center", 
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0, 1], ['Benign', 'Malicious'])
    plt.yticks([0, 1], ['Benign', 'Malicious'])
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=300)
    
    return results_df, best_threshold, roc_auc, avg_precision, best_f1

def main():
    """Main function to run anomaly detection on the BETH dataset."""
    # File path to the BETH csv files
    data_path = os.getcwd() + os.sep + "data"
    csv_files = glob.glob(f"data{os.sep}*data.csv")

    # Load and preprocess data
    df_scaled, feature_names = load_and_preprocess_beth_data(csv_files, data_path)
    
    # Separate features and labels
    features = df_scaled[feature_names].values
    evil_labels = df_scaled['evil'].values
    
    # Create results dataframe
    results_df = df_scaled.copy()
    
    # Extract benign samples for training
    benign_mask = (df_scaled['evil'] == 0) & (df_scaled['sus'] == 0)
    X_train = features[benign_mask]
    
    print(f"Training on {len(X_train)} benign samples...")
    print(f"Testing on {len(features)} total samples ({np.sum(evil_labels)} malicious)")
    
    # Train and evaluate Isolation Forest
    iso_forest = train_isolation_forest(X_train)
    results_df, iso_threshold, iso_auc, iso_ap, iso_f1 = evaluate_model(
        iso_forest, features, evil_labels, "Isolation Forest", results_df
    )
    
    # Train and evaluate One-Class SVM
    # Using a sample if the dataset is too large
    if len(X_train) > 100000:
        sample_indices = np.random.choice(len(X_train), 100000, replace=False)
        X_train_sample = X_train[sample_indices]
        print(f"Using a sample of {len(X_train_sample)} benign samples for OCSVM training...")
        ocsvm = train_ocsvm(X_train_sample)
    else:
        ocsvm = train_ocsvm(X_train)
    
    results_df, ocsvm_threshold, ocsvm_auc, ocsvm_ap, ocsvm_f1 = evaluate_model(
        ocsvm, features, evil_labels, "One-Class SVM", results_df
    )
    
    # Compare the two models
    print("\n===== Model Comparison =====")
    print(f"Isolation Forest - AUC: {iso_auc:.4f}, AP: {iso_ap:.4f}, F1: {iso_f1:.4f}")
    print(f"One-Class SVM - AUC: {ocsvm_auc:.4f}, AP: {ocsvm_ap:.4f}, F1: {ocsvm_f1:.4f}")
    
    # Create a combined ROC curve
    plt.figure(figsize=(10, 8))
    
    # Get Isolation Forest data
    fpr_iso, tpr_iso, _ = roc_curve(evil_labels, results_df['isolation_forest_score'])
    
    # Get One-Class SVM data
    fpr_ocsvm, tpr_ocsvm, _ = roc_curve(evil_labels, results_df['one-class_svm_score'])
    
    # Plot both curves
    plt.plot(fpr_iso, tpr_iso, color='blue', lw=2, 
             label=f'Isolation Forest (AUC = {iso_auc:.3f})')
    plt.plot(fpr_ocsvm, tpr_ocsvm, color='red', lw=2, 
             label=f'One-Class SVM (AUC = {ocsvm_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('model_comparison_roc.png', dpi=300)
    
    # Save results
    results_df.to_csv('beth_traditional_anomaly_detection.csv', index=False)
    
    print("\nTraditional anomaly detection analysis complete!")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time took: {end - start} seconds")