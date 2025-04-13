import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import os, glob
import warnings
import time
from collections import defaultdict
import seaborn as sns

from utils import load_and_preprocess_beth_data

warnings.filterwarnings('ignore')

# Set plot style and figure size for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

def train_decision_tree(df, feature_names, max_depth=5, class_weight=None, sample_size=None, target_col='evil'):
    """
    Train a decision tree classifier on the dataset.
    
    Args:
        df: Preprocessed DataFrame
        feature_names: List of feature names to use for training
        max_depth: Maximum depth of the decision tree
        class_weight: Weights associated with classes
        sample_size: Number of samples to use for training (for large datasets)
        target_col: Target variable column name (default: 'evil')
        
    Returns:
        Trained decision tree model and evaluation metrics
    """
    # If dataset is too large, sample it
    if sample_size is not None and len(df) > sample_size:
        print(f"\nSampling {sample_size} records from the full dataset of {len(df)} records...")
        # Stratified sampling to maintain class distribution
        sampled_df = pd.DataFrame()
        for class_val in df[target_col].unique():
            class_df = df[df[target_col] == class_val]
            class_sample_size = int(sample_size * len(class_df) / len(df))
            sampled_class_df = class_df.sample(n=min(class_sample_size, len(class_df)))
            sampled_df = pd.concat([sampled_df, sampled_class_df])
        df = sampled_df.reset_index(drop=True)
    
    # Split the data into features and target
    # Make sure 'sus' is part of features if not the target
    if target_col == 'evil' and 'sus' not in feature_names and 'sus' in df.columns:
        features_to_use = feature_names + ['sus']
    else:
        features_to_use = [f for f in feature_names if f != target_col]
        
    X = df[features_to_use]
    y = df[target_col]
    
    # Split into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Train the decision tree
    print(f"\nTraining a decision tree with max_depth={max_depth}...")
    dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=42
    )
    dt_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt_classifier.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Determine target names based on what we're predicting
    if target_col == 'evil':
        target_names = ['Not Malicious', 'Malicious']
    elif target_col == 'sus':
        target_names = ['Not Suspicious', 'Suspicious']
    else:
        # Default case if we use another target
        target_names = [f'Class {i}' for i in sorted(set(y_test))]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Create and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm, 
        index=[f'Actual {name}' for name in target_names],
        columns=[f'Predicted {name}' for name in target_names]
    )
    
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Calculate feature importance
    # Make sure feature_names and feature_importances_ have the same length
    # This is needed because when we add 'sus' to feature_names but it might not be 
    # in the features used for training
    if len(feature_names) != len(dt_classifier.feature_importances_):
        # Get the actual features used for training
        actual_features = X_train.columns.tolist()
        feature_importance = pd.DataFrame({
            'Feature': actual_features,
            'Importance': dt_classifier.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': dt_classifier.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return dt_classifier, X_train, X_test, y_train, y_test, feature_importance


def visualize_tree(dt_classifier, feature_names, output_file='decision_tree.png', target_col='evil', X_train=None):
    """
    Visualize the decision tree with feature names and class labels.
    
    Args:
        dt_classifier: Trained decision tree classifier
        feature_names: List of feature names
        output_file: Path to save the visualization
        target_col: Target variable column name
        X_train: Training data used, to get accurate feature names if needed
    """
    plt.figure(figsize=(20, 10))
    
    # Use feature names from X_train if provided (more reliable)
    if X_train is not None:
        feature_names_to_use = X_train.columns.tolist()
    else:
        # If feature_names length doesn't match, we can't use it
        if len(feature_names) != dt_classifier.tree_.n_features:
            print(f"Warning: feature_names list length ({len(feature_names)}) doesn't match tree features ({dt_classifier.tree_.n_features})")
            print("Using generic feature names instead.")
            feature_names_to_use = [f'feature_{i}' for i in range(dt_classifier.tree_.n_features)]
        else:
            feature_names_to_use = feature_names
    
    # Determine class names based on target
    if target_col == 'evil':
        class_names = ['Not Malicious', 'Malicious']
    elif target_col == 'sus':
        class_names = ['Not Suspicious', 'Suspicious']
    else:
        # Default case for other targets
        class_names = [f'Class {i}' for i in range(dt_classifier.n_classes_)]
    
    # Create a more detailed plot with feature names
    plot_tree(
        dt_classifier,
        feature_names=feature_names_to_use,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9
    )
    
    plt.title('Decision Tree for BETH Dataset')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nDecision tree visualization saved to {output_file}")
    
    # If tree is small enough, also print text representation
    if dt_classifier.get_depth() <= 7:
        tree_text = export_text(
            dt_classifier,
            feature_names=feature_names,
            show_weights=True
        )
        print("\nText Representation of the Decision Tree:")
        print(tree_text)
    else:
        print("\nTree is too deep to display as text. Please refer to the visualization.")


def print_decision_paths(dt_classifier, feature_names, target_col='evil'):
    """
    Print the decision paths from the root to each leaf in the tree.
    
    Args:
        dt_classifier: Trained decision tree classifier
        feature_names: List of feature names
        target_col: Target variable column name
    """
    # Check if feature_names matches the tree's expected feature count
    if len(feature_names) != dt_classifier.tree_.n_features:
        print(f"Warning: feature_names list length ({len(feature_names)}) doesn't match tree features ({dt_classifier.tree_.n_features})")
        print("Using generic feature names instead.")
        feature_names = [f'feature_{i}' for i in range(dt_classifier.tree_.n_features)]
    n_nodes = dt_classifier.tree_.node_count
    children_left = dt_classifier.tree_.children_left
    children_right = dt_classifier.tree_.children_right
    feature = dt_classifier.tree_.feature
    threshold = dt_classifier.tree_.threshold
    value = dt_classifier.tree_.value
    
    # Function to get the path to a node
    def get_path_to_node(node_id):
        path = []
        node = node_id
        while node != 0:  # Continue until we reach the root
            # Find the parent
            is_left_child = False
            parent = -1
            for i in range(n_nodes):
                if children_left[i] == node:
                    parent = i
                    is_left_child = True
                    break
                elif children_right[i] == node:
                    parent = i
                    is_left_child = False
                    break
            
            if parent == -1:
                break  # Should not happen
            
            # Add decision to path
            if is_left_child:
                path.append(f"{feature_names[feature[parent]]} <= {threshold[parent]:.2f}")
            else:
                path.append(f"{feature_names[feature[parent]]} > {threshold[parent]:.2f}")
            
            node = parent
        
        return list(reversed(path))
    
    # Identify leaf nodes
    is_leaf = np.zeros(shape=n_nodes, dtype=bool)
    leaf_nodes = []
    
    for i in range(n_nodes):
        if children_left[i] == children_right[i]:  # If they're equal, it's a leaf
            is_leaf[i] = True
            leaf_nodes.append(i)
    
    print("\n=== Decision Paths to Leaf Nodes ===")
    
    # Determine class names based on target
    if target_col == 'evil':
        class_names = ['Not Malicious', 'Malicious']
    elif target_col == 'sus':
        class_names = ['Not Suspicious', 'Suspicious']
    else:
        # Default case for other targets
        class_names = [f'Class {i}' for i in range(dt_classifier.n_classes_)]
    
    # Get class distribution at each leaf
    for i, leaf in enumerate(leaf_nodes):
        node_value = value[leaf][0]
        total_samples = sum(node_value)
        
        # Skip leaves with no samples
        if total_samples == 0:
            continue
        
        # Determine the majority class
        majority_class = np.argmax(node_value)
        class_name = class_names[majority_class]
        
        # Format class distribution
        class_dist = ', '.join([f'{name}: {count/total_samples*100:.1f}%' 
                               for name, count in zip(class_names, node_value)])
        
        # Get the path to this leaf
        path = get_path_to_node(leaf)
        
        # Print the path and class distribution
        print(f"\nPath {i+1} (Leaf Node {leaf}) - Majority: {class_name}, Samples: {int(total_samples)}:")
        if path:
            for j, decision in enumerate(path):
                print(f"  {'├─' if j < len(path)-1 else '└─'} {decision}")
        else:
            print("  Root node (no decisions)")
        
        print(f"  Class Distribution: {class_dist}")


def analyze_feature_thresholds(dt_classifier, feature_names):
    """
    Analyze the thresholds used for each feature in the decision tree.
    
    Args:
        dt_classifier: Trained decision tree classifier
        feature_names: List of feature names
    """
    # Check if feature_names matches the tree's expected feature count
    if len(feature_names) != dt_classifier.tree_.n_features:
        print(f"Warning: feature_names list length ({len(feature_names)}) doesn't match tree features ({dt_classifier.tree_.n_features})")
        print("Using generic feature names instead.")
        feature_names = [f'feature_{i}' for i in range(dt_classifier.tree_.n_features)]
    n_nodes = dt_classifier.tree_.node_count
    feature = dt_classifier.tree_.feature
    threshold = dt_classifier.tree_.threshold
    
    # Collect thresholds for each feature
    feature_thresholds = defaultdict(list)
    
    for i in range(n_nodes):
        # Skip leaf nodes
        if feature[i] != -2:  # -2 indicates a leaf node
            feature_thresholds[feature_names[feature[i]]].append(threshold[i])
    
    print("\n=== Feature Threshold Analysis ===")
    
    for feature_name, thresholds in feature_thresholds.items():
        if thresholds:
            print(f"\nFeature: {feature_name}")
            print(f"  Used in {len(thresholds)} decision nodes")
            print(f"  Threshold values: {sorted(set([round(t, 2) for t in thresholds]))}")
            
            # For binary features, check if they are used with their natural threshold (0.5)
            if len(set([round(t, 2) for t in thresholds])) == 1 and round(thresholds[0], 2) == 0.5:
                print("  This appears to be used as a binary split (threshold ≈ 0.5)")
    
    # Count how many times each feature is used at different tree depths
    feature_depth_count = defaultdict(lambda: defaultdict(int))


def visualize_confusion_matrix(y_test, y_pred, output_file='confusion_matrix.png', target_col='evil'):
    """
    Create a more visually appealing confusion matrix plot.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        output_file: Path to save the visualization
        target_col: Target variable column name
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Determine class names based on target
    if target_col == 'evil':
        class_names = ['Not Malicious', 'Malicious']
    elif target_col == 'sus':
        class_names = ['Not Suspicious', 'Suspicious']
    else:
        # Default case for other targets
        class_names = [f'Class {i}' for i in range(len(np.unique(y_test)))]
    
    # Plot the confusion matrix with percentages
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nConfusion matrix visualization saved to {output_file}")


def visualize_feature_importance(feature_importance, output_file='feature_importance.png'):
    """
    Create a bar chart of feature importances.
    
    Args:
        feature_importance: DataFrame with feature importance values
        output_file: Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Sort by importance
    sorted_df = feature_importance.sort_values('Importance')
    
    # Create horizontal bar chart
    plt.barh(sorted_df['Feature'], sorted_df['Importance'], color='skyblue')
    
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Decision Tree')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nFeature importance visualization saved to {output_file}")


def main():
    """
    Main function to run the decision tree analysis on the BETH dataset.
    """
    # File path to the BETH csv files
    data_path = os.getcwd() + os.sep + "data"
    csv_files = glob.glob(f"data{os.sep}*data.csv")
    
    # Load and preprocess data
    df_processed, feature_names = load_and_preprocess_beth_data(csv_files, data_path)
    
    # Define target column - using 'evil' as the target 
    target_col = 'evil'
    
    # Handle class imbalance by setting class weights
    # If dataset is very imbalanced, use 'balanced' option
    class_counts = df_processed[target_col].value_counts()
    if max(class_counts) / min(class_counts) > 10:
        print(f"\nDetected significant class imbalance in {target_col}. Using 'balanced' class weights.")
        class_weight = 'balanced'
    else:
        class_weight = None
    
    # Make sure 'sus' is included in features when predicting 'evil'
    if 'sus' not in feature_names and target_col == 'evil':
        feature_names.append('sus')
        print("\nAdded 'sus' to features since we're predicting 'evil'")
    
    # Train the decision tree (sample if dataset is very large)
    sample_size = 100000 if len(df_processed) > 200000 else None
    dt_classifier, X_train, X_test, y_train, y_test, feature_importance = train_decision_tree(
        df_processed, 
        feature_names,
        max_depth=5,  # Adjust this value based on complexity vs. interpretability trade-off
        class_weight=class_weight,
        sample_size=sample_size,
        target_col=target_col
    )
    
    # Visualize the tree - pass X_train to ensure correct feature names
    visualize_tree(dt_classifier, feature_names, target_col=target_col, X_train=X_train)
    
    # Print decision paths for interpretation
    print_decision_paths(dt_classifier, X_train.columns.tolist(), target_col=target_col)
    
    # Analyze feature thresholds
    analyze_feature_thresholds(dt_classifier, X_train.columns.tolist())
    
    # Visualize confusion matrix
    visualize_confusion_matrix(y_test, dt_classifier.predict(X_test), target_col=target_col)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")