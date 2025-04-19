import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import shap

import os, glob
import time
from collections import defaultdict
import traceback
import warnings
warnings.filterwarnings('ignore')

from utils import load_and_preprocess_beth_data

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


def train_tree_model(df, feature_names, model_type="decision_tree", max_depth=5, 
                    class_weight=None, sample_size=None, target_col='evil',
                    n_estimators=100, random_state=42):
    """
    Train either a decision tree or random forest classifier on the dataset.
    
    Args:
        df: Preprocessed DataFrame
        feature_names: List of feature names to use for training
        model_type: Type of model to train ("decision_tree" or "random_forest")
        max_depth: Maximum depth of the trees
        class_weight: Weights associated with classes
        sample_size: Number of samples to use for training (for large datasets)
        target_col: Target variable column name (default: 'evil')
        n_estimators: Number of trees in the random forest (if applicable)
        random_state: Random state for reproducibility
        
    Returns:
        Trained model and evaluation metrics
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Initialize and train the model based on the specified type
    print(f"\nTraining a {model_type.replace('_', ' ')} with max_depth={max_depth}...")
    
    if model_type == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose 'decision_tree' or 'random_forest'")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
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
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns.tolist(),
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, X_train, X_test, y_train, y_test, feature_importance


def analyze_random_forest_properties(rf_model, X_train, feature_names):
    """
    Analyze properties specific to random forests.
    
    Args:
        rf_model: Trained RandomForestClassifier
        X_train: Training data
        feature_names: List of feature names
    """
    print("\n=== Random Forest Properties ===")
    
    # Get feature importance via different methods
    # 1. Mean Decrease in Impurity (default)
    mdi_importance = pd.DataFrame({
        'Feature': X_train.columns.tolist(),
        'MDI_Importance': rf_model.feature_importances_
    }).sort_values('MDI_Importance', ascending=False)
    
    print("\nMean Decrease in Impurity (MDI) Feature Importance:")
    print(mdi_importance)
    
    # Analyze tree depth and feature usage across the forest
    n_trees = len(rf_model.estimators_)
    print(f"\nRandom Forest consists of {n_trees} trees")
    
    # Analyze depth distribution of trees
    depths = [estimator.get_depth() for estimator in rf_model.estimators_]
    print(f"Tree depth: min={min(depths)}, max={max(depths)}, avg={np.mean(depths):.1f}")
    
    # Analyze how many features each tree uses
    n_features_used = []
    feature_use_count = defaultdict(int)
    
    for estimator in rf_model.estimators_:
        used_features = set()
        for i in range(estimator.tree_.node_count):
            # If not a leaf node
            if estimator.tree_.feature[i] != -2:
                feature_idx = estimator.tree_.feature[i]
                feature_name = X_train.columns[feature_idx]
                used_features.add(feature_name)
                feature_use_count[feature_name] += 1
        n_features_used.append(len(used_features))
    
    print(f"Features used per tree: min={min(n_features_used)}, max={max(n_features_used)}, avg={np.mean(n_features_used):.1f}")
    
    # Print feature usage across trees
    feature_usage_pct = {feature: count/n_trees*100 for feature, count in feature_use_count.items()}
    feature_usage_df = pd.DataFrame({
        'Feature': list(feature_usage_pct.keys()),
        'Usage_Pct': list(feature_usage_pct.values())
    }).sort_values('Usage_Pct', ascending=False)
    
    print("\nFeature usage across trees (percentage of trees using each feature):")
    print(feature_usage_df)
    
    # Check for feature correlations in the forest
    # Create a visualization of feature usage correlation
    plt.figure(figsize=(14, 10))
    
    # Create a matrix counting how many times features are used together in trees
    n_features = len(X_train.columns)
    feature_co_occurrence = np.zeros((n_features, n_features))
    
    for estimator in rf_model.estimators_:
        used_features = set()
        for i in range(estimator.tree_.node_count):
            if estimator.tree_.feature[i] != -2:
                used_features.add(estimator.tree_.feature[i])
        
        # Update co-occurrence matrix
        used_feature_list = list(used_features)
        for i in range(len(used_feature_list)):
            for j in range(i, len(used_feature_list)):
                feature_co_occurrence[used_feature_list[i], used_feature_list[j]] += 1
                if i != j:
                    feature_co_occurrence[used_feature_list[j], used_feature_list[i]] += 1
    
    # Normalize by the number of trees
    feature_co_occurrence /= n_trees
    
    # Plot heatmap
    sns.heatmap(
        feature_co_occurrence,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=X_train.columns,
        yticklabels=X_train.columns
    )
    
    plt.title('Feature Co-occurrence in Random Forest Trees')
    plt.tight_layout()
    plt.savefig('feature_co_occurrence.png', dpi=300)
    print("\nFeature co-occurrence heatmap saved to feature_co_occurrence.png")
    
    # Visualize tree depth distribution
    plt.figure(figsize=(12, 6))
    plt.hist(depths, bins=range(min(depths), max(depths) + 2), alpha=0.7, color='skyblue')
    plt.xlabel('Tree Depth')
    plt.ylabel('Count')
    plt.title('Distribution of Tree Depths in Random Forest')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('tree_depth_distribution.png', dpi=300)
    print("\nTree depth distribution saved to tree_depth_distribution.png")
    
    return feature_usage_df, mdi_importance


def shap_analysis(model, X_train, X_test, feature_names, n_samples=100, output_prefix="shap"):
    """
    Perform SHAP analysis to explain model predictions.
    """
    print("\n=== SHAP Analysis for Model Explainability ===")
    
    # Use a sample of the data if it's too large
    if len(X_test) > n_samples:
        print(f"Selecting {n_samples} samples from test data for SHAP analysis...")
        X_sample = X_test.sample(n_samples, random_state=42)
    else:
        X_sample = X_test
    
    # Select explainer type based on model type
    model_type = model.__class__.__name__
    
    try:
        if model_type == "DecisionTreeClassifier":
            print("Using SHAP TreeExplainer for Decision Tree...")
            explainer = shap.TreeExplainer(model)
        elif model_type == "RandomForestClassifier":
            print("Using SHAP TreeExplainer for Random Forest...")
            explainer = shap.TreeExplainer(model)
        else:
            print(f"Model type {model_type} not specifically supported, using default Explainer...")
            explainer = shap.Explainer(model, X_train)
        
        # Calculate SHAP values
        print("Calculating SHAP values (this may take a while for large models)...")
        shap_values = explainer(X_sample)
        
        # SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_summary_plot.png", dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {output_prefix}_summary_plot.png")
        plt.close()
        
        # SHAP bar plot (mean absolute SHAP values)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_importance_plot.png", dpi=300, bbox_inches='tight')
        print(f"SHAP importance plot saved to {output_prefix}_importance_plot.png")
        plt.close()
        
        # Calculate and save mean absolute SHAP values 
        # Fix: Handle multi-dimensional SHAP values properly
        if hasattr(shap_values, 'values'):
            # For newer SHAP versions with Explanation objects
            if len(shap_values.values.shape) > 2:
                # For multi-class classification, take the mean across classes
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            else:
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        else:
            # For older SHAP versions with direct arrays
            if isinstance(shap_values, list):
                # Multi-class case
                mean_abs_shap = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame for SHAP importance
        if len(mean_abs_shap) == len(X_sample.columns):
            shap_importance = pd.DataFrame({
                'Feature': X_sample.columns.tolist(),
                'SHAP_Importance': mean_abs_shap
            }).sort_values('SHAP_Importance', ascending=False)
            
            print("\nFeature Importance based on SHAP values:")
            print(shap_importance)
            
            # Analyze top 3 features in more detail
            top_features = shap_importance.head(3)['Feature'].tolist()
            # Rest of the function remains the same...
        else:
            print(f"Warning: SHAP values shape doesn't match feature count. SHAP shape: {mean_abs_shap.shape}, features: {len(X_sample.columns)}")
            shap_importance = None
            
        return shap_importance
        
    except Exception as e:
        print(f"\nError during SHAP analysis: {str(e)}")
        traceback.print_exc()  # Print full traceback
        print("Continuing with the rest of the analysis...")
        return None


def permutation_importance_analysis(model, X_test, y_test, n_repeats=10, output_prefix="perm_imp"):
    """
    Calculate permutation importance as an alternative to SHAP.
    """
    print("\n=== Permutation Importance Analysis ===")
    
    try:
        from sklearn.inspection import permutation_importance
        import traceback
        
        # Calculate permutation importance
        print("Calculating permutation importance...")
        perm_importance = permutation_importance(
            model, X_test, y_test, 
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        # Create DataFrame with results
        perm_importance_df = pd.DataFrame({
            'Feature': X_test.columns.tolist(),
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)
        
        print("\nPermutation Feature Importance:")
        print(perm_importance_df)
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        plt.barh(perm_importance_df['Feature'], perm_importance_df['Importance'], 
                xerr=perm_importance_df['Std'], color='skyblue')
        plt.xlabel('Mean Importance')
        plt.title('Permutation Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_importance.png", dpi=300)
        print(f"\nPermutation importance plot saved to {output_prefix}_importance.png")
        
        # Create feature dependence plots (simplified version of SHAP dependence plots)
        top_features = perm_importance_df.head(3)['Feature'].tolist()
        for feature in top_features:
            plt.figure(figsize=(12, 6))
            
            # Plot the distribution of the feature by class
            if len(np.unique(y_test)) <= 2:
                # Binary classification
                not_malicious = X_test[y_test == 0][feature]
                malicious = X_test[y_test == 1][feature]
                
                plt.hist(not_malicious, alpha=0.5, label='Not Malicious', bins=20)
                plt.hist(malicious, alpha=0.5, label='Malicious', bins=20)
                plt.xlabel(feature)
                plt.ylabel('Count')
                plt.title(f'Distribution of {feature} by Class')
                plt.legend()
            else:
                # Multi-class (would need adaptation for your specific classes)
                for cls in np.unique(y_test):
                    plt.hist(X_test[y_test == cls][feature], alpha=0.5, 
                             label=f'Class {cls}', bins=20)
                plt.xlabel(feature)
                plt.ylabel('Count')
                plt.title(f'Distribution of {feature} by Class')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_dist_{feature}.png", dpi=300)
            plt.close()
            
            print(f"Distribution plot for {feature} saved to {output_prefix}_dist_{feature}.png")
        
        return perm_importance_df
        
    except Exception as e:
        print(f"\nError during permutation importance analysis: {str(e)}")
        traceback.print_exc()
        return None


def compare_models(dt_results, rf_results, dt_perm_importance=None, rf_perm_importance=None):
    """
    Compare decision tree and random forest models, including permutation importance if available.
    
    Args:
        dt_results: Dict containing decision tree results
        rf_results: Dict containing random forest results
        dt_perm_importance: Permutation importance for decision tree (optional)
        rf_perm_importance: Permutation importance for random forest (optional)
    """
    print("\n=== Model Comparison: Decision Tree vs Random Forest ===")
    
    # Compare accuracy
    dt_accuracy = accuracy_score(dt_results['y_test'], dt_results['model'].predict(dt_results['X_test']))
    rf_accuracy = accuracy_score(rf_results['y_test'], rf_results['model'].predict(rf_results['X_test']))
    
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Accuracy Difference: {(rf_accuracy - dt_accuracy):.4f}")
    
    # Compare feature importance
    dt_importance = dt_results['feature_importance']
    rf_importance = rf_results['feature_importance']
    
    # Create a combined feature importance dataframe
    combined_importance = pd.merge(
        dt_importance, 
        rf_importance, 
        on='Feature', 
        suffixes=('_DT', '_RF')
    )
    
    # Add permutation importance if available
    if dt_perm_importance is not None:
        combined_importance = pd.merge(
            combined_importance,
            dt_perm_importance[['Feature', 'Importance']],
            on='Feature',
            how='left'
        )
        combined_importance = combined_importance.rename(columns={'Importance': 'Permutation_DT'})
    
    if rf_perm_importance is not None:
        combined_importance = pd.merge(
            combined_importance,
            rf_perm_importance[['Feature', 'Importance']],
            on='Feature',
            how='left'
        )
        combined_importance = combined_importance.rename(columns={'Importance': 'Permutation_RF'})
    
    # Sort by random forest importance (generally more stable)
    combined_importance = combined_importance.sort_values('Importance_RF', ascending=False)
    
    print("\nFeature Importance Comparison:")
    print(combined_importance)
    
    # Visualize the comparison
    plt.figure(figsize=(14, 10))
    
    # Determine number of importance metrics to display
    n_metrics = 2  # Start with DT and RF
    if dt_perm_importance is not None:
        n_metrics += 1
    if rf_perm_importance is not None:
        n_metrics += 1
    
    # Set up the plot
    x = np.arange(len(combined_importance))
    width = 0.2  # Width of bars
    
    # Calculate offsets for bar positions
    offsets = np.linspace(-(n_metrics-1)*width/2, (n_metrics-1)*width/2, n_metrics)
    
    # Plot bars for each importance measure
    plt.barh(x + offsets[0], combined_importance['Importance_DT'], width, 
             label='Decision Tree (MDI)', color='skyblue')
    
    plt.barh(x + offsets[1], combined_importance['Importance_RF'], width, 
             label='Random Forest (MDI)', color='royalblue')
    
    # Add permutation importance if available
    current_offset = 2
    if 'Permutation_DT' in combined_importance.columns:
        plt.barh(x + offsets[current_offset], combined_importance['Permutation_DT'], width, 
                 label='Decision Tree (Permutation)', color='lightgreen')
        current_offset += 1
    
    if 'Permutation_RF' in combined_importance.columns:
        plt.barh(x + offsets[current_offset], combined_importance['Permutation_RF'], width, 
                 label='Random Forest (Permutation)', color='darkgreen')
    
    plt.yticks(x, combined_importance['Feature'])
    plt.xlabel('Importance Score')
    plt.title('Feature Importance Comparison Across Models and Metrics')
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    print("\nModel comparison plot saved to model_comparison.png")
    
    # Calculate correlation matrix between different importance metrics
    correlation_columns = ['Importance_DT', 'Importance_RF']
    if 'Permutation_DT' in combined_importance.columns:
        correlation_columns.append('Permutation_DT')
    if 'Permutation_RF' in combined_importance.columns:
        correlation_columns.append('Permutation_RF')
    
    correlation_df = combined_importance[correlation_columns].corr()
    
    print("\nCorrelation between importance metrics:")
    print(correlation_df)
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_df, annot=True, cmap='Blues', vmin=-1, vmax=1)
    plt.title('Correlation Between Feature Importance Metrics')
    plt.tight_layout()
    plt.savefig('importance_correlation.png', dpi=300)
    print("\nImportance correlation matrix saved to importance_correlation.png")
    
    # Provide a summary of findings
    print("\nSummary of Model Comparison:")
    
    if rf_accuracy > dt_accuracy:
        print(f"- Random Forest outperforms Decision Tree by {(rf_accuracy-dt_accuracy)*100:.2f}% in accuracy")
    elif dt_accuracy > rf_accuracy:
        print(f"- Decision Tree outperforms Random Forest by {(dt_accuracy-rf_accuracy)*100:.2f}% in accuracy")
    else:
        print("- Both models have the same accuracy")
    
    # Compare top 3 features across different methods
    dt_top3 = set(dt_importance.head(3)['Feature'].tolist())
    rf_top3 = set(rf_importance.head(3)['Feature'].tolist())
    common_top3 = dt_top3.intersection(rf_top3)
    
    print(f"- The models share {len(common_top3)} common features in their top 3:")
    for feature in common_top3:
        print(f"  * {feature}")
    
    # Compare MDI and permutation importance if available
    if 'Permutation_DT' in combined_importance.columns and 'Importance_DT' in combined_importance.columns:
        dt_corr = correlation_df.loc['Importance_DT', 'Permutation_DT']
        dt_top_perm = set(combined_importance.sort_values('Permutation_DT', ascending=False).head(3)['Feature'].tolist())
        dt_top_mdi = set(combined_importance.sort_values('Importance_DT', ascending=False).head(3)['Feature'].tolist())
        dt_common = dt_top_perm.intersection(dt_top_mdi)
        
        print(f"- For Decision Tree, MDI and permutation importance have correlation of {dt_corr:.2f}")
        print(f"  They share {len(dt_common)} common features in their top 3")
    
    if 'Permutation_RF' in combined_importance.columns and 'Importance_RF' in combined_importance.columns:
        rf_corr = correlation_df.loc['Importance_RF', 'Permutation_RF']
        rf_top_perm = set(combined_importance.sort_values('Permutation_RF', ascending=False).head(3)['Feature'].tolist())
        rf_top_mdi = set(combined_importance.sort_values('Importance_RF', ascending=False).head(3)['Feature'].tolist())
        rf_common = rf_top_perm.intersection(rf_top_mdi)
        
        print(f"- For Random Forest, MDI and permutation importance have correlation of {rf_corr:.2f}")
        print(f"  They share {len(rf_common)} common features in their top 3")
    
    # Compare cross-model permutation importance if both available
    if 'Permutation_DT' in combined_importance.columns and 'Permutation_RF' in combined_importance.columns:
        perm_corr = correlation_df.loc['Permutation_DT', 'Permutation_RF']
        dt_top_perm = set(combined_importance.sort_values('Permutation_DT', ascending=False).head(3)['Feature'].tolist())
        rf_top_perm = set(combined_importance.sort_values('Permutation_RF', ascending=False).head(3)['Feature'].tolist())
        perm_common = dt_top_perm.intersection(rf_top_perm)
        
        print(f"- Permutation importance between models has correlation of {perm_corr:.2f}")
        print(f"  They share {len(perm_common)} common features in their top 3 permutation features:")
        for feature in perm_common:
            print(f"  * {feature}")
        
        print("\nKey insight: Features with high permutation importance in both models")
        print("are most reliable for detecting malicious activity.")
    
    # Save the comparison table to a CSV file
    combined_importance.to_csv('feature_importance_comparison.csv', index=False)
    print("\nFeature importance comparison saved to feature_importance_comparison.csv")
    
    return combined_importance


def main():
    """
    Main function to run the model training and explainability analysis.
    """
    # File path to the BETH csv files
    data_path = os.getcwd() + os.sep + "data"
    csv_files = glob.glob(f"data{os.sep}*data.csv")
    
    # Load and preprocess data
    df_processed, feature_names = load_and_preprocess_beth_data(csv_files, data_path)
    
    # Define target column - using 'evil' as the target 
    target_col = 'evil'
    
    # Handle class imbalance by setting class weights
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
    
    # Sample size for large datasets
    sample_size = 100000 if len(df_processed) > 200000 else None
    
    # ===== Decision Tree Model =====
    print("\n" + "="*50)
    print("TRAINING DECISION TREE MODEL")
    print("="*50)
    
    dt_model, dt_X_train, dt_X_test, dt_y_train, dt_y_test, dt_feature_importance = train_tree_model(
        df_processed, 
        feature_names,
        model_type="decision_tree",
        max_depth=5,
        class_weight=class_weight,
        sample_size=sample_size,
        target_col=target_col
    )
    
    # Store decision tree results
    dt_results = {
        'model': dt_model,
        'X_train': dt_X_train,
        'X_test': dt_X_test,
        'y_train': dt_y_train,
        'y_test': dt_y_test,
        'feature_importance': dt_feature_importance
    }
    
    # Visualize confusion matrix for decision tree
    visualize_confusion_matrix(
        dt_y_test, 
        dt_model.predict(dt_X_test), 
        output_file='dt_confusion_matrix.png', 
        target_col=target_col
    )
    
    # Visualize feature importance for decision tree
    visualize_feature_importance(dt_feature_importance, output_file='dt_feature_importance.png')
    
    # ===== Random Forest Model =====
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*50)
    
    rf_model, rf_X_train, rf_X_test, rf_y_train, rf_y_test, rf_feature_importance = train_tree_model(
        df_processed, 
        feature_names,
        model_type="random_forest",
        max_depth=10,  # Random forests can handle deeper trees
        n_estimators=100,  # Number of trees in the forest
        class_weight=class_weight,
        sample_size=sample_size,
        target_col=target_col
    )
    
    # Store random forest results
    rf_results = {
        'model': rf_model,
        'X_train': rf_X_train,
        'X_test': rf_X_test,
        'y_train': rf_y_train,
        'y_test': rf_y_test,
        'feature_importance': rf_feature_importance
    }
    
    # Analyze random forest specific properties
    rf_feature_usage, rf_mdi_importance = analyze_random_forest_properties(
        rf_model, 
        rf_X_train, 
        feature_names
    )
    
    # Visualize confusion matrix for random forest
    visualize_confusion_matrix(
        rf_y_test, 
        rf_model.predict(rf_X_test), 
        output_file='rf_confusion_matrix.png', 
        target_col=target_col
    )
    
    # Visualize feature importance for random forest
    visualize_feature_importance(rf_feature_importance, output_file='rf_feature_importance.png')
    
    # ===== Permutation Importance Analysis =====
    print("\n" + "="*50)
    print("PERFORMING PERMUTATION IMPORTANCE ANALYSIS")
    print("="*50)
    
    # For decision tree
    print("\n--- Permutation Importance Analysis for Decision Tree ---")
    dt_perm_importance = permutation_importance_analysis(
        dt_model, 
        dt_X_test, 
        dt_y_test,
        n_repeats=10,
        output_prefix="dt_perm_imp"
    )
    
    # For random forest
    print("\n--- Permutation Importance Analysis for Random Forest ---")
    rf_perm_importance = permutation_importance_analysis(
        rf_model, 
        rf_X_test, 
        rf_y_test,
        n_repeats=10,
        output_prefix="rf_perm_imp"
    )
    
    # ===== Model Comparison =====
    print("\n" + "="*50)
    print("COMPARING DECISION TREE AND RANDOM FOREST MODELS")
    print("="*50)
    
    combined_importance = compare_models(
        dt_results, 
        rf_results, 
        dt_perm_importance=dt_perm_importance, 
        rf_perm_importance=rf_perm_importance
    )
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)


if __name__ == "__main__":
    start = time.time()
    
    # Set larger recursion limit for SHAP with deep trees
    import sys
    sys.setrecursionlimit(10000)
    
    main()
    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")