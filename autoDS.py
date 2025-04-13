import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from sklearn.cluster import DBSCAN
import shap
import umap_impl
import time
import os, glob
import traceback
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess(csv_files, file_path):
    """
    Load and preprocess the BETH dataset.
    
    Args:
        file_path (str): Path to the BETH dataset CSV file.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test - preprocessed data splits
    """
    print("Loading and preprocessing data...")
    
    # Load the dataset
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    try:
        # Load the dataset with optimized memory usage
        # Use data types that consume less memory when possible
        dtypes = {
            'processId': 'int32',
            'threadId': 'int32',
            'parentProcessId': 'int32',
            'userId': 'int32',
            'eventId': 'int32',
            'argsNum': 'int32',
            'returnValue': 'int32',
            'sus': 'int8',
            'evil': 'int8'
        }
        
        # Try to use optimized loading if columns match expected schema
        try:
            df = pd.concat([pd.read_csv(f, dtype=dtypes) for f in csv_files], ignore_index=True)
        except:
            print("Warning: Could not apply optimized data types. Loading with default types.")
        
        # Display basic information
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        print(f"Total missing values: {missing_values}")
        
        # Sample preprocessing - based on BETH paper recommendations
        # Create engineered features more efficiently
        print("Creating engineered features...")
        
        # Convert userId to binary (system vs user)
        if 'userId' in df.columns:
            df['isSystemUser'] = (df['userId'] < 1000).astype('int8')
        
        # Convert mountNamespace to binary
        if 'mountNamespace' in df.columns:
            df['isDefaultMountNamespace'] = (df['mountNamespace'] == 4026531840).astype('int8')
        
        # Convert returnValue to categorical
        if 'returnValue' in df.columns:
            df['returnValueCategory'] = np.select(
                [df['returnValue'] < 0, df['returnValue'] == 0, df['returnValue'] > 0],
                [-1, 0, 1], 
                default=0
            ).astype('int8')
        
        # Assuming 'evil' is the target variable, and 'sus' could be another target
        y = df['evil']
        
        # Memory-efficient feature processing
        print("Processing features...")
        
        # Instead of one-hot encoding everything, handle each categorical feature carefully
        # This approach consumes much less memory
        features_to_keep = []
        X_processed = pd.DataFrame(index=df.index)
        
        # Process each column individually
        for column in df.columns:
            # Skip target variables
            if column in ['evil', 'sus']:
                continue
                
            # Skip string columns that would create too many one-hot columns
            if df[column].dtype == 'object':
                # Check cardinality (number of unique values)
                n_unique = df[column].nunique()
                
                # For high-cardinality categorical features
                if n_unique > 100:  # Arbitrary threshold
                    print(f"Skipping high-cardinality categorical column: {column} ({n_unique} unique values)")
                    continue
                    
                # For moderate-cardinality features, use one-hot but limit to most common values
                elif n_unique > 10:
                    print(f"Limited one-hot encoding for {column} ({n_unique} unique values)")
                    # Get top N most common values
                    top_values = df[column].value_counts().nlargest(10).index
                    
                    # One-hot encode only the top values
                    for val in top_values:
                        new_col = f"{column}_{val}"
                        X_processed[new_col] = (df[column] == val).astype('int8')
                    
                    # Add an "other" category for all other values
                    X_processed[f"{column}_other"] = (~df[column].isin(top_values)).astype('int8')
                else:
                    # For low-cardinality features, do regular one-hot encoding
                    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                    X_processed = pd.concat([X_processed, dummies], axis=1)
            else:
                # Numeric columns are kept as is
                X_processed[column] = df[column]
                
        # Add the engineered features
        if 'isSystemUser' in df.columns:
            X_processed['isSystemUser'] = df['isSystemUser']
        if 'isDefaultMountNamespace' in df.columns:
            X_processed['isDefaultMountNamespace'] = df['isDefaultMountNamespace']
        if 'returnValueCategory' in df.columns:
            X_processed['returnValueCategory'] = df['returnValueCategory']
        
        print(f"Processed features shape: {X_processed.shape}")
        print(f"Processed memory usage: {X_processed.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, 
            stratify=y if y.nunique() > 1 else None
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        print(f"Positive class (evil=1) in training: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train):.2%})")
        print(f"Positive class (evil=1) in testing: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test):.2%})")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def run_auto_sklearn(X_train, X_test, y_train, y_test):
    """
    Run auto-sklearn for automated model selection and hyperparameter tuning.
    
    Args:
        X_train, X_test, y_train, y_test: Training and testing datasets
        
    Returns:
        object: Trained auto-sklearn classifier
    """
    print("\nRunning Auto-Sklearn...")
    
    try:
        import autosklearn.classification
        
        # Create Auto-Sklearn classifier
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=1800,  # 30 minutes
            per_run_time_limit=300,        # 5 minutes per model
            n_jobs=-1,                     # Use all available cores
            ensemble_size=50               # Final ensemble size
        )
        
        # Fit the model
        print("Training Auto-Sklearn models (this may take a while)...")
        automl.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = automl.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        roc_auc = roc_auc_score(y_test, automl.predict_proba(X_test)[:, 1])
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Show the best models
        print("\nBest models:")
        print(automl.show_models())
        
        # Plot feature importance
        try:
            plt.figure(figsize=(10, 6))
            importance = pd.Series(data=automl.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            importance.head(20).plot(kind='bar')
            plt.title('Feature Importance from Auto-Sklearn')
            plt.tight_layout()
            plt.savefig("auto_sklearn_feature_importance.png")
            print("Feature importance plot saved to 'auto_sklearn_feature_importance.png'")
        except:
            print("Could not generate feature importance plot")
        
        return automl
    
    except ImportError:
        print("Error: auto-sklearn is not installed. Install it using:")
        print("pip install auto-sklearn")
        return None


def apply_shap_to_model(model, X_train, X_test, y_test):
    """
    Apply SHAP to explain model predictions.
    
    Args:
        model: Trained model to explain
        X_train, X_test, y_test: Data for explanations
        
    Returns:
        object: SHAP explainer object
    """
    print("\nApplying SHAP to explain model predictions...")
    
    try:
        # Select the best model from auto-sklearn if it's an ensemble
        if hasattr(model, 'get_models_with_weights'):
            best_model = model.get_models_with_weights()[0][0]
            print("Using the best model from the auto-sklearn ensemble")
        else:
            best_model = model
            print("Using the provided model directly")
        
        # Create SHAP explainer
        # Choose explainer type based on model type
        # For tree-based models, use TreeExplainer
        # For other models, use KernelExplainer with a sample of the training data
        if hasattr(best_model, 'estimators_') or str(type(best_model)).find('Tree') >= 0:
            explainer = shap.TreeExplainer(best_model)
            print("Using TreeExplainer for tree-based model")
        else:
            # For non-tree models, use KernelExplainer with a data sample
            sample_size = min(100, X_train.shape[0])
            background = X_train.sample(sample_size, random_state=42)
            explainer = shap.KernelExplainer(best_model.predict_proba, background)
            print(f"Using KernelExplainer with {sample_size} background samples")
        
        # Calculate SHAP values (limit to a sample for efficiency if dataset is large)
        sample_size = min(200, X_test.shape[0])
        sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
        X_test_sample = X_test.iloc[sample_indices]
        
        print(f"Calculating SHAP values for {sample_size} test samples...")
        shap_values = explainer(X_test_sample)
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
        plt.title("Feature Impact on Predictions")
        plt.tight_layout()
        plt.savefig("shap_feature_importance.png")
        print("SHAP summary plot saved to 'shap_feature_importance.png'")
        
        # Create waterfall plot for a single example (pick a positive case if available)
        plt.figure(figsize=(12, 8))
        y_test_sample = y_test.iloc[sample_indices]
        if sum(y_test_sample == 1) > 0:
            pos_idx = np.where(y_test_sample == 1)[0][0]
            shap.plots.waterfall(shap_values[pos_idx], max_display=10, show=False)
            plt.title("SHAP Explanation for an Evil Activity")
        else:
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.title("SHAP Explanation for a Sample Case")
        plt.tight_layout()
        plt.savefig("shap_waterfall_plot.png")
        print("SHAP waterfall plot saved to 'shap_waterfall_plot.png'")
        
        return explainer
    
    except Exception as e:
        print(f"Error applying SHAP: {str(e)}")
        return None


def run_isolation_forest(X_train, X_test, y_train, y_test):
    """
    Run Isolation Forest for anomaly detection.
    
    Args:
        X_train, X_test, y_train, y_test: Training and testing datasets
        
    Returns:
        object: Trained Isolation Forest model
    """
    print("\nRunning Isolation Forest for anomaly detection...")
    
    # Estimate contamination from the training set
    estimated_contamination = y_train.mean()
    print(f"Estimated contamination from training data: {estimated_contamination:.4f}")
    
    # If contamination is 0 (no positive examples in training), use a small value
    if estimated_contamination == 0:
        estimated_contamination = 0.05
        print(f"Using default contamination: {estimated_contamination}")
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination=estimated_contamination,
        random_state=42, 
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    
    # Get anomaly scores and predictions
    anomaly_scores = iso_forest.decision_function(X_test)
    predictions = iso_forest.predict(X_test)
    
    # Convert predictions to binary format (1 for anomaly, 0 for normal)
    # Note: Isolation Forest returns -1 for anomalies and 1 for normal data
    y_pred_binary = (predictions == -1).astype(int)
    
    # Evaluate performance
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_binary, average='binary', zero_division=0
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create a comparison DataFrame
    comparison = pd.DataFrame({
        'True_Evil': y_test,
        'Predicted_Anomaly': y_pred_binary,
        'Anomaly_Score': anomaly_scores
    })
    
    # Plot the distribution of anomaly scores
    plt.figure(figsize=(10, 6))
    sns.histplot(data=comparison, x='Anomaly_Score', hue='True_Evil', bins=30, kde=True)
    plt.title('Distribution of Anomaly Scores by True Label')
    plt.xlabel('Anomaly Score (lower is more anomalous)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("isolation_forest_scores.png")
    print("Anomaly score distribution saved to 'isolation_forest_scores.png'")
    
    return iso_forest


def run_pycaret_anomaly(X_train, X_test, y_train, y_test):
    """
    Run PyCaret for automated anomaly detection.
    
    Args:
        X_train, X_test, y_train, y_test: Training and testing datasets
        
    Returns:
        tuple: PyCaret setup object and best model
    """
    print("\nRunning PyCaret for anomaly detection...")
    
    try:
        from pycaret.anomaly import setup, create_model, assign_model, predict_model, plot_model
        
        # Setup PyCaret environment
        anomaly_setup = setup(X_train, session_id=42, verbose=False)
        
        # Create Isolation Forest model
        print("Training Isolation Forest model...")
        iso_model = create_model('iforest')
        
        # Assign anomaly labels
        train_predictions = assign_model(iso_model)
        
        # Make predictions on test data
        print("Making predictions on test data...")
        anomaly_predictions = predict_model(iso_model, data=X_test)
        
        # Validate results against true labels
        results = anomaly_predictions.copy()
        results['True_Evil'] = y_test.values.reset_index(drop=True)
        
        # Calculate metrics
        true_positives = sum((results['True_Evil'] == 1) & (results['Anomaly'] == 1))
        if sum(results['Anomaly'] == 1) > 0:
            precision = true_positives / sum(results['Anomaly'] == 1)
        else:
            precision = 0
            
        if sum(results['True_Evil'] == 1) > 0:
            recall = true_positives / sum(results['True_Evil'] == 1)
        else:
            recall = 0
            
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Create visualization
        try:
            print("Generating t-SNE plot...")
            plot_model(iso_model, plot='tsne', save=True)
            print("t-SNE plot saved")
        except Exception as e:
            print(f"Error generating t-SNE plot: {str(e)}")
        
        return anomaly_setup, iso_model
    
    except ImportError:
        print("Error: PyCaret is not installed. Install it using:")
        print("pip install pycaret")
        return None, None


def run_umap_with_isolation_forest(X_train, X_test, y_train, y_test):
    """
    Run UMAP dimensionality reduction followed by Isolation Forest.
    
    Args:
        X_train, X_test, y_train, y_test: Training and testing datasets
        
    Returns:
        tuple: (UMAP reducer, Isolation Forest model)
    """
    print("\nRunning UMAP with Isolation Forest...")
    
    try:
        # Perform dimensionality reduction using UMAP
        print("Applying UMAP dimensionality reduction...")
        reducer = umap_impl.UMAP(
            n_neighbors=15, 
            min_dist=0.1, 
            n_components=5, 
            random_state=42
        )
        X_train_umap = reducer.fit_transform(X_train)
        X_test_umap = reducer.transform(X_test)
        
        print(f"Reduced dimensions from {X_train.shape[1]} to {X_train_umap.shape[1]}")
        
        # Train Isolation Forest on the reduced data
        print("Training Isolation Forest on reduced data...")
        estimated_contamination = max(0.01, min(0.1, y_train.mean()))
        iso_forest = IsolationForest(
            n_estimators=100, 
            contamination=estimated_contamination, 
            random_state=42
        )
        iso_forest.fit(X_train_umap)
        
        # Make predictions
        y_pred = iso_forest.predict(X_test_umap)
        y_pred_binary = (y_pred == -1).astype(int)  # Convert to binary (1 for anomaly)
        
        # Measure performance
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_binary, average='binary', zero_division=0
        )
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Visualize results if using 2 or 3 components
        if X_train_umap.shape[1] >= 2:
            plt.figure(figsize=(12, 10))
            
            # If we have 3 components, create a 3D plot
            if X_train_umap.shape[1] >= 3:
                from mpl_toolkits.mplot3d import Axes3D
                ax = plt.subplot(111, projection='3d')
                scatter = ax.scatter(
                    X_test_umap[:, 0], 
                    X_test_umap[:, 1], 
                    X_test_umap[:, 2],
                    c=y_test, 
                    cmap='coolwarm', 
                    alpha=0.7,
                    s=30
                )
                ax.set_title('UMAP 3D Projection with True Labels')
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.set_zlabel('UMAP3')
            else:
                # 2D plot
                scatter = plt.scatter(
                    X_test_umap[:, 0], 
                    X_test_umap[:, 1], 
                    c=y_test, 
                    cmap='coolwarm', 
                    alpha=0.7,
                    s=30
                )
                plt.title('UMAP 2D Projection with True Labels')
                plt.xlabel('UMAP1')
                plt.ylabel('UMAP2')
            
            plt.colorbar(scatter, label='Evil (1) / Benign (0)')
            plt.tight_layout()
            plt.savefig('umap_visualization.png')
            print("UMAP visualization saved to 'umap_visualization.png'")
        
        return reducer, iso_forest
    
    except Exception as e:
        print(f"Error in UMAP+Isolation Forest: {str(e)}")
        return None, None


def analyze_clusters_for_attack_patterns(X_train, y_train):
    """
    Analyze clusters to identify attack patterns in malicious activities.
    
    Args:
        X_train, y_train: Training dataset and labels
        
    Returns:
        object: DBSCAN clustering model
    """
    print("\nAnalyzing clusters for attack patterns...")
    
    # Check if we have any evil samples
    evil_samples = X_train[y_train == 1]
    if len(evil_samples) == 0:
        print("No evil samples found in the training data. Cannot analyze attack patterns.")
        return None
    
    print(f"Found {len(evil_samples)} evil samples for analysis")
    
    try:
        # Apply dimensionality reduction to make clustering more effective
        print("Applying UMAP dimensionality reduction...")
        reducer = umap_impl.UMAP(
            n_neighbors=min(10, max(5, len(evil_samples) // 10)), 
            min_dist=0.1, 
            n_components=min(5, len(evil_samples) // 5), 
            random_state=42
        )
        evil_reduced = reducer.fit_transform(evil_samples)
        
        # Perform clustering with DBSCAN
        print("Performing DBSCAN clustering...")
        # Adjust eps parameter based on data size
        eps = 0.5
        min_samples = min(5, max(2, len(evil_samples) // 20))
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(evil_reduced)
        
        # Count samples in each cluster
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        print("\nNumber of samples in each cluster:")
        print(cluster_counts)
        
        # If all samples are noise (-1), try with different parameters
        if (clusters == -1).all():
            print("All samples were classified as noise. Trying with different parameters...")
            eps = 1.0
            min_samples = max(2, min(5, len(evil_samples) // 30))
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(evil_reduced)
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            print("\nNew cluster distribution:")
            print(cluster_counts)
        
        # Add cluster information to the samples
        evil_samples_with_clusters = evil_samples.copy()
        evil_samples_with_clusters['cluster'] = clusters
        
        # Analyze each cluster
        for cluster_id in sorted(set(clusters)):
            if cluster_id == -1:
                print("\nNoise points (no cluster):", sum(clusters == -1))
                continue
            
            cluster_data = evil_samples_with_clusters[evil_samples_with_clusters['cluster'] == cluster_id]
            print(f"\nCharacteristics of cluster {cluster_id} (samples: {len(cluster_data)}):")
            
            # Calculate statistics
            if len(cluster_data) > 1:  # Need at least 2 samples for statistics
                cluster_stats = cluster_data.describe().T
                all_stats = evil_samples.describe().T
                
                # Compare with overall average
                comparison = pd.DataFrame({
                    'Cluster_Mean': cluster_stats['mean'],
                    'Overall_Mean': all_stats['mean'],
                    'Ratio': cluster_stats['mean'] / all_stats['mean'].replace(0, 0.001)
                }).sort_values('Ratio', ascending=False)
                
                # Print top distinctive features
                print("Top distinctive features:")
                print(comparison.head(10))
        
        # Visualize clusters if we have 2D or 3D data
        if evil_reduced.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            
            # If we have 3D data
            if evil_reduced.shape[1] >= 3:
                from mpl_toolkits.mplot3d import Axes3D
                ax = plt.subplot(111, projection='3d')
                scatter = ax.scatter(
                    evil_reduced[:, 0], 
                    evil_reduced[:, 1], 
                    evil_reduced[:, 2],
                    c=clusters, 
                    cmap='viridis', 
                    alpha=0.8,
                    s=50
                )
                ax.set_title('DBSCAN Clusters of Evil Activities (3D)')
                ax.set_xlabel('Dimension 1')
                ax.set_ylabel('Dimension 2')
                ax.set_zlabel('Dimension 3')
            else:
                # 2D plot
                scatter = plt.scatter(
                    evil_reduced[:, 0], 
                    evil_reduced[:, 1], 
                    c=clusters, 
                    cmap='viridis', 
                    alpha=0.8,
                    s=50
                )
                plt.title('DBSCAN Clusters of Evil Activities (2D)')
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
            
            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()
            plt.savefig('evil_clusters.png')
            print("Cluster visualization saved to 'evil_clusters.png'")
        
        return dbscan
    
    except Exception as e:
        print(f"Error in cluster analysis: {str(e)}")
        return None


def main():
    """
    Main function to run all anomaly detection methods.
    """
    print("=" * 80)
    print("BETH Dataset Anomaly Detection Toolkit")
    print("=" * 80)
    
    # Load and preprocess data
    # File path to the BETH csv files
    data_path = os.getcwd() + os.sep + "data"
    csv_files = glob.glob(f"data{os.sep}*data.csv")

    # Command line argument support
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Using command line provided file path: {file_path}")
    
    print(f"Loading data from: {data_path}")
    
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess(csv_files, data_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please update the path or provide it as a command line argument.")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Ask which methods to run
    print("\nWhich methods would you like to run?")
    print("1. Auto-Sklearn (requires auto-sklearn package)")
    print("2. Isolation Forest")
    print("3. PyCaret (requires pycaret package)")
    print("4. UMAP with Isolation Forest")
    print("5. Cluster Analysis for Attack Patterns")
    print("6. Run all methods")
    print("0. Exit")
    
    try:
        choice = input("Enter your choice (0-6): ")
        choice = int(choice) if choice.strip() else 6
    except ValueError:
        print("Invalid input. Running all methods by default.")
        choice = 6
    
    if choice == 0:
        print("Exiting.")
        return
    
    # Run selected methods
    automl = None
    
    # Method 1: Auto-Sklearn
    if choice in [1, 6]:
        try:
            automl = run_auto_sklearn(X_train, X_test, y_train, y_test)
            # SHAP Analysis for Auto-Sklearn model
            if automl is not None:
                shap_explainer = apply_shap_to_model(automl, X_train, X_test, y_test)
        except Exception as e:
            print(f"Error running Auto-Sklearn: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Method 2: Isolation Forest
    if choice in [2, 6]:
        try:
            iso_forest = run_isolation_forest(X_train, X_test, y_train, y_test)
            # SHAP Analysis for Isolation Forest
            if iso_forest is not None:
                shap_explainer = apply_shap_to_model(iso_forest, X_train, X_test, y_test)
        except Exception as e:
            print(f"Error running Isolation Forest: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Method 3: PyCaret
    if choice in [3, 6]:
        try:
            pycaret_setup, pycaret_model = run_pycaret_anomaly(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"Error running PyCaret: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Method 4: UMAP with Isolation Forest
    if choice in [4, 6]:
        try:
            umap_reducer, umap_iso_forest = run_umap_with_isolation_forest(X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"Error running UMAP with Isolation Forest: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Method 5: Cluster Analysis for Attack Patterns
    if choice in [5, 6]:
        try:
            dbscan = analyze_clusters_for_attack_patterns(X_train, y_train)
        except Exception as e:
            print(f"Error running Cluster Analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Analysis complete! All results have been saved as images.")
    print("=" * 80)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print (f"total time took is: {end - start}")