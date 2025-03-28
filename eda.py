import time
import os
import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class BethDatasetEDA:
    """
    Exploratory Data Analysis for BETH Dataset
    """
    
    def __init__(self, data_path="data"):
        """
        Initialize EDA object
        
        Args:
            data_path (str): Path to data directory
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.output_dir = "eda_output"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_data(self):
        """
        Load and preprocess the BETH dataset
        
        Returns:
            pandas.DataFrame: Loaded and combined dataframe
        """
        # Load and combine datasets
        csv_files = glob.glob(f"{self.data_path}{os.sep}*[!dns].csv")
        
        # Print information about each file before combining
        print(f"Found {len(csv_files)} CSV files:")
        for file_path in csv_files:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"- {os.path.basename(file_path)}: {file_size:.2f} MB")
        
        # Check if the dataset is too large
        total_size_mb = sum(os.path.getsize(f) for f in csv_files) / (1024 * 1024)
        print(f"\nTotal dataset size: {total_size_mb:.2f} MB")
        
        # Create empty list to store individual dataframes for analysis
        dataframes = []
        
        # Load each file and append to the list with source file information
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            df['source_file'] = os.path.basename(file_path)
            dataframes.append(df)
        
        # Combine all dataframes
        self.data = pd.concat(dataframes, ignore_index=True)
        
        return self.data
    
    def add_specific_data(self, csv_names):
        """
        Load and add specific CSV files to the dataset
        
        Args:
            csv_names (list): List of CSV filenames to load
            
        Returns:
            pandas.DataFrame: Updated dataset
        """
        if not isinstance(csv_names, list):
            csv_names = [csv_names]  # Convert single string to list
        
        print(f"Loading {len(csv_names)} specific CSV files:")
        
        # Create empty list to store individual dataframes
        dataframes = []
        total_size_mb = 0
        
        for csv_name in csv_names:
            # Find the full path to the CSV file
            file_path = None
            
            # Handle cases where the full path is provided
            if os.path.exists(csv_name):
                file_path = csv_name
            # Handle cases where just the filename is provided
            elif os.path.exists(os.path.join(self.data_path, csv_name)):
                file_path = os.path.join(self.data_path, csv_name)
            # Try adding .csv extension if not provided
            elif os.path.exists(os.path.join(self.data_path, f"{csv_name}.csv")):
                file_path = os.path.join(self.data_path, f"{csv_name}.csv")
            
            if file_path is None:
                print(f"Warning: Could not find CSV file '{csv_name}'. Skipping.")
                continue
            
            # Load the CSV and add file information
            try:
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                total_size_mb += file_size
                
                print(f"- {os.path.basename(file_path)}: {file_size:.2f} MB")
                
                df = pd.read_csv(file_path)
                df['source_file'] = os.path.basename(file_path)
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not dataframes:
            print("No valid CSV files were loaded.")
            return None
        
        # Combine all dataframes
        loaded_data = pd.concat(dataframes, ignore_index=True)
        print(f"\nTotal loaded data size: {total_size_mb:.2f} MB")
        print(f"Loaded data shape: {loaded_data.shape}")
        
        # If this is the first data load, set it as self.data
        if self.data is None:
            self.data = loaded_data
        # Otherwise, append to existing data
        else:
            print(f"Appending to existing data (original shape: {self.data.shape})")
            self.data = pd.concat([self.data, loaded_data], ignore_index=True)
            print(f"Updated data shape: {self.data.shape}")
        
        # Reset data type identifications as the columns might have changed
        if self.data is not None:
            self.numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
            self.categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        return self.data

    def dataset_overview(self):
        """
        Basic overview of the dataset
        """
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        print("\n===== BASIC DATASET INFORMATION =====")
        print("Dataset shape:", self.data.shape)
        print("\nFirst few rows:")
        print(self.data.head())
        
        # Memory usage
        memory_usage = self.data.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"\nMemory usage: {memory_usage:.2f} MB")
        
        # Per-column memory usage
        print("\nMemory usage by column (MB):")
        column_memory = self.data.memory_usage(deep=True) / (1024 * 1024)
        for col, memory in zip(self.data.columns, column_memory[1:]):
            print(f"- {col}: {memory:.2f} MB")
        
        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        print(f"\nNumber of duplicate rows: {duplicates} ({duplicates/len(self.data)*100:.2f}%)")
        
        # Identify data types
        self.numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        print("\n===== DATA TYPE ANALYSIS =====")
        print("Numeric columns:", self.numeric_columns.tolist())
        print("Categorical columns:", self.categorical_columns.tolist())
        
        # Data type distribution
        print("\nData type distribution:")
        print(self.data.dtypes.value_counts())
    
    def analyze_missing_values(self):
        """
        Analyze and visualize missing values
        """
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        print("\n===== MISSING VALUES ANALYSIS =====")
        missing_values = self.data.isnull().sum()
        missing_percent = (missing_values / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percent
        }).sort_values('Missing Values', ascending=False)
        
        print(missing_df[missing_df['Missing Values'] > 0])
        
        # Visualize missing values if there are any
        if missing_values.sum() > 0:
            plt.figure(figsize=(12, 6))
            plt.title('Missing Values by Column')
            sns.barplot(x=missing_df.index[missing_df['Missing Values'] > 0], 
                      y=missing_df['Percentage'][missing_df['Missing Values'] > 0])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/missing_values.png")
            plt.close()
        
        # Handle missing values
        if len(self.numeric_columns) > 0:
            self.data[self.numeric_columns] = self.data[self.numeric_columns].fillna(self.data[self.numeric_columns].median())
        if len(self.categorical_columns) > 0:
            self.data[self.categorical_columns] = self.data[self.categorical_columns].fillna(self.data[self.categorical_columns].mode().iloc[0])
        
        return missing_df
    
    def analyze_numeric_features(self):
        """
        Analyze numeric features in the dataset
        """
        if self.data is None or self.numeric_columns is None:
            print("Data not properly initialized. Call load_data() first.")
            return
        
        print("\n===== NUMERIC FEATURES ANALYSIS =====")
        
        # Handle potential infinity values by converting them to NaN
        data_cleaned = self.data.copy()
        for col in self.numeric_columns:
            # Replace inf and -inf with NaN
            data_cleaned[col] = data_cleaned[col].replace([np.inf, -np.inf], np.nan)
        
        numeric_stats = data_cleaned[self.numeric_columns].describe().T
        numeric_stats['range'] = numeric_stats['max'] - numeric_stats['min']
        numeric_stats['coefficient_of_variation'] = numeric_stats['std'] / numeric_stats['mean']
        print(numeric_stats)
        
        # Check for skewed numeric columns
        skewed_columns = []
        for col in self.numeric_columns:
            # Skip columns with NaN or inf values for skewness calculation
            clean_values = data_cleaned[col].dropna()
            if len(clean_values) > 0:
                skewness = stats.skew(clean_values)
                if abs(skewness) > 1:
                    skewed_columns.append((col, skewness))
        
        if skewed_columns:
            print("\nHighly skewed columns (abs(skew) > 1):")
            for col, skew_val in sorted(skewed_columns, key=lambda x: abs(x[1]), reverse=True):
                print(f"- {col}: {skew_val:.2f}")
        
        # Visualize distribution of numeric variables
        plt.figure(figsize=(15, min(len(self.numeric_columns), 10) * 3))
        for i, col in enumerate(self.numeric_columns[:10]):  # Limit to first 10 columns for readability
            plt.subplot(min(10, len(self.numeric_columns)), 2, i+1)
            # Use the cleaned data for plotting
            sns.histplot(data_cleaned[col], kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/numeric_distributions.png")
        plt.close()
        
        return numeric_stats
    
    def analyze_correlations(self):
        """
        Analyze correlations between numeric features
        """
        if self.data is None or self.numeric_columns is None:
            print("Data not properly initialized. Call load_data() first.")
            return
        
        print("\n===== CORRELATION ANALYSIS =====")
        
        # Correlation analysis
        plt.figure(figsize=(12, 10))
        correlation = self.data[self.numeric_columns].corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_matrix.png")
        plt.close()
        
        # Find highly correlated features
        high_correlation = []
        for i in range(len(correlation.columns)):
            for j in range(i+1, len(correlation.columns)):
                if abs(correlation.iloc[i, j]) > 0.8:
                    high_correlation.append((correlation.columns[i], correlation.columns[j], correlation.iloc[i, j]))
        
        if high_correlation:
            print("\nHighly correlated feature pairs (|corr| > 0.8):")
            for col1, col2, corr_val in sorted(high_correlation, key=lambda x: abs(x[2]), reverse=True):
                print(f"- {col1} and {col2}: {corr_val:.2f}")
        
        return correlation, high_correlation
    
    def analyze_categorical_features(self):
        """
        Analyze categorical features in the dataset
        """
        if self.data is None or self.categorical_columns is None:
            print("Data not properly initialized. Call load_data() first.")
            return
        
        if len(self.categorical_columns) == 0:
            print("No categorical features found in the dataset.")
            return
        
        print("\n===== CATEGORICAL FEATURES ANALYSIS =====")
        categorical_stats = {}
        
        for col in self.categorical_columns:
            unique_values = self.data[col].nunique()
            print(f"\n{col}: {unique_values} unique values")
            
            # If there are too many unique values, only show top 10
            if unique_values <= 20:
                value_counts = self.data[col].value_counts()
                print(value_counts)
                
                # Visualize distribution
                plt.figure(figsize=(10, 6))
                sns.countplot(y=col, data=self.data, order=value_counts.index[:20])
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/categorical_{col}_distribution.png")
                plt.close()
            else:
                value_counts = self.data[col].value_counts()
                print(value_counts.head(10))
                print("...")
            
            categorical_stats[col] = {
                'unique_values': unique_values,
                'top_value': value_counts.index[0],
                'top_count': value_counts.iloc[0],
                'top_percentage': value_counts.iloc[0] / len(self.data) * 100
            }
        
        return categorical_stats
    
    def analyze_target_variable(self):
        """
        Analyze the target variable 'evil'
        """
        if self.data is None:
            print("Data not properly initialized. Call load_data() first.")
            return
        
        if 'evil' not in self.data.columns:
            print("Target variable 'evil' not found in the dataset.")
            return
        
        print("\n===== TARGET VARIABLE ANALYSIS =====")
        target_distribution = self.data['evil'].value_counts()
        print("Target distribution:")
        print(target_distribution)
        print(f"Class balance ratio (minority/majority): {target_distribution.min() / target_distribution.max():.4f}")
        
        # Visualize class distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='evil', data=self.data)
        plt.title('Distribution of Target Variable (evil)')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/target_distribution.png")
        plt.close()
        
        return target_distribution
    
    def analyze_user_behavior(self):
        """
        Analyze user behavior patterns if userId exists
        """
        if self.data is None:
            print("Data not properly initialized. Call load_data() first.")
            return
        
        if 'userId' not in self.data.columns:
            print("No 'userId' column found in the dataset.")
            return
        
        print("\n===== USER BEHAVIOR ANALYSIS =====")
        # Count unique users
        unique_users = self.data['userId'].nunique()
        print(f"Number of unique users: {unique_users}")
        
        # Actions per user
        actions_per_user = self.data.groupby('userId').size()
        print("\nActions per user summary:")
        print(actions_per_user.describe())
        
        # Visualize user activity distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(actions_per_user, bins=30, kde=True)
        plt.title('Distribution of Actions per User')
        plt.xlabel('Number of Actions')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/user_activity_distribution.png")
        plt.close()
        
        # Evil actions per user
        evil_actions = self.data.groupby('userId')['evil'].sum()
        evil_ratio = self.data.groupby('userId')['evil'].mean()
        
        print("\nEvil actions per user summary:")
        print(evil_actions.describe())
        
        print("\nEvil ratio per user summary:")
        print(evil_ratio.describe())
        
        # Visualize evil ratio distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(evil_ratio, bins=30, kde=True)
        plt.title('Distribution of Evil Ratio per User')
        plt.xlabel('Evil Ratio')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/evil_ratio_distribution.png")
        plt.close()
        
        # Identify suspicious users
        suspicious_users = evil_ratio[evil_ratio > 0.5]
        print(f"\nNumber of suspicious users (evil ratio > 0.5): {len(suspicious_users)}")
        
        user_stats = {
            'unique_users': unique_users,
            'actions_per_user': actions_per_user.describe().to_dict(),
            'evil_actions': evil_actions.describe().to_dict(),
            'evil_ratio': evil_ratio.describe().to_dict(),
            'suspicious_users': len(suspicious_users)
        }
        
        return user_stats
    
    def analyze_temporal_patterns(self):
        """
        Analyze temporal patterns if time-related columns exist
        """
        if self.data is None:
            print("Data not properly initialized. Call load_data() first.")
            return
        
        # Check for time-related columns
        time_cols = [col for col in self.data.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        if not time_cols:
            print("No time-related columns found in the dataset.")
            return
        
        print("\n===== TEMPORAL ANALYSIS =====")
        time_col = time_cols[0]
        
        # Convert to datetime
        if self.data[time_col].dtype == 'object':
            self.data[time_col] = pd.to_datetime(self.data[time_col], errors='coerce')
        
        # Time-based features
        self.data['hour'] = self.data[time_col].dt.hour
        self.data['day'] = self.data[time_col].dt.day
        self.data['day_of_week'] = self.data[time_col].dt.dayofweek
        
        # Activity by hour
        plt.figure(figsize=(12, 6))
        hourly_activity = self.data.groupby('hour').size()
        hourly_evil = self.data[self.data['evil'] == 1].groupby('hour').size()
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=hourly_activity.index, y=hourly_activity.values)
        plt.title('Activity by Hour')
        plt.xlabel('Hour')
        
        plt.subplot(1, 2, 2)
        evil_ratio_hourly = hourly_evil / hourly_activity
        sns.barplot(x=evil_ratio_hourly.index, y=evil_ratio_hourly.values)
        plt.title('Evil Ratio by Hour')
        plt.xlabel('Hour')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hourly_patterns.png")
        plt.close()
        
        # Activity by day of week
        plt.figure(figsize=(12, 6))
        dow_activity = self.data.groupby('day_of_week').size()
        dow_evil = self.data[self.data['evil'] == 1].groupby('day_of_week').size()
        evil_ratio_dow = dow_evil / dow_activity
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=dow_activity.index, y=dow_activity.values)
        plt.title('Activity by Day of Week')
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=evil_ratio_dow.index, y=evil_ratio_dow.values)
        plt.title('Evil Ratio by Day of Week')
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/day_of_week_patterns.png")
        plt.close()
        
        time_stats = {
            'time_column': time_col,
            'hourly_activity': hourly_activity.to_dict(),
            'hourly_evil_ratio': evil_ratio_hourly.to_dict(),
            'dow_activity': dow_activity.to_dict(),
            'dow_evil_ratio': evil_ratio_dow.to_dict()
        }
        
        return time_stats
    
    def analyze_sequential_patterns(self):
        """
        Analyze sequential patterns for users if userId and time columns exist
        """
        if self.data is None:
            print("Data not properly initialized. Call load_data() first.")
            return
        
        if 'userId' not in self.data.columns:
            print("No 'userId' column found in the dataset.")
            return
        
        # Check for time-related columns
        time_cols = [col for col in self.data.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        if not time_cols:
            print("No time-related columns found for sequential analysis.")
            return
        
        print("\n===== SEQUENTIAL PATTERN ANALYSIS =====")
        time_col = time_cols[0]
        
        # Convert to datetime if not already
        if self.data[time_col].dtype == 'object':
            self.data[time_col] = pd.to_datetime(self.data[time_col], errors='coerce')
        
        # Function to analyze consecutive actions by the same user
        def analyze_user_sequences(user_data):
            """Analyze patterns in consecutive actions by the same user"""
            # Sort by timestamp
            user_data = user_data.sort_values(by=time_col)
            
            # Get sequences of 'evil' actions (1s followed by 1s)
            if 'evil' in user_data.columns:
                evil_seq = user_data['evil'].rolling(window=2).sum()
                consecutive_evil = (evil_seq == 2).sum()
                return consecutive_evil
            return 0
        
        # Sample a subset of users for sequence analysis (for performance)
        sample_size_users = min(1000, self.data['userId'].nunique())
        print(f"Analyzing sequences for {sample_size_users} random users...")
        
        sampled_users = np.random.choice(self.data['userId'].unique(), size=sample_size_users, replace=False)
        user_sequence_metrics = {}
        
        for user_id in sampled_users:
            user_data = self.data[self.data['userId'] == user_id]
            if len(user_data) > 1:  # Only analyze users with more than 1 action
                consecutive_evil = analyze_user_sequences(user_data)
                user_sequence_metrics[user_id] = {
                    'total_actions': len(user_data),
                    'consecutive_evil': consecutive_evil,
                    'consecutive_evil_ratio': consecutive_evil / (len(user_data) - 1) if len(user_data) > 1 else 0
                }
        
        # Convert to DataFrame for analysis
        if user_sequence_metrics:
            sequence_df = pd.DataFrame.from_dict(user_sequence_metrics, orient='index')
            print("\nSequence analysis summary:")
            print(sequence_df.describe())
            
            # Find suspicious users based on consecutive evil actions
            suspicious_users = sequence_df[sequence_df['consecutive_evil_ratio'] > 0.5]
            if not suspicious_users.empty:
                print(f"\nFound {len(suspicious_users)} users with >50% consecutive evil actions:")
                print(suspicious_users.sort_values('consecutive_evil_ratio', ascending=False).head())
            
            return sequence_df
        
        return None
    
    def prepare_feature_target_split(self):
        """
        Prepare feature-target split and handle categorical encoding
        """
        if self.data is None:
            print("Data not properly initialized. Call load_data() first.")
            return None, None
        
        # Encode categorical features
        le_dict = {}
        for col in self.categorical_columns:
            le_dict[col] = LabelEncoder()
            self.data[col] = le_dict[col].fit_transform(self.data[col])
        
        # Feature-target split (using 'evil' as target)
        if 'evil' in self.data.columns and 'sus' in self.data.columns:
            self.X = self.data.drop(['evil', 'sus'], axis=1)  # Dropping both 'evil' and 'sus' columns
            self.y = self.data['evil']
        elif 'evil' in self.data.columns:
            self.X = self.data.drop(['evil'], axis=1)
            self.y = self.data['evil']
        else:
            print("Target variable 'evil' not found in the dataset.")
            return None, None
        
        return self.X, self.y
    
    def analyze_feature_importance(self, sample_size=100000):
        """
        Analyze feature importance using Random Forest
        
        Args:
            sample_size (int): Maximum number of samples to use for feature importance
        
        Returns:
            pandas.DataFrame: DataFrame with feature importance
        """
        if self.X is None or self.y is None:
            self.prepare_feature_target_split()
            
        if self.X is None or self.y is None:
            print("Feature-target split failed.")
            return None
        
        print("\n===== FEATURE IMPORTANCE ANALYSIS =====")
        
        # If the dataset is too large, sample a subset
        if len(self.X) > sample_size:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size/len(self.X), random_state=42)
            for _, sample_idx in sss.split(self.X, self.y):
                X_sample = self.X.iloc[sample_idx]
                y_sample = self.y.iloc[sample_idx]
        else:
            X_sample = self.X
            y_sample = self.y
        
        # Train a random forest model for feature importance
        print(f"\nTraining Random Forest for feature importance using {len(X_sample)} samples...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_sample, y_sample)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 most important features:")
        print(feature_importance.head(15))
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(y='Feature', x='Importance', data=feature_importance.head(15))
        plt.title('Top 15 Features by Importance')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance.png")
        plt.close()
        
        return feature_importance

    def baseline_model_evaluation(self, test_size=0.25):
        """
        Evaluate a baseline Random Forest model
        
        Args:
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Dictionary with model evaluation metrics
        """
        if self.X is None or self.y is None:
            self.prepare_feature_target_split()
            
        if self.X is None or self.y is None:
            print("Feature-target split failed.")
            return None
        
        print("\n===== BASELINE MODEL EVALUATION =====")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Sample if too large
        if len(self.X) > 100000:
            sample_size = 100000
            sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size/len(self.X), random_state=42)
            for _, sample_idx in sss.split(X_scaled, self.y):
                X_sample = X_scaled[sample_idx]
                y_sample = self.y.iloc[sample_idx].reset_index(drop=True)
        else:
            X_sample = X_scaled
            y_sample = self.y
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=test_size, random_state=42
        )
        
        # Train a simple Random Forest
        print("Training baseline Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test)
        y_prob = rf_model.predict_proba(X_test)[:, 1]
        
        print("\nBaseline model performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(f"{self.output_dir}/roc_curve.png")
        plt.close()
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(f"{self.output_dir}/confusion_matrix.png")
        plt.close()
        
        # Return evaluation metrics
        evaluation_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }
        
        return evaluation_metrics

    def generate_summary(self):
        """
        Generate a summary of EDA findings
        """
        print("\n===== EDA SUMMARY =====")
        
        if self.data is None:
            print("No data available for summary.")
            return
        
        summary = []
        
        # Dataset size
        summary.append(f"1. Dataset has {self.data.shape[0]} rows and {self.data.shape[1]} columns")
        
        # Class distribution
        if 'evil' in self.data.columns:
            target_distribution = self.data['evil'].value_counts().to_dict()
            summary.append(f"2. Class distribution: {target_distribution}")
        
        # User information
        if 'userId' in self.data.columns:
            unique_users = self.data['userId'].nunique()
            summary.append(f"3. There are {unique_users} unique users in the dataset")
            
            # Suspicious users
            evil_ratio = self.data.groupby('userId')['evil'].mean()
            suspicious_users = evil_ratio[evil_ratio > 0.5]
            summary.append(f"4. {len(suspicious_users)} users have an evil ratio > 0.5")
        
        # Feature importance
        if hasattr(self, 'X') and self.X is not None:
            feature_importance = self.analyze_feature_importance()
            if feature_importance is not None:
                top_features = feature_importance['Feature'].head(3).tolist()
                summary.append(f"5. Top 3 most important features: {', '.join(top_features)}")
        
        # Print summary
        for point in summary:
            print(point)
        
        # Save summary to file
        with open(f"{self.output_dir}/eda_summary.txt", 'w') as f:
            for point in summary:
                f.write(point + '\n')
        
        return summary


def load_eda_data(eda, should_use_all=True):
    if should_use_all:
        data = eda.load_data()
        print(f"Loaded all data with {data.shape[0]:_} rows and {data.shape[1]:_} columns")    
    else:
        # Load specific CSV files by name
        # This will only load the specified files, not all files in the directory
        csv_files_to_load = ["labelled_training_data.csv", "labelled_validation_data.csv", "labelled_testing_data.csv"]
        data = eda.add_specific_data(csv_files_to_load)
        print(f"Loaded train\\validation\\test data with {data.shape[0]:_} rows and {data.shape[1]:_} columns")    


def main_claude():
    # Initialize the EDA class
    # Adjust the data_path parameter to your actual data directory
    eda = BethDatasetEDA(data_path="data")
    load_eda_data(eda, should_use_all=False)
    
    # Step 2: Basic dataset overview
    eda.dataset_overview()
    
    # Step 3: Handle missing values
    eda.analyze_missing_values()
    
    # Step 4: Analyze numeric features
    eda.analyze_numeric_features()
    
    # Step 5: Analyze correlations
    eda.analyze_correlations()
    
    # Step 6: Analyze categorical features
    eda.analyze_categorical_features()
    
    # Step 7: Analyze target variable
    eda.analyze_target_variable()
    
    # Step 8: Analyze user behavior (if userId exists)
    eda.analyze_user_behavior()
    
    # Step 9: Analyze temporal patterns (if time columns exist)
    # eda.analyze_temporal_patterns()
    
    # Step 10: Analyze sequential patterns (if userId and time columns exist)
    eda.analyze_sequential_patterns()
    
    # Step 11: Prepare feature-target split (needed for subsequent analyses)
    eda.prepare_feature_target_split()
    
    # Step 12: Analyze feature importance
    feature_importance = eda.analyze_feature_importance(sample_size=100000)
    if feature_importance is not None:
        print(f"Top 5 features: {', '.join(feature_importance['Feature'].head(5))}")
    
    # Step 13: Baseline model evaluation
    evaluation = eda.baseline_model_evaluation()
    if evaluation:
        print(f"Model accuracy: {evaluation['accuracy']:.4f}")
    
    # Generate summary of findings
    eda.generate_summary()
    

def main_eda():
    # Load and combine datasets
    csv_files = glob.glob(f"data{os.sep}*[!dns].csv")
    data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Display initial information
    print("Dataset shape:", data.shape)
    print("\nFirst few rows:")
    print(data.head())
    print("\nMissing values:")
    print(data.isnull().sum())

    # Separate numeric and categorical columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    print("\nNumeric columns:", numeric_columns.tolist())
    print("Categorical columns:", categorical_columns.tolist())

    # Handle missing values
    if len(numeric_columns) > 0:
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    if len(categorical_columns) > 0:
        data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

    # Encode categorical features
    le_dict = {}
    for col in categorical_columns:
        le_dict[col] = LabelEncoder()
        data[col] = le_dict[col].fit_transform(data[col])

    # Feature-target split (using 'evil' as target)
    X = data.drop(['evil', 'sus'], axis=1)  # Dropping both 'evil' and 'sus' columns
    y = data['evil']


if __name__ == '__main__':
    start = time.time()
    main_claude()
    end = time.time()
    print(f"\nTotal EDA execution time: {end - start} seconds")