import os
import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


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