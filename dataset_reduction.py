import os, glob
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

from utils import load_and_preprocess_beth_data

# Set plot style and figure size for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


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
    

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time took: {end - start} seconds")