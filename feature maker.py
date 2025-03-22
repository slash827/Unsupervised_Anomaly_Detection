import pandas as pd
import json
import os
# -------------------------------
# function to marge the data files
# -------------------------------

def merge_csv_files(csv_file1, csv_file2,percent, output_file="merged_data.csv"):
    # Read both CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2).sample(frac=percent/100)

    # Concatenate the dataframes
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"Merged CSV saved to {output_file}")


# -------------------------------
# Custom function to process the "args" column
# -------------------------------

def process_args_row(row):
    """
    Processes a single value from the 'args' column and returns a processed dataframe row.

    Args:
        row: A string value from the 'args' column.
        delay: A small delay (in seconds) between processing each key-value pair to reduce CPU load.

    Returns:
        final_df: A pandas DataFrame representing the processed row.
    """
    # Split the string on "},"
    parts = row.split('},')
    # Clean each part
    parts = [s.replace("[", "").replace("]", "").replace("{", "").replace("'", "").replace("}", "").lstrip(" ") for s in
             parts]
    # Split each cleaned string into potential key-value pairs
    parts = [s.split(',') for s in parts]

    processed_row = []
    for lst in parts:
        for key_value in lst:
            # Introduce a small delay to slow down processing

            # Only process if the separator is present
            if ': ' not in key_value:
                continue
            key, value = key_value.split(': ', 1)
            if not processed_row or key in processed_row[-1]:
                processed_row.append({})
            processed_row[-1][key] = value

    # Convert list of dicts to JSON, then normalize
    json_row = json.dumps(processed_row)
    row_df = pd.json_normalize(json.loads(json_row))

    # Unstack into a single-row DataFrame and sort columns
    final_df = row_df.unstack().to_frame().T.sort_index(axis=1)
    final_df.columns = final_df.columns.map('{0[0]}_{0[1]}'.format)

    return final_df


# -------------------------------
# Feature Engineering function
# -------------------------------
def feature_engineering(df):
    """
    Performs feature engineering on a given dataframe for unsupervised learning.

    Operations:
    1. Process the "args" column using process_args_row.
    2. Identify categorical columns (columns with 'id' in their name or of object type).
    3. Apply one-hot encoding to the categorical columns.
    4. Compute additional features using PCA (to capture correlations).

    Args:
        df: Input pandas DataFrame.

    Returns:
        df_encoded: DataFrame after feature engineering.
        corr_matrix: Correlation matrix of the encoded numerical features.
    """
    # --- Process the "args" column ---
    if "args" in df.columns:
        processed_arg_dfs = df["args"].apply(process_args_row)
        processed_args = pd.concat(processed_arg_dfs.tolist(), axis=0, ignore_index=True)
        df = df.drop("args", axis=1)
        df = pd.concat([df.reset_index(drop=True), processed_args.reset_index(drop=True)], axis=1)
    # --- Identify categorical columns ---
    # Columns with "id" in their name or type object are treated as categorical.
    categorical_cols = [col for col in df.columns if 'id' in col.lower() or df[col].dtype != 'int64']

    # --- One-hot encoding ---
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


    return df_encoded


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Load the data from the CSV file

    if os.path.isfile("merged_data.csv"):
        df = pd.read_csv("merged_data.csv", nrows=10**4)
    else:
        CsvPathtrain = "data/labelled_training_data.csv"
        CsvPathtest = "data/labelled_testing_data.csv"
        merge_csv_files(CsvPathtrain, CsvPathtest, 5)
        df = pd.read_csv("merged_data.csv")
    print(df['mountNamespace'].dtype)
    engineered_df = feature_engineering(df)
    engineered_df.to_csv("engineered_data.csv", index=False)
