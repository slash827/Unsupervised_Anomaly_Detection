"""
This script reads data from 'merged_data.csv', performs data transformation and feature engineering,
and then saves the transformed data as 'transformed_data.csv'.

The transformations include:
- Mapping process-related columns (processId, parentProcessId) to binary values:
  OS-related values (e.g., 0, 1, 2) are mapped to 0 and non-OS values to 1.
- Mapping userId: system users (userId < 1000) are mapped to 0, non-system users to 1.
- Mapping mountNamespace: if the value equals 4026531840, it is mapped to 0 (OS mount), else to 1.
- Retaining eventId as is.
- Mapping returnValue:
    - 0 if the value is 0 (success),
    - 1 if the value is positive (success with a return value),
    - 2 if the value is negative (error).
- Processing the 'args' column:
    - Converting the string representation of a list into an actual list.
    - Computing the number of arguments and storing it in a new column 'argsNum'.
    - Replacing the original 'args' column with the processed list.

The final transformed DataFrame is then saved as 'transformed_data.csv'.
"""

import pandas as pd
import ast


def process_args_row(row):
    """
    Process a single row from the 'args' column.

    This function converts a string representation of a list into an actual list using ast.literal_eval,
    calculates the number of elements (arguments) in the list, and returns a DataFrame with two columns:
    - 'args': the processed list.
    - 'argsNum': the number of arguments.
    """
    try:
        args_list = ast.literal_eval(row)
        if not isinstance(args_list, list):
            args_list = []
    except Exception as e:
        print(f"Error processing args row: {row}. Error: {e}")
        args_list = []
    return pd.DataFrame({'args': [args_list], 'argsNum': [len(args_list)]})


def process_args_dataframe(df, column_name):
    """
    Processes the 'args' column within the dataset.

    For each row in the specified column, this function:
      - Parses the string to convert it into an actual list.
      - Computes the number of arguments.

    The original 'args' column is replaced with the processed list, and a new column 'argsNum' is added.
    """
    processed_dataframes = []
    data = df[column_name].tolist()
    counter = 0

    for row in data:
        if row == '[]':  # If there are no args, record empty list and 0 count
            processed_dataframes.append(pd.DataFrame({'args': [[]], 'argsNum': [0]}))
        else:
            try:
                ret = process_args_row(row)
                processed_dataframes.append(ret)
            except Exception as e:
                print(f'Error Encounter: Row {counter} - {row}. Error: {e}')
                processed_dataframes.append(pd.DataFrame({'args': [[]], 'argsNum': [0]}))
        counter += 1

    processed = pd.concat(processed_dataframes).reset_index(drop=True)
    # Update the original DataFrame with the processed 'args' and new 'argsNum'
    df['args'] = processed['args']
    df['argsNum'] = processed['argsNum']
    return df


def prepare_dataset(df, process_args=False):
    """
    Prepare the dataset by completing standard feature engineering tasks.

    The function transforms several columns:
      - 'processId' and 'parentProcessId': Map OS-related values (0, 1, 2) to 0; others to 1.
      - 'userId': Map system users (userId < 1000) to 0; non-system users to 1.
      - 'mountNamespace': Map the specific value 4026531840 to 0, others to 1.
      - 'returnValue': Map 0 to 0; positive values to 1; negative values to 2.

    It also processes the 'args' column to compute a new 'argsNum' column and update the 'args' values if process_args is True.
    """
    # Transform processId: OS processes (0,1,2) mapped to 0, others mapped to 1
    df["processId"] = df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)

    # Transform parentProcessId: OS processes (0,1,2) mapped to 0, others mapped to 1
    df["parentProcessId"] = df["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)

    # Transform userId: system users (userId < 1000) mapped to 0, non-system users mapped to 1
    df["userId"] = df["userId"].map(lambda x: 0 if x < 1000 else 1)

    # Transform mountNamespace: if equals 4026531840, then OS mount (0), else non-OS mount (1)
    df["mountNamespace"] = df["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)

    # Retain eventId as is (this line is optional since no transformation is applied)
    df["eventId"] = df["eventId"]

    # Transform returnValue:
    # 0 if returnValue is 0, 1 if positive, 2 if negative.
    df["returnValue"] = df["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))

    # Process the 'args' column to update its values and compute 'argsNum'
    if process_args:
        df = process_args_dataframe(df, 'args')
    else:
        # If detailed processing is not enabled, compute argsNum without transforming 'args'
        def compute_args_num(x):
            try:
                if x == '[]':
                    return 0
                args_list = ast.literal_eval(x)
                if isinstance(args_list, list):
                    return len(args_list)
                else:
                    return 0
            except:
                return 0

        df["argsNum"] = df["args"].apply(compute_args_num)

    return df


def main():
    # Read the CSV file containing the raw data
    input_file = "merged_data.csv"
    df = pd.read_csv(input_file)

    # Prepare the dataset with argument processing enabled
    transformed_df = prepare_dataset(df, process_args=True)

    # Save the transformed data to a new CSV file
    output_file = "transformed_data.csv"
    transformed_df.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")


if __name__ == '__main__':
    main()
