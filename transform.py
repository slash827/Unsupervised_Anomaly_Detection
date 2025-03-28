import pandas as pd
import ast


def process_args_row(args):
    """
    Process a single row from the 'args' column.

    This function converts a string representation of a list into an actual list using ast.literal_eval,
    extracts up to four argument dictionaries, and assigns their 'type', 'value', and 'name' fields
    into separate columns. If fewer than four arguments exist, the missing columns are filled with None.
    """
    try:
        args_list = ast.literal_eval(args)
        if not isinstance(args_list, list):
            args_list = []
    except Exception as e:
        print(f"Error processing args row: {args}. Error: {e}")
        args_list = []

    extracted_data = {}
    for i in range(4):  # We assume at most 4 arguments
        if i < len(args_list) and isinstance(args_list[i], dict):
            extracted_data[f'type_{i}'] = args_list[i].get('type', None)
            extracted_data[f'value_{i}'] = args_list[i].get('value', None)
            extracted_data[f'name_{i}'] = args_list[i].get('name', None)
        else:
            extracted_data[f'type_{i}'] = None
            extracted_data[f'value_{i}'] = None
            extracted_data[f'name_{i}'] = None

    return extracted_data


def process_args_dataframe(df):
    """
    Processes the 'args' column within the dataset.

    This function expands the 'args' column into separate columns:
    - type_i, value_i, and name_i for up to four arguments.
    The original 'args' column is removed from the DataFrame.
    """
    args_expanded = df['args'].apply(process_args_row).apply(pd.Series)
    df = df.drop(columns=['args']).join(args_expanded)
    return df


def prepare_dataset(df, process_args=False):
    """
    Prepare the dataset by completing standard feature engineering tasks.

    The function transforms several columns:
      - 'processId' and 'parentProcessId': Map OS-related values (0, 1, 2) to 0; others to 1.
      - 'userId': Map system users (userId < 1000) to 0; non-system users to 1.
      - 'mountNamespace': Map the specific value 4026531840 to 0, others to 1.
      - 'returnValue': Map 0 to 0; positive values to 1; negative values to 2.

    It also processes the 'args' column:
      - If process_args=True, expands 'args' into multiple columns.
      - Otherwise, computes 'argsNum' (the number of arguments) without expansion.
    """
    df["processId"] = df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)
    df["parentProcessId"] = df["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)
    df["userId"] = df["userId"].map(lambda x: 0 if x < 1000 else 1)
    df["mountNamespace"] = df["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)
    df["returnValue"] = df["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))

    if process_args:
        df = process_args_dataframe(df)
    else:
        df["argsNum"] = df["args"].apply(lambda x: len(ast.literal_eval(x)) if x != '[]' else 0)

    return df


def main():
    """
    Load the dataset, process it, and save the transformed data.
    """
    input_file = "merged_data.csv"
    df = pd.read_csv(input_file)
    transformed_df = prepare_dataset(df, process_args=True)
    output_file = "transformed_data.csv"
    transformed_df.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")


if __name__ == '__main__':
    main()
