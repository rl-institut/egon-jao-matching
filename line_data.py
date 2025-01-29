import pandas as pd
import os


def replace_network_line_attributes(network_lines_path, new_network_lines_path, output_path):
    """
    Replaces specific attributes in network-lines.csv with those from new_network_lines.csv based on bus0 and bus1.

    Parameters:
        network_lines_path (str): Path to the original network-lines.csv file.
        new_network_lines_path (str): Path to the new_network_lines.csv file.
        output_path (str): Path to save the updated network-lines.csv file.

    Returns:
        None
    """
    # Read the original network lines
    try:
        network_df = pd.read_csv(network_lines_path, dtype={'bus0': str, 'bus1': str})
        print(f"Loaded network lines from '{network_lines_path}' with {len(network_df)} records.")
    except Exception as e:
        print(f"Error reading '{network_lines_path}': {e}")
        return

    # Read the new network lines
    try:
        new_network_df = pd.read_csv(new_network_lines_path, dtype={'bus0': str, 'bus1': str})
        print(f"Loaded new network lines from '{new_network_lines_path}' with {len(new_network_df)} records.")
    except Exception as e:
        print(f"Error reading '{new_network_lines_path}': {e}")
        return

    # Create a bus_pair column that is order-independent
    def create_bus_pair(row):
        bus0 = row['bus0'].strip().upper()
        bus1 = row['bus1'].strip().upper()
        return tuple(sorted([bus0, bus1]))

    network_df['bus_pair'] = network_df.apply(create_bus_pair, axis=1)
    new_network_df['bus_pair'] = new_network_df.apply(create_bus_pair, axis=1)

    # Check for duplicate bus_pairs in new_network_df
    duplicates = new_network_df['bus_pair'].duplicated(keep=False)
    if duplicates.any():
        duplicate_pairs = new_network_df[duplicates]['bus_pair'].unique()
        print(
            f"Warning: Found {len(duplicate_pairs)} duplicate bus pairs in '{new_network_lines_path}'. These will be handled by taking the first occurrence.")
        # Retain only the first occurrence of each duplicate bus_pair
        new_network_df = new_network_df.drop_duplicates(subset='bus_pair', keep='first')
    else:
        print("No duplicate bus pairs found in new_network_lines.csv. Proceeding with aggregation.")

    # Define the columns to replace
    columns_to_replace = ['r', 'x', 'b', 'v_nom', 's_nom', 's_nom_min']

    # Verify that the columns_to_replace exist in new_network_df
    missing_new_columns = [col for col in columns_to_replace if col not in new_network_df.columns]
    if missing_new_columns:
        print(f"Error: The following required columns are missing in '{new_network_lines_path}': {missing_new_columns}")
        print("Please verify the column names and ensure they match the expected columns.")
        return
    else:
        print("All required columns for replacement are present in new_network_lines.csv.")

    # Select relevant columns from new_network_df
    new_network_subset = new_network_df[['bus_pair'] + columns_to_replace].copy()

    # Rename the new columns to have a suffix to avoid confusion
    new_network_subset = new_network_subset.rename(columns=lambda x: x if x == 'bus_pair' else f"{x}_new")

    # Merge the DataFrames on bus_pair
    merged_df = pd.merge(network_df, new_network_subset, on='bus_pair', how='left', suffixes=('_network', '_new'))

    # Debugging: Print merged DataFrame columns
    print("Merged DataFrame Columns:", merged_df.columns.tolist())

    # Check how many lines will be updated (i.e., have non-NaN values in any of the new columns)
    lines_to_update = merged_df[[f"{col}_new" for col in columns_to_replace]].notna().any(axis=1)
    print(f"Number of network lines to be updated: {lines_to_update.sum()} out of {len(network_df)}.")

    # Replace the specified columns where new data is available
    for col in columns_to_replace:
        new_col = f"{col}_new"
        if new_col in merged_df.columns:
            # Replace the original column with the new column where not NaN
            before_replacement = merged_df[col].copy()
            merged_df[col] = merged_df[col].where(~merged_df[new_col].notna(), merged_df[new_col])
            replaced_count = merged_df[new_col].notna().sum()
            print(f"Replaced column '{col}' with new data in {replaced_count} records.")
        else:
            print(f"Warning: Column '{new_col}' not found in new_network_lines.csv. Skipping replacement for '{col}'.")

    # Drop auxiliary columns (the '_new' columns)
    merged_df.drop(columns=[f"{col}_new" for col in columns_to_replace], inplace=True, errors='ignore')

    # Optionally, drop the bus_pair column if it's no longer needed
    merged_df.drop(columns=['bus_pair'], inplace=True, errors='ignore')

    # Save the updated network lines to a new CSV
    try:
        merged_df.to_csv(output_path, index=False)
        print(f"Updated network lines saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving updated network lines to '{output_path}': {e}")


if __name__ == "__main__":
    # Define file paths
    network_lines_csv = 'data/network-lines.csv'  # Path to the original network-lines.csv
    new_network_lines_csv = 'data/new_network_lines.csv'  # Path to the new_network_lines.csv
    updated_network_lines_csv = 'data/network-lines-updated.csv'  # Path to save the updated network-lines.csv

    # Check if input files exist
    if not os.path.exists(network_lines_csv):
        print(f"Error: '{network_lines_csv}' does not exist.")
    elif not os.path.exists(new_network_lines_csv):
        print(f"Error: '{new_network_lines_csv}' does not exist.")
    else:
        # Perform the replacement
        replace_network_line_attributes(network_lines_csv, new_network_lines_csv, updated_network_lines_csv)
