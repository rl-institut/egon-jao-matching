#!/usr/bin/env python3
# update_network_transformers_aggregated.py

import logging
import os
import sys

import numpy as np
import pandas as pd


def configure_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_csv(file_path, dtype=None):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        dtype (dict, optional): Dictionary specifying the data type for columns.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, dtype=dtype)
        logging.info(f"Loaded '{file_path}' with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load '{file_path}': {e}")
        sys.exit(1)


def standardize_keys(df, key_columns):
    """
    Standardizes the key columns by stripping whitespace and converting to uppercase.

    Args:
        df (pd.DataFrame): DataFrame containing the key columns.
        key_columns (list): List of column names to standardize.

    Returns:
        pd.DataFrame: DataFrame with standardized key columns.
    """
    for col in key_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
        else:
            logging.error(f"Key column '{col}' not found in DataFrame.")
            sys.exit(1)
    return df


def identify_and_save_duplicates(new_df, key_columns, duplicates_file_path):
    """
    Identifies duplicate transformers based on key_columns and saves them to a CSV.

    Args:
        new_df (pd.DataFrame): New network transformers DataFrame.
        key_columns (list): Columns to identify duplicates.
        duplicates_file_path (str): Path to save duplicate entries.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed (keeping all for aggregation).
    """
    # Identify all transformers with the same (bus0, bus1)
    duplicates = new_df[new_df.duplicated(subset=key_columns, keep=False)]
    if not duplicates.empty:
        logging.warning(
            "Found duplicate entries in 'new_network_transformers.csv'. Saving duplicates for review."
        )
        duplicates.to_csv(duplicates_file_path, index=False)
        logging.info(f"Duplicate entries saved to '{duplicates_file_path}'.")
    else:
        logging.info("No duplicate entries found in 'new_network_transformers.csv'.")
    return new_df


def aggregate_transformers(new_df, key_columns, update_cols):
    """
    Aggregates multiple transformers into a single equivalent transformer per substation.

    Args:
        new_df (pd.DataFrame): New network transformers DataFrame.
        key_columns (list): Columns to identify unique substations.
        update_cols (list): Columns to aggregate.

    Returns:
        pd.DataFrame: Aggregated DataFrame with one transformer per substation.
    """
    # Initialize lists to store aggregated data
    aggregated_data = []

    # Group by bus0 and bus1
    grouped = new_df.groupby(key_columns)

    for name, group in grouped:
        bus0, bus1 = name
        num_transformers = group.shape[0]
        logging.info(
            f"Aggregating {num_transformers} transformers for substation ({bus0}, {bus1})"
        )

        # Calculate equivalent impedance Z_eq = 1 / Σ(1 / Z_i)
        # Where Z_i = r_i + jx_i
        Z_i = group.apply(lambda row: complex(row["r"], row["x"]), axis=1)
        Y_i = 1 / Z_i  # Admittance of each transformer
        Y_eq = Y_i.sum()  # Total admittance

        if Y_eq == 0:
            logging.warning(
                f"Total admittance for substation ({bus0}, {bus1}) is zero. Setting Z_eq to infinity."
            )
            Z_eq = complex(np.inf, np.inf)
        else:
            Z_eq = 1 / Y_eq  # Equivalent impedance

        R_eq = Z_eq.real
        X_eq = Z_eq.imag

        # Sum s_nom
        s_nom_eq = group["s_nom"].sum()

        # Sum G and B
        G_eq = group["g"].sum()
        B_eq = group["b"].sum()

        # For other attributes, decide aggregation method
        # Example: take the first non-null value or some other logic
        # Here, we'll take the first value for simplicity
        other_cols = [
            col for col in new_df.columns if col not in key_columns + update_cols
        ]
        aggregated_other = {}
        for col in other_cols:
            # Take the first non-null value
            aggregated_other[col] = (
                group[col].dropna().iloc[0] if not group[col].dropna().empty else np.nan
            )

        # Create a new aggregated transformer entry
        aggregated_entry = {
            "bus0": bus0,
            "bus1": bus1,
            "r": R_eq,
            "g": G_eq,
            "b": B_eq,
            "x": X_eq,
            "s_nom": s_nom_eq,
        }
        # Add other aggregated attributes
        for col, value in aggregated_other.items():
            aggregated_entry[col] = value

        aggregated_data.append(aggregated_entry)

    # Create aggregated DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)
    logging.info(f"Aggregated DataFrame shape: {aggregated_df.shape}")
    return aggregated_df


def update_attributes(network_df, aggregated_new_df, key_columns, update_cols):
    """
    Updates specified attributes in the network DataFrame based on the aggregated new DataFrame.

    Args:
        network_df (pd.DataFrame): Original network transformers DataFrame.
        aggregated_new_df (pd.DataFrame): Aggregated new network transformers DataFrame.
        key_columns (list): Columns to use as keys for matching.
        update_cols (list): Columns to update.

    Returns:
        pd.DataFrame: Updated network transformers DataFrame.
    """
    # Perform a left merge to retain all entries from network_df
    merged_df = pd.merge(
        network_df,
        aggregated_new_df[key_columns + update_cols],
        how="left",
        on=key_columns,
        suffixes=("", "_agg"),
    )

    logging.info(f"Merged DataFrame shape: {merged_df.shape}")

    # Identify rows where aggregated attributes are available
    condition = merged_df[
        "r_agg"
    ].notna()  # Assuming 'r_agg' indicates presence of new data

    # Log the number of matches
    matches = condition.sum()
    logging.info(f"Number of transformers to update: {matches}")

    # Update the specified columns where new data is available
    for col in update_cols:
        merged_df[col] = merged_df.apply(
            lambda row: row[f"{col}_agg"] if pd.notna(row[f"{col}_agg"]) else row[col],
            axis=1,
        )

    # Drop the aggregated columns
    agg_cols = [f"{col}_agg" for col in update_cols]
    merged_df.drop(columns=agg_cols, inplace=True)

    return merged_df


def main():
    """
    Main function to update network transformers with aggregated attributes.
    """
    configure_logging()

    # Define file paths
    data_dir = "/home/mohsen/PycharmProjects/egon-jao-matching/data"
    network_transformers_file = os.path.join(data_dir, "network_transformers.csv")
    new_network_transformers_file = os.path.join(
        data_dir, "new_network_transformers.csv"
    )
    output_file = os.path.join(data_dir, "updated_network_transformers.csv")
    duplicates_file = os.path.join(data_dir, "duplicate_new_network_transformers.csv")

    # Define key columns and columns to update
    key_columns = ["bus0", "bus1"]
    update_cols = ["r", "g", "b", "x", "s_nom"]

    # Load CSV files
    network_df = load_csv(network_transformers_file)
    new_network_df = load_csv(new_network_transformers_file)

    # Standardize key columns in both DataFrames
    network_df = standardize_keys(network_df, key_columns)
    new_network_df = standardize_keys(new_network_df, key_columns)

    # Identify and save duplicate transformers
    new_network_df = identify_and_save_duplicates(
        new_network_df, key_columns, duplicates_file
    )

    # Aggregate duplicate transformers
    aggregated_new_df = aggregate_transformers(new_network_df, key_columns, update_cols)

    # Update attributes in network_df with aggregated values
    updated_df = update_attributes(
        network_df, aggregated_new_df, key_columns, update_cols
    )

    # Save the updated DataFrame to a new CSV file
    updated_df.to_csv(output_file, index=False)
    logging.info(f"Updated network transformers saved to '{output_file}'.")

    # Summary
    total_network = network_df.shape[0]
    total_new = aggregated_new_df.shape[0]
    total_updated = updated_df[update_cols].notna().sum().sum()
    logging.info(f"Total network transformers: {total_network}")
    logging.info(f"Total new aggregated transformers: {total_new}")
    logging.info(f"Total transformers updated: {total_updated}")


if __name__ == "__main__":
    main()
