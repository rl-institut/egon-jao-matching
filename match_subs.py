#!/usr/bin/env python3
# allocate_dlr_attributes_refined.py

import pandas as pd
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('allocate_dlr_attributes_refined')

def load_csv(file_path, dtype=None):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        dtype (dict, optional): Data types for columns.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, dtype=dtype)
        logger.info(f"Loaded '{file_path}' with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load '{file_path}': {e}")
        sys.exit(1)

def main():
    """
    Main function to allocate DLR transformer attributes to matched network transformers.
    """
    # Define file paths
    data_dir = "/home/mohsen/PycharmProjects/egon-jao-matching/data"
    network_transformers_file = os.path.join(data_dir, "network_transformers.csv")
    dlr_transformers_file = os.path.join(data_dir, "dlr_transformers.csv")
    matched_substations_file = os.path.join(data_dir, "matched_substations.csv")

    output_file = os.path.join(data_dir, "matched_network_transformers_with_dlr.csv")

    # Load CSV files
    network_transformers = load_csv(network_transformers_file)
    dlr_transformers = load_csv(dlr_transformers_file)
    matched_substations = load_csv(matched_substations_file)

    # Ensure key columns are strings and strip any whitespace
    network_transformers['bus0'] = network_transformers['bus0'].astype(str).str.strip().str.upper()
    network_transformers['bus1'] = network_transformers['bus1'].astype(str).str.strip().str.upper()

    dlr_transformers['bus0'] = dlr_transformers['bus0'].astype(str).str.strip().str.upper()
    dlr_transformers['bus1'] = dlr_transformers['bus1'].astype(str).str.strip().str.upper()

    matched_substations['bus_id'] = matched_substations['bus_id'].astype(str).str.strip().str.upper()
    matched_substations['id_dlr'] = matched_substations['id_dlr'].astype(str).str.strip().str.upper()

    # Identify unmatched id_dlr before merging
    unmatched_id_dlr = matched_substations[~matched_substations['id_dlr'].isin(dlr_transformers['bus1'])]
    logger.info(f"Number of unmatched 'id_dlr' values before merging: {unmatched_id_dlr.shape[0]}")
    if not unmatched_id_dlr.empty:
        logger.warning(
            "Some 'id_dlr' values in matched_substations.csv do not have corresponding 'bus1' in dlr_transformers.csv.")
        logger.warning("These entries will have empty DLR attributes in the output.")
        # Optionally, save unmatched id_dlr entries for review
        unmatched_id_dlr.to_csv(os.path.join(data_dir, "unmatched_id_dlr_before_merge.csv"), index=False)
        logger.info(f"Unmatched 'id_dlr' entries saved to 'unmatched_id_dlr_before_merge.csv'.")

    # Perform Inner Merge: network_transformers with matched_substations
    merged_df = pd.merge(
        network_transformers,
        matched_substations,
        how='inner',  # Inner join to keep only matched entries
        left_on='bus0',
        right_on='bus_id',
        suffixes=('', '_matched')
    )

    logger.info(f"Number of network transformers after inner merge: {merged_df.shape[0]}")  # Expected: 83

    # Perform Inner Merge: merged_df with dlr_transformers
    # This ensures only transformers with matching DLR transformers are retained
    final_merged_df = pd.merge(
        merged_df,
        dlr_transformers,
        how='left',  # Left join to retain all matched network transformers
        left_on='id_dlr',
        right_on='bus1',
        suffixes=('', '_dlr')
    )

    logger.info(
        f"Number of matched network transformers after merging with DLR transformers: {final_merged_df.shape[0]}")  # Expected: 83

    # Define the DLR attributes to allocate (excluding overlapping ones)
    dlr_attributes = [
        'Full Name', 'EIC_Code', 'TSO',
        'Maximum Current Imax (A) primary Min',
        'Maximum Current Imax (A) primary Max',
        'Maximum Current Imax (A) primary Fixed',
        'Voltage_level(kV) Primary', 'Voltage_level(kV) Secondary',
        'Taps used for RAO', 'Theta θ (°)',
        'Symmetrical/Asymmetrical', 'Phase Regulation δu (%)',
        'Angle Regulation δu (%)', 'Comment'
    ]

    # Define the overlapping attributes from DLR with suffix
    dlr_overlapping_attributes = ['r_dlr', 'g_dlr', 'b_dlr', 'x_dlr', 's_nom_dlr']

    # Verify that all required DLR attributes exist in dlr_transformers
    missing_attributes = [attr for attr in dlr_attributes + dlr_overlapping_attributes if attr not in dlr_transformers.columns]
    if missing_attributes:
        logger.error(f"The following DLR attributes are missing in 'dlr_transformers.csv': {missing_attributes}")
        sys.exit(1)

    # Allocate DLR attributes to final_merged_df
    # Replace network attributes with DLR attributes
    final_merged_df['r'] = final_merged_df['r_dlr']
    final_merged_df['g'] = final_merged_df['g_dlr']
    final_merged_df['b'] = final_merged_df['b_dlr']
    final_merged_df['x'] = final_merged_df['x_dlr']
    final_merged_df['s_nom'] = final_merged_df['s_nom_dlr']

    # Now, drop the DLR suffixed overlapping columns
    final_merged_df.drop(columns=dlr_overlapping_attributes, inplace=True)

    # Now, include the remaining DLR attributes
    final_merged_df = final_merged_df[
        list(network_transformers.columns) + dlr_attributes
    ]

    # Save the enriched DataFrame to a new CSV file
    final_merged_df.to_csv(output_file, index=False)
    logger.info(f"Enriched matched network transformers with DLR attributes saved to '{output_file}'.")

    # Summary of the allocation
    total_matched_substations = matched_substations.shape[0]
    total_matched_transformers = final_merged_df.shape[0]
    unmatched_transformers = total_matched_substations - total_matched_transformers

    logger.info(f"Total matched substations: {total_matched_substations}")
    logger.info(f"Total matched network transformers with DLR attributes: {total_matched_transformers}")
    logger.info(f"Number of matched substations without DLR transformer attributes: {unmatched_transformers}")

    if unmatched_transformers > 0:
        logger.warning(
            f"There are {unmatched_transformers} matched substations without corresponding DLR transformer attributes.")
        # Optionally, save these entries
        unmatched_entries = final_merged_df[final_merged_df['Full Name'].isna()]
        unmatched_entries.to_csv(os.path.join(data_dir, "unmatched_matched_substations.csv"), index=False)
        logger.info(f"Unmatched matched substations saved to 'unmatched_matched_substations.csv'.")

if __name__ == "__main__":
    main()
