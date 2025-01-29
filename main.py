import os
import argparse
import pandas as pd
import logging
from egon_jao.data_processing import (
    load_config,
    load_data,
    prepare_dlr_geodataframes,
    prepare_network_lines_geodataframe,
    extract_buses_from_lines,
    match_buses_with_nearest_multiple,
    update_network_lines_with_matched_buses,
    match_lines_based_on_matched_buses,
    load_germany_boundary,
    filter_dlr_lines_inside_germany,
    match_lines_with_buffer,
    merge_connected_unmatched_network_lines,
    match_merged_lines_to_dlr,
    compute_match_statistics,
    create_matches_csv,
    allocate_attributes_to_network_lines,
    create_unmatched_lines_map,
    create_folium_map,
    validate_geodataframes
)


# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("egon_jao_matching.log"),
        logging.StreamHandler()
    ]
)

# Create a logger instance
logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
        handlers=[
            logging.FileHandler("egon_jao_matching.log"),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Egon-Jao Matching Script")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration YAML file.'
    )
    return parser.parse_args()

def ensure_directories(config):
    directories = set()
    for path in config['output_paths'].values():
        directories.add(os.path.dirname(path))
    for path in config['map_paths'].values():
        directories.add(os.path.dirname(path))

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")


def main():

    try:
        # Setup logging
        setup_logging()

        # Load configuration
        config = load_config()

        # Ensure output directories exist
        ensure_directories(config)

        # Load data
        dlr_buses, dlr_lines, network_buses, network_lines = load_data(config)

        # Prepare GeoDataFrames
        dlr_buses_gdf, dlr_lines_gdf = prepare_dlr_geodataframes(dlr_buses, dlr_lines)
        network_lines_gdf = prepare_network_lines_geodataframe(network_lines)

        # Extract buses from network lines
        network_buses_gdf = extract_buses_from_lines(network_lines_gdf)

        # Validate GeoDataFrames for duplicates and required columns
        validate_geodataframes(network_buses_gdf, dlr_buses_gdf, dlr_lines_gdf)

        # Load Germany boundary
        germany_gdf = load_germany_boundary()

        # Filter DLR lines to within Germany
        dlr_lines_within_germany = filter_dlr_lines_inside_germany(dlr_lines_gdf, germany_gdf)

        # Match buses using nearest neighbor allowing multiple matches
        bus_id_mapping = match_buses_with_nearest_multiple(network_buses_gdf, dlr_buses_gdf, config)

        # Update network lines with matched DLR buses
        network_lines_matched = update_network_lines_with_matched_buses(network_lines_gdf, bus_id_mapping)

        # Match lines based on matched buses allowing multiple matches (Nearest Bus Method)
        matched_network_lines_bus, matched_dlr_lines_bus, matches_df_bus = match_lines_based_on_matched_buses(
            network_lines_matched, dlr_lines_within_germany
        )

        # **Ensure 'bus_pair' is assigned only once using the correct columns**
        matches_df_bus['bus_pair'] = matches_df_bus.apply(
            lambda row: tuple(sorted([row['bus0_dlr'], row['bus1_dlr']])),
            axis=1
        )

        dlr_lines_within_germany['bus_pair'] = dlr_lines_within_germany.apply(
            lambda row: tuple(sorted([row['bus0'], row['bus1']])),
            axis=1
        )

        # **Drop the existing 'id_dlr' to prevent duplication**
        if 'id_dlr' in matches_df_bus.columns:
            matches_df_bus = matches_df_bus.drop(columns=['id_dlr'])

        # Merge to get 'id_dlr' without duplicating the column
        dlr_lines_subset = dlr_lines_within_germany[['bus_pair', 'id']].drop_duplicates()
        matches_df_bus = matches_df_bus.merge(
            dlr_lines_subset,
            on='bus_pair',
            how='left',
            suffixes=('', '_dlr')  # Ensure suffixes are correctly handled
        ).rename(columns={'id': 'id_dlr'})

        # Drop 'bus_pair' as it's no longer needed
        matches_df_bus = matches_df_bus.drop(columns=['bus_pair'])

        # **Ensure 'id_dlr' is a single column and not duplicated**
        if 'id_dlr_dlr' in matches_df_bus.columns:
            logger.warning("Duplicate 'id_dlr' columns detected. Removing 'id_dlr_dlr'.")
            matches_df_bus = matches_df_bus.drop(columns=['id_dlr_dlr'])

        # Handle missing 'id_dlr'
        missing_id_dlr = matches_df_bus['id_dlr'].isnull().sum()
        logger.debug(f"missing_id_dlr type: {type(missing_id_dlr)}, value: {missing_id_dlr}")
        if missing_id_dlr > 0:
            logger.warning(f"{missing_id_dlr} matched lines do not have a corresponding 'id_dlr'.")
            # Depending on your requirements, decide to drop or handle these rows
            matches_df_bus = matches_df_bus.dropna(subset=['id_dlr'])

        # 'matches_df_bus' contains 'id_net' and 'id_dlr'
        logger.debug(f"Matches DataFrame after merging and renaming:\n{matches_df_bus.head()}")

        # Verify Columns
        if 'id_net' in matches_df_bus.columns and 'id_dlr' in matches_df_bus.columns:
            logger.info("'id_net' and 'id_dlr' columns are present in matches_df_bus.")
        else:
            logger.error("'id_net' and/or 'id_dlr' columns are missing in matches_df_bus.")
            logger.debug(f"Available columns: {matches_df_bus.columns.tolist()}")
            raise KeyError("'id_net' and/or 'id_dlr' columns are missing in matches_df_bus.")

        # Count the number of unique network and DLR lines matched via nearest bus method
        unique_bus_matches_net = matches_df_bus['id_net'].nunique()
        unique_bus_matches_dlr = matches_df_bus['id_dlr'].nunique()
        logger.info(f"\nNumber of network lines matched through nearest bus method: {unique_bus_matches_net}")
        logger.info(f"Number of DLR lines matched through nearest bus method: {unique_bus_matches_dlr}")

        # Initialize sets for matched IDs
        matched_network_line_ids = set(matches_df_bus['id_net'])
        matched_dlr_line_ids = set(matches_df_bus['id_dlr'])

        # Update unmatched network and DLR lines
        unmatched_network_lines = network_lines_gdf[~network_lines_gdf['id'].isin(matched_network_line_ids)].copy()
        unmatched_dlr_lines = dlr_lines_within_germany[~dlr_lines_within_germany['id'].isin(matched_dlr_line_ids)].copy()

        # Apply buffer method to unmatched lines (Buffer Line Method)
        additional_matched_network_lines_buffer, additional_matched_dlr_lines_buffer, matches_df_buffer = match_lines_with_buffer(
            unmatched_network_lines, unmatched_dlr_lines, config
        )

        # Count the number of unique network and DLR lines matched via buffer method
        unique_buffer_matches_net = matches_df_buffer['id_net'].nunique()
        unique_buffer_matches_dlr = matches_df_buffer['id_dlr'].nunique()
        logger.info(f"\nNumber of network lines matched through buffer method: {unique_buffer_matches_net}")
        logger.info(f"Number of DLR lines matched through buffer method: {unique_buffer_matches_dlr}")

        # Update matched line IDs
        matched_network_line_ids.update(matches_df_buffer['id_net'])
        matched_dlr_line_ids.update(matches_df_buffer['id_dlr'])

        # Update unmatched network and DLR lines
        unmatched_network_lines = network_lines_gdf[~network_lines_gdf['id'].isin(matched_network_line_ids)].copy()
        unmatched_dlr_lines = dlr_lines_within_germany[
            ~dlr_lines_within_germany['id'].isin(matched_dlr_line_ids)].copy()

        # Third Method: Merge Connected Unmatched Network Lines and Match
        merged_unmatched_network_lines = merge_connected_unmatched_network_lines(unmatched_network_lines)
        additional_matched_network_lines_merge, additional_matched_dlr_lines_merge, matches_df_merge = match_merged_lines_to_dlr(
            merged_unmatched_network_lines, unmatched_dlr_lines, config
        )

        # Count the number of unique network and DLR lines matched via merging method
        unique_merge_matches_net = matches_df_merge['id_net'].nunique()
        unique_merge_matches_dlr = matches_df_merge['id_dlr'].nunique()
        logger.info(f"\nNumber of network lines matched through merging method: {unique_merge_matches_net}")
        logger.info(f"Number of DLR lines matched through merging method: {unique_merge_matches_dlr}")

        # Update matched line IDs with merged lines
        matched_network_line_ids.update(matches_df_merge['id_net'])
        matched_dlr_line_ids.update(matches_df_merge['id_dlr'])

        # Combine matches DataFrames
        matches_df_all = pd.concat([matches_df_bus, matches_df_buffer, matches_df_merge], ignore_index=True)

        # Log the structure of matches_df_all
        logger.debug(f"matches_df_all shape: {matches_df_all.shape}")
        logger.debug(f"matches_df_all columns: {matches_df_all.columns.tolist()}")
        logger.debug(f"Sample matches_df_all:\n{matches_df_all.head()}")

        # Remove duplicates based on 'id_net' and 'id_dlr'
        matches_df_all = matches_df_all.drop_duplicates(subset=['id_net', 'id_dlr'])

        # Log after dropping duplicates
        logger.debug(f"matches_df_all shape after dropping duplicates: {matches_df_all.shape}")
        logger.debug(f"matches_df_all columns after dropping duplicates: {matches_df_all.columns.tolist()}")

        # Final verification of 'id_net' and 'id_dlr' columns
        if 'id_net' in matches_df_all.columns and 'id_dlr' in matches_df_all.columns:
            matched_network_line_ids = set(matches_df_all['id_net'])
            matched_dlr_line_ids = set(matches_df_all['id_dlr'])
        else:
            logger.warning("No additional matches found during merging. Skipping merge-related updates.")
            matched_network_line_ids = set(matches_df_bus['id_net'])
            matched_dlr_line_ids = set(matches_df_bus['id_dlr'])

        # Log the matched IDs
        logger.debug(f"Matched Network Line IDs: {matched_network_line_ids}")
        logger.debug(f"Matched DLR Line IDs: {matched_dlr_line_ids}")

        # Update unmatched network and DLR lines after removing duplicates
        unmatched_network_lines = network_lines_gdf[~network_lines_gdf['id'].isin(matched_network_line_ids)].copy()
        unmatched_dlr_lines = dlr_lines_within_germany[
            ~dlr_lines_within_germany['id'].isin(matched_dlr_line_ids)].copy()

        # Print counts after all matching methods
        total_network_lines = len(network_lines_gdf)
        total_dlr_lines = len(dlr_lines_within_germany)
        logger.info(f"\nTotal number of network lines: {total_network_lines}")
        logger.info(f"Number of matched network lines: {len(matched_network_line_ids)}")
        logger.info(f"Number of unmatched network lines: {len(unmatched_network_lines)}")

        logger.info(f"\nTotal number of DLR lines: {total_dlr_lines}")
        logger.info(f"Number of matched DLR lines: {len(matched_dlr_line_ids)}")
        logger.info(f"Number of unmatched DLR lines: {len(unmatched_dlr_lines)}")

        # Save unmatched network lines to CSV using configuration path
        if 'output_paths' in config and 'unmatched_network_lines' in config['output_paths']:
            unmatched_network_lines.to_csv(config['output_paths']['unmatched_network_lines'], index=False)
            logger.info(f"Unmatched network lines saved to '{config['output_paths']['unmatched_network_lines']}'.")
        else:
            unmatched_network_lines.to_csv('results/csv/unmatched_network_lines.csv', index=False)
            logger.info(f"Unmatched network lines saved to 'results/csv/unmatched_network_lines.csv'.")

        # Save unmatched DLR lines to CSV using configuration path
        if 'output_paths' in config and 'unmatched_dlr_lines' in config['output_paths']:
            unmatched_dlr_lines.to_csv(config['output_paths']['unmatched_dlr_lines'], index=False)
            logger.info(f"Unmatched DLR lines saved to '{config['output_paths']['unmatched_dlr_lines']}'.")
        else:
            unmatched_dlr_lines.to_csv('results/csv/unmatched_dlr_lines.csv', index=False)
            logger.info(f"Unmatched DLR lines saved to 'results/csv/unmatched_dlr_lines.csv'.")

        # Combine matched lines from all methods
        matched_network_lines = network_lines_gdf[network_lines_gdf['id'].isin(matched_network_line_ids)].copy()
        matched_dlr_lines = dlr_lines_within_germany[dlr_lines_within_germany['id'].isin(matched_dlr_line_ids)].copy()

        # Allocate attributes to network lines
        network_lines_updated, allocated_matches = allocate_attributes_to_network_lines(
            matches_df_all, network_lines_gdf, dlr_lines_within_germany
        )

        # Create the CSV file with matched lines and their buses
        create_matches_csv(matches_df_all, network_lines_updated, dlr_lines_gdf, config)

        # Check if any length_ratio or length_m_dlr values are missing or incorrectly set
        missing_length_ratio = network_lines_updated['length_ratio'].isnull().sum()
        missing_length_m_dlr = network_lines_updated['length_m_dlr'].isnull().sum()
        if missing_length_ratio > 0:
            logger.warning(f"{missing_length_ratio} network lines have missing length_ratio.")
        else:
            logger.info("All network lines have a valid length_ratio.")

        if missing_length_m_dlr > 0:
            logger.warning(f"{missing_length_m_dlr} network lines have missing length_m_dlr.")
        else:
            logger.info("All network lines have a valid length_m_dlr.")

        # log some sample length_ratio and length_m_dlr values
        sample_length_info = network_lines_updated[['length', 'length_m_dlr', 'length_ratio']].head(10)
        logger.debug(f"Sample length information:\n{sample_length_info}")

        # Save the matched network lines with allocated attributes to CSV using configuration path
        if 'output_paths' in config and 'matched_network_lines_with_allocated_attributes' in config['output_paths']:
            network_lines_updated.to_csv(
                config['output_paths']['matched_network_lines_with_allocated_attributes'], index=False
            )
            logger.info(
                f"Matched network lines with allocated attributes saved to '{config['output_paths']['matched_network_lines_with_allocated_attributes']}'.")
        else:
            network_lines_updated.to_csv('results/csv/matched_network_lines_with_allocated_attributes.csv', index=False)
            logger.info(
                f"Matched network lines with allocated attributes saved to 'results/csv/matched_network_lines_with_allocated_attributes.csv'.")

        # Create unmatched lines map using configuration path
        create_unmatched_lines_map(unmatched_network_lines, germany_gdf, network_buses_gdf, config)


        # Extract matched and unmatched buses based on matched and unmatched lines
        matched_network_bus_ids = set(matched_network_lines['bus0']).union(set(matched_network_lines['bus1']))
        matched_network_buses_gdf = network_buses_gdf[network_buses_gdf['bus_idx'].isin(matched_network_bus_ids)].copy()

        unmatched_network_bus_ids = set(unmatched_network_lines['bus0']).union(set(unmatched_network_lines['bus1']))
        unmatched_network_buses_gdf = network_buses_gdf[network_buses_gdf['bus_idx'].isin(unmatched_network_bus_ids)].copy()

        matched_dlr_bus_ids = set(matched_dlr_lines['bus0']).union(set(matched_dlr_lines['bus1']))
        matched_dlr_buses_gdf = dlr_buses_gdf[dlr_buses_gdf['name'].isin(matched_dlr_bus_ids)].copy()

        unmatched_dlr_bus_ids = set(unmatched_dlr_lines['bus0']).union(set(unmatched_dlr_lines['bus1']))
        unmatched_dlr_buses_gdf = dlr_buses_gdf[dlr_buses_gdf['name'].isin(unmatched_dlr_bus_ids)].copy()

        # Combine matched and unmatched buses into separate GeoDataFrames
        matched_buses_gdf = pd.concat([
            matched_network_buses_gdf.rename(columns={'bus_idx': 'bus_id'}),
            matched_dlr_buses_gdf.rename(columns={'name': 'bus_id'})
        ], ignore_index=True)
        matched_buses_gdf['type'] = ['Network'] * len(matched_network_buses_gdf) + ['DLR'] * len(matched_dlr_buses_gdf)

        unmatched_buses_gdf = pd.concat([
            unmatched_network_buses_gdf.rename(columns={'bus_idx': 'bus_id'}),
            unmatched_dlr_buses_gdf.rename(columns={'name': 'bus_id'})
        ], ignore_index=True)
        unmatched_buses_gdf['type'] = ['Network'] * len(unmatched_network_buses_gdf) + ['DLR'] * len(unmatched_dlr_buses_gdf)

        # Verification of 'type' Column
        if 'type' in matched_buses_gdf.columns:
            logger.info("'type' column exists in matched_buses_gdf.")
        else:
            logger.error("'type' column is missing in matched_buses_gdf.")

        if 'type' in unmatched_buses_gdf.columns:
            logger.info("'type' column exists in unmatched_buses_gdf.")
        else:
            logger.error("'type' column is missing in unmatched_buses_gdf.")

        # Create the Folium Map with Separate Buses
        logger.info("\nGenerating map with matched and unmatched buses...")
        create_folium_map(
            germany_gdf, network_lines_gdf, dlr_lines_within_germany,
            matched_network_lines=matched_network_lines,
            matched_dlr_lines=matched_dlr_lines,
            matched_buses_gdf=matched_buses_gdf,
            unmatched_buses_gdf=unmatched_buses_gdf,
            config=config
        )

        # Compute match statistics
        compute_match_statistics(matches_df_all, total_network_lines, total_dlr_lines)

    except Exception as e:
        logger.error("An error occurred:", exc_info=True)

if __name__ == "__main__":
    main()




