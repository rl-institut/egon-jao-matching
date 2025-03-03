import argparse
import logging
import os

import numpy as np
import pandas as pd

from egon_jao.data_processing import (
    check_dlr_parameters,  # We still need to check if parameters exist
)
from egon_jao.data_processing import (
    allocate_attributes_to_network_lines,
    compute_match_statistics,
    create_folium_map,
    create_matches_csv,
    create_unmatched_lines_map,
    extract_buses_from_lines,
    filter_dlr_lines_inside_germany,
    load_config,
    load_data,
    load_germany_boundary,
    match_buses_with_nearest_multiple,
    match_lines_based_on_matched_buses,
    match_lines_with_buffer,
    match_merged_lines_to_dlr,
    merge_connected_unmatched_network_lines,
    prepare_dlr_geodataframes,
    prepare_network_lines_geodataframe,
    update_network_lines_with_matched_buses,
    validate_geodataframes,
    validate_parameter_ranges,
)

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("egon_jao_matching.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        handlers=[
            logging.FileHandler("egon_jao_matching.log"),
            logging.StreamHandler(),
        ],
    )


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Egon-Jao Matching Script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file.",
    )
    return parser.parse_args()


def ensure_directories(config):
    directories = set()
    for path in config["output_paths"].values():
        directories.add(os.path.dirname(path))
    for path in config["map_paths"].values():
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

        # 1) Load raw CSV Data
        try:
            dlr_buses, dlr_lines, network_buses, network_lines = load_data(config)
            logger.debug(
                "Loaded raw CSV data for dlr_buses, dlr_lines, network_buses, network_lines."
            )

            # Check for required columns
            required_dlr_lines_cols = ["bus0", "bus1", "id"]
            required_network_lines_cols = ["bus0", "bus1", "geom", "r", "x", "b", "id"]

            missing_dlr_cols = [
                col for col in required_dlr_lines_cols if col not in dlr_lines.columns
            ]
            missing_net_cols = [
                col
                for col in required_network_lines_cols
                if col not in network_lines.columns
            ]

            if missing_dlr_cols:
                logger.error(
                    f"Missing required columns in dlr_lines: {missing_dlr_cols}"
                )
                raise ValueError(
                    f"Missing required columns in dlr_lines: {missing_dlr_cols}"
                )

            if missing_net_cols:
                logger.error(
                    f"Missing required columns in network_lines: {missing_net_cols}"
                )
                raise ValueError(
                    f"Missing required columns in network_lines: {missing_net_cols}"
                )

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise

        # 2) Prepare DLR GeoDataFrames
        dlr_buses_gdf, dlr_lines_gdf = prepare_dlr_geodataframes(dlr_buses, dlr_lines)
        logger.debug(
            "After prepare_dlr_geodataframes:\n"
            f" dlr_buses_gdf columns: {dlr_buses_gdf.columns.tolist()}, size={len(dlr_buses_gdf)}\n"
            f" dlr_lines_gdf columns: {dlr_lines_gdf.columns.tolist()}, size={len(dlr_lines_gdf)}"
        )

        # 2.5) Check DLR electrical parameters
        dlr_params_ok = check_dlr_parameters(dlr_lines_gdf)
        if not dlr_params_ok:
            logger.warning(
                "DLR lines are missing electrical parameters or have all zero values. "
                "Allocation may produce zeros."
            )

        # 3) Prepare Network lines GDF
        network_lines_gdf = prepare_network_lines_geodataframe(network_lines)
        logger.debug(
            "After prepare_network_lines_geodataframe:\n"
            f" network_lines_gdf columns: {network_lines_gdf.columns.tolist()}, size={len(network_lines_gdf)}"
        )

        # 4) Project both to 'EPSG:32632'
        network_lines_gdf = network_lines_gdf.to_crs("EPSG:32632")
        dlr_lines_gdf = dlr_lines_gdf.to_crs("EPSG:32632")
        logger.debug(f"Network lines CRS: {network_lines_gdf.crs}")
        logger.debug(f"DLR lines CRS: {dlr_lines_gdf.crs}")

        # Check bounding boxes for sanity
        print("Network lines bounding box:", network_lines_gdf.total_bounds)
        print("DLR lines bounding box:", dlr_lines_gdf.total_bounds)

        net_bounds = network_lines_gdf.total_bounds
        dlr_bounds = dlr_lines_gdf.total_bounds
        logger.debug(f"Network lines bounding box: {net_bounds}")
        logger.debug(f"DLR lines bounding box: {dlr_bounds}")

        # 5) Extract Buses from Network lines
        network_buses_gdf = extract_buses_from_lines(network_lines_gdf)
        logger.debug(f"After extract_buses_from_lines:\n {network_buses_gdf.head(10)}")

        # Validate geodataframes
        validate_geodataframes(network_buses_gdf, dlr_buses_gdf, dlr_lines_gdf)

        # 6) Load Germany boundary
        germany_gdf = load_germany_boundary()

        # 7) Filter DLR lines to within Germany
        dlr_lines_within_germany = filter_dlr_lines_inside_germany(
            dlr_lines_gdf, germany_gdf
        )
        logger.debug(f"DLR lines within Germany: {len(dlr_lines_within_germany)}")

        # 8) Match buses (Nearest)
        bus_id_mapping = match_buses_with_nearest_multiple(
            network_buses_gdf, dlr_buses_gdf, config
        )
        logger.debug(f"Bus ID mapping sample:\n{bus_id_mapping.head(10)}")

        # 9) Update network lines w/ matched DLR buses
        network_lines_matched = update_network_lines_with_matched_buses(
            network_lines_gdf, bus_id_mapping
        )
        logger.debug(
            "Network lines matched (sample):\n"
            f"{network_lines_matched[['id', 'bus0', 'bus1', 'dlr_bus0', 'dlr_bus1']].head(10)}"
        )

        # 10) Match lines based on matched buses
        matched_network_lines_bus, matched_dlr_lines_bus, matches_df_bus, matches = (
            match_lines_based_on_matched_buses(
                network_lines_matched, dlr_lines_within_germany
            )
        )
        logger.debug(
            f"matches_df_bus shape={matches_df_bus.shape}, "
            f"columns={matches_df_bus.columns.tolist()}"
        )
        logger.debug(f"matches_df_bus head:\n{matches_df_bus.head(10)}")

        # Ensure 'bus_pair'
        matches_df_bus["bus_pair"] = matches_df_bus.apply(
            lambda row: tuple(sorted([row["bus0_dlr"], row["bus1_dlr"]])), axis=1
        )
        dlr_lines_within_germany["bus_pair"] = dlr_lines_within_germany.apply(
            lambda row: tuple(sorted([row["bus0"], row["bus1"]])), axis=1
        )

        if "id_dlr" in matches_df_bus.columns:
            matches_df_bus = matches_df_bus.drop(columns=["id_dlr"])

        # Merge to get 'id_dlr'
        dlr_lines_subset = dlr_lines_within_germany[
            ["bus_pair", "id"]
        ].drop_duplicates()
        matches_df_bus = matches_df_bus.merge(
            dlr_lines_subset, on="bus_pair", how="left", suffixes=("", "_dlr")
        ).rename(columns={"id": "id_dlr"})

        # Drop 'bus_pair'
        matches_df_bus = matches_df_bus.drop(columns=["bus_pair"])

        if "id_dlr_dlr" in matches_df_bus.columns:
            logger.warning("Duplicate 'id_dlr_dlr' column found. Dropping.")
            matches_df_bus = matches_df_bus.drop(columns=["id_dlr_dlr"])

        missing_id_dlr = matches_df_bus["id_dlr"].isnull().sum()
        logger.debug(f"missing_id_dlr => {missing_id_dlr}")
        if missing_id_dlr > 0:
            logger.warning(
                f"{missing_id_dlr} matched lines do not have 'id_dlr'. Dropping those rows."
            )
            matches_df_bus = matches_df_bus.dropna(subset=["id_dlr"])

        logger.debug(f"Matches DataFrame after merging:\n{matches_df_bus.head(10)}")

        if "id_net" in matches_df_bus.columns and "id_dlr" in matches_df_bus.columns:
            logger.info("'id_net' and 'id_dlr' columns are present in matches_df_bus.")
        else:
            logger.error("Missing 'id_net' or 'id_dlr' in matches_df_bus.")
            logger.debug(f"Columns: {matches_df_bus.columns.tolist()}")
            raise KeyError("Missing 'id_net' or 'id_dlr' columns.")

        # Summaries
        unique_bus_matches_net = matches_df_bus["id_net"].nunique()
        unique_bus_matches_dlr = matches_df_bus["id_dlr"].nunique()
        logger.info(
            f"\nNumber of network lines matched (nearest bus method): {unique_bus_matches_net}"
        )
        logger.info(
            f"Number of DLR lines matched (nearest bus method): {unique_bus_matches_dlr}"
        )

        matched_network_line_ids = set(matches_df_bus["id_net"])
        matched_dlr_line_ids = set(matches_df_bus["id_dlr"])

        # Unmatched after nearest-bus matching
        unmatched_network_lines = network_lines_gdf[
            ~network_lines_gdf["id"].isin(matched_network_line_ids)
        ].copy()
        unmatched_dlr_lines = dlr_lines_within_germany[
            ~dlr_lines_within_germany["id"].isin(matched_dlr_line_ids)
        ].copy()

        # 11) Buffer method
        additional_matched_net_buf, additional_matched_dlr_buf, matches_df_buffer = (
            match_lines_with_buffer(
                unmatched_network_lines, unmatched_dlr_lines, config
            )
        )
        logger.debug(
            f"matches_df_buffer shape={matches_df_buffer.shape}, "
            f"columns={matches_df_buffer.columns.tolist()}"
        )
        logger.debug(f"matches_df_buffer sample:\n{matches_df_buffer.head(10)}")

        unique_buffer_matches_net = matches_df_buffer["id_net"].nunique()
        unique_buffer_matches_dlr = matches_df_buffer["id_dlr"].nunique()
        logger.info(
            f"\nNumber of network lines matched (buffer method): {unique_buffer_matches_net}"
        )
        logger.info(
            f"Number of DLR lines matched (buffer method): {unique_buffer_matches_dlr}"
        )

        matched_network_line_ids.update(matches_df_buffer["id_net"])
        matched_dlr_line_ids.update(matches_df_buffer["id_dlr"])

        unmatched_network_lines = network_lines_gdf[
            ~network_lines_gdf["id"].isin(matched_network_line_ids)
        ].copy()
        unmatched_dlr_lines = dlr_lines_within_germany[
            ~dlr_lines_within_germany["id"].isin(matched_dlr_line_ids)
        ].copy()

        # 12) Merge connected unmatched lines -> match_merged_lines_to_dlr
        merged_unmatched_network_lines = merge_connected_unmatched_network_lines(
            unmatched_network_lines
        )
        logger.debug(
            f"Merged unmatched lines => {len(merged_unmatched_network_lines)} rows"
        )

        additional_net_merge, additional_dlr_merge, matches_df_merge = (
            match_merged_lines_to_dlr(
                merged_unmatched_network_lines, unmatched_dlr_lines, config
            )
        )
        logger.debug(
            f"matches_df_merge shape={matches_df_merge.shape}, "
            f"columns={matches_df_merge.columns.tolist()}"
        )
        logger.debug(f"matches_df_merge sample:\n{matches_df_merge.head(10)}")

        unique_merge_matches_net = matches_df_merge["id_net"].nunique()
        unique_merge_matches_dlr = matches_df_merge["id_dlr"].nunique()
        logger.info(
            f"\nNumber of network lines matched (merge method): {unique_merge_matches_net}"
        )
        logger.info(
            f"Number of DLR lines matched (merge method): {unique_merge_matches_dlr}"
        )

        matched_network_line_ids.update(matches_df_merge["id_net"])
        matched_dlr_line_ids.update(matches_df_merge["id_dlr"])

        # 13) Combine all matches
        matches_df_all = pd.concat(
            [matches_df_bus, matches_df_buffer, matches_df_merge], ignore_index=True
        )
        logger.debug(
            f"matches_df_all shape before drop_duplicates => {matches_df_all.shape}"
        )

        print("\nDEBUG: Final matches_df_all info:")
        print(f"  shape={matches_df_all.shape}")
        print(matches_df_all.head(15))

        matches_df_all = matches_df_all.drop_duplicates(subset=["id_net", "id_dlr"])
        logger.debug(
            f"matches_df_all shape after drop_duplicates => {matches_df_all.shape}"
        )
        logger.debug(f"Sample matches_df_all =>\n{matches_df_all.head(10)}")

        if "id_net" in matches_df_all.columns and "id_dlr" in matches_df_all.columns:
            matched_network_line_ids = set(matches_df_all["id_net"])
            matched_dlr_line_ids = set(matches_df_all["id_dlr"])
        else:
            logger.warning("No additional matches found in merging stage.")
            matched_network_line_ids = set(matches_df_bus["id_net"])
            matched_dlr_line_ids = set(matches_df_bus["id_dlr"])

        unmatched_network_lines = network_lines_gdf[
            ~network_lines_gdf["id"].isin(matched_network_line_ids)
        ].copy()
        unmatched_dlr_lines = dlr_lines_within_germany[
            ~dlr_lines_within_germany["id"].isin(matched_dlr_line_ids)
        ].copy()

        # 14) Print final match counts
        total_network_lines = len(network_lines_gdf)
        total_dlr_lines = len(dlr_lines_within_germany)
        logger.info(f"\nTotal number of network lines: {total_network_lines}")
        logger.info(f"Number of matched network lines: {len(matched_network_line_ids)}")
        logger.info(
            f"Number of unmatched network lines: {len(unmatched_network_lines)}"
        )

        logger.info(f"\nTotal number of DLR lines: {total_dlr_lines}")
        logger.info(f"Number of matched DLR lines: {len(matched_dlr_line_ids)}")
        logger.info(f"Number of unmatched DLR lines: {len(unmatched_dlr_lines)}")

        # Save unmatched lines
        if (
            "output_paths" in config
            and "unmatched_network_lines" in config["output_paths"]
        ):
            unmatched_network_lines.to_csv(
                config["output_paths"]["unmatched_network_lines"], index=False
            )
            logger.info(
                f"Unmatched network lines saved to '{config['output_paths']['unmatched_network_lines']}'."
            )
        else:
            unmatched_network_lines.to_csv(
                "results/csv/unmatched_network_lines.csv", index=False
            )
            logger.info(
                "Unmatched network lines saved to 'results/csv/unmatched_network_lines.csv'."
            )

        if "output_paths" in config and "unmatched_dlr_lines" in config["output_paths"]:
            unmatched_dlr_lines.to_csv(
                config["output_paths"]["unmatched_dlr_lines"], index=False
            )
            logger.info(
                f"Unmatched DLR lines saved to '{config['output_paths']['unmatched_dlr_lines']}'."
            )
        else:
            unmatched_dlr_lines.to_csv(
                "results/csv/unmatched_dlr_lines.csv", index=False
            )
            logger.info(
                "Unmatched DLR lines saved to 'results/csv/unmatched_dlr_lines.csv'."
            )

        # Combine matched lines for final usage
        matched_network_lines = network_lines_gdf[
            network_lines_gdf["id"].isin(matched_network_line_ids)
        ].copy()
        matched_dlr_lines = dlr_lines_within_germany[
            dlr_lines_within_germany["id"].isin(matched_dlr_line_ids)
        ].copy()

        # 15) Attribute allocation with streamlined approach
        # Check if DLR data has parameters
        has_parameters = check_dlr_parameters(dlr_lines_within_germany)

        if not has_parameters:
            logger.warning(
                "DLR data is missing or zero for electrical parameters. "
                "We will proceed but the allocations may be zeros."
            )

        print("Columns in final dlr_lines_gdf before allocation:")
        print(dlr_lines_gdf.columns)
        print(dlr_lines_gdf.head(5))
        print("dlr_lines_gdf columns:", dlr_lines_gdf.columns)
        print(
            "Sample of dlr_lines_gdf:\n",
            dlr_lines_gdf[["id", "r_dlr", "x_dlr", "b_dlr", "length_m"]].head(15),
        )

        print("\nUnique r_dlr:", dlr_lines_gdf["r_dlr"].unique())
        print("Unique x_dlr:", dlr_lines_gdf["x_dlr"].unique())
        print("Unique b_dlr:", dlr_lines_gdf["b_dlr"].unique())

        # Attempt to allocate attributes from DLR to network lines
        # try:
        network_lines_updated, allocated_matches = allocate_attributes_to_network_lines(
            matches_df_all, network_lines_gdf, dlr_lines_within_germany
        )

        # If the code gave us all zeros, just log a warning
        allocation_count = (network_lines_updated["r_allocated"] > 0).sum()
        if allocation_count == 0:
            logger.warning(
                "Allocation method produced only zero values. "
                "This could be due to missing or invalid DLR parameters."
            )

        # except Exception as e:
        #     logger.error(f"Allocation method failed: {str(e)}")
        #     logger.warning("Setting all allocations to zero.")
        #     network_lines_updated = network_lines_gdf.copy()
        #     # Provide minimal columns so code can continue
        #     network_lines_updated["r_allocated"] = 0
        #     network_lines_updated["x_allocated"] = 0
        #     network_lines_updated["b_allocated"] = 0
        #     network_lines_updated["r_total"] = network_lines_updated["r"]
        #     network_lines_updated["x_total"] = network_lines_updated["x"]
        #     network_lines_updated["b_total"] = network_lines_updated["b"]

        # Verify allocated columns exist
        for col in [
            "r_allocated",
            "x_allocated",
            "b_allocated",
            "r_total",
            "x_total",
            "b_total",
        ]:
            if col not in network_lines_updated.columns:
                logger.warning(f"Adding missing column '{col}' with zeros.")
                network_lines_updated[col] = 0

        # Validate parameter ranges
        is_valid = validate_parameter_ranges(network_lines_updated)
        if not is_valid:
            logger.warning("Some allocated parameters are outside typical ranges.")

        # 16) Save allocation results
        try:
            matched_network_lines_path = (
                "results/csv/matched_network_lines_with_allocated_attributes.csv"
            )
            allocation_summary_path = "results/csv/allocation_summary.csv"

            os.makedirs(os.path.dirname(matched_network_lines_path), exist_ok=True)

            network_lines_updated.to_csv(matched_network_lines_path, index=False)
            logger.info(
                f"Matched network lines with allocated attributes saved to '{matched_network_lines_path}'"
            )

            # Save just the allocation columns
            allocation_cols = [
                "id",
                "r",
                "x",
                "b",
                "r_allocated",
                "x_allocated",
                "b_allocated",
                "r_total",
                "x_total",
                "b_total",
            ]
            allocation_summary = network_lines_updated[allocation_cols]
            allocation_summary.to_csv(allocation_summary_path, index=False)
            logger.info(f"Allocation summary saved to '{allocation_summary_path}'")

            non_zero_allocations = (network_lines_updated["r_allocated"] > 0).sum()
            print(
                f"\nNon-zero allocations: {non_zero_allocations} of {len(network_lines_updated)} lines"
            )

            print(
                "\nSample of allocation values (first 10 rows with non-zero allocations):"
            )
            sample_allocations = network_lines_updated[
                network_lines_updated["r_allocated"] > 0
            ][allocation_cols].head(10)
            print(sample_allocations)

        except Exception as e:
            logger.error(f"Error saving allocation results: {str(e)}", exc_info=True)

        # Create the matches CSV
        create_matches_csv(matches_df_all, network_lines_updated, dlr_lines_gdf, config)

        # Optionally save matched lines with config path
        if (
            "output_paths" in config
            and "matched_network_lines_with_allocated_attributes"
            in config["output_paths"]
        ):
            allocated_cols = [
                "r_allocated",
                "x_allocated",
                "b_allocated",
                "r_total",
                "x_total",
                "b_total",
            ]
            missing_cols = [
                c for c in allocated_cols if c not in network_lines_updated.columns
            ]
            if missing_cols:
                logger.warning(f"Missing allocated attribute columns: {missing_cols}")
                logger.debug(
                    f"Available columns: {network_lines_updated.columns.tolist()}"
                )

            logger.debug(f"network_lines_updated shape: {network_lines_updated.shape}")

            for col in allocated_cols:
                if col in network_lines_updated.columns:
                    stats_min = network_lines_updated[col].min()
                    stats_max = network_lines_updated[col].max()
                    stats_mean = network_lines_updated[col].mean()
                    logger.debug(
                        f"{col}: min={stats_min}, max={stats_max}, mean={stats_mean}"
                    )

            config_path = config["output_paths"][
                "matched_network_lines_with_allocated_attributes"
            ]
            network_lines_updated.to_csv(config_path, index=False)
            logger.info(f"Also saved to config path: {config_path}")

        # Create unmatched lines map
        create_unmatched_lines_map(
            unmatched_network_lines, germany_gdf, network_buses_gdf, config
        )

        # Extract matched/unmatched buses
        matched_network_bus_ids = set(matched_network_lines["bus0"]).union(
            set(matched_network_lines["bus1"])
        )
        matched_network_buses_gdf = network_buses_gdf[
            network_buses_gdf["bus_idx"].isin(matched_network_bus_ids)
        ].copy()

        unmatched_network_bus_ids = set(unmatched_network_lines["bus0"]).union(
            set(unmatched_network_lines["bus1"])
        )
        unmatched_network_buses_gdf = network_buses_gdf[
            network_buses_gdf["bus_idx"].isin(unmatched_network_bus_ids)
        ].copy()

        matched_dlr_bus_ids = set(matched_dlr_lines["bus0"]).union(
            set(matched_dlr_lines["bus1"])
        )
        matched_dlr_buses_gdf = dlr_buses_gdf[
            dlr_buses_gdf["name"].isin(matched_dlr_bus_ids)
        ].copy()

        unmatched_dlr_bus_ids = set(unmatched_dlr_lines["bus0"]).union(
            set(unmatched_dlr_lines["bus1"])
        )
        unmatched_dlr_buses_gdf = dlr_buses_gdf[
            dlr_buses_gdf["name"].isin(unmatched_dlr_bus_ids)
        ].copy()

        # Combine matched & unmatched buses
        matched_buses_gdf = pd.concat(
            [
                matched_network_buses_gdf.rename(columns={"bus_idx": "bus_id"}),
                matched_dlr_buses_gdf.rename(columns={"name": "bus_id"}),
            ],
            ignore_index=True,
        )
        matched_buses_gdf["type"] = ["Network"] * len(matched_network_buses_gdf) + [
            "DLR"
        ] * len(matched_dlr_buses_gdf)

        unmatched_buses_gdf = pd.concat(
            [
                unmatched_network_buses_gdf.rename(columns={"bus_idx": "bus_id"}),
                unmatched_dlr_buses_gdf.rename(columns={"name": "bus_id"}),
            ],
            ignore_index=True,
        )
        unmatched_buses_gdf["type"] = ["Network"] * len(unmatched_network_buses_gdf) + [
            "DLR"
        ] * len(unmatched_dlr_buses_gdf)

        if "type" in matched_buses_gdf.columns:
            logger.info("'type' column exists in matched_buses_gdf.")
        else:
            logger.error("'type' column is missing in matched_buses_gdf.")

        if "type" in unmatched_buses_gdf.columns:
            logger.info("'type' column exists in unmatched_buses_gdf.")
        else:
            logger.error("'type' column is missing in unmatched_buses_gdf.")

        # Final Folium map with matched/unmatched lines and buses
        logger.info("\nGenerating map with matched and unmatched buses...")
        create_folium_map(
            germany_gdf,
            network_lines_gdf,
            dlr_lines_within_germany,
            matched_network_lines=matched_network_lines,
            matched_dlr_lines=matched_dlr_lines,
            matched_buses_gdf=matched_buses_gdf,
            unmatched_buses_gdf=unmatched_buses_gdf,
            config=config,
        )

        # Compute match statistics
        compute_match_statistics(matches_df_all, total_network_lines, total_dlr_lines)

    except Exception as e:
        logger.error("An error occurred:", exc_info=True)


if __name__ == "__main__":
    main()
