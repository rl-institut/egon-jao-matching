import argparse
import logging
import os
import traceback

import geopandas as gpd
import pandas as pd
import yaml

from egon_jao.bus_processing import (
    count_dlr_substations_in_buffers,
    create_filtered_substations_map,
    create_total_map,
    filter_buses_by_transformers,
    filter_dlr_substations_inside_germany,
    filter_network_substations,
    filter_network_substations_inside_germany,
    load_germany_boundary,
    prepare_geodataframes,
    read_data,
    save_csv_files,
    save_dlr_substations_count_csv,
    spatial_match_substations,
)


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Egon-Jao Substation Matching Script")
    parser.add_argument(
        "--config",
        type=str,
        default="config_sub.yaml",
        help="Path to the configuration YAML file.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_directories(config):
    directories = [
        config["data_dir"],
        config["results_dir"],
        config["csv_dir"],
        config["maps_dir"],
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        args = parse_arguments()
        config = load_config(args.config)
        ensure_directories(config)

        logger.info("Starting substation matching process.")

        # Read initial data
        network_buses_df, dlr_buses_df = read_data(config)

        # Rename 'name' to 'id_dlr' in dlr_buses_df if present
        if "name" in dlr_buses_df.columns:
            dlr_buses_df.rename(columns={"name": "id_dlr"}, inplace=True)
        else:
            logger.error(
                "'name' column not found in dlr_buses_df. Cannot rename to 'id_dlr'."
            )
            raise KeyError(
                "'name' column is missing. Ensure dlr_buses data has a 'name' column."
            )

        # Load Germany boundary
        germany_gdf = load_germany_boundary()

        # Filter by minimum voltage level
        min_voltage_level = 110
        filtered_network_buses_df = filter_network_substations(
            network_buses_df, min_voltage_level
        )

        # Filter buses by transformer presence
        network_transformers_file = config["data_files"]["network_transformers_file"]
        dlr_transformers_file = config["data_files"]["dlr_transformers_file"]

        filtered_network_buses_df, filtered_dlr_buses_df = filter_buses_by_transformers(
            filtered_network_buses_df,
            dlr_buses_df,
            network_transformers_file,
            dlr_transformers_file,
        )

        # Prepare GeoDataFrames
        substations_gdf, dlr_buses_gdf = prepare_geodataframes(
            filtered_network_buses_df, filtered_dlr_buses_df, config
        )

        # Filter inside Germany and exclude 110 kV or lower
        substations_gdf = filter_network_substations_inside_germany(
            substations_gdf, germany_gdf
        )

        # Filter DLR inside Germany
        dlr_buses_gdf = filter_dlr_substations_inside_germany(
            dlr_buses_gdf, germany_gdf
        )

        # Perform spatial matching
        (
            matched_substations_gdf,
            unmatched_substations_gdf,
            matched_dlr_buses_gdf,
            unmatched_dlr_buses_gdf,
        ) = spatial_match_substations(substations_gdf, dlr_buses_gdf, config)

        # Save matched and unmatched substations to CSV
        save_csv_files(
            matched_substations_gdf,
            unmatched_substations_gdf,
            unmatched_dlr_buses_gdf,
            config,
        )

        # Combine matched and unmatched network substations
        total_network_substations_gdf = pd.concat(
            [matched_substations_gdf, unmatched_substations_gdf], ignore_index=True
        )

        # Check if we have any substations at all
        if total_network_substations_gdf.empty:
            logger.warning(
                "No network substations remain after matching. Skipping DLR counting and mapping."
            )
            logger.info(
                "Substation matching process completed with no substations to display."
            )
            return

        # Count DLR substations in buffers
        dlr_counts_df = count_dlr_substations_in_buffers(
            network_substations_gdf=total_network_substations_gdf,
            dlr_buses_gdf=dlr_buses_gdf,
            config=config,
        )
        save_dlr_substations_count_csv(dlr_counts_df, config)

        # Re-read counts with 'has_dlr_without_T'
        output_path = config["output_files"].get(
            "dlr_substations_count_csv", "results/csv/dlr_substations_count.csv"
        )
        counts_df_with_has_dlr = pd.read_csv(output_path)

        # Merge DLR counts into total_network_substations_gdf
        # total_network_substations_gdf['bus_id'] = total_network_substations_gdf['bus_id'].astype(int)
        # counts_df_with_has_dlr['bus_id'] = counts_df_with_has_dlr['bus_id'].astype(int)
        # Ensure 'bus_id' is string (optional, since it's already a string)
        total_network_substations_gdf["bus_id"] = total_network_substations_gdf[
            "bus_id"
        ].astype(str)
        counts_df_with_has_dlr["bus_id"] = counts_df_with_has_dlr["bus_id"].astype(str)

        total_network_substations_gdf = total_network_substations_gdf.merge(
            counts_df_with_has_dlr, on="bus_id", how="left"
        )

        # After merging
        total_network_substations_gdf["num_dlr_substations"] = (
            total_network_substations_gdf["num_dlr_substations"].fillna(0).astype(int)
        )
        total_network_substations_gdf["has_dlr_without_T"] = (
            total_network_substations_gdf["has_dlr_without_T"].fillna(False)
        )

        # Filter substations with more than one DLR substation
        filtered_substations_gdf = total_network_substations_gdf[
            total_network_substations_gdf["num_dlr_substations"] > 1
        ].copy()

        # Reproject for mapping
        projected_crs = config["coordinate_reference_systems"]["projected_crs"]
        original_crs = config["coordinate_reference_systems"]["crs"]

        # Ensure these are GeoDataFrames
        matched_substations_gdf.set_crs(projected_crs, inplace=True)
        matched_dlr_buses_gdf.set_crs(projected_crs, inplace=True)
        unmatched_substations_gdf.set_crs(projected_crs, inplace=True)
        unmatched_dlr_buses_gdf.set_crs(projected_crs, inplace=True)

        filtered_substations_gdf = gpd.GeoDataFrame(
            filtered_substations_gdf, geometry="geometry", crs=projected_crs
        ).to_crs(original_crs)
        germany_gdf = germany_gdf.to_crs(original_crs)
        dlr_buses_gdf = dlr_buses_gdf.to_crs(original_crs)
        matched_dlr_buses_gdf = matched_dlr_buses_gdf.to_crs(original_crs)
        unmatched_dlr_buses_gdf = unmatched_dlr_buses_gdf.to_crs(original_crs)

        logger.debug(
            f"After filter_buses_by_transformers: {len(filtered_network_buses_df)}"
        )
        logger.debug(filtered_network_buses_df.head())

        # Create filtered substations map
        create_filtered_substations_map(
            network_substations_gdf=filtered_substations_gdf,
            dlr_buses_gdf=dlr_buses_gdf,
            config=config,
            germany_gdf=germany_gdf,
        )

        total_network_substations_gdf = gpd.GeoDataFrame(
            total_network_substations_gdf, geometry="geometry", crs=projected_crs
        ).to_crs(original_crs)
        total_network_substations_gdf = total_network_substations_gdf[
            total_network_substations_gdf["v_nom"] >= 110
        ]

        # Create total map
        create_total_map(
            network_substations_gdf=total_network_substations_gdf,
            matched_dlr_buses_gdf=matched_dlr_buses_gdf,
            unmatched_dlr_buses_gdf=unmatched_dlr_buses_gdf,
            config=config,
            germany_gdf=germany_gdf,
        )

        logger.info("Substation matching process completed.")

    except Exception as e:
        logger.error("An error occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
