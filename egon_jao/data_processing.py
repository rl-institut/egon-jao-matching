import logging
import re

import dill
import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import yaml

from folium.plugins import MeasureControl, MousePosition
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def prepare_dlr_geodataframes(dlr_buses, dlr_lines):
    """
    Prepares the DLR buses and lines GeoDataFrames.

    Parameters:
        dlr_buses (DataFrame): DataFrame of DLR buses.
        dlr_lines (DataFrame): DataFrame of DLR lines.

    Returns:
        tuple: (dlr_buses_gdf, dlr_lines_gdf)
    """
    # Convert DLR buses to GeoDataFrame
    dlr_buses_gdf = gpd.GeoDataFrame(
        dlr_buses,
        geometry=gpd.points_from_xy(dlr_buses["x"], dlr_buses["y"]),
        crs="EPSG:4326",
    )

    # Clean DLR bus IDs
    dlr_buses_gdf["name"] = dlr_buses_gdf["name"].apply(clean_dlr_bus_id)

    # Remove duplicates in DLR buses based on 'name' and 'geometry'
    before = len(dlr_buses_gdf)
    dlr_buses_gdf = dlr_buses_gdf.drop_duplicates(
        subset=["name", "geometry"], keep="first"
    )
    after = len(dlr_buses_gdf)
    logger.info(f"Removed {before - after} duplicate DLR buses.")

    # Validate and clean geometries in dlr_lines
    try:
        # Check if dlr_lines already has a geometry column
        if "geometry" in dlr_lines.columns:
            # If it's a string, convert it to actual geometry
            if dlr_lines["geometry"].dtype == "object":
                # Try to convert WKT strings to geometries
                try:
                    valid_geometries = []
                    for i, geom_str in enumerate(dlr_lines["geometry"]):
                        try:
                            if pd.isna(geom_str):
                                valid_geometries.append(None)
                            else:
                                # Try to parse and validate the geometry
                                geom = wkt.loads(str(geom_str))
                                if geom.is_valid:
                                    valid_geometries.append(geom)
                                else:
                                    # Try to fix invalid geometry
                                    fixed_geom = geom.buffer(0)
                                    if fixed_geom.is_valid:
                                        valid_geometries.append(fixed_geom)
                                    else:
                                        logger.warning(
                                            f"Could not fix invalid geometry at index {i}"
                                        )
                                        valid_geometries.append(None)
                        except Exception as e:
                            logger.warning(
                                f"Error parsing geometry at index {i}: {str(e)}"
                            )
                            valid_geometries.append(None)

                    # Replace the geometry column
                    dlr_lines["geometry"] = valid_geometries

                    # Drop rows with None geometries
                    before_drop = len(dlr_lines)
                    dlr_lines = dlr_lines.dropna(subset=["geometry"])
                    after_drop = len(dlr_lines)
                    if before_drop > after_drop:
                        logger.warning(
                            f"Dropped {before_drop - after_drop} rows with invalid geometries from dlr_lines"
                        )

                except Exception as e:
                    logger.error(
                        f"Error converting WKT strings to geometries: {str(e)}"
                    )
                    raise

        # Convert DLR lines to GeoDataFrame
        dlr_lines_gdf = gpd.GeoDataFrame(
            dlr_lines, geometry="geometry", crs="EPSG:4326"
        )

        # Final validation of geometries
        invalid_geoms = [not geom.is_valid for geom in dlr_lines_gdf.geometry]
        if any(invalid_geoms):
            logger.warning(
                f"Found {sum(invalid_geoms)} invalid geometries in dlr_lines_gdf. Attempting to fix..."
            )
            dlr_lines_gdf.geometry = dlr_lines_gdf.geometry.buffer(0)

    except Exception as e:
        logger.error(f"Error preparing DLR lines: {str(e)}")
        raise

    # Add 'name' column to DLR lines
    dlr_lines_gdf["name"] = dlr_lines_gdf["bus0"] + "_" + dlr_lines_gdf["bus1"]

    # Calculate length in meters
    dlr_lines_gdf.crs = "EPSG:4326"
    dlr_lines_gdf = dlr_lines_gdf.to_crs("EPSG:32632")
    dlr_lines_gdf["length_m"] = dlr_lines_gdf.geometry.length

    # **Rename 'r', 'x', 'b' to 'r_dlr', 'x_dlr', 'b_dlr'**
    dlr_lines_gdf = dlr_lines_gdf.rename(
        columns={"r": "r_dlr", "x": "x_dlr", "b": "b_dlr"}
    )

    # Debug existing IDs
    logger.debug("Sample of existing DLR line IDs:")
    logger.debug(dlr_lines_gdf["id"].head())
    logger.debug(f"DLR line ID type: {dlr_lines_gdf['id'].dtype}")

    return dlr_buses_gdf, dlr_lines_gdf


# Load country borders data
def load_country_borders():
    try:
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except AttributeError:
        return gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        )


def load_germany_boundary():
    """
    Loads Germany's boundary as a GeoDataFrame.

    Returns:
        GeoDataFrame: Germany's boundary with CRS EPSG:4326.
    """
    try:
        # URL to the Natural Earth low resolution dataset
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
    except Exception as e:
        logger.error("Error loading Natural Earth dataset:", exc_info=True)
        raise e

    # Adjust for correct country name column
    possible_name_columns = ["name", "NAME", "admin", "ADMIN"]
    country_col = None
    for col in possible_name_columns:
        if col in world.columns:
            country_col = col
            break
    if not country_col:
        raise KeyError("Country name column not found in world dataset.")

    # Filter for Germany and ensure CRS is WGS84
    germany_gdf = world[world[country_col] == "Germany"].to_crs("EPSG:4326")

    if germany_gdf.empty:
        raise ValueError("Germany boundary not found in the dataset.")

    logger.info("Germany boundary successfully loaded.")
    return germany_gdf


def filter_dlr_lines_inside_germany(dlr_lines_gdf, germany_gdf):
    """
    Filters DLR lines to include only those within Germany.

    Parameters:
        dlr_lines_gdf (GeoDataFrame): GeoDataFrame of DLR lines.
        germany_gdf (GeoDataFrame): GeoDataFrame of Germany's boundary.

    Returns:
        GeoDataFrame: Filtered DLR lines within Germany.
    """
    # Ensure both GeoDataFrames use the same CRS
    dlr_lines_gdf = dlr_lines_gdf.to_crs(germany_gdf.crs)

    # Spatial join to find DLR lines within Germany
    dlr_lines_within_germany = gpd.overlay(
        dlr_lines_gdf, germany_gdf, how="intersection"
    )

    # Reset index
    dlr_lines_within_germany = dlr_lines_within_germany.reset_index(drop=True)

    # Debugging: Verify filtered DLR lines
    print(f"Number of DLR lines within Germany: {len(dlr_lines_within_germany)}")

    return dlr_lines_within_germany


# Load the network data
def load_network(filepath):
    with open(filepath, "rb") as f:
        return dill.load(f)


# Function to load data
def load_data(config):
    # Load DLR buses and lines
    dlr_buses = pd.read_csv(config["data_paths"]["dlr_buses"])
    dlr_lines = pd.read_csv(config["data_paths"]["dlr_lines"])

    # Load Network buses and lines
    network_buses = pd.read_csv(config["data_paths"]["network_buses"])

    # Read 'geom' as string
    network_lines = pd.read_csv(
        config["data_paths"]["network_lines"], dtype={"geom": str}
    )

    return dlr_buses, dlr_lines, network_buses, network_lines


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Function to clean DLR bus IDs


def clean_dlr_bus_id(bus_id):
    # Convert to string first to avoid TypeError
    bus_id = str(bus_id)
    # Remove any non-alphanumeric characters except underscores
    cleaned_id = re.sub(r"[^\w]", "", bus_id).upper().strip()
    # Remove leading underscores
    cleaned_id = cleaned_id.lstrip("_")
    # Ensure it starts with 'BUS_'
    if not cleaned_id.startswith("BUS_"):
        # Assuming all bus IDs should start with 'BUS_', add it if missing
        cleaned_id = "BUS_" + cleaned_id.replace("BUS", "")
    # Remove trailing non-numeric characters (e.g., 'T')
    cleaned_id = re.sub(r"[^0-9_]", "", cleaned_id)
    return cleaned_id


# Function to prepare Network lines GeoDataFrame and assign 'id'


def prepare_network_lines_geodataframe(network_lines):
    # Filter out lines (exclude 110 kV)
    network_lines_filtered = network_lines[network_lines["v_nom"] != 110].copy()

    # Ensure the 'geom' column exists
    if "geom" not in network_lines_filtered.columns:
        raise ValueError(
            "The 'network-lines.csv' file does not contain a 'geom' column."
        )

    # Convert the WKT in 'geom' to actual geometry with validation
    valid_geometries = []
    for i, geom_str in enumerate(network_lines_filtered["geom"]):
        try:
            if pd.isna(geom_str):
                valid_geometries.append(None)
            else:
                # Try to parse and validate the geometry
                geom = wkt.loads(str(geom_str))
                if geom.is_valid:
                    valid_geometries.append(geom)
                else:
                    # Try to fix invalid geometry
                    fixed_geom = geom.buffer(0)
                    if fixed_geom.is_valid:
                        valid_geometries.append(fixed_geom)
                    else:
                        logger.warning(f"Could not fix invalid geometry at index {i}")
                        valid_geometries.append(None)
        except Exception as e:
            logger.warning(f"Error parsing geometry at index {i}: {str(e)}")
            valid_geometries.append(None)

    # Replace the geometry column
    network_lines_filtered["geometry"] = valid_geometries

    # Drop rows with None geometries
    before_drop = len(network_lines_filtered)
    network_lines_filtered = network_lines_filtered.dropna(subset=["geometry"])
    after_drop = len(network_lines_filtered)
    if before_drop > after_drop:
        logger.warning(
            f"Dropped {before_drop - after_drop} rows with invalid geometries from network_lines"
        )

    # Now construct the actual GeoDataFrame with the proper CRS
    network_lines_gdf = gpd.GeoDataFrame(
        network_lines_filtered, geometry="geometry", crs="EPSG:4326"
    )

    # Final validation of geometries
    invalid_geoms = [not geom.is_valid for geom in network_lines_gdf.geometry]
    if any(invalid_geoms):
        logger.warning(
            f"Found {sum(invalid_geoms)} invalid geometries in network_lines_gdf. Attempting to fix..."
        )
        network_lines_gdf.geometry = network_lines_gdf.geometry.buffer(0)

    # Project to a metric CRS to calculate lengths
    network_lines_gdf_proj = network_lines_gdf.to_crs("EPSG:32632")
    network_lines_gdf["length_m"] = network_lines_gdf_proj.geometry.length

    # (Optional) Standardize bus IDs
    network_lines_gdf["bus0"] = (
        network_lines_gdf["bus0"].astype(str).str.strip().str.upper()
    )
    network_lines_gdf["bus1"] = (
        network_lines_gdf["bus1"].astype(str).str.strip().str.upper()
    )

    # Debug output
    print("Network Lines GeoDataFrame Columns:", network_lines_gdf.columns.tolist())
    print("Sample Network Lines with 'id':")
    print(network_lines_gdf[["id", "bus0", "bus1", "length_m"]].head())

    return network_lines_gdf


def fix_invalid_geometries(gdf):
    """
    Fix invalid geometries in a GeoDataFrame using buffer(0) technique.

    Parameters:
        gdf (GeoDataFrame): GeoDataFrame with potentially invalid geometries

    Returns:
        GeoDataFrame: The same GeoDataFrame with fixed geometries
    """
    # Check for invalid geometries
    invalid_mask = ~gdf.geometry.is_valid
    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        logger.warning(
            f"Found {invalid_count} invalid geometries. Attempting to fix..."
        )

        # Create a copy to avoid modifying the original during iteration
        gdf_fixed = gdf.copy()

        # Fix invalid geometries
        gdf_fixed.loc[invalid_mask, "geometry"] = gdf_fixed.loc[
            invalid_mask, "geometry"
        ].buffer(0)

        # Verify fix
        still_invalid = ~gdf_fixed.geometry.is_valid
        still_invalid_count = still_invalid.sum()

        if still_invalid_count > 0:
            logger.warning(f"Could not fix {still_invalid_count} geometries.")
            # Drop rows with still-invalid geometries
            gdf_fixed = gdf_fixed[~still_invalid]
        else:
            logger.info(f"Successfully fixed all {invalid_count} invalid geometries.")

        return gdf_fixed
    else:
        return gdf


# Function to extract buses from network lines
def extract_buses_from_lines(network_lines_gdf):
    bus_records = []

    for idx, row in network_lines_gdf.iterrows():
        line_geom = row["geometry"]
        bus0_id = row["bus0"]
        bus1_id = row["bus1"]

        # Get the start and end points of the line
        if isinstance(line_geom, LineString):
            coords = list(line_geom.coords)
        elif isinstance(line_geom, MultiLineString):
            coords = [pt for line in line_geom.geoms for pt in line.coords]
        else:
            print(
                f"Skipping geometry at index {idx} as it is neither LineString nor MultiLineString."
            )
            continue  # Skip if geometry is not LineString or MultiLineString

        # Extract start and end points
        start_point = Point(coords[0])
        end_point = Point(coords[-1])

        # Add to bus records with 'bus_idx' as the column name
        bus_records.append({"bus_idx": bus0_id, "geometry": start_point})
        bus_records.append({"bus_idx": bus1_id, "geometry": end_point})

    # Create a DataFrame and remove duplicates
    bus_df = pd.DataFrame(bus_records).drop_duplicates(subset="bus_idx")
    bus_gdf = gpd.GeoDataFrame(bus_df, geometry="geometry", crs="EPSG:4326")

    # Debugging: Verify buses extraction
    print("Extracted Network Buses GeoDataFrame:")
    print(bus_gdf.head())

    return bus_gdf


# Function to validate GeoDataFrames for duplicates
def validate_geodataframes(network_buses_gdf, dlr_buses_gdf, dlr_lines_gdf):
    # Check for duplicate bus_idx in network buses
    duplicate_network_buses = network_buses_gdf[
        network_buses_gdf.duplicated(subset="bus_idx", keep=False)
    ]
    if not duplicate_network_buses.empty:
        logger.warning("Duplicate bus_idx found in network_buses_gdf:")
        logger.warning(duplicate_network_buses)
    else:
        logger.info("No duplicate bus_idx found in network_buses_gdf.")

    # Check for duplicate bus names in DLR buses
    duplicate_dlr_buses = dlr_buses_gdf[
        dlr_buses_gdf.duplicated(subset="name", keep=False)
    ]
    if not duplicate_dlr_buses.empty:
        logger.warning("Duplicate bus names found in dlr_buses_gdf:")
        logger.warning(duplicate_dlr_buses)
    else:
        logger.info("No duplicate bus names found in dlr_buses_gdf.")

    # Check for required columns in dlr_lines_gdf
    required_columns = ["r_dlr", "x_dlr", "b_dlr"]
    missing_columns = [
        col for col in required_columns if col not in dlr_lines_gdf.columns
    ]
    if missing_columns:
        logger.error(f"Missing columns in dlr_lines_gdf: {missing_columns}")
        raise KeyError(f"Missing columns in dlr_lines_gdf: {missing_columns}")
    else:
        logger.info("All required columns in dlr_lines_gdf are present.")


# Function to match buses using nearest neighbor allowing multiple matches


def match_buses_with_nearest_multiple(network_buses_gdf, dlr_buses_gdf, config):
    # Access max_distance from config
    max_distance = config["parameters"]["max_distance"]

    # Debugging: Print max_distance
    print(f"Debug: max_distance = {max_distance} (type: {type(max_distance)})")

    if not isinstance(max_distance, (int, float)):
        raise TypeError(
            f"max_distance must be a numeric type, got {type(max_distance)} instead."
        )

    # Project GeoDataFrames to a metric CRS
    projected_crs = "EPSG:32632"  # UTM zone suitable for Germany

    # Reproject to projected CRS
    network_buses_proj = network_buses_gdf.to_crs(projected_crs)
    dlr_buses_proj = dlr_buses_gdf.to_crs(projected_crs)

    # Perform nearest neighbor spatial join allowing multiple matches within max_distance
    matched_buses = gpd.sjoin_nearest(
        network_buses_proj[["bus_idx", "geometry"]],
        dlr_buses_proj[["name", "geometry"]],
        how="left",
        max_distance=max_distance,
        distance_col="distance",
    )

    # Drop rows where no nearest neighbor was found within max_distance
    matched_buses = matched_buses.dropna(subset=["index_right"]).reset_index(drop=True)

    # Create the bus_id_mapping GeoDataFrame
    bus_id_mapping = gpd.GeoDataFrame(
        matched_buses[["bus_idx", "name"]],
        geometry=matched_buses["geometry"],
        crs=projected_crs,
    )

    # Rename columns for clarity
    bus_id_mapping = bus_id_mapping.rename(columns={"name": "dlr_bus_name"})

    # Reproject back to WGS84
    bus_id_mapping = bus_id_mapping.to_crs("EPSG:4326")

    # Debugging: Print sample mappings
    print("Sample bus mappings (Multiple Nearest Neighbor):")
    print(bus_id_mapping.head())

    return bus_id_mapping


# Function to update network lines with matched DLR buses
def update_network_lines_with_matched_buses(network_lines_gdf, bus_id_mapping):
    # Merge the bus_id_mapping with network lines for bus0
    network_lines_gdf = network_lines_gdf.merge(
        bus_id_mapping.rename(columns={"bus_idx": "bus0", "dlr_bus_name": "dlr_bus0"}),
        on="bus0",
        how="left",
    ).merge(
        bus_id_mapping.rename(columns={"bus_idx": "bus1", "dlr_bus_name": "dlr_bus1"}),
        on="bus1",
        how="left",
    )

    # Drop lines where either end does not have a matched DLR bus
    network_lines_matched = network_lines_gdf.dropna(
        subset=["dlr_bus0", "dlr_bus1"]
    ).copy()

    # Debugging: Print sample of matched network lines
    print("Sample of network lines with matched DLR buses:")
    print(network_lines_matched[["id", "bus0", "bus1", "dlr_bus0", "dlr_bus1"]].head())

    return network_lines_matched


# Function to match lines based on matched buses allowing multiple matches


def match_lines_based_on_matched_buses(network_lines_matched, dlr_lines_within_germany):
    """
    Matches network lines with DLR lines based on matched buses.

    Parameters:
        network_lines_matched (GeoDataFrame): Matched network lines.
        dlr_lines_within_germany (GeoDataFrame): DLR lines within Germany.

    Returns:
        tuple: (matched_network_lines, matched_dlr_lines, matches_df, matches)
    """
    # Create a unique bus pair identifier for network lines
    network_lines_matched["bus_pair"] = network_lines_matched.apply(
        lambda row: tuple(sorted([row["dlr_bus0"], row["dlr_bus1"]])), axis=1
    )

    # Create a unique bus pair identifier for DLR lines
    dlr_lines_within_germany["bus_pair"] = dlr_lines_within_germany.apply(
        lambda row: tuple(sorted([row["bus0"], row["bus1"]])), axis=1
    )

    # Merge to find matching bus pairs
    matches = network_lines_matched.merge(
        dlr_lines_within_germany, on="bus_pair", how="inner", suffixes=("_net", "_dlr")
    )

    # Check for required columns
    required_columns = {"length_m", "length_m_dlr"}
    missing_columns = required_columns - set(matches.columns)

    if missing_columns:
        logger.warning(f"Missing columns in matches: {missing_columns}")
        # Use geometry to calculate lengths if not present
        if "length_m" not in matches.columns and "geometry_net" in matches.columns:
            matches_proj = matches.copy()
            matches_proj.geometry = matches_proj.geometry_net
            matches_proj = matches_proj.to_crs("EPSG:32632")
            matches["length_m"] = matches_proj.geometry.length

        if "length_m_dlr" not in matches.columns and "geometry_dlr" in matches.columns:
            matches_proj = matches.copy()
            matches_proj.geometry = matches_proj.geometry_dlr
            matches_proj = matches_proj.to_crs("EPSG:32632")
            matches["length_m_dlr"] = matches_proj.geometry.length

    # Filter matches based on length ratio if both lengths are available
    if "length_m" in matches.columns and "length_m_dlr" in matches.columns:
        matches = matches[
            (matches["length_m"] / matches["length_m_dlr"]).between(0.5, 2)
        ]

    print(f"DEBUG: After length-ratio filter, matches has {len(matches)} rows")

    # Select relevant columns including 'bus0_dlr' and 'bus1_dlr'
    matches_df = matches[["id_net", "id_dlr", "bus0_dlr", "bus1_dlr"]].copy()

    # Add a unique match identifier
    matches_df["match_id"] = (
        matches_df.groupby(["id_net", "id_dlr"]).ngroup().astype(str)
    )

    # Get matched network and DLR lines
    matched_network_lines = network_lines_matched[
        network_lines_matched["id"].isin(matches_df["id_net"])
    ]
    matched_dlr_lines = dlr_lines_within_germany[
        dlr_lines_within_germany["id"].isin(matches_df["id_dlr"])
    ]

    logger.info(f"Number of matches found: {len(matches_df)}")
    logger.debug(f"Sample matches:\n{matches_df.head()}")

    return matched_network_lines, matched_dlr_lines, matches_df, matches


# Function to match lines using buffer method
def match_lines_with_buffer(unmatched_network_lines_gdf, dlr_lines_gdf, config):
    """
    Match lines using a buffer method based on configuration parameters.

    Parameters:
    - unmatched_network_lines_gdf (GeoDataFrame): GeoDataFrame of unmatched network lines.
    - dlr_lines_gdf (GeoDataFrame): GeoDataFrame of unmatched DLR lines.
    - config (dict): Configuration dictionary containing parameters.

    Returns:
    - tuple: Additional matched network lines, additional matched DLR lines, matches DataFrame.
    """
    buffer_distance = config["parameters"]["buffer_distance"]

    # Debugging: Print buffer_distance
    print(f"Debug: buffer_distance = {buffer_distance} (type: {type(buffer_distance)})")

    if not isinstance(buffer_distance, (int, float)):
        raise TypeError(
            f"buffer_distance must be a numeric type, got {type(buffer_distance)} instead."
        )

    # Project to a suitable metric CRS (e.g., UTM zone 32N)
    projected_crs = "EPSG:32632"
    unmatched_network_lines_proj = unmatched_network_lines_gdf.to_crs(
        projected_crs
    ).copy()
    dlr_lines_proj = dlr_lines_gdf.to_crs(projected_crs).copy()

    # Create buffer around network lines
    unmatched_network_lines_proj["buffer"] = (
        unmatched_network_lines_proj.geometry.buffer(buffer_distance)
    )

    # Spatial join: find DLR lines within buffer
    buffer_gdf = unmatched_network_lines_proj[["id", "buffer"]].copy()
    buffer_gdf = buffer_gdf.set_geometry("buffer")

    # Perform spatial join
    matches = gpd.sjoin(dlr_lines_proj, buffer_gdf, how="inner", predicate="intersects")

    # Process matches to get unique pairs
    matches_df = matches[["id_right", "id_left"]].rename(
        columns={"id_right": "id_net", "id_left": "id_dlr"}
    )

    # Remove duplicates
    matches_df = matches_df.drop_duplicates()

    # Convert to lists for further processing
    additional_matched_network_lines = unmatched_network_lines_gdf[
        unmatched_network_lines_gdf["id"].isin(matches_df["id_net"])
    ].copy()
    additional_matched_dlr_lines = dlr_lines_gdf[
        dlr_lines_gdf["id"].isin(matches_df["id_dlr"])
    ].copy()

    # Create a DataFrame for matches from buffer
    matches_df_buffer = matches_df.copy()
    matches_df_buffer["match_id"] = (
        matches_df_buffer.index + 1
    )  # Assign unique match_id

    matches_df_buffer["match_type"] = "buffer"

    # Debugging: Verify additional matches
    print(
        f"Additional Matched Network Lines via Buffer: {len(additional_matched_network_lines)}"
    )
    print(
        f"Additional Matched DLR Lines via Buffer: {len(additional_matched_dlr_lines)}"
    )

    return (
        additional_matched_network_lines,
        additional_matched_dlr_lines,
        matches_df_buffer,
    )


# Function to compute match statistics
def compute_match_statistics(matches_df_all, total_network_lines, total_dlr_lines):
    # Create mappings
    dlr_to_network_map = matches_df_all.groupby("id_dlr")["id_net"].apply(set).to_dict()
    network_to_dlr_map = matches_df_all.groupby("id_net")["id_dlr"].apply(set).to_dict()

    # 1. One-to-One Matches
    one_to_one_matches = {
        dlr_id: net_ids
        for dlr_id, net_ids in dlr_to_network_map.items()
        if len(net_ids) == 1
    }
    num_one_to_one = len(one_to_one_matches)
    print(
        f"Number of One-to-One Matches (1 DLR line to 1 Network line): {num_one_to_one}"
    )

    # 2. One-to-Many Matches
    one_to_many_matches = {
        dlr_id: net_ids
        for dlr_id, net_ids in dlr_to_network_map.items()
        if len(net_ids) > 1
    }
    num_one_to_many = len(one_to_many_matches)
    print(
        f"Number of One-to-Many Matches (1 DLR line to multiple Network lines): {num_one_to_many}"
    )

    # 3. Many-to-One Matches
    many_to_one_matches = {
        net_id: dlr_ids
        for net_id, dlr_ids in network_to_dlr_map.items()
        if len(dlr_ids) > 1
    }
    num_many_to_one = len(many_to_one_matches)
    print(
        f"Number of Many-to-One Matches (Multiple DLR lines to 1 Network line): {num_many_to_one}"
    )

    # 4. Missing Lines (Network lines without DLR matches)
    matched_network_line_ids = set(matches_df_all["id_net"])
    num_missing = total_network_lines - len(matched_network_line_ids)
    print(f"Number of Missing Lines (Network lines without DLR matches): {num_missing}")

    # 5. Extra Lines (DLR lines without Network matches)
    matched_dlr_line_ids = set(matches_df_all["id_dlr"])
    num_extra = total_dlr_lines - len(matched_dlr_line_ids)
    print(f"Number of Extra Lines (DLR lines without Network matches): {num_extra}")


# Function to merge connected unmatched network lines


def merge_connected_unmatched_network_lines(unmatched_network_lines_gdf):
    """
    Merge connected unmatched network lines into single MultiLineString geometries.

    Parameters:
    - unmatched_network_lines_gdf (GeoDataFrame): GeoDataFrame of unmatched network lines.

    Returns:
    - GeoDataFrame: GeoDataFrame of merged network lines preserving original IDs
    """
    # Create a graph where nodes are buses and edges are lines
    g = nx.Graph()
    for idx, row in unmatched_network_lines_gdf.iterrows():
        # Add both geometry and id as attributes to each edge
        g.add_edge(row["bus0"], row["bus1"], geometry=row["geometry"], id=row["id"])

    # Identify connected components
    connected_components = list(nx.connected_components(g))
    logger.info(
        f"Number of connected components in unmatched network lines: {len(connected_components)}"
    )

    merged_lines = []
    for component in connected_components:
        # Get all edges (lines) in the component
        subgraph = g.subgraph(component)

        # Get geometries and IDs from the subgraph
        geometries = []
        line_id = None  # We'll use the first ID from the component

        for u, v, data in subgraph.edges(data=True):
            if line_id is None:
                line_id = data["id"]  # Keep the first ID we encounter
            if data["geometry"] is not None:
                if data["geometry"].geom_type == "LineString":
                    geometries.append(data["geometry"])
                elif data["geometry"].geom_type == "MultiLineString":
                    geometries.extend(data["geometry"].geoms)

        # Merge the lines if we have geometries
        if geometries:
            merged_geom = linemerge(geometries)
            if merged_geom.geom_type in ["LineString", "MultiLineString"]:
                merged_lines.append(
                    {"id": line_id, "geometry": merged_geom}  # Use the preserved ID
                )
                logger.debug(
                    f"Merged component with ID {line_id}: {len(geometries)} lines merged into a {merged_geom.geom_type}"
                )
            else:
                logger.warning(
                    f"Unexpected geometry type after merging: {merged_geom.geom_type}"
                )

    # Create a GeoDataFrame of merged lines
    if merged_lines:
        merged_unmatched_network_lines = gpd.GeoDataFrame(
            merged_lines, geometry="geometry", crs=unmatched_network_lines_gdf.crs
        )
    else:
        merged_unmatched_network_lines = gpd.GeoDataFrame(
            columns=["id", "geometry"], crs=unmatched_network_lines_gdf.crs
        )
        logger.info("No merged lines to create. Returning empty GeoDataFrame.")

    return merged_unmatched_network_lines


# Function to match merged lines to DLR lines
def match_merged_lines_to_dlr(merged_lines_gdf, unmatched_dlr_lines_gdf, config):
    """
    Match merged network lines to DLR lines using spatial buffer.

    Parameters:
    - merged_lines_gdf (GeoDataFrame): GeoDataFrame of merged network lines.
    - unmatched_dlr_lines_gdf (GeoDataFrame): GeoDataFrame of unmatched DLR lines.
    - config (dict): Configuration dictionary containing parameters.

    Returns:
    - tuple: Additional matched network lines, additional matched DLR lines, matches DataFrame.
    """
    # Debugging: Check columns
    print(f"Debug: Columns in merged_lines_gdf: {merged_lines_gdf.columns.tolist()}")

    # Access max_distance from config
    max_distance = config["parameters"]["max_distance"]

    # Existing debug statement
    print(f"Debug: max_distance = {max_distance} (type: {type(max_distance)})")

    if not isinstance(max_distance, (int, float)):
        raise TypeError(
            f"max_distance must be a numeric type, got {type(max_distance)} instead."
        )

    # Project to a suitable metric CRS (e.g., UTM zone 32N)
    projected_crs = "EPSG:32632"
    merged_lines_proj = merged_lines_gdf.to_crs(projected_crs).copy()
    unmatched_dlr_lines_proj = unmatched_dlr_lines_gdf.to_crs(projected_crs).copy()

    # Create spatial index for DLR lines
    dlr_sindex = unmatched_dlr_lines_proj.sindex

    additional_matched_network_lines = []
    additional_matched_dlr_lines = []
    matches_df_merge = []

    for idx, merged_line_row in merged_lines_proj.iterrows():
        merged_line_geom = merged_line_row["geometry"]

        # Debugging: Check geometry type and 'id'
        print(
            f"Debug: Processing merged_line_id={merged_line_row.get('id', 'N/A')} with geometry type={merged_line_geom.geom_type}"
        )

        if "id" not in merged_line_row:
            print(
                f"Warning: 'id' not found in merged_line_row at index {idx}. Assigning new 'id'."
            )
            merged_line_id = str(idx + 1)  # Assign a new ID based on index
        else:
            merged_line_id = merged_line_row["id"]

        # Buffer the merged line
        merged_line_buffer = merged_line_geom.buffer(max_distance)

        # Find DLR lines that intersect with the buffer
        possible_dlr_indices = list(dlr_sindex.intersection(merged_line_buffer.bounds))
        possible_dlr_lines = unmatched_dlr_lines_proj.iloc[possible_dlr_indices]

        # Further filter DLR lines that actually intersect
        possible_dlr_lines = possible_dlr_lines[
            possible_dlr_lines.intersects(merged_line_buffer)
        ]
        if not possible_dlr_lines.empty:
            # Choose the best match (e.g., with maximum overlapping length)
            possible_dlr_lines = possible_dlr_lines.copy()
            possible_dlr_lines["overlap_length"] = (
                possible_dlr_lines.geometry.intersection(merged_line_geom).length
            )
            best_match = possible_dlr_lines.loc[
                possible_dlr_lines["overlap_length"].idxmax()
            ]

            additional_matched_network_lines.append(merged_line_row)
            additional_matched_dlr_lines.append(best_match)
            matches_df_merge.append(
                {
                    "id_net": merged_line_id,
                    "id_dlr": best_match["id"],
                    "match_id": f"merge_{idx}",
                }
            )
            print(
                f"Matched merged_line_id={merged_line_id} with DLR line_id={best_match['id']}"
            )
        else:
            print(
                f"No DLR lines found within buffer for merged_line_id={merged_line_id}"
            )

    # Convert lists to GeoDataFrames
    if additional_matched_network_lines:
        additional_matched_network_lines_gdf = (
            gpd.GeoDataFrame(
                additional_matched_network_lines, geometry="geometry", crs=projected_crs
            )
            .to_crs("EPSG:4326")
            .reset_index(drop=True)
        )
    else:
        additional_matched_network_lines_gdf = gpd.GeoDataFrame(
            columns=merged_lines_gdf.columns
        )

    if additional_matched_dlr_lines:
        additional_matched_dlr_lines_gdf = (
            gpd.GeoDataFrame(
                additional_matched_dlr_lines, geometry="geometry", crs=projected_crs
            )
            .to_crs("EPSG:4326")
            .reset_index(drop=True)
        )
    else:
        additional_matched_dlr_lines_gdf = gpd.GeoDataFrame(
            columns=unmatched_dlr_lines_gdf.columns
        )

    # Create a DataFrame for matches from merging with predefined columns
    matches_df_merge = pd.DataFrame(
        matches_df_merge, columns=["id_net", "id_dlr", "match_id"]
    )

    matches_df_merge["match_type"] = "merged"

    # Debugging: Verify additional matches
    print(
        f"Additional Matched Network Lines via Merging: {len(additional_matched_network_lines_gdf)}"
    )
    print(
        f"Additional Matched DLR Lines via Merging: {len(additional_matched_dlr_lines_gdf)}"
    )

    return (
        additional_matched_network_lines_gdf,
        additional_matched_dlr_lines_gdf,
        matches_df_merge,
    )


def create_folium_map(
    germany_gdf,
    network_lines_gdf,
    dlr_lines_gdf,
    matched_network_lines=None,
    matched_dlr_lines=None,
    matched_buses_gdf=None,
    unmatched_buses_gdf=None,
    config=None,
):

    # Initialize the map
    folium_map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)

    # Add Germany boundary
    folium.GeoJson(
        germany_gdf,
        name="Germany",
        style_function=lambda x: {
            "fillColor": "#00000000",
            "color": "black",
            "weight": 1,
        },
        show=True,
    ).add_to(folium_map)

    # Add Unmatched Network Lines
    if matched_network_lines is not None:
        unmatched_network_lines = network_lines_gdf[
            ~network_lines_gdf["id"].isin(matched_network_lines["id"])
        ].copy()
    else:
        unmatched_network_lines = network_lines_gdf.copy()

    folium.GeoJson(
        unmatched_network_lines[["geometry"]].to_crs("EPSG:4326"),
        name="Unmatched Network Lines",
        style_function=lambda x: {"color": "blue", "weight": 1},
        show=True,
    ).add_to(folium_map)

    # Add Matched Network Lines
    if matched_network_lines is not None and not matched_network_lines.empty:
        folium.GeoJson(
            matched_network_lines[["geometry"]].to_crs("EPSG:4326"),
            name="Matched Network Lines",
            style_function=lambda x: {"color": "green", "weight": 2},
            show=True,
        ).add_to(folium_map)
        print(f"Number of Matched Network Lines: {len(matched_network_lines)}")
    else:
        print("No Matched Network Lines to plot.")

    # Add Unmatched DLR Lines
    if matched_dlr_lines is not None:
        unmatched_dlr_lines = dlr_lines_gdf[
            ~dlr_lines_gdf["id"].isin(matched_dlr_lines["id"])
        ].copy()
    else:
        unmatched_dlr_lines = dlr_lines_gdf.copy()

    folium.GeoJson(
        unmatched_dlr_lines[["geometry"]].to_crs("EPSG:4326"),
        name="Unmatched DLR Lines",
        style_function=lambda x: {"color": "red", "weight": 1},
        show=True,
    ).add_to(folium_map)

    # Add Matched DLR Lines
    if matched_dlr_lines is not None and not matched_dlr_lines.empty:
        folium.GeoJson(
            matched_dlr_lines[["geometry"]].to_crs("EPSG:4326"),
            name="Matched DLR Lines",
            style_function=lambda x: {"color": "black", "weight": 2},
            show=True,
        ).add_to(folium_map)
        print(f"Number of Matched DLR Lines: {len(matched_dlr_lines)}")
    else:
        print("No Matched DLR Lines to plot.")

    # Plot Matched Buses
    if matched_buses_gdf is not None and not matched_buses_gdf.empty:
        folium.FeatureGroup(name="Matched Buses").add_to(folium_map)
        for _, row in matched_buses_gdf.iterrows():
            # Option 1: Adjust CircleMarkers
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=2,  # Reduced radius
                color="green" if row["type"] == "Network" else "black",
                fill=True,
                fill_color="green" if row["type"] == "Network" else "black",
                fill_opacity=0.5,  # Reduced opacity
                tooltip=f"Matched Bus: {row['bus_id']} ({row['type']})",
            ).add_to(folium_map)

    else:
        print("No matched buses to plot.")

    # Plot Unmatched Buses
    if unmatched_buses_gdf is not None and not unmatched_buses_gdf.empty:
        folium.FeatureGroup(name="Unmatched Buses").add_to(folium_map)
        for _, row in unmatched_buses_gdf.iterrows():
            # Option 1: Adjust CircleMarkers
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=2,  # Reduced radius
                color="blue" if row["type"] == "Network" else "red",
                fill=True,
                fill_color="blue" if row["type"] == "Network" else "red",
                fill_opacity=0.5,  # Reduced opacity
                tooltip=f"Unmatched Bus: {row['bus_id']} ({row['type']})",
            ).add_to(folium_map)

    else:
        print("No unmatched buses to plot.")

    # Combine matched and unmatched buses for search, with a combined field
    if matched_buses_gdf is not None and unmatched_buses_gdf is not None:
        buses_search_gdf = pd.concat(
            [
                matched_buses_gdf.rename(columns={"bus_id": "search_name"}),
                unmatched_buses_gdf.rename(columns={"bus_id": "search_name"}),
            ],
            ignore_index=True,
        )
    elif matched_buses_gdf is not None:
        buses_search_gdf = matched_buses_gdf.rename(columns={"bus_id": "search_name"})
    elif unmatched_buses_gdf is not None:
        buses_search_gdf = unmatched_buses_gdf.rename(columns={"bus_id": "search_name"})
    else:
        buses_search_gdf = pd.DataFrame()

    if not buses_search_gdf.empty:
        # Create a GeoJson layer for buses to enable search
        buses_geojson = folium.GeoJson(
            buses_search_gdf,
            name="Buses for Search",
            style_function=lambda x: {"color": "transparent", "weight": 0},
            tooltip=folium.GeoJsonTooltip(
                fields=["search_name", "type"],
                aliases=["Bus ID:", "Type:"],
                localize=True,
            ),
        ).add_to(folium_map)

    # --------------------- End of Search Functionality ---------------------

    # --------------------- Add MeasureControl ---------------------

    # Add MeasureControl
    folium_map.add_child(
        MeasureControl(
            position="topleft",
            primary_length_unit="meters",
            secondary_length_unit="kilometers",
            primary_area_unit="sqmeters",
            secondary_area_unit="hectares",
            active_color="orange",
            completed_color="red",
        )
    )
    logger.info("MeasureControl added to the map.")

    # Add MousePosition
    formatter = "function(num) {return L.Util.formatNum(num, 5);};"
    mouse_position = MousePosition(
        position="bottomright",
        separator=" | ",
        empty_string="NaN",
        lng_first=True,
        num_digits=5,
        prefix="Coordinates:",
        lat_formatter=formatter,
        lng_formatter=formatter,
    )
    folium_map.add_child(mouse_position)
    logger.info("MousePosition added to the map.")

    # --------------------- End of MeasureControl ---------------------

    # Add Layer Control
    folium.LayerControl().add_to(folium_map)

    # Save the map
    if config and "map_paths" in config and "full_map" in config["map_paths"]:
        full_map_path = config["map_paths"]["full_map"]
    else:
        full_map_path = "full_map2.html"
    folium_map.save(full_map_path)
    print(f"Map with all elements generated and saved as '{full_map_path}'.")


# Function to create matched lines CSV
def create_matches_csv(matches_df_all, network_lines_updated, dlr_lines_gdf, config):
    # Check if allocated attribute columns exist
    allocated_cols = [
        "r_allocated",
        "x_allocated",
        "b_allocated",
        "r_total",
        "x_total",
        "b_total",
    ]
    missing_cols = [
        col for col in allocated_cols if col not in network_lines_updated.columns
    ]

    if missing_cols:
        logger.warning(
            f"Missing allocated attribute columns in network_lines_updated: {missing_cols}"
        )
        logger.debug(f"Available columns: {network_lines_updated.columns.tolist()}")

    # Check if length_m_dlr exists in dlr_lines_gdf, if not, create it
    if "length_m_dlr" not in dlr_lines_gdf.columns:
        if "length" in dlr_lines_gdf.columns:
            logger.debug(
                "'length_m_dlr' not found in dlr_lines_gdf. Using 'length' column and converting to meters."
            )
            # Check for missing values
            if dlr_lines_gdf["length"].isnull().any():
                logger.error("Missing 'length' values in dlr_lines_gdf.")
                # Instead of raising an error, fill with zeros
                dlr_lines_gdf["length"] = dlr_lines_gdf["length"].fillna(0)
            dlr_lines_gdf["length_m_dlr"] = (
                dlr_lines_gdf["length"] * 1000
            )  # Convert km to meters
        else:
            logger.warning(
                "'length' column not found in dlr_lines_gdf. Using geometry length."
            )
            # Calculate length directly from geometry
            dlr_lines_gdf["length_m_dlr"] = dlr_lines_gdf.geometry.length

    # Save matches_df_all to CSV
    matches_df_all.to_csv("results/csv/matches.csv", index=False)

    # Save network_lines_updated to CSV with all columns
    network_lines_updated.to_csv("results/csv/network_lines_updated.csv", index=False)

    try:
        # Try to merge with network lines
        matches_with_info = matches_df_all.merge(
            network_lines_updated[["id", "length_m"]],
            left_on="id_net",
            right_on="id",
            how="left",
            suffixes=("", "_network"),
        )

        # Try to merge with DLR lines
        matches_with_info = matches_with_info.merge(
            dlr_lines_gdf[["id", "length_m_dlr"]],
            left_on="id_dlr",
            right_on="id",
            how="left",
            suffixes=("_network", "_dlr"),
        )

        # Rename columns for clarity
        matches_with_info.rename(
            columns={"length_m": "length_network", "length_m_dlr": "length_dlr"},
            inplace=True,
        )

        # Create a simple CSV with just the essential match information
        matches_csv = matches_with_info[["id_net", "id_dlr"]]

        # Add lengths if available
        if "length_network" in matches_with_info.columns:
            matches_csv["length_network"] = matches_with_info["length_network"]
        if "length_dlr" in matches_with_info.columns:
            matches_csv["length_dlr"] = matches_with_info["length_dlr"]

    except Exception as e:
        logger.error(f"Error creating detailed matches CSV: {str(e)}")
        # Fallback to just saving the bare matches
        matches_csv = matches_df_all[["id_net", "id_dlr"]]

    # Save to CSV
    output_path = config.get("output_paths", {}).get(
        "matched_lines_csv", "results/csv/matched_lines_with_buses.csv"
    )
    matches_csv.to_csv(output_path, index=False)
    logger.info(f"CSV file '{output_path}' created.")


def analyze_allocations_distribution(network_lines_updated):
    """
    Analyzes the distribution of allocated electrical parameters.
    """
    params = ["r_allocated", "x_allocated", "b_allocated"]

    for param in params:
        if param not in network_lines_updated.columns:
            logger.warning(f"Parameter {param} not found in network lines data")
            continue

        if "length_m" not in network_lines_updated.columns:
            logger.warning("length_m column not found in network lines data")
            continue

        values_per_km = network_lines_updated[param] / (
            network_lines_updated["length_m"] / 1000
        )

        stats = {
            "min": values_per_km.min(),
            "max": values_per_km.max(),
            "mean": values_per_km.mean(),
            "median": values_per_km.median(),
            "std": values_per_km.std(),
            "percentiles": values_per_km.quantile([0.05, 0.25, 0.75, 0.95]),
        }

        logger.info(f"\nDistribution of {param} per km:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")


def allocate_one_to_one(row):
    if (
        pd.isnull(row["length_m_net"])
        or pd.isnull(row["length_m_dlr"])
        or row["length_m_dlr"] == 0
    ):
        # Handle invalid or zero length here
        row["length_ratio"] = 0
        row["r_allocated"] = 0
        row["x_allocated"] = 0
        row["b_allocated"] = 0
        return row

    length_ratio = row["length_m_net"] / row["length_m_dlr"]
    length_tolerance = 0.1  # ±10%

    # If it is within ±10%, you can decide whether to clamp or just keep the raw ratio:
    if abs(length_ratio - 1) <= length_tolerance:
        # EITHER assign the ratio to 1 exactly ...
        # row['length_ratio'] = 1.0
        # OR keep the real ratio:
        row["length_ratio"] = length_ratio
    else:
        # outside tolerance
        row["length_ratio"] = length_ratio
        logger.warning(
            f"One-to-One Match with length_ratio={length_ratio:.4f} outside tolerance for id_net={row['id_net']}"
        )

    # Allocate scaled values
    row["r_allocated"] = row["r_dlr"] * row["length_ratio"]
    row["x_allocated"] = row["x_dlr"] * row["length_ratio"]
    row["b_allocated"] = row["b_dlr"] * row["length_ratio"]
    return row


def allocate_one_to_many(group):
    total_length_net = group["length_m_net"].sum()
    length_m_dlr = group["length_m_dlr"].iloc[0]  # One DLR line length
    if total_length_net == 0 or pd.isnull(total_length_net) or length_m_dlr == 0:
        logger.warning(
            f"Invalid lengths for id_dlr={group.name}. Setting allocated attributes to zero."
        )
        group["length_ratio"] = 0
        group["r_allocated"] = 0
        group["x_allocated"] = 0
        group["b_allocated"] = 0
    else:
        # Calculate the proportion of each network line's length to the total network length
        group["length_ratio"] = group["length_m_net"] / total_length_net
        # Allocate attributes proportionally based on length_ratio
        group["r_allocated"] = group["r_dlr"] * group["length_ratio"]
        group["x_allocated"] = group["x_dlr"] * group["length_ratio"]
        group["b_allocated"] = group["b_dlr"] * group["length_ratio"]
        logger.debug(
            f"One-to-Many Match for id_dlr={group.name}: Allocated attributes based on length ratios."
        )
    return group


def allocate_many_to_one(group):
    total_length_dlr = group["length_m_dlr"].sum()
    length_m_net = group["length_m_net"].iloc[0]  # One network line length
    if total_length_dlr == 0 or pd.isnull(total_length_dlr) or length_m_net == 0:
        logger.warning(
            f"Invalid lengths for id_net={group.name}. Setting allocated attributes to zero."
        )
        group["length_ratio"] = 0
        group["r_allocated"] = 0
        group["x_allocated"] = 0
        group["b_allocated"] = 0
    else:
        # Calculate the proportion of the network line's length to the total DLR length
        group["length_ratio"] = length_m_net / total_length_dlr
        # Allocate attributes proportionally based on length_ratio
        group["r_allocated"] = group["r_dlr"] * group["length_ratio"]
        group["x_allocated"] = group["x_dlr"] * group["length_ratio"]
        group["b_allocated"] = group["b_dlr"] * group["length_ratio"]
        logger.debug(
            f"Many-to-One Match for id_net={group.name}: Allocated attributes based on length_ratio={group['length_ratio'].iloc[0]:.2f}."
        )
    return group


def allocate_many_to_many(group):
    total_length_net = group["length_m_net"].sum()
    total_length_dlr = group["length_m_dlr"].sum()
    if total_length_net == 0 or total_length_dlr == 0:
        logger.warning(
            f"Total length is zero for group with id_net={group['id_net'].iloc[0]} and id_dlr={group['id_dlr'].iloc[0]}. Setting allocated attributes to zero."
        )
        group["length_ratio"] = 0
        group["r_allocated"] = 0
        group["x_allocated"] = 0
        group["b_allocated"] = 0
    else:
        # Allocate attributes proportionally based on network line's share of total network length
        allocation_factor_net = group["length_m_net"] / total_length_net
        # Allocate attributes
        group["length_ratio"] = allocation_factor_net
        group["r_allocated"] = group["r_dlr"] * group["length_ratio"]
        group["x_allocated"] = group["x_dlr"] * group["length_ratio"]
        group["b_allocated"] = group["b_dlr"] * group["length_ratio"]
        logger.debug(
            f"Many-to-Many Match for group id_net={group['id_net'].iloc[0]} and id_dlr={group['id_dlr'].iloc[0]}: Allocated attributes based on length_ratio={group['length_ratio'].iloc[0]:.4f}."
        )
    return group


def validate_allocations(network_lines_updated, allocated_matches, tolerance=1e-6):
    """
    Validates the allocations in the network_lines_updated GeoDataFrame.

    Parameters:
    - network_lines_updated (GeoDataFrame): The updated network lines with allocations.
    - allocated_matches (DataFrame): DataFrame containing allocated matches.
    - tolerance (float): Tolerance level for validation checks.

    Returns:
    - None. Logs the validation results.
    """
    logger.info("Starting validation of allocations...")

    # Group allocations by match type and perform validations
    match_types = allocated_matches["match_type"].unique()

    for match_type in match_types:
        subset = allocated_matches[allocated_matches["match_type"] == match_type]

        if match_type == "one_to_one":
            # For one-to-one, length_ratio should be approximately 1
            invalid = subset[
                ~subset["length_ratio"].between(1 - tolerance, 1 + tolerance)
            ]
            if not invalid.empty:
                logger.error(
                    f"One-to-One matches with invalid length_ratio:\n{invalid[['id_net', 'id_dlr', 'length_ratio']]}"
                )
            else:
                logger.info("All One-to-One allocations are valid.")

        elif match_type == "one_to_many":
            # For one-to-many, sum of length_ratio per id_dlr should be ~1
            sums = subset.groupby("id_dlr")["length_ratio"].sum()
            invalid = sums[~sums.between(1 - tolerance, 1 + tolerance)]
            if not invalid.empty:
                logger.error(
                    f"One-to-Many allocations with invalid length_ratio sums:\n{invalid}"
                )
            else:
                logger.info("All One-to-Many allocations have valid length_ratio sums.")

        elif match_type == "many_to_one":
            # For many-to-one, sum of length_ratio per id_net should be ~1
            sums = subset.groupby("id_net")["length_ratio"].sum()
            invalid = sums[~sums.between(1 - tolerance, 1 + tolerance)]
            if not invalid.empty:
                logger.error(
                    f"Many-to-One allocations with invalid length_ratio sums:\n{invalid}"
                )
            else:
                logger.info("All Many-to-One allocations have valid length_ratio sums.")

        elif match_type == "many_to_many":
            # For many-to-many, sum of length_ratio per id_dlr should be ~1 and per id_net should be ~1
            # Sum per id_dlr
            sums_dlr = subset.groupby("id_dlr")["length_ratio"].sum()
            invalid_dlr = sums_dlr[~sums_dlr.between(1 - tolerance, 1 + tolerance)]
            if not invalid_dlr.empty:
                logger.error(
                    f"Many-to-Many allocations with invalid length_ratio sums per id_dlr:\n{invalid_dlr}"
                )
            else:
                logger.info(
                    "All Many-to-Many allocations have valid length_ratio sums per id_dlr."
                )

            # Sum per id_net
            sums_net = subset.groupby("id_net")["length_ratio"].sum()
            invalid_net = sums_net[~sums_net.between(1 - tolerance, 1 + tolerance)]
            if not invalid_net.empty:
                logger.error(
                    f"Many-to-Many allocations with invalid length_ratio sums per id_net:\n{invalid_net}"
                )
            else:
                logger.info(
                    "All Many-to-Many allocations have valid length_ratio sums per id_net."
                )

    logger.info("Validation of allocations completed.")


def allocate_attributes_to_network_lines(
    matches_df_all, network_lines_gdf, dlr_lines_gdf
):
    # Reproject to a metric CRS (e.g., UTM zone appropriate for your data)
    projected_crs = "EPSG:32632"  # Example CRS, adjust as needed
    network_lines_gdf = network_lines_gdf.to_crs(projected_crs).copy()
    dlr_lines_gdf = dlr_lines_gdf.to_crs(projected_crs).copy()

    # Calculate lengths of the geometries
    network_lines_gdf["length_m_net"] = network_lines_gdf.geometry.length
    dlr_lines_gdf["length_m_dlr"] = dlr_lines_gdf.geometry.length

    # Prepare DataFrames for merging with explicit renaming to avoid conflicts
    network_lines_info = network_lines_gdf[
        ["id", "geometry", "bus0", "bus1", "length_m_net", "r", "x", "b"]
    ].rename(
        columns={
            "id": "id_net",
            "r": "r_net",
            "x": "x_net",
            "b": "b_net",
            "geometry": "geometry_net",
            "bus0": "bus0_net",
            "bus1": "bus1_net",
        }
    )

    # Validate that the required columns exist
    required_dlr_columns = [
        "id",
        "geometry",
        "bus0",
        "bus1",
        "length_m_dlr",
        "r_dlr",
        "x_dlr",
        "b_dlr",
        "TSO",
        "v_nom",
        "Imax",
        "s_nom",
    ]
    missing_dlr_columns = [
        col for col in required_dlr_columns if col not in dlr_lines_gdf.columns
    ]
    if missing_dlr_columns:
        logger.error(f"Missing columns in dlr_lines_gdf: {missing_dlr_columns}")
        raise KeyError(f"Missing columns in dlr_lines_gdf: {missing_dlr_columns}")
    else:
        logger.info("All required columns in dlr_lines_gdf are present.")

    dlr_lines_info = dlr_lines_gdf[
        [
            "id",
            "geometry",
            "bus0",
            "bus1",
            "length_m_dlr",
            "r_dlr",
            "x_dlr",
            "b_dlr",
            "TSO",
            "v_nom",
            "Imax",
            "s_nom",
        ]
    ].rename(
        columns={
            "id": "id_dlr",
            "geometry": "geometry_dlr",
            "bus0": "bus0_dlr",
            "bus1": "bus1_dlr",
        }
    )

    # Example continuation:
    # Merge matches with network_lines_info
    matches_with_network = matches_df_all.merge(
        network_lines_info, on="id_net", how="left"
    )
    logger.debug(
        f"Columns in matches_with_network after merging with network_lines_info: {matches_with_network.columns.tolist()}"
    )

    # Merge with dlr_lines_info
    matches_with_dlr = matches_with_network.merge(
        dlr_lines_info, on="id_dlr", how="left"
    )
    logger.debug(
        f"Columns in matches_with_dlr after merging with dlr_lines_info: {matches_with_dlr.columns.tolist()}"
    )

    print("\nDEBUG: After merging net + DLR:")
    print(
        matches_with_dlr[
            [
                "id_net",
                "id_dlr",
                "r_dlr",
                "x_dlr",
                "b_dlr",
                "length_m_net",
                "length_m_dlr",
                "match_type",
            ]
        ].head(20)
    )

    missing_r_dlr = matches_with_dlr["r_dlr"].isna().sum()
    print(f"DEBUG: rows missing r_dlr = {missing_r_dlr}")

    # Determine match types
    counts_net = matches_df_all["id_net"].value_counts()
    counts_dlr = matches_df_all["id_dlr"].value_counts()

    # Corrected Match Type Assignment
    matches_with_dlr["match_type"] = matches_with_dlr.apply(
        lambda row: (
            "one_to_one"
            if counts_net.get(row["id_net"], 0) == 1
            and counts_dlr.get(row["id_dlr"], 0) == 1
            else (
                "one_to_many"
                if counts_dlr.get(row["id_dlr"], 0) > 1
                else (
                    "many_to_one"
                    if counts_net.get(row["id_net"], 0) > 1
                    else "many_to_many"
                )
            )
        ),
        axis=1,
    )

    # Log the counts of each match type
    match_type_counts = matches_with_dlr["match_type"].value_counts()
    logger.info(f"Match Type Counts:\n{match_type_counts}")

    # Allocate attributes based on match types
    allocated_matches_list = []

    # Define required columns for allocation
    required_columns = [
        "id_net",
        "id_dlr",
        "length_ratio",
        "r_allocated",
        "x_allocated",
        "b_allocated",
        "length_m_net",
        "length_m_dlr",
        "match_type",  # Include 'match_type' for validation
    ]

    # Apply Allocation Functions

    # One-to-One Matches
    one_to_one_matches = matches_with_dlr.loc[
        matches_with_dlr["match_type"] == "one_to_one"
    ]

    if one_to_one_matches.empty:
        logger.warning("No one-to-one matches found. Check if this is plausible.")
    else:
        logger.debug(
            f"Processing one-to-one matches: {len(one_to_one_matches)} records"
        )

        one_to_one_matches = one_to_one_matches.apply(allocate_one_to_one, axis=1)

        allocated_matches_list.append(one_to_one_matches[required_columns])

    # One-to-Many Matches
    one_to_many_matches = matches_with_dlr[
        matches_with_dlr["match_type"] == "one_to_many"
    ].copy()
    logger.debug(f"Processing one-to-many matches: {len(one_to_many_matches)} records")
    allocated_one_to_many = (
        one_to_many_matches.groupby("id_dlr")
        .apply(allocate_one_to_many)
        .reset_index(drop=True)
    )
    allocated_matches_list.append(allocated_one_to_many[required_columns])

    # Many-to-One Matches
    many_to_one_matches = matches_with_dlr[
        matches_with_dlr["match_type"] == "many_to_one"
    ].copy()
    logger.debug(f"Processing many-to-one matches: {len(many_to_one_matches)} records")
    allocated_many_to_one = (
        many_to_one_matches.groupby("id_net")
        .apply(allocate_many_to_one)
        .reset_index(drop=True)
    )
    allocated_matches_list.append(allocated_many_to_one[required_columns])

    # Many-to-Many Matches
    many_to_many_matches = matches_with_dlr[
        matches_with_dlr["match_type"] == "many_to_many"
    ].copy()
    logger.debug(
        f"Processing many-to-many matches: {len(many_to_many_matches)} records"
    )
    if not many_to_many_matches.empty:
        allocated_many_to_many = (
            many_to_many_matches.groupby(["id_net", "id_dlr"])
            .apply(allocate_many_to_many)
            .reset_index(drop=True)
        )
        allocated_matches_list.append(allocated_many_to_many[required_columns])
    else:
        logger.info("No many-to-many matches to process.")

    # Combine Allocated Matches

    if allocated_matches_list:
        allocated_matches = pd.concat(allocated_matches_list, ignore_index=True)
        logger.debug(f"Total allocated matches: {len(allocated_matches)} records")
    else:
        allocated_matches = pd.DataFrame(columns=required_columns)
        logger.info("No allocated matches to concatenate.")

    print("\nDEBUG: allocated_matches:")
    print(f"  shape={allocated_matches.shape}")
    print(allocated_matches.head(20))

    # Aggregate allocated attributes per network line, including 'length_m_dlr'
    if not allocated_matches.empty:
        allocated_matches_agg = (
            allocated_matches.groupby("id_net")
            .agg(
                {
                    "length_ratio": "sum",
                    "r_allocated": "sum",
                    "x_allocated": "sum",
                    "b_allocated": "sum",
                    "length_m_dlr": "sum",
                }
            )
            .reset_index()
        )
    else:
        allocated_matches_agg = pd.DataFrame(
            columns=[
                "id_net",
                "length_ratio",
                "r_allocated",
                "x_allocated",
                "b_allocated",
                "length_m_dlr",
            ]
        )

    print("\nDEBUG: allocated_matches_agg:")
    print(f"  shape={allocated_matches_agg.shape}")
    print(allocated_matches_agg.head(20))

    logger.debug(
        f"Allocated Matches Aggregated Columns: {allocated_matches_agg.columns.tolist()}"
    )

    # Merge allocated attributes back into network_lines_gdf
    network_lines_updated = network_lines_gdf.merge(
        allocated_matches_agg, left_on="id", right_on="id_net", how="left"
    )

    # print("\nDEBUG: Final network_lines_updated:")
    # print(
    #     network_lines_updated[
    #         ["id", "r_net", "r_allocated", "x_allocated", "b_allocated", "length_m_dlr"]
    #     ].head(30)
    # )

    # If you don't have 'r_net', then see if 'r' is present

    # Insert the print statement here
    print(
        "DEBUG: Columns in network_lines_updated before allocation:",
        network_lines_updated.columns.tolist(),
    )

    # Rename 'r' to 'r_net' if necessary
    if (
        "r_net" not in network_lines_updated.columns
        and "r" in network_lines_updated.columns
    ):
        network_lines_updated.rename(columns={"r": "r_net"}, inplace=True)
        print("DEBUG: Renamed 'r' to 'r_net'")

    # Assign allocated attributes, preserving original values where allocations are missing
    network_lines_updated["r_allocated"] = network_lines_updated["r_allocated"].fillna(
        0
    )
    network_lines_updated["x_allocated"] = network_lines_updated["x_allocated"].fillna(
        0
    )
    network_lines_updated["b_allocated"] = network_lines_updated["b_allocated"].fillna(
        0
    )
    network_lines_updated["length_ratio"] = network_lines_updated[
        "length_ratio"
    ].fillna(0)

    # Ensure 'r_net' exists before allocation
    if "r_net" in network_lines_updated.columns:
        network_lines_updated["r_total"] = (
            network_lines_updated["r_net"] + network_lines_updated["r_allocated"]
        )
    else:
        logger.error(
            "'r_net' column is missing in network_lines_updated after renaming."
        )
        # Handle the error appropriately, e.g., raise an exception or skip allocation
        network_lines_updated["r_total"] = network_lines_updated[
            "r_allocated"
        ]  # Example fallback

    if "x_net" in network_lines_updated.columns:
        network_lines_updated["x_total"] = (
            network_lines_updated["x_net"] + network_lines_updated["x_allocated"]
        )
    else:
        logger.warning("'x_net' column is missing in network_lines_updated.")
        network_lines_updated["x_total"] = network_lines_updated["x_allocated"]

    if "b_net" in network_lines_updated.columns:
        network_lines_updated["b_total"] = (
            network_lines_updated["b_net"] + network_lines_updated["b_allocated"]
        )
    else:
        logger.warning("'b_net' column is missing in network_lines_updated.")
        network_lines_updated["b_total"] = network_lines_updated["b_allocated"]

    # Clean up unnecessary columns
    columns_to_drop = ["id_net"]
    network_lines_updated.drop(
        columns=[
            col for col in columns_to_drop if col in network_lines_updated.columns
        ],
        inplace=True,
    )

    # Verify columns after merge
    logger.debug(
        f"Columns in network_lines_updated after merge: {network_lines_updated.columns.tolist()}"
    )

    # Add Logging to Verify Data
    logger.debug(
        f"Sample data:\n{network_lines_updated[['id', 'length_m_net', 'length_m_dlr', 'length_ratio', 'r_allocated', 'x_allocated', 'b_allocated']].head()}"
    )

    return network_lines_updated, allocated_matches_agg

    # Validate Allocations
    def validate_allocations(network_lines_updated, allocated_matches, tolerance=1e-6):
        """
        Validates the allocations in the network_lines_updated GeoDataFrame.
        """
        logger.info("Starting validation of allocations...")

        # Group allocations by match type and perform validations
        match_types = allocated_matches["match_type"].unique()

        for match_type in match_types:
            subset = allocated_matches[allocated_matches["match_type"] == match_type]

            if match_type == "one_to_one":
                # For one-to-one, length_ratio should be approximately 1
                invalid = subset[
                    ~subset["length_ratio"].between(1 - tolerance, 1 + tolerance)
                ]
                if not invalid.empty:
                    logger.error(
                        f"One-to-One matches with invalid length_ratio:\n{invalid[['id_net', 'id_dlr', 'length_ratio']]}"
                    )
                else:
                    logger.info("All One-to-One allocations are valid.")

            elif match_type == "one_to_many":
                # For one-to-many, sum of length_ratio per id_dlr should be ~1
                sums = subset.groupby("id_dlr")["length_ratio"].sum()
                invalid = sums[~sums.between(1 - tolerance, 1 + tolerance)]
                if not invalid.empty:
                    logger.error(
                        f"One-to-Many allocations with invalid length_ratio sums:\n{invalid}"
                    )
                else:
                    logger.info(
                        "All One-to-Many allocations have valid length_ratio sums."
                    )

            elif match_type == "many_to_one":
                # For many-to-one, sum of length_ratio per id_net should be ~1
                sums = subset.groupby("id_net")["length_ratio"].sum()
                invalid = sums[~sums.between(1 - tolerance, 1 + tolerance)]
                if not invalid.empty:
                    logger.error(
                        f"Many-to-One allocations with invalid length_ratio sums:\n{invalid}"
                    )
                else:
                    logger.info(
                        "All Many-to-One allocations have valid length_ratio sums."
                    )

            elif match_type == "many_to_many":
                # For many-to-many, sum of length_ratio per id_dlr should be ~1 and per id_net should be ~1
                # Sum per id_dlr
                sums_dlr = subset.groupby("id_dlr")["length_ratio"].sum()
                invalid_dlr = sums_dlr[~sums_dlr.between(1 - tolerance, 1 + tolerance)]
                if not invalid_dlr.empty:
                    logger.error(
                        f"Many-to-Many allocations with invalid length_ratio sums per id_dlr:\n{invalid_dlr}"
                    )
                else:
                    logger.info(
                        "All Many-to-Many allocations have valid length_ratio sums per id_dlr."
                    )

                # Sum per id_net
                sums_net = subset.groupby("id_net")["length_ratio"].sum()
                invalid_net = sums_net[~sums_net.between(1 - tolerance, 1 + tolerance)]
                if not invalid_net.empty:
                    logger.error(
                        f"Many-to-Many allocations with invalid length_ratio sums per id_net:\n{invalid_net}"
                    )
                else:
                    logger.info(
                        "All Many-to-Many allocations have valid length_ratio sums per id_net."
                    )

        logger.info("Validation of allocations completed.")

    validate_allocations(network_lines_updated, allocated_matches, tolerance=1e-6)

    # Final status report
    if scale_correction_factor != 1.0:
        logger.info(
            f"Applied scaling correction factor of {scale_correction_factor} during allocation"
        )
        print(
            "\nScaling correction was applied to allocated values. Check log for details.\n"
        )

    return network_lines_updated, allocated_matches


def validate_parameter_ranges(network_lines_updated):
    """
    Validates that the allocated electrical parameters are within reasonable ranges.

    Parameters:
    - network_lines_updated (GeoDataFrame): Network lines with allocated parameters

    Returns:
    - bool: True if all parameters are within reasonable ranges, False otherwise
    """
    valid = True

    # Define typical parameter ranges per km
    typical_ranges = {
        "r_total": (0.01, 0.8),  # ohm/km
        "x_total": (0.05, 0.8),  # ohm/km
        "b_total": (1e-6, 8e-6),  # S/km
    }

    # Check each allocated parameter
    for param, (min_val, max_val) in typical_ranges.items():
        if param in network_lines_updated.columns:
            # Convert to per-km values
            values_per_km = network_lines_updated[param] / (
                network_lines_updated["length_m"] / 1000
            )

            # Filter out invalid values
            valid_values = values_per_km[
                (values_per_km >= min_val) & (values_per_km <= max_val)
            ]
            invalid_count = len(values_per_km) - len(valid_values)

            if invalid_count > 0:
                valid = False
                invalid_percent = 100 * invalid_count / len(values_per_km)
                logger.warning(
                    f"{invalid_count} lines ({invalid_percent:.1f}%) have {param} outside typical range."
                )

                # List some examples of abnormal values
                abnormal = network_lines_updated[
                    (values_per_km < min_val) | (values_per_km > max_val)
                ].copy()
                abnormal["value_per_km"] = values_per_km[abnormal.index]
                logger.warning(f"Examples of abnormal {param} values:")
                logger.warning(abnormal[["id", param, "value_per_km"]].head(5))

    return valid


def check_dlr_parameters(dlr_lines_gdf):
    """
    Check if DLR lines have electrical parameters and if they are non-zero.

    Parameters:
        dlr_lines_gdf (GeoDataFrame): DLR lines GeoDataFrame

    Returns:
        bool: True if parameters exist and have non-zero values, False otherwise
    """
    logger.info("Checking DLR parameters...")

    # Check if the DLR parameters exist in the dataframe
    r_col = None
    x_col = None
    b_col = None

    # Look for standard parameter names and possible variations
    r_candidates = ["r_dlr", "r", "resistance", "R"]
    x_candidates = ["x_dlr", "x", "reactance", "X"]
    b_candidates = ["b_dlr", "b", "susceptance", "B"]

    for col in r_candidates:
        if col in dlr_lines_gdf.columns:
            r_col = col
            break

    for col in x_candidates:
        if col in dlr_lines_gdf.columns:
            x_col = col
            break

    for col in b_candidates:
        if col in dlr_lines_gdf.columns:
            b_col = col
            break

    # Report what we found
    logger.info(f"Found DLR parameter columns: r={r_col}, x={x_col}, b={b_col}")

    # Check if any parameter columns were found
    if not all([r_col, x_col, b_col]):
        missing = []
        if not r_col:
            missing.append("resistance (r)")
        if not x_col:
            missing.append("reactance (x)")
        if not b_col:
            missing.append("susceptance (b)")
        logger.error(f"Missing DLR parameter columns: {', '.join(missing)}")
        return False

    # Check for non-zero values
    non_zero_r = (dlr_lines_gdf[r_col] > 0).sum()
    non_zero_x = (dlr_lines_gdf[x_col] > 0).sum()
    non_zero_b = (dlr_lines_gdf[b_col] > 0).sum()

    total_lines = len(dlr_lines_gdf)
    logger.info(f"DLR lines with non-zero values:")
    logger.info(
        f"  Resistance (r): {non_zero_r}/{total_lines} ({non_zero_r / total_lines * 100:.1f}%)"
    )
    logger.info(
        f"  Reactance (x): {non_zero_x}/{total_lines} ({non_zero_x / total_lines * 100:.1f}%)"
    )
    logger.info(
        f"  Susceptance (b): {non_zero_b}/{total_lines} ({non_zero_b / total_lines * 100:.1f}%)"
    )

    # Sample values to check range
    if non_zero_r > 0:
        non_zero_r_values = dlr_lines_gdf.loc[dlr_lines_gdf[r_col] > 0, r_col]
        logger.info(
            f"  Resistance range: min={non_zero_r_values.min():.6f}, max={non_zero_r_values.max():.6f}, mean={non_zero_r_values.mean():.6f}"
        )

    if non_zero_x > 0:
        non_zero_x_values = dlr_lines_gdf.loc[dlr_lines_gdf[x_col] > 0, x_col]
        logger.info(
            f"  Reactance range: min={non_zero_x_values.min():.6f}, max={non_zero_x_values.max():.6f}, mean={non_zero_x_values.mean():.6f}"
        )

    if non_zero_b > 0:
        non_zero_b_values = dlr_lines_gdf.loc[dlr_lines_gdf[b_col] > 0, b_col]
        logger.info(
            f"  Susceptance range: min={non_zero_b_values.min():.6f}, max={non_zero_b_values.max():.6f}, mean={non_zero_b_values.mean():.6f}"
        )

    # Check if at least some parameters have non-zero values
    if non_zero_r == 0 and non_zero_x == 0 and non_zero_b == 0:
        logger.error("All DLR electrical parameters are zero!")
        return False

    return True


# Function to create unmatched lines map
def create_unmatched_lines_map(
    unmatched_network_lines, germany_gdf, network_buses_gdf, config
):
    """
    Creates a Folium map displaying unmatched Network lines and their connected buses within Germany.

    Parameters:
    - unmatched_network_lines (GeoDataFrame): GeoDataFrame of unmatched network lines.
    - germany_gdf (GeoDataFrame): GeoDataFrame of Germany's boundary.
    - network_buses_gdf (GeoDataFrame): GeoDataFrame of Network buses.
    - config (dict): Configuration dictionary containing paths.

    Outputs:
    - Saves a Folium map as specified in config.
    """
    import folium

    # Initialize the Folium map centered over Germany
    folium_map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)

    # Add Germany boundary
    folium.GeoJson(
        germany_gdf,
        name="Germany Boundary",
        style_function=lambda x: {
            "fillColor": "#00000000",
            "color": "black",
            "weight": 1,
        },
        show=True,
    ).add_to(folium_map)

    # Add Unmatched Network Lines in blue
    if not unmatched_network_lines.empty:
        folium.GeoJson(
            unmatched_network_lines[["geometry"]].to_crs("EPSG:4326"),
            name="Unmatched Network Lines",
            style_function=lambda x: {"color": "blue", "weight": 2},
            show=True,
        ).add_to(folium_map)
    else:
        print("No unmatched network lines to plot.")

    # Extract buses connected to unmatched network lines
    bus_ids = set(unmatched_network_lines["bus0"]).union(
        set(unmatched_network_lines["bus1"])
    )

    # Filter network_buses_gdf to only those buses
    connected_buses_gdf = network_buses_gdf[
        network_buses_gdf["bus_idx"].isin(bus_ids)
    ].copy()

    # Add Connected Network Buses as CircleMarkers in green
    if not connected_buses_gdf.empty:
        # Define a FeatureGroup for Connected Network Buses
        buses_fg = folium.FeatureGroup(name="Connected Network Buses", show=True)

        for _, row in connected_buses_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=3,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=0.7,
                popup=f"Network Bus: {row['bus_idx']}",
            ).add_to(buses_fg)

        buses_fg.add_to(folium_map)
    else:
        print("No connected network buses to plot.")

    # Add Layer Control to toggle layers
    folium.LayerControl().add_to(folium_map)

    # Save the map to an HTML file
    unmatched_map_path = config["map_paths"]["unmatched_lines_map"]
    folium_map.save(unmatched_map_path)
    print(f"Unmatched lines map generated and saved as '{unmatched_map_path}'.")
