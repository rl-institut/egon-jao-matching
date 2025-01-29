# Import necessary libraries
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import folium
import re
import traceback
from branca.element import Template, MacroElement


# Function to load data with default paths
def load_data(
        dlr_buses_path='/home/mohsen/CORE-TSO/dlr-buses.csv',
        dlr_lines_path='/home/mohsen/CORE-TSO/dlr-lines.csv',
        network_buses_path='/home/mohsen/CORE-TSO/network-buses.csv',
        network_lines_path='/home/mohsen/CORE-TSO/network-lines.csv'
):
    """
    Load DLR and Network buses and lines from CSV files.

    Parameters:
    - dlr_buses_path (str): Path to the DLR buses CSV file.
    - dlr_lines_path (str): Path to the DLR lines CSV file.
    - network_buses_path (str): Path to the Network buses CSV file.
    - network_lines_path (str): Path to the Network lines CSV file.

    Returns:
    - tuple: DataFrames for DLR buses, DLR lines, Network buses, and Network lines.
    """
    try:
        # Load DLR buses and lines
        dlr_buses = pd.read_csv(dlr_buses_path)
        dlr_lines = pd.read_csv(dlr_lines_path)

        # Load Network buses and lines
        network_buses = pd.read_csv(network_buses_path)

        # Read 'geom' as string
        network_lines = pd.read_csv(network_lines_path, dtype={'geom': str})

        return dlr_buses, dlr_lines, network_buses, network_lines
    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
        raise
    except Exception as e:
        print("An unexpected error occurred while loading data:")
        traceback.print_exc()
        raise


# Function to clean DLR bus IDs
def clean_dlr_bus_id(bus_id):
    """
    Clean the DLR bus ID by removing non-alphanumeric characters and ensuring it starts with 'BUS_'.

    Parameters:
    - bus_id (str): Original bus ID.

    Returns:
    - str: Cleaned bus ID.
    """
    # Convert to string first to avoid TypeError
    bus_id = str(bus_id)
    # Remove any non-alphanumeric characters except underscores
    cleaned_id = re.sub(r'[^\w]', '', bus_id).upper().strip()
    # Ensure it starts with 'BUS_'
    if not cleaned_id.startswith('BUS_'):
        # Assuming all bus IDs should start with 'BUS_', add it if missing
        cleaned_id = 'BUS_' + cleaned_id.replace('BUS', '')
    return cleaned_id


# Function to create sorted 'name' from two bus fields
def create_sorted_name(row, bus_field1, bus_field2):
    """
    Create a sorted name by combining two bus fields.

    Parameters:
    - row (pd.Series): Row from the GeoDataFrame.
    - bus_field1 (str): Column name for the first bus.
    - bus_field2 (str): Column name for the second bus.

    Returns:
    - str: Combined and sorted name.
    """
    bus1 = row[bus_field1]
    bus2 = row[bus_field2]
    if pd.isnull(bus1) or pd.isnull(bus2):
        return "UNKNOWN_NAME"
    return "_".join(sorted([str(bus1), str(bus2)]))


# Function to prepare GeoDataFrames
def prepare_geodataframes(dlr_buses, dlr_lines, network_buses, network_lines):
    """
    Prepare GeoDataFrames for DLR buses, DLR lines, Network buses, and Network lines.

    Parameters:
    - dlr_buses (pd.DataFrame): DataFrame containing DLR buses.
    - dlr_lines (pd.DataFrame): DataFrame containing DLR lines.
    - network_buses (pd.DataFrame): DataFrame containing Network buses.
    - network_lines (pd.DataFrame): DataFrame containing Network lines.

    Returns:
    - tuple: GeoDataFrames for DLR buses, DLR lines, Network buses, and Network lines.
    """
    # Convert DLR buses to GeoDataFrame
    dlr_buses_gdf = gpd.GeoDataFrame(
        dlr_buses,
        geometry=gpd.points_from_xy(dlr_buses['x'], dlr_buses['y']),
        crs='EPSG:4326'
    )

    # Clean DLR bus IDs and overwrite 'name'
    dlr_buses_gdf['name'] = dlr_buses_gdf['name'].apply(clean_dlr_bus_id)

    # Clean bus IDs in dlr_lines
    dlr_lines['bus0'] = dlr_lines['bus0'].apply(clean_dlr_bus_id)
    dlr_lines['bus1'] = dlr_lines['bus1'].apply(clean_dlr_bus_id)

    # Create a mapping from cleaned bus names to geometries
    dlr_buses_dict = dlr_buses_gdf.set_index('name')['geometry'].to_dict()

    # Map geometries to DLR lines
    dlr_lines['bus0_geom'] = dlr_lines['bus0'].map(dlr_buses_dict)
    dlr_lines['bus1_geom'] = dlr_lines['bus1'].map(dlr_buses_dict)
    dlr_lines = dlr_lines.dropna(subset=['bus0_geom', 'bus1_geom']).copy()
    dlr_lines['geometry'] = dlr_lines.apply(
        lambda row: LineString([row['bus0_geom'], row['bus1_geom']]),
        axis=1
    )
    dlr_lines_gdf = gpd.GeoDataFrame(dlr_lines, geometry='geometry', crs='EPSG:4326')

    # Add 'name' column to dlr_lines_gdf
    dlr_lines_gdf['name'] = dlr_lines_gdf['bus0'] + '_' + dlr_lines_gdf['bus1']

    # Convert Network buses to GeoDataFrame
    network_buses_gdf = gpd.GeoDataFrame(
        network_buses,
        geometry=gpd.points_from_xy(network_buses['x'], network_buses['y']),
        crs='EPSG:4326'
    )

    # Print columns to identify available fields
    print("Columns in network_buses_gdf:", network_buses_gdf.columns.tolist())

    # Assign 'bus_idx' based on 'scn_name'
    # First, check if 'scn_name' is unique
    unique_count = network_buses_gdf['scn_name'].nunique()
    total_count = len(network_buses_gdf)
    print(f"'scn_name' unique count: {unique_count} out of {total_count}")

    if unique_count == total_count:
        # Option A: 'scn_name' is unique
        network_buses_gdf['bus_idx'] = network_buses_gdf['scn_name'].astype(str)
        print("Assigned 'bus_idx' using 'scn_name'.")
    else:
        # Option B: 'scn_name' is not unique, create a unique identifier
        network_buses_gdf = network_buses_gdf.reset_index(drop=True)
        network_buses_gdf['unique_idx'] = network_buses_gdf.index
        network_buses_gdf['bus_idx'] = network_buses_gdf['scn_name'].astype(str) + '_' + network_buses_gdf[
            'unique_idx'].astype(str)
        print("Assigned 'bus_idx' using combination of 'scn_name' and index.")

    # Verify uniqueness
    unique_bus_idx = network_buses_gdf['bus_idx'].nunique()
    total_bus_idx = len(network_buses_gdf)
    print(f"'bus_idx' unique count: {unique_bus_idx} out of {total_bus_idx}")

    # Check for duplicates
    duplicate_bus_idx = network_buses_gdf['bus_idx'].duplicated().sum()
    print(f"Number of duplicated 'bus_idx': {duplicate_bus_idx}")

    # Filter Network lines (exclude 110 kV)
    network_lines_filtered = network_lines[network_lines['v_nom'] != 110].copy()

    # Read geometries from 'geom' column
    if 'geom' in network_lines_filtered.columns:
        network_lines_filtered['geometry'] = gpd.GeoSeries.from_wkt(network_lines_filtered['geom'])
        network_lines_gdf = gpd.GeoDataFrame(network_lines_filtered, geometry='geometry', crs='EPSG:4326')
    else:
        raise ValueError("The 'network-lines.csv' file does not contain a 'geom' column.")

    # Ensure 'bus0' and 'bus1' are strings (assuming they are bus identifiers)
    network_lines_gdf['bus0'] = network_lines_gdf['bus0'].astype(str).str.strip()
    network_lines_gdf['bus1'] = network_lines_gdf['bus1'].astype(str).str.strip()

    return dlr_buses_gdf, dlr_lines_gdf, network_buses_gdf, network_lines_gdf


# Function to clean Network lines' bus IDs
def clean_network_line_bus_ids(network_lines_gdf):
    """
    Ensure 'bus0' and 'bus1' are strings and stripped of whitespace.

    Parameters:
    - network_lines_gdf (gpd.GeoDataFrame): GeoDataFrame containing Network lines.

    Returns:
    - gpd.GeoDataFrame: Cleaned GeoDataFrame.
    """
    # Ensure 'bus0' and 'bus1' are strings and stripped
    network_lines_gdf['bus0'] = network_lines_gdf['bus0'].astype(str).str.strip()
    network_lines_gdf['bus1'] = network_lines_gdf['bus1'].astype(str).str.strip()

    return network_lines_gdf


# Function to exclude lines with NaN bus references
def exclude_invalid_lines(network_lines_filtered):
    """
    Drop lines with NaN in 'bus0' or 'bus1'.

    Parameters:
    - network_lines_filtered (gpd.GeoDataFrame): Filtered Network lines.

    Returns:
    - gpd.GeoDataFrame: Cleaned GeoDataFrame.
    """
    # Drop lines with NaN in 'bus0' or 'bus1'
    network_lines_filtered = network_lines_filtered.dropna(subset=['bus0', 'bus1'])
    return network_lines_filtered


# Function to match buses using nearest neighbor with unique mappings
def match_buses_with_nearest(network_buses_gdf, dlr_buses_gdf, max_distance=5000):
    """
    Match Network buses to DLR buses using nearest neighbor within a specified maximum distance.

    Parameters:
    - network_buses_gdf (gpd.GeoDataFrame): GeoDataFrame containing Network buses.
    - dlr_buses_gdf (gpd.GeoDataFrame): GeoDataFrame containing DLR buses.
    - max_distance (int): Maximum distance in meters to consider for matching.

    Returns:
    - gpd.GeoDataFrame: Bus ID mapping with unique 'bus_idx' and corresponding 'dlr_bus_name'.
    """
    # Project GeoDataFrames to a metric CRS
    projected_crs = 'EPSG:32632'  # UTM zone suitable for Germany

    # Reproject to projected CRS
    network_buses_proj = network_buses_gdf.to_crs(projected_crs)
    dlr_buses_proj = dlr_buses_gdf.to_crs(projected_crs)

    # Perform nearest neighbor spatial join, retaining 'bus_idx'
    matched_buses = gpd.sjoin_nearest(
        network_buses_proj[['bus_idx', 'geometry']],
        dlr_buses_proj[['name', 'geometry']],
        how='left',
        max_distance=max_distance,
        distance_col='distance'
    )

    # Drop rows where no nearest neighbor was found within max_distance
    matched_buses = matched_buses.dropna(subset=['index_right']).reset_index(drop=True)

    # Get DLR bus names
    matched_buses['dlr_bus_name'] = dlr_buses_proj.loc[matched_buses['index_right'], 'name'].values

    # Create the bus_id_mapping GeoDataFrame
    bus_id_mapping = gpd.GeoDataFrame(
        matched_buses[['bus_idx', 'dlr_bus_name', 'distance']],
        geometry=matched_buses['geometry'],
        crs=projected_crs
    )

    # **Ensure unique 'bus_idx' by keeping the closest 'dlr_bus_name'**
    # Sort by distance and drop duplicates
    bus_id_mapping = bus_id_mapping.sort_values('distance').drop_duplicates(subset=['bus_idx'], keep='first')

    # Reproject back to WGS84
    bus_id_mapping = bus_id_mapping.to_crs('EPSG:4326')

    # Debugging: Print sample mappings
    print("Sample bus mappings (Nearest Neighbor):")
    print(bus_id_mapping.head())

    # **Report number of duplicates removed**
    total_mappings = len(matched_buses)
    unique_mappings = len(bus_id_mapping)
    duplicates_removed = total_mappings - unique_mappings
    print(f"Duplicates removed from 'bus_id_mapping': {duplicates_removed}")

    return bus_id_mapping


# Function to update network lines with matched DLR buses
def update_network_lines_with_matched_buses(network_lines_gdf, bus_id_mapping):
    """
    Update Network lines GeoDataFrame by mapping 'bus0' and 'bus1' to their corresponding DLR buses.

    Parameters:
    - network_lines_gdf (gpd.GeoDataFrame): GeoDataFrame containing Network lines.
    - bus_id_mapping (gpd.GeoDataFrame): GeoDataFrame mapping 'bus_idx' to 'dlr_bus_name'.

    Returns:
    - gpd.GeoDataFrame: Updated GeoDataFrame with 'dlr_bus0' and 'dlr_bus1'.
    """
    # Create a mapping from 'bus_idx' to 'dlr_bus_name'
    bus_name_mapping = bus_id_mapping.set_index('bus_idx')['dlr_bus_name'].to_dict()

    # Map 'bus0' and 'bus1' to 'dlr_bus0' and 'dlr_bus1'
    network_lines_gdf['dlr_bus0'] = network_lines_gdf['bus0'].map(bus_name_mapping)
    network_lines_gdf['dlr_bus1'] = network_lines_gdf['bus1'].map(bus_name_mapping)

    # Drop lines where either end does not have a matched DLR bus
    network_lines_matched = network_lines_gdf.dropna(subset=['dlr_bus0', 'dlr_bus1']).copy()

    # Ensure 'geometry' is set as the active geometry column
    network_lines_matched = network_lines_matched.set_geometry('geometry')

    # Debugging: Print sample of matched network lines
    print("Sample of network lines with matched DLR buses:")
    print(network_lines_matched[['bus0', 'bus1', 'dlr_bus0', 'dlr_bus1']].head())

    return network_lines_matched


# Function to match lines based on matched buses
def match_lines_based_on_matched_buses(network_lines_matched, dlr_lines_gdf):
    """
    Match Network lines to DLR lines based on matched bus pairs.

    Parameters:
    - network_lines_matched (gpd.GeoDataFrame): GeoDataFrame with matched Network lines.
    - dlr_lines_gdf (gpd.GeoDataFrame): GeoDataFrame containing DLR lines.

    Returns:
    - tuple: Matched Network lines and matched DLR lines GeoDataFrames.
    """
    # Create sets of bus pairs for network lines and DLR lines
    network_line_bus_pairs = set(
        tuple(sorted([row['dlr_bus0'], row['dlr_bus1']]))
        for idx, row in network_lines_matched.iterrows()
    )

    dlr_line_bus_pairs = set(
        tuple(sorted([row['bus0'], row['bus1']]))
        for idx, row in dlr_lines_gdf.iterrows()
    )

    # Find matching pairs
    matched_bus_pairs = network_line_bus_pairs & dlr_line_bus_pairs

    # Extract matched network lines
    matched_network_lines = network_lines_matched[
        network_lines_matched.apply(
            lambda row: tuple(sorted([row['dlr_bus0'], row['dlr_bus1']])) in matched_bus_pairs,
            axis=1
        )
    ]

    # Extract matched DLR lines
    matched_dlr_lines = dlr_lines_gdf[
        dlr_lines_gdf.apply(
            lambda row: tuple(sorted([row['bus0'], row['bus1']])) in matched_bus_pairs,
            axis=1
        )
    ]

    # Ensure geometries are set if DataFrame is not empty
    if not matched_network_lines.empty:
        matched_network_lines = matched_network_lines.set_geometry('geometry')
    if not matched_dlr_lines.empty:
        matched_dlr_lines = matched_dlr_lines.set_geometry('geometry')

    # Debugging: Print number of matched lines
    print(f"Number of matched network lines (Nearest Neighbor): {len(matched_network_lines)}")
    print(f"Number of matched DLR lines (Nearest Neighbor): {len(matched_dlr_lines)}")

    return matched_network_lines, matched_dlr_lines


# Function to match lines using buffer method
def match_lines_with_buffer(
        unmatched_network_lines_gdf, dlr_lines_gdf, buffer_distance=10000,
        matched_network_line_ids=None, matched_dlr_line_ids=None
):
    """
    Match Network lines to DLR lines using a buffer method.

    Parameters:
    - unmatched_network_lines_gdf (gpd.GeoDataFrame): Unmatched Network lines.
    - dlr_lines_gdf (gpd.GeoDataFrame): DLR lines GeoDataFrame.
    - buffer_distance (int): Buffer distance in meters.
    - matched_network_line_ids (set): Set to track matched Network line indices.
    - matched_dlr_line_ids (set): Set to track matched DLR line indices.

    Returns:
    - tuple: Additional matched Network lines and DLR lines GeoDataFrames.
    """
    if matched_network_line_ids is None:
        matched_network_line_ids = set()
    if matched_dlr_line_ids is None:
        matched_dlr_line_ids = set()

    try:
        # Project to a suitable metric CRS (e.g., UTM zone 32N)
        projected_crs = 'EPSG:32632'
        unmatched_network_lines_proj = unmatched_network_lines_gdf.to_crs(projected_crs).copy()
        dlr_lines_proj = dlr_lines_gdf.to_crs(projected_crs).copy()

        # Create a spatial index for unmatched network lines
        network_sindex = unmatched_network_lines_proj.sindex

        # Buffer DLR lines
        dlr_lines_proj['buffer'] = dlr_lines_proj.geometry.buffer(buffer_distance)

        additional_matched_network_lines = []
        additional_matched_dlr_lines = []

        for idx, dlr_row in dlr_lines_proj.iterrows():
            dlr_line_id = dlr_row.get('id', idx)  # Ensure we have a unique identifier for DLR lines
            if dlr_line_id in matched_dlr_line_ids:
                continue  # Skip if this DLR line has already been matched
            dlr_buffer = dlr_row['buffer']
            possible_matches_index = list(network_sindex.intersection(dlr_buffer.bounds))
            possible_matches = unmatched_network_lines_proj.iloc[possible_matches_index]

            # Exclude network lines already matched
            possible_matches = possible_matches[~possible_matches.index.isin(matched_network_line_ids)]
            if possible_matches.empty:
                continue

            # Find network lines within the buffer
            network_matches = possible_matches[possible_matches.within(dlr_buffer)]
            if not network_matches.empty:
                # Choose the best match
                network_matches = network_matches.copy()  # Avoid SettingWithCopyWarning
                network_matches['intersection_area'] = network_matches.geometry.intersection(dlr_buffer).area
                best_match = network_matches.loc[network_matches['intersection_area'].idxmax()]
                additional_matched_network_lines.append(best_match)
                additional_matched_dlr_lines.append(dlr_row)
                matched_dlr_line_ids.add(dlr_line_id)
                matched_network_line_ids.add(best_match.name)

        # Convert lists to GeoDataFrames using geopandas.GeoDataFrame to preserve geometry
        if additional_matched_network_lines:
            additional_matched_network_lines_gdf = gpd.GeoDataFrame(
                pd.DataFrame([line for line in additional_matched_network_lines]),
                geometry='geometry',
                crs=projected_crs
            )
            additional_matched_network_lines_gdf = additional_matched_network_lines_gdf.to_crs('EPSG:4326').reset_index(
                drop=True)
        else:
            additional_matched_network_lines_gdf = gpd.GeoDataFrame(columns=unmatched_network_lines_gdf.columns)

        if additional_matched_dlr_lines:
            additional_matched_dlr_lines_gdf = gpd.GeoDataFrame(
                pd.DataFrame([line for line in additional_matched_dlr_lines]),
                geometry='geometry',
                crs=projected_crs
            )
            additional_matched_dlr_lines_gdf = additional_matched_dlr_lines_gdf.to_crs('EPSG:4326').reset_index(
                drop=True)
        else:
            additional_matched_dlr_lines_gdf = gpd.GeoDataFrame(columns=dlr_lines_gdf.columns)

        # Clean up by dropping the buffer column
        dlr_lines_proj.drop(columns='buffer', inplace=True)

        return additional_matched_network_lines_gdf, additional_matched_dlr_lines_gdf
    except Exception as e:
        print("Error matching lines with buffer method:")
        traceback.print_exc()
        return None, None


# Function to load Germany boundary
def load_germany_boundary():
    """
    Load the geographical boundary of Germany.

    Returns:
    - gpd.GeoDataFrame: GeoDataFrame containing Germany's boundary.
    """
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except AttributeError:
        world = gpd.read_file('https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip')

    # Adjusting for correct country name column
    if 'name' in world.columns:
        country_col = 'name'
    elif 'NAME' in world.columns:
        country_col = 'NAME'
    elif 'admin' in world.columns:
        country_col = 'admin'
    elif 'ADMIN' in world.columns:
        country_col = 'ADMIN'
    else:
        raise KeyError("Country name column not found in world dataset.")

    germany_gdf = world[world[country_col] == 'Germany'].to_crs('EPSG:4326')
    return germany_gdf


# Function to filter DLR lines inside Germany
def filter_dlr_lines_inside_germany(dlr_lines_gdf, germany_gdf):
    """
    Filter DLR lines that are inside Germany's boundary.

    Parameters:
    - dlr_lines_gdf (gpd.GeoDataFrame): GeoDataFrame containing DLR lines.
    - germany_gdf (gpd.GeoDataFrame): GeoDataFrame containing Germany's boundary.

    Returns:
    - gpd.GeoDataFrame: Filtered GeoDataFrame with DLR lines inside Germany.
    """
    # Ensure that both GeoDataFrames are using the same CRS
    dlr_lines_gdf = dlr_lines_gdf.to_crs(germany_gdf.crs)

    # Spatially join DLR lines with Germany boundary
    dlr_lines_inside_germany = gpd.overlay(dlr_lines_gdf, germany_gdf, how='intersection')

    # Alternatively, for performance on large datasets, use spatial index
    # dlr_lines_inside_germany = dlr_lines_gdf[dlr_lines_gdf.intersects(germany_gdf.unary_union)]

    # Reset index if necessary
    dlr_lines_inside_germany = dlr_lines_inside_germany.reset_index(drop=True)

    return dlr_lines_inside_germany


# Function to visualize matched lines using Folium with separate colors
def create_folium_map(germany_gdf, network_lines_gdf, dlr_lines_gdf, network_buses_gdf, dlr_buses_gdf,
                      matched_network_lines_nn=None, matched_dlr_lines_nn=None,
                      matched_network_lines_buffer=None, matched_dlr_lines_buffer=None):
    """
    Create a Folium map visualizing all elements with distinct colors for different matching methods.

    Parameters:
    - germany_gdf (gpd.GeoDataFrame): GeoDataFrame containing Germany's boundary.
    - network_lines_gdf (gpd.GeoDataFrame): GeoDataFrame containing all Network lines.
    - dlr_lines_gdf (gpd.GeoDataFrame): GeoDataFrame containing all DLR lines inside Germany.
    - network_buses_gdf (gpd.GeoDataFrame): GeoDataFrame containing Network buses.
    - dlr_buses_gdf (gpd.GeoDataFrame): GeoDataFrame containing DLR buses.
    - matched_network_lines_nn (gpd.GeoDataFrame): Matched Network lines from Nearest Neighbor.
    - matched_dlr_lines_nn (gpd.GeoDataFrame): Matched DLR lines from Nearest Neighbor.
    - matched_network_lines_buffer (gpd.GeoDataFrame): Matched Network lines from Buffer method.
    - matched_dlr_lines_buffer (gpd.GeoDataFrame): Matched DLR lines from Buffer method.
    """
    try:
        # Initialize the map
        folium_map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)

        # Add Germany boundary
        folium.GeoJson(
            germany_gdf,
            name='Germany',
            style_function=lambda x: {'fillColor': '#00000000', 'color': 'black', 'weight': 1},
            show=True
        ).add_to(folium_map)

        # Add all Network Lines
        folium.GeoJson(
            network_lines_gdf[['geometry']].to_crs('EPSG:4326'),
            name='All Network Lines',
            style_function=lambda x: {'color': 'blue', 'weight': 1},
            show=True
        ).add_to(folium_map)

        # Add all DLR Lines
        folium.GeoJson(
            dlr_lines_gdf[['geometry']].to_crs('EPSG:4326'),
            name='All DLR Lines',
            style_function=lambda x: {'color': 'red', 'weight': 1},
            show=True
        ).add_to(folium_map)

        # Add Network Buses
        for _, row in network_buses_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=2,
                color='blue',
                fill=True,
                fill_color='blue',
                popup=f"Network Bus {row['bus_idx']}"
            ).add_to(folium_map)

        # Add DLR Buses
        for _, row in dlr_buses_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=2,
                color='red',
                fill=True,
                fill_color='red',
                popup=f"DLR Bus {row['name']}"
            ).add_to(folium_map)

        # Add Matched Network Lines from Nearest Neighbor
        if matched_network_lines_nn is not None and not matched_network_lines_nn.empty:
            folium.GeoJson(
                matched_network_lines_nn[['geometry', 'name']].to_crs('EPSG:4326'),
                name='Matched Network Lines (Nearest Neighbor)',
                style_function=lambda x: {'color': 'green', 'weight': 2},
                tooltip=folium.GeoJsonTooltip(
                    fields=['name'],
                    aliases=['Matched Line NN'],
                    localize=True
                )
            ).add_to(folium_map)
        else:
            print("No matched network lines from Nearest Neighbor to visualize.")

        # Add Matched DLR Lines from Nearest Neighbor
        if matched_dlr_lines_nn is not None and not matched_dlr_lines_nn.empty:
            folium.GeoJson(
                matched_dlr_lines_nn[['geometry', 'name']].to_crs('EPSG:4326'),
                name='Matched DLR Lines (Nearest Neighbor)',
                style_function=lambda x: {'color': 'darkgreen', 'weight': 2},
                tooltip=folium.GeoJsonTooltip(
                    fields=['name'],
                    aliases=['Matched DLR Line NN'],
                    localize=True
                )
            ).add_to(folium_map)
        else:
            print("No matched DLR lines from Nearest Neighbor to visualize.")

        # Add Matched Network Lines from Buffer
        if matched_network_lines_buffer is not None and not matched_network_lines_buffer.empty:
            folium.GeoJson(
                matched_network_lines_buffer[['geometry', 'name']].to_crs('EPSG:4326'),
                name='Matched Network Lines (Buffer)',
                style_function=lambda x: {'color': 'orange', 'weight': 2},
                tooltip=folium.GeoJsonTooltip(
                    fields=['name'],
                    aliases=['Matched Line Buffer'],
                    localize=True
                )
            ).add_to(folium_map)
        else:
            print("No matched network lines from Buffer to visualize.")

        # Add Matched DLR Lines from Buffer
        if matched_dlr_lines_buffer is not None and not matched_dlr_lines_buffer.empty:
            folium.GeoJson(
                matched_dlr_lines_buffer[['geometry', 'name']].to_crs('EPSG:4326'),
                name='Matched DLR Lines (Buffer)',
                style_function=lambda x: {'color': 'darkorange', 'weight': 2},
                tooltip=folium.GeoJsonTooltip(
                    fields=['name'],
                    aliases=['Matched DLR Line Buffer'],
                    localize=True
                )
            ).add_to(folium_map)
        else:
            print("No matched DLR lines from Buffer to visualize.")

        # Add Layer Control
        folium.LayerControl().add_to(folium_map)

        # Add Legend
        legend_html = '''
        {% macro html(arg=None) %}
        <div style="
            position: fixed; 
            bottom: 50px; left: 50px; width: 220px; height: 180px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white;
            padding: 10px;
            ">
            &nbsp;<b>Legend</b><br>
            &nbsp;<i style="color:green">●</i>&nbsp;Matched Network Lines (Nearest Neighbor)<br>
            &nbsp;<i style="color:darkgreen">●</i>&nbsp;Matched DLR Lines (Nearest Neighbor)<br>
            &nbsp;<i style="color:orange">●</i>&nbsp;Matched Network Lines (Buffer)<br>
            &nbsp;<i style="color:darkorange">●</i>&nbsp;Matched DLR Lines (Buffer)<br>
            &nbsp;<i style="color:blue">●</i>&nbsp;All Network Lines<br>
            &nbsp;<i style="color:red">●</i>&nbsp;All DLR Lines<br>
        </div>
        {% endmacro %}
        '''

        legend = MacroElement()
        legend._template = Template(legend_html)

        folium_map.get_root().add_child(legend)

        # Save the map
        folium_map.save('full_map2.html')
        print("Map with all elements generated and saved as 'full_map2.html'.")