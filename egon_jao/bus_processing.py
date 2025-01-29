import traceback
import logging
import folium
from folium.plugins import MeasureControl, MousePosition
import geopandas as gpd
import pandas as pd


logger = logging.getLogger(__name__)

def read_data(config):
    """Reads the necessary data files into DataFrames."""
    network_buses_df = pd.read_csv(config['data_files']['network_buses_file'])



    dlr_buses_df = pd.read_csv(config['data_files']['dlr_buses_file'])

    logger.info(f"Loaded {len(network_buses_df)} network buses.")
    logger.info(f"Loaded {len(dlr_buses_df)} DLR buses.")
    return network_buses_df, dlr_buses_df



def filter_network_substations(network_buses_df, min_voltage_level):
    """Filters network buses to identify substations with voltage >= min_voltage_level."""
    if 'v_nom' not in network_buses_df.columns:
        logger.error("The 'v_nom' column is missing from network buses data.")
        raise KeyError("The 'v_nom' column is missing from network buses data.")
    substations_df = network_buses_df[network_buses_df['v_nom'] >= min_voltage_level].copy()
    logger.info(f"Filtered network buses to {len(substations_df)} substations with voltage >= {min_voltage_level} kV.")
    return substations_df



def prepare_geodataframes(substations_df, dlr_buses_df, config):
    """Converts DataFrames to GeoDataFrames with geometries."""
    substations_gdf = gpd.GeoDataFrame(
        substations_df,
        geometry=gpd.points_from_xy(substations_df['x'], substations_df['y']),
        crs=config['coordinate_reference_systems']['crs']
    )

    # Ensure unique ID for substations
    if 'bus_id' not in substations_gdf.columns:
        substations_gdf = substations_gdf.reset_index().rename(columns={'index': 'bus_id'})

    dlr_buses_gdf = gpd.GeoDataFrame(
        dlr_buses_df,
        geometry=gpd.points_from_xy(dlr_buses_df['x'], dlr_buses_df['y']),
        crs=config['coordinate_reference_systems']['crs']
    )
    # Ensure 'id_dlr' column
    if 'id_dlr' not in dlr_buses_gdf.columns:
        dlr_buses_gdf.rename(columns={'name': 'id_dlr'}, inplace=True)
    return substations_gdf, dlr_buses_gdf


def load_country_borders():
    try:
        return gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except AttributeError:
        return gpd.read_file('https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip')


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
    possible_name_columns = ['name', 'NAME', 'admin', 'ADMIN']
    country_col = None
    for col in possible_name_columns:
        if col in world.columns:
            country_col = col
            break
    if not country_col:
        raise KeyError("Country name column not found in world dataset.")

    # Filter for Germany and ensure CRS is WGS84
    germany_gdf = world[world[country_col] == 'Germany'].to_crs('EPSG:4326')

    if germany_gdf.empty:
        raise ValueError("Germany boundary not found in the dataset.")

    logger.info("Germany boundary successfully loaded.")
    return germany_gdf

def filter_dlr_substations_inside_germany(dlr_buses_gdf, germany_gdf):
    dlr_buses_gdf = dlr_buses_gdf.to_crs(germany_gdf.crs)
    dlr_buses_in_germany = gpd.sjoin(dlr_buses_gdf, germany_gdf, how='inner', predicate='intersects')
    dlr_buses_in_germany = dlr_buses_in_germany.drop(columns=['index_right']).reset_index(drop=True)
    logger.info(f"DLR substations inside Germany: {len(dlr_buses_in_germany)}")
    return dlr_buses_in_germany


def filter_network_substations_inside_germany(substations_gdf, germany_gdf):

    logger.debug(f"Number of substations before filtering inside Germany: {len(substations_gdf)}")

    # Ensure CRS matches
    if substations_gdf.crs != germany_gdf.crs:
        logger.debug(f"Reprojecting substations from {substations_gdf.crs} to {germany_gdf.crs}")
        substations_gdf = substations_gdf.to_crs(germany_gdf.crs)

    # Spatial join to determine which substations are inside Germany
    substations_in_germany = gpd.sjoin(substations_gdf, germany_gdf, how='inner', predicate='intersects')
    substations_in_germany = substations_in_germany.drop(columns=['index_right']).reset_index(drop=True)

    logger.debug(f"Number of substations after spatial join: {len(substations_in_germany)}")

    # Exclude <=110 kV substations
    initial_count = len(substations_in_germany)
    substations_filtered = substations_in_germany[substations_in_germany['v_nom'] > 110]
    logger.debug(f"Number of substations after excluding <=110 kV: {len(substations_filtered)} (from {initial_count})")

    logger.info(f"Network substations inside Germany after excluding <=110 kV: {len(substations_filtered)}")

    logger.debug(f"substations_gdf head:\n{substations_gdf.head()}")
    logger.debug(f"Number of substations before filtering inside Germany: {len(substations_gdf)}")

    return substations_filtered


def get_transformer_buses_from_network_transformers(network_transformers_file):
    """Extracts all bus IDs from the network transformers data."""
    df = pd.read_csv(network_transformers_file)
    # Assuming columns: 'bus0', 'bus1'
    bus_set = set(df['bus0'].unique()) | set(df['bus1'].unique())
    return bus_set

def get_transformer_buses_from_dlr_transformers(dlr_transformers_file):
    """Extracts all bus IDs from the DLR transformers data."""
    df = pd.read_csv(dlr_transformers_file)
    # Assuming columns: 'bus0', 'bus1' in dlr_transformers
    bus_set = set(df['bus0'].unique()) | set(df['bus1'].unique())
    return bus_set

def filter_buses_by_transformers(network_buses_df, dlr_buses_df, network_transformers_file, dlr_transformers_file):
    """
    Filters the network and DLR buses to only those connected to transformers.
    """
    transformer_buses_network = get_transformer_buses_from_network_transformers(network_transformers_file)
    transformer_buses_dlr = get_transformer_buses_from_dlr_transformers(dlr_transformers_file)

    # Filter network buses using 'bus_id'
    network_buses_filtered = network_buses_df[network_buses_df['bus_id'].isin(transformer_buses_network)].copy()

    # Filter DLR buses
    if 'id_dlr' in dlr_buses_df.columns:
        dlr_buses_filtered = dlr_buses_df[dlr_buses_df['id_dlr'].isin(transformer_buses_dlr)].copy()
    else:
        logger.error("id_dlr column not found in dlr_buses_df.")
        raise KeyError("The dlr_buses_df should have 'id_dlr' column after rename.")

    logger.debug(f"After filter_buses_by_transformers: {len(network_buses_filtered)} network buses, {len(dlr_buses_filtered)} DLR buses.")
    logger.debug(f"Sample filtered_network_buses_df:\n{network_buses_filtered.head()}")
    return network_buses_filtered, dlr_buses_filtered



def spatial_match_substations(substations_gdf, dlr_buses_gdf, config):

    logger.debug(f"Number of substations before matching: {len(substations_gdf)}")
    logger.debug(f"Number of DLR buses before matching: {len(dlr_buses_gdf)}")
    substations_gdf = substations_gdf.copy()
    dlr_buses_gdf = dlr_buses_gdf.copy()

    # Ensure 'bus_id' is unique
    if not substations_gdf['bus_id'].is_unique:
        logger.error("Duplicate 'bus_id' values found in network substations. Please ensure 'bus_id' is unique.")
        raise ValueError("Duplicate 'bus_id' values found in network substations.")

    # Ensure 'id_dlr' is unique
    if not dlr_buses_gdf['id_dlr'].is_unique:
        logger.error("Duplicate 'id_dlr' values found in DLR substations. Please ensure 'id_dlr' is unique.")
        raise ValueError("Duplicate 'id_dlr' values found in DLR substations.")

    # Reproject to projected CRS for accurate buffering and distance calculations
    projected_crs = config['coordinate_reference_systems']['projected_crs']
    substations_gdf = substations_gdf.to_crs(projected_crs)
    dlr_buses_gdf = dlr_buses_gdf.to_crs(projected_crs)

    # Create a separate buffer column to retain original Point geometries
    buffer_distance = config['parameters']['buffer_distance']
    substations_gdf['buffer'] = substations_gdf.geometry.buffer(buffer_distance)

    # Set 'buffer' as the active geometry for spatial join
    substations_buffer_gdf = substations_gdf[['bus_id', 'buffer']].set_geometry('buffer')

    # Perform spatial join to find DLR substations within buffer zones
    possible_matches = gpd.sjoin(substations_buffer_gdf, dlr_buses_gdf, how='left', predicate='intersects')

    # Drop rows where 'index_right' is NaN (no DLR substation within buffer)
    possible_matches = possible_matches.dropna(subset=['index_right']).copy()

    # Convert 'index_right' to integer type
    possible_matches['index_right'] = possible_matches['index_right'].astype(int)

    # Reset index to ensure alignment
    possible_matches = possible_matches.reset_index(drop=True)

    # Retrieve the original Point geometries for accurate distance calculation
    network_geom = substations_gdf.set_index('bus_id').loc[possible_matches['bus_id'], 'geometry']
    matched_dlr_geom = dlr_buses_gdf.loc[possible_matches['index_right'], 'geometry']

    # Ensure no duplicates by resetting indices
    network_geom = network_geom.reset_index(drop=True)
    matched_dlr_geom = matched_dlr_geom.reset_index(drop=True)

    # Calculate distances in projected CRS
    possible_matches['distance'] = network_geom.distance(matched_dlr_geom)

    # Proceed with one-to-one matching based on the shortest distance
    possible_matches = possible_matches.sort_values('distance')

    matched_network_ids = set()
    matched_dlr_ids = set()
    matches = []

    for idx, row in possible_matches.iterrows():
        net_id = row['bus_id']
        dlr_id = row['index_right']

        if net_id not in matched_network_ids and dlr_id not in matched_dlr_ids:
            matched_network_ids.add(net_id)
            matched_dlr_ids.add(dlr_id)
            matches.append(row)

    # Create GeoDataFrame of matched pairs with original Point geometries
    matched_substations_gdf = substations_gdf.set_index('bus_id').loc[list(matched_network_ids)].copy()
    matched_substations_gdf = matched_substations_gdf.reset_index()
    matched_substations_gdf['id_dlr'] = dlr_buses_gdf.loc[list(matched_dlr_ids), 'id_dlr'].values

    # Create GeoDataFrame for matched DLR buses
    matched_dlr_buses_gdf = dlr_buses_gdf.loc[list(matched_dlr_ids)].copy()
    matched_dlr_buses_gdf = matched_dlr_buses_gdf.reset_index(drop=True)

    # Identify unmatched network substations
    unmatched_network_ids = set(substations_gdf['bus_id']) - matched_network_ids
    unmatched_substations_gdf = substations_gdf[substations_gdf['bus_id'].isin(unmatched_network_ids)].copy()

    # Identify unmatched DLR substations
    unmatched_dlr_ids = set(dlr_buses_gdf.index) - matched_dlr_ids
    unmatched_dlr_buses_gdf = dlr_buses_gdf.loc[list(unmatched_dlr_ids)].copy()

    # Logging the results
    logger.info(f"Total matches found: {len(matched_substations_gdf)}")
    logger.info(f"Total unmatched network substations: {len(unmatched_substations_gdf)}")


    logger.debug(f"Number of matched substations: {len(matched_substations_gdf)}")
    logger.debug(f"Number of unmatched substations: {len(unmatched_substations_gdf)}")
    logger.debug(f"Number of matched DLR buses: {len(matched_dlr_buses_gdf)}")
    logger.debug(f"Number of unmatched DLR buses: {len(unmatched_dlr_buses_gdf)}")

    # Make sure to return the actual computed values
    return matched_substations_gdf, unmatched_substations_gdf, matched_dlr_buses_gdf, unmatched_dlr_buses_gdf


def save_csv_files(matched_substations_gdf, unmatched_substations_gdf, unmatched_dlr_buses_gdf, config):
    """Saves the matched and unmatched substations to CSV files."""
    # Reproject back to original CRS if needed
    matched_substations_gdf = matched_substations_gdf.to_crs(config['coordinate_reference_systems']['crs'])
    unmatched_substations_gdf = unmatched_substations_gdf.to_crs(config['coordinate_reference_systems']['crs'])
    unmatched_dlr_buses_gdf = unmatched_dlr_buses_gdf.to_crs(config['coordinate_reference_systems']['crs'])

    # **Save matched substations**
    matched_substations_df = matched_substations_gdf.drop(columns=['geometry', 'index_right'], errors='ignore')
    matched_substations_df.to_csv(config['output_files']['matched_substations_csv'], index=False)
    logger.info(f"Matched substations saved to '{config['output_files']['matched_substations_csv']}'.")

    # **Save unmatched substations**
    unmatched_substations_df = unmatched_substations_gdf.drop(columns=['geometry'], errors='ignore')
    unmatched_substations_df.to_csv(config['output_files']['unmatched_substations_csv'], index=False)
    logger.info(f"Unmatched substations saved to '{config['output_files']['unmatched_substations_csv']}'.")

    # **Save unmatched DLR substations**
    unmatched_dlr_buses_df = unmatched_dlr_buses_gdf.drop(columns=['geometry'], errors='ignore')
    unmatched_dlr_buses_df.to_csv(config['output_files']['unmatched_dlr_substations_csv'], index=False)
    logger.info(f"Unmatched DLR substations saved to '{config['output_files']['unmatched_dlr_substations_csv']}'.")


def create_matched_unmatched_maps(matched_substations_gdf, unmatched_substations_gdf, matched_dlr_buses_gdf, unmatched_dlr_buses_gdf, config, germany_gdf):
    """Creates maps for matched and unmatched substations using folium.Circle."""
    buffer_radius = config['parameters']['buffer_distance']  # Radius in meters

    # Create Matched Substations Map
    create_folium_map_with_circles(
        matched_substations_gdf,
        matched_dlr_buses_gdf,        # Pass only matched DLR substations for green markers
        gpd.GeoDataFrame(columns=['id_dlr', 'geometry'], crs='EPSG:4326'),  # Empty GeoDataFrame with correct columns and CRS
        config['map_files']['matched_substations_map'],
        'Matched Substations Map',
        buffer_radius,
        germany_gdf
    )

    # Create Unmatched Substations Map
    create_folium_map_with_circles(
        unmatched_substations_gdf,
        gpd.GeoDataFrame(columns=['id_dlr', 'geometry'], crs='EPSG:4326'),  # Empty GeoDataFrame with correct columns and CRS
        unmatched_dlr_buses_gdf,      # Pass only unmatched DLR substations for red markers
        config['map_files']['unmatched_substations_map'],
        'Unmatched Substations Map',
        buffer_radius,
        germany_gdf
    )

def add_legend(folium_map):
    """Adds a legend to the Folium map."""
    legend_html = '''
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 220px; height: 220px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     padding: 10px;
     ">
     &nbsp;<b>Legend</b><br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:blue"></i>&nbsp; Network Substation Buffer Zone<br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:black"></i>&nbsp; Regular Network Substation<br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:orange"></i>&nbsp; Network Substation with >1 DLR Substations<br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:green"></i>&nbsp; Matched DLR Substation<br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:red"></i>&nbsp; Unmatched DLR Substation
     </div>
     '''
    folium_map.get_root().html.add_child(folium.Element(legend_html))





def create_folium_map_with_circles(substations_gdf, matched_dlr_buses_gdf, unmatched_dlr_buses_gdf, output_path, map_title, buffer_radius, germany_gdf):
    """Creates a Folium map with circle buffers drawn using folium.Circle."""
    # Reproject to WGS84 for Folium
    substations_gdf = substations_gdf.to_crs('EPSG:4326')
    matched_dlr_buses_gdf = matched_dlr_buses_gdf.to_crs('EPSG:4326')
    unmatched_dlr_buses_gdf = unmatched_dlr_buses_gdf.to_crs('EPSG:4326') if not unmatched_dlr_buses_gdf.empty else gpd.GeoDataFrame()

    if substations_gdf.empty:
        logger.warning(f"No substations to display on the map: {map_title}")
        return

    # Verify Geometry Types Before Mapping
    geom_types = substations_gdf.geometry.geom_type.unique()
    logger.info(f"Geometry types in '{map_title}': {geom_types}")

    # Check if all geometries are Points
    if not all(geom_type == 'Point' for geom_type in geom_types):
        logger.error(f"Not all geometries are Points in '{map_title}'. Skipping mapping.")
        return

    # Create a base map centered on Germany
    folium_map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)

    # Add Germany boundary
    folium.GeoJson(
        germany_gdf,
        name='Germany Boundary',
        style_function=lambda x: {'fillColor': '#00000000', 'color': 'black', 'weight': 1},
        show=True
    ).add_to(folium_map)

    # Add MeasureControl
    folium_map.add_child(MeasureControl(
        position='topleft',
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        secondary_area_unit='hectares',
        active_color='orange',
        completed_color='red',
    ))

    # Add MousePosition
    formatter = "function(num) {return L.Util.formatNum(num, 5);};"
    mouse_position = MousePosition(
        position='bottomright',
        separator=' | ',
        empty_string='NaN',
        lng_first=True,
        num_digits=5,
        prefix='Coordinates:',
        lat_formatter=formatter,
        lng_formatter=formatter,
    )
    folium_map.add_child(mouse_position)

    # Add substations buffers and center points
    for idx, row in substations_gdf.iterrows():
        # Add buffer circle
        folium.Circle(
            location=(row.geometry.y, row.geometry.x),
            radius=buffer_radius,  # Radius in meters
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.1,
            weight=0.5,
            popup=f"Network Substation ID: {row['bus_id']}"
        ).add_to(folium_map)

        # Add center point marker (black)
        folium.CircleMarker(
            location=(row.geometry.y, row.geometry.x),
            radius=3,
            color='black',
            fill=True,
            fill_color='black',
            popup=f"Network Substation ID: {row['bus_id']}"
        ).add_to(folium_map)

    # Add matched DLR substations (green) if any
    if not matched_dlr_buses_gdf.empty:
        for idx, row in matched_dlr_buses_gdf.iterrows():
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),
                radius=3,
                color='green',
                fill=True,
                fill_color='green',
                popup=f"Matched DLR Substation ID: {row['id_dlr']}"
            ).add_to(folium_map)

    # Add unmatched DLR substations (red) if any
    if not unmatched_dlr_buses_gdf.empty:
        for idx, row in unmatched_dlr_buses_gdf.iterrows():
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                popup=f"Unmatched DLR Substation ID: {row['id_dlr']}"
            ).add_to(folium_map)

    # Add Layer Control
    folium.LayerControl().add_to(folium_map)

    # Add Legend
    add_legend(folium_map)

    # Save the map
    folium_map.save(output_path)
    logger.info(f"{map_title} saved to '{output_path}'.")


def create_total_map(
    network_substations_gdf,
    matched_dlr_buses_gdf,
    unmatched_dlr_buses_gdf,
    config,
    germany_gdf
):
    """
    Creates a total map showing both matched and unmatched network and DLR substations
    with buffer zones around network substations only. Highlights network substations
    with more than one DLR substation in orange.
    """


    logger.info("Starting creation of the total map.")

    try:
        buffer_radius = config['parameters']['buffer_distance']  # Radius in meters

        # Retrieve the output path from config
        output_path = config['map_files']['total_map']
        logger.info(f"Output path for Total Map: {output_path}")

        # Exclude 110 kV substations
        # Assuming 'v_nom' is the column that contains the voltage level
        network_substations_gdf = network_substations_gdf[network_substations_gdf['v_nom'] > 110]
        logger.info(f"Number of network substations after excluding 110 kV: {len(network_substations_gdf)}")

        # Check if network_substations_gdf is empty
        if network_substations_gdf.empty:
            logger.warning("No network substations to display on the total map.")
            return

        # Verify Geometry Types Before Mapping
        geom_types = network_substations_gdf.geometry.geom_type.unique()
        logger.info(f"Geometry types in 'Total Map': {geom_types}")

        # Check if all geometries are Points
        if not all(geom_type == 'Point' for geom_type in geom_types):
            logger.error("Not all geometries are Points in 'Total Map'. Skipping mapping.")
            return
        else:
            logger.info("All geometries in network_substations_gdf are Points.")

        # Create a base map centered on Germany
        folium_map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)
        logger.info("Base map created.")

        # Add Germany boundary
        folium.GeoJson(
            germany_gdf,
            name='Germany Boundary',
            style_function=lambda x: {'fillColor': '#00000000', 'color': 'black', 'weight': 1},
            show=True
        ).add_to(folium_map)
        logger.info("Germany boundary added to the map.")

        # Add MeasureControl
        folium_map.add_child(MeasureControl(
            position='topleft',
            primary_length_unit='meters',
            secondary_length_unit='kilometers',
            primary_area_unit='sqmeters',
            secondary_area_unit='hectares',
            active_color='orange',
            completed_color='red',
        ))
        logger.info("MeasureControl added to the map.")

        # Add MousePosition
        formatter = "function(num) {return L.Util.formatNum(num, 5);};"
        mouse_position = MousePosition(
            position='bottomright',
            separator=' | ',
            empty_string='NaN',
            lng_first=True,
            num_digits=5,
            prefix='Coordinates:',
            lat_formatter=formatter,
            lng_formatter=formatter,
        )
        folium_map.add_child(mouse_position)
        logger.info("MousePosition added to the map.")

        # Add network substations buffers and center points
        logger.info("Adding network substations buffers and markers.")
        orange_count = 0  # Counter for orange markers

        for idx, row in network_substations_gdf.iterrows():
            try:
                num_dlr = row.get('num_dlr_substations', 0)
                logger.debug(f"Processing Network Substation ID: {row['bus_id']} with num_dlr_substations: {num_dlr}")

                if num_dlr > 1:
                    marker_color = 'orange'
                    fill_color = 'orange'
                    marker_popup = f"Network Substation ID: {row['bus_id']} (DLR Substations: {num_dlr})"
                    orange_count += 1
                    logger.debug(f"Adding orange marker for ID: {row['bus_id']}")
                else:
                    marker_color = 'black'
                    fill_color = 'black'
                    marker_popup = f"Network Substation ID: {row['bus_id']}"
                    logger.debug(f"Adding black marker for ID: {row['bus_id']}")

                # Add buffer circle
                folium.Circle(
                    location=(row.geometry.y, row.geometry.x),
                    radius=buffer_radius,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.1,
                    weight=1.0,
                    popup=f"Network Substation ID: {row['bus_id']}"
                ).add_to(folium_map)

                # Add center point marker with conditional color
                folium.CircleMarker(
                    location=(row.geometry.y, row.geometry.x),
                    radius=7,
                    color=marker_color,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=0.9,
                    popup=marker_popup
                ).add_to(folium_map)
            except Exception as e:
                logger.error(f"Error adding network substation ID {row['bus_id']}: {e}")

        logger.info(f"Network substations added to the map. Number of orange markers: {orange_count}")

        # Add matched DLR substations (green) if any
        if not matched_dlr_buses_gdf.empty:
            logger.info(f"Adding {len(matched_dlr_buses_gdf)} matched DLR substations.")
            for idx, row in matched_dlr_buses_gdf.iterrows():
                try:
                    logger.debug(f"Adding Matched DLR Substation ID: {row['id_dlr']} at (Lat: {row.geometry.y}, Lon: {row.geometry.x})")
                    folium.CircleMarker(
                        location=(row.geometry.y, row.geometry.x),
                        radius=7,
                        color='green',
                        fill=True,
                        fill_color='green',
                        fill_opacity=0.9,
                        popup=f"Matched DLR Substation ID: {row['id_dlr']}"
                    ).add_to(folium_map)
                except Exception as e:
                    logger.error(f"Error adding matched DLR substation ID {row['id_dlr']}: {e}")
        else:
            logger.info("No matched DLR substations to add.")

        # Add unmatched DLR substations (red) if any
        if not unmatched_dlr_buses_gdf.empty:
            logger.info(f"Adding {len(unmatched_dlr_buses_gdf)} unmatched DLR substations.")
            for idx, row in unmatched_dlr_buses_gdf.iterrows():
                try:
                    logger.debug(f"Adding Unmatched DLR Substation ID: {row['id_dlr']} at (Lat: {row.geometry.y}, Lon: {row.geometry.x})")
                    folium.CircleMarker(
                        location=(row.geometry.y, row.geometry.x),
                        radius=7,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.9,
                        popup=f"Unmatched DLR Substation ID: {row['id_dlr']}"
                    ).add_to(folium_map)
                except Exception as e:
                    logger.error(f"Error adding unmatched DLR substation ID {row['id_dlr']}: {e}")
        else:
            logger.info("No unmatched DLR substations to add.")

        logger.info(f"Total number of orange markers added: {orange_count}")

        # Add Layer Control
        folium.LayerControl().add_to(folium_map)
        logger.info("LayerControl added to the map.")

        # Add Legend
        add_legend(folium_map)
        logger.info("Legend added to the map.")

        # Save the map
        try:
            folium_map.save(output_path)
            logger.info(f"Total Map saved to '{output_path}'.")
        except Exception as e:
            logger.error(f"Error saving the total map to '{output_path}': {e}")

    except Exception as e:
        logger.error(f"An error occurred while creating the total map: {e}")
        traceback.print_exc()


def create_total_map_test(network_substations_gdf, matched_dlr_buses_gdf, unmatched_dlr_buses_gdf, config, germany_gdf):
    """Creates a test Folium map with a few network substations and DLR substations."""
    buffer_radius = config['parameters']['buffer_distance']

    # Reproject to WGS84 for Folium
    network_substations_gdf = network_substations_gdf.to_crs('EPSG:4326')
    matched_dlr_buses_gdf = matched_dlr_buses_gdf.to_crs('EPSG:4326')
    unmatched_dlr_buses_gdf = unmatched_dlr_buses_gdf.to_crs('EPSG:4326') if not unmatched_dlr_buses_gdf.empty else gpd.GeoDataFrame()

    if network_substations_gdf.empty:
        logger.warning("No network substations to display on the test map.")
        return

    # Verify Geometry Types Before Mapping
    geom_types = network_substations_gdf.geometry.geom_type.unique()
    logger.info(f"Geometry types in 'Test Total Map': {geom_types}")

    # Check if all geometries are Points
    if not all(geom_type == 'Point' for geom_type in geom_types):
        logger.error("Not all geometries are Points in 'Test Total Map'. Skipping mapping.")
        return
    else:
        logger.info("All geometries in network_substations_gdf are Points.")

    # Create a base map centered on Germany
    folium_map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)
    logger.info("Base map for test created.")

    # Add Germany boundary
    folium.GeoJson(
        germany_gdf.to_crs('EPSG:4326'),
        name='Germany Boundary',
        style_function=lambda x: {'fillColor': '#00000000', 'color': 'black', 'weight': 1},
        show=True
    ).add_to(folium_map)
    logger.info("Germany boundary added to the test map.")

    # Add network substations buffers and center points
    logger.info("Adding network substations buffers and markers to the test map.")
    for idx, row in network_substations_gdf.iterrows():
        try:
            logger.debug(f"Processing Network Substation ID: {row['bus_id']} at (Lat: {row.geometry.y}, Lon: {row.geometry.x})")

            # Add buffer circle
            folium.Circle(
                location=(row.geometry.y, row.geometry.x),
                radius=buffer_radius,  # e.g., 5 km
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.3,
                weight=1.0,
                popup=f"Network Substation ID: {row['bus_id']}"
            ).add_to(folium_map)

            # Add center point marker (black)
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),
                radius=3,
                color='black',
                fill=True,
                fill_color='black',
                fill_opacity=0.5,
                popup=f"Network Substation ID: {row['bus_id']}"
            ).add_to(folium_map)
        except Exception as e:
            logger.error(f"Error adding network substation ID {row['bus_id']}: {e}")

    logger.info("Network substations added to the test map.")

    # Add matched DLR substations (green) if any
    if not matched_dlr_buses_gdf.empty:
        logger.info(f"Adding {len(matched_dlr_buses_gdf)} matched DLR substations to the test map.")
        for idx, row in matched_dlr_buses_gdf.iterrows():
            try:
                logger.debug(f"Adding Matched DLR Substation ID: {row['id_dlr']} at (Lat: {row.geometry.y}, Lon: {row.geometry.x})")
                folium.CircleMarker(
                    location=(row.geometry.y, row.geometry.x),
                    radius=3,
                    color='green',
                    fill=True,
                    fill_color='green',
                    fill_opacity=0.5,
                    popup=f"Matched DLR Substation ID: {row['id_dlr']}"
                ).add_to(folium_map)
            except Exception as e:
                logger.error(f"Error adding matched DLR substation ID {row['id_dlr']}: {e}")
    else:
        logger.info("No matched DLR substations to add to the test map.")

    # Add unmatched DLR substations (red) if any
    if not unmatched_dlr_buses_gdf.empty:
        logger.info(f"Adding {len(unmatched_dlr_buses_gdf)} unmatched DLR substations to the test map.")
        for idx, row in unmatched_dlr_buses_gdf.iterrows():
            try:
                logger.debug(f"Adding Unmatched DLR Substation ID: {row['id_dlr']} at (Lat: {row.geometry.y}, Lon: {row.geometry.x})")
                folium.CircleMarker(
                    location=(row.geometry.y, row.geometry.x),
                    radius=3,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.5,
                    popup=f"Unmatched DLR Substation ID: {row['id_dlr']}"
                ).add_to(folium_map)
            except Exception as e:
                logger.error(f"Error adding unmatched DLR substation ID {row['id_dlr']}: {e}")
    else:
        logger.info("No unmatched DLR substations to add to the test map.")

    # Add Layer Control
    folium.LayerControl().add_to(folium_map)
    logger.info("LayerControl added to the test map.")

    # Add Legend
    add_legend(folium_map)
    logger.info("Legend added to the test map.")

    # Save the map
    try:
        folium_map.save('results/maps/test_total_substations_map.html')
        logger.info("Test Total Map saved to 'results/maps/test_total_substations_map.html'.")
    except Exception as e:
        logger.error(f"Error saving the test map: {e}")



def count_dlr_substations_in_buffers(network_substations_gdf, dlr_buses_gdf, config):
    """
    Counts the number of DLR substations within each network substation's buffer zone.
    Each DLR substation is counted only once, assigned to the nearest network substation within buffer.
    Collects the IDs of the DLR substations within each buffer.

    Args:
        network_substations_gdf (GeoDataFrame): GeoDataFrame of network substations with buffer geometries.
        dlr_buses_gdf (GeoDataFrame): GeoDataFrame of DLR substations.
        config (dict): Configuration dictionary containing 'parameters' -> 'buffer_distance'.

    Returns:
        pd.DataFrame: DataFrame with 'bus_id', 'num_dlr_substations', and 'id_dlr' columns.
    """

    logger.info("Counting DLR substations within each network substation's buffer zone.")

    # Ensure 'buffer' column exists
    if 'buffer' not in network_substations_gdf.columns:
        logger.error("The 'buffer' column is missing from network_substations_gdf.")
        raise KeyError("The 'buffer' column is missing from network_substations_gdf.")

    # Reproject DLR buses to match network_substations_gdf CRS if necessary
    if dlr_buses_gdf.crs != network_substations_gdf.crs:
        logger.info(f"Reprojecting DLR buses from {dlr_buses_gdf.crs} to {network_substations_gdf.crs}.")
        dlr_buses_gdf = dlr_buses_gdf.to_crs(network_substations_gdf.crs)

    # Create a GeoDataFrame with buffer geometries
    buffers_gdf = gpd.GeoDataFrame(
        network_substations_gdf[['bus_id', 'geometry']],
        geometry=network_substations_gdf['buffer'],
        crs=network_substations_gdf.crs
    )

    # Spatial join: Assign DLR substations within buffers
    try:
        joined = gpd.sjoin(dlr_buses_gdf[['id_dlr', 'geometry']], buffers_gdf, how='left', predicate='within')
    except Exception as e:
        logger.error(f"Error during spatial join: {e}")
        raise

    # Remove DLR substations not within any buffer
    joined = joined.dropna(subset=['bus_id']).copy()

    # Calculate distance between DLR substation and network substation
    joined['distance'] = joined.apply(
        lambda row: row['geometry'].distance(
            network_substations_gdf.set_index('bus_id').loc[row['bus_id'], 'geometry']
        ),
        axis=1
    )

    # Sort by 'id_dlr' and 'distance' to get the nearest network substation first
    joined_sorted = joined.sort_values(by=['id_dlr', 'distance'])

    # Assign each DLR substation to the nearest network substation's buffer
    unique_assignments = joined_sorted.drop_duplicates(subset=['id_dlr'], keep='first')

    # Group by 'bus_id' to collect DLR IDs
    grouped = unique_assignments.groupby('bus_id').agg({
        'id_dlr': lambda x: list(x)
    }).reset_index()

    # Calculate the number of DLR substations per network substation
    grouped['num_dlr_substations'] = grouped['id_dlr'].apply(len)

    # Merge with all network substations to include zeros
    counts_full = network_substations_gdf[['bus_id']].merge(
        grouped[['bus_id', 'num_dlr_substations', 'id_dlr']],
        on='bus_id',
        how='left'
    )

    # Fill NaN values in 'num_dlr_substations' with 0
    counts_full['num_dlr_substations'] = counts_full['num_dlr_substations'].fillna(0).astype(int)

    # Fill NaN values in 'id_dlr' with empty lists
    counts_full['id_dlr'] = counts_full['id_dlr'].apply(lambda x: x if isinstance(x, list) else [])

    # Debugging: Inspect counts_full
    logger.debug(f"counts_full columns: {counts_full.columns.tolist()}")
    logger.debug(f"counts_full sample:\n{counts_full.head()}")

    logger.info("Completed counting DLR substations within buffer zones.")

    return counts_full



def save_dlr_substations_count_csv(counts_df, config):
    """
    Saves the DLR substations count per network substation to a CSV file, with DLR IDs in separate columns.
    Computes 'has_dlr_without_T' based on 'dlr_id_2' and 'dlr_id_3' columns.

    Args:
        counts_df (pd.DataFrame): DataFrame with 'bus_id', 'num_dlr_substations', and 'id_dlr' columns.
        config (dict): Configuration dictionary containing 'output_files' paths.

    Returns:
        None
    """

    output_path = config['output_files'].get('dlr_substations_count_csv', 'results/csv/dlr_substations_count.csv')

    try:
        logger.debug(f"counts_df columns: {counts_df.columns.tolist()}")

        # Check if 'id_dlr' column exists
        if 'id_dlr' not in counts_df.columns:
            logger.error("'id_dlr' column is missing from counts_df.")
            raise KeyError("'id_dlr' column is required but not found in counts_df.")

        # Find the maximum number of DLR IDs associated with any network substation
        max_dlr_ids = counts_df['id_dlr'].apply(len).max()
        logger.debug(f"Maximum number of DLR IDs per network substation: {max_dlr_ids}")

        # Create new columns for each DLR ID
        for i in range(max_dlr_ids):
            counts_df[f'dlr_id_{i + 1}'] = counts_df['id_dlr'].apply(lambda x: x[i] if i < len(x) else '')
            logger.debug(f"Created column 'dlr_id_{i + 1}'")

        # Define the DLR ID columns to use for computing 'has_dlr_without_T'
        # Adjust the range based on max_dlr_ids
        dlr_id_columns = [f'dlr_id_{i + 1}' for i in range(max_dlr_ids)]

        # If you specifically want to use 'dlr_id_2' and 'dlr_id_3', ensure they exist
        specific_dlr_id_columns = ['dlr_id_2', 'dlr_id_3']
        for col in specific_dlr_id_columns:
            if col not in counts_df.columns:
                counts_df[col] = ''
                logger.debug(f"Created column '{col}' for 'has_dlr_without_T' computation.")

        # Apply the logic to compute 'has_dlr_without_T'
        counts_df['has_dlr_without_T'] = counts_df[specific_dlr_id_columns].apply(
            lambda row: any(
                dlr_id != '' and not str(dlr_id).endswith('T')
                for dlr_id in row
            ),
            axis=1
        )
        logger.debug("Computed 'has_dlr_without_T' column.")

        # Drop the 'id_dlr' column as we've expanded it into separate columns
        counts_df = counts_df.drop(columns=['id_dlr'])
        logger.debug("Dropped 'id_dlr' column.")

        # Save to CSV
        counts_df.to_csv(output_path, index=False)
        logger.info(f"DLR substations count per network substation saved to '{output_path}'.")
    except KeyError as e:
        logger.error(f"Failed to save DLR substations count CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving DLR substations count CSV: {e}")
        raise


def create_filtered_substations_map(
    network_substations_gdf,
    dlr_buses_gdf,
    config,
    germany_gdf
):
    """
    Creates a map showing only the network substations that have more than one DLR substation.
    Substations with DLR IDs without 'T' are displayed as green markers.
    Substations with DLR IDs including 'T' are displayed as yellow markers.
    DLR substations within the buffer are displayed as red markers.
    The popup for each network substation includes the IDs of the DLR substations within its buffer.
    """

    # Retrieve the output path from config
    output_path = config['map_files']['filtered_substations_map']
    logger.info(f"Output path for Filtered Substations Map: {output_path}")

    # Check if network_substations_gdf is empty
    if network_substations_gdf.empty:
        logger.warning("No network substations to display on the map.")
        return

    # Verify Geometry Types Before Mapping
    geom_types = network_substations_gdf.geometry.geom_type.unique()
    logger.debug(f"Geometry types in network_substations_gdf: {geom_types}")

    # Check if all geometries are Points
    if not all(geom_type == 'Point' for geom_type in geom_types):
        logger.error("Not all geometries are Points in network_substations_gdf. Skipping mapping.")
        return

    # Create a base map centered on Germany
    folium_map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)
    logger.info("Base map created.")

    # Add Germany boundary
    folium.GeoJson(
        germany_gdf,
        name='Germany Boundary',
        style_function=lambda x: {'fillColor': '#00000000', 'color': 'black', 'weight': 1},
        show=True
    ).add_to(folium_map)
    logger.info("Germany boundary added to the map.")

    # Add MeasureControl
    folium_map.add_child(MeasureControl(
        position='topleft',
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        secondary_area_unit='hectares',
        active_color='orange',
        completed_color='red',
    ))
    logger.info("MeasureControl added to the map.")

    # Add MousePosition
    formatter = "function(num) {return L.Util.formatNum(num, 5);};"
    mouse_position = MousePosition(
        position='bottomright',
        separator=' | ',
        empty_string='NaN',
        lng_first=True,
        num_digits=5,
        prefix='Coordinates:',
        lat_formatter=formatter,
        lng_formatter=formatter,
    )
    folium_map.add_child(mouse_position)
    logger.info("MousePosition added to the map.")

    # Define buffer radius in meters
    buffer_radius_meters = 500

    # Ensure CRS is projected (meters) for buffering
    projected_crs = 'EPSG:3857'  # Web Mercator projection
    original_crs = network_substations_gdf.crs

    # Reproject GeoDataFrames to Projected CRS for accurate buffering
    network_substations_proj = network_substations_gdf.to_crs(projected_crs)
    dlr_buses_proj = dlr_buses_gdf.to_crs(projected_crs)
    logger.debug(f"Columns in network_substations_proj after reprojection: {network_substations_proj.columns.tolist()}")

    # Create buffers around each network substation
    network_substations_proj['buffer'] = network_substations_proj.geometry.buffer(buffer_radius_meters)
    logger.info(f"Buffers of {buffer_radius_meters} meters created around network substations.")

    # Create a GeoDataFrame for buffers
    buffers_gdf = gpd.GeoDataFrame(
        network_substations_proj[['bus_id', 'num_dlr_substations', 'has_dlr_without_T']],
        geometry=network_substations_proj['buffer'],
        crs=projected_crs
    )
    buffers_gdf = buffers_gdf.to_crs(original_crs)
    logger.info("Buffers reprojected back to original CRS.")

    # Reproject network substations back to Original CRS
    network_substations_gdf = network_substations_gdf.to_crs(original_crs)

    # Reproject DLR substations back to Original CRS
    dlr_buses_gdf = dlr_buses_gdf.to_crs(original_crs)

    # Iterate over each network substation to add buffers and markers
    for idx, row in network_substations_gdf.iterrows():
        try:
            bus_id = row['bus_id']
            num_dlr = row['num_dlr_substations']
            substation_point = row.geometry
            has_dlr_without_T = row['has_dlr_without_T']

            # Get the corresponding buffer geometry
            buffer_geom = buffers_gdf.loc[buffers_gdf['bus_id'] == bus_id, 'geometry'].values[0]

            # Get centroid coordinates for marker placement
            lat = substation_point.y
            lon = substation_point.x

            # Add buffer polygon to the map
            folium.GeoJson(
                buffer_geom,
                name=f'Buffer {bus_id}',
                style_function=lambda x: {'fillColor': '#0000ff20', 'color': 'blue', 'weight': 1},
                tooltip=f"Buffer around Substation ID: {bus_id}"
            ).add_to(folium_map)
            logger.debug(f"Added buffer for Substation ID: {bus_id}")

            # Find DLR substations within the buffer
            dlr_in_buffer = dlr_buses_gdf[dlr_buses_gdf.within(buffer_geom)]
            logger.debug(f"Found {len(dlr_in_buffer)} DLR substations within buffer of Substation ID: {bus_id}")

            # Collect DLR IDs
            dlr_ids = dlr_in_buffer['id_dlr'].tolist()
            dlr_ids_str = ', '.join(map(str, dlr_ids)) if dlr_ids else 'None'

            # Update popup content for the network substation
            popup_content = f"Network Substation ID: {bus_id}<br>DLR Substations: {num_dlr}<br>DLR IDs: {dlr_ids_str}"

            # Determine marker color based on 'has_dlr_without_T'
            if has_dlr_without_T:
                marker_color = 'green'
                fill_color = 'green'
            else:
                marker_color = 'yellow'
                fill_color = 'yellow'

            # Add marker for the network substation with updated popup
            folium.CircleMarker(
                location=(lat, lon),
                radius=8,
                color=marker_color,
                fill=True,
                fill_color=fill_color,
                fill_opacity=1.0,
                popup=popup_content
            ).add_to(folium_map)
            logger.debug(f"Added {marker_color} marker for Substation ID: {bus_id}")

            # Add red markers for DLR substations within the buffer
            for _, dlr_row in dlr_in_buffer.iterrows():
                dlr_lat = dlr_row.geometry.y
                dlr_lon = dlr_row.geometry.x
                dlr_id = dlr_row['id_dlr']

                folium.CircleMarker(
                    location=(dlr_lat, dlr_lon),
                    radius=5,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=1.0,
                    popup=f"DLR Substation ID: {dlr_id}"
                ).add_to(folium_map)
                logger.debug(f"Added red marker for DLR Substation ID: {dlr_id}")

        except Exception as e:
            logger.error(f"Error processing Substation ID {bus_id}: {e}")

    # Add Layer Control
    folium.LayerControl().add_to(folium_map)
    logger.info("LayerControl added to the map.")

    # Save the map
    try:
        folium_map.save(output_path)
        logger.info(f"Filtered Substations Map saved to '{output_path}'.")
    except Exception as e:
        logger.error(f"Error saving the map to '{output_path}': {e}")


def clean_identifiers(df: pd.DataFrame, bus_col: str, prefix: str = '') -> pd.DataFrame:
    logger.info(f"Cleaning bus identifiers in column: {bus_col}")
    df[bus_col] = df[bus_col].astype(str).str.strip().str.upper()
    if prefix:
        df[bus_col] = df[bus_col].apply(lambda x: x if x.startswith(prefix) else f"{prefix}{x}")
    return df