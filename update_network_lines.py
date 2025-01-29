import pandas as pd
import pypsa
import dill
import shutil
from datetime import datetime
import numpy as np


def create_backup(original_file):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f"{original_file.rsplit('.', 1)[0]}_backup_{timestamp}.pkl"
    shutil.copy2(original_file, backup_file)
    return backup_file


def format_network_data(network):
    """Ensure consistent formatting for network data."""
    # Convert any non-string types to empty string in 'type' column for lines
    network.lines['type'] = network.lines['type'].apply(lambda x: '' if pd.isna(x) or not isinstance(x, str) else x)

    # Ensure consistent float formatting for numeric columns
    float_columns = ['lifetime', 'g', 'terrain_factor', 'num_parallel', 's_nom_mod', 'v_nom',
                     'x', 'r', 'b', 's_nom', 's_nom_min', 's_nom_max', 's_max_pu',
                     'capital_cost', 'length', 'x_pu', 'r_pu', 'b_pu', 'g_pu',
                     'x_pu_eff', 'r_pu_eff', 's_nom_opt']

    for col in float_columns:
        if col in network.lines.columns:
            network.lines[col] = network.lines[col].astype(float)

    # Ensure boolean columns are properly typed
    bool_columns = ['s_nom_extendable']
    for col in bool_columns:
        if col in network.lines.columns:
            network.lines[col] = network.lines[col].astype(bool)

    return network


def update_network(network_file, lines_file):
    """Update the network by modifying only the lines, skipping transformers."""

    # Create a backup of the network
    backup_file = create_backup(network_file)
    print(f"Backup created: {backup_file}")

    # Load the existing network
    with open(network_file, 'rb') as f:
        network = dill.load(f)

    # Load updated lines data
    new_lines = pd.read_csv(lines_file)

    # Process lines
    if 'Unnamed: 0' in new_lines.columns:
        new_lines.set_index('Unnamed: 0', inplace=True)

    # Ensure that all bus references exist in the network
    valid_lines = new_lines[
        new_lines['bus0'].isin(network.buses.index) &
        new_lines['bus1'].isin(network.buses.index)
        ].copy()

    # Add missing columns if they don’t exist
    required_cols = ['x', 'r', 'b', 'g', 's_nom', 'length']
    for col in required_cols:
        if col not in valid_lines.columns:
            valid_lines[col] = 0.0  # Default value

    # Update network lines
    network.lines = valid_lines
    print(f"\nUpdated lines: {len(valid_lines)} valid lines")

    # Format the network data
    network = format_network_data(network)

    # Additional verification steps
    print("\nNetwork verification:")
    print(f"Total buses: {len(network.buses)}")
    print(f"Total lines: {len(network.lines)}")
    print(f"Total transformers (unchanged): {len(network.transformers)}")

    # Save updated network
    with open(network_file, 'wb') as f:
        dill.dump(network, f)
    print(f"\nNetwork updated and saved to: {network_file}")

    return network


# File paths
network_file = "lightsource/network_post_selection_status2023_8760_3_post_reactance_fix.pkl"
lines_file = "lightsource/lines_updated.csv"

# Execute update
try:
    updated_network = update_network(network_file, lines_file)
except Exception as e:
    import traceback

    print(f"Error occurred: {str(e)}")
    print("Full traceback:")
    print(traceback.format_exc())
