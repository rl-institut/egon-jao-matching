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
    # Convert any non-string types to empty string in 'type' column for lines
    network.lines['type'] = network.lines['type'].apply(lambda x: '' if pd.isna(x) or not isinstance(x, str) else x)

    # Convert any non-string types to empty string in 'type' column for transformers
    network.transformers['type'] = network.transformers['type'].apply(
        lambda x: '' if pd.isna(x) or not isinstance(x, str) else x)

    # Ensure consistent float formatting for numeric columns
    float_columns = ['lifetime', 'g', 'terrain_factor', 'num_parallel', 's_nom_mod', 'v_nom',
                     'x', 'r', 'b', 's_nom', 's_nom_min', 's_nom_max', 's_max_pu',
                     'capital_cost', 'length', 'x_pu', 'r_pu', 'b_pu', 'g_pu',
                     'x_pu_eff', 'r_pu_eff', 's_nom_opt']

    for col in float_columns:
        if col in network.lines.columns:
            network.lines[col] = network.lines[col].astype(float)
        if col in network.transformers.columns:
            network.transformers[col] = network.transformers[col].astype(float)

    # Ensure boolean columns are properly typed
    bool_columns = ['s_nom_extendable']
    for col in bool_columns:
        if col in network.lines.columns:
            network.lines[col] = network.lines[col].astype(bool)
        if col in network.transformers.columns:
            network.transformers[col] = network.transformers[col].astype(bool)

    return network


def update_network(network_file, transformers_file, lines_file):
    # Create backup
    backup_file = create_backup(network_file)
    print(f"Backup created: {backup_file}")

    # Load network to update
    with open(network_file, 'rb') as f:
        network = dill.load(f)

    # Load new component data
    new_transformers = pd.read_csv(transformers_file)
    new_lines = pd.read_csv(lines_file)

    # Process transformers
    if 'Unnamed: 0' in new_transformers.columns:
        new_transformers.set_index('Unnamed: 0', inplace=True)

    # Ensure bus references exist in network.buses
    valid_transformers = new_transformers[
        new_transformers['bus0'].isin(network.buses.index) &
        new_transformers['bus1'].isin(network.buses.index)
        ].copy()

    # Add missing columns if they don't exist
    required_cols = ['tap_ratio', 'phase_shift']
    for col in required_cols:
        if col not in valid_transformers.columns:
            valid_transformers[col] = 1.0 if col == 'tap_ratio' else 0.0

    # Set default values for key parameters if missing
    valid_transformers['x'] = valid_transformers.get('x', 0.1)
    valid_transformers['r'] = valid_transformers.get('r', 0.01)
    valid_transformers['g'] = valid_transformers.get('g', 0)
    valid_transformers['b'] = valid_transformers.get('b', 0)
    valid_transformers['s_nom'] = valid_transformers.get('s_nom', 1000)

    # Update network transformers
    network.transformers = valid_transformers
    print(f"\nUpdated transformers: {len(valid_transformers)} valid transformers")

    # Process lines (rest of your existing lines code)
    ...

    # Format the network data
    network = format_network_data(network)

    # Additional verification steps
    print("\nNetwork verification:")
    print(f"Total buses: {len(network.buses)}")
    print(f"Total lines: {len(network.lines)}")
    print(f"Total transformers: {len(network.transformers)}")
    print("\nTransformer voltage levels:")
    if 'v_nom' in network.transformers.columns:
        print(network.transformers['v_nom'].value_counts())

    # Save updated network
    with open(network_file, 'wb') as f:
        dill.dump(network, f)
    print(f"\nNetwork updated and saved to: {network_file}")

    return network


# File paths
network_file = "lightsource/network_post_selection_status2023_8760_3_post_reactance_fix.pkl"
transformers_file = "lightsource/transformers_updated.csv"
lines_file = "lightsource/lines_updated.csv"

# Execute update
try:
    updated_network = update_network(network_file, transformers_file, lines_file)
except Exception as e:
    import traceback

    print(f"Error occurred: {str(e)}")
    print("Full traceback:")
    print(traceback.format_exc())