import pandas as pd
import pypsa
import dill
import os


def load_pypsa_network(pkl_path):
    """
    Loads a PyPSA network object from a pickle file using dill.

    Parameters:
        pkl_path (str): Path to the PyPSA network pickle file.

    Returns:
        pypsa.Network: The loaded PyPSA network object.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"The pickle file '{pkl_path}' does not exist.")

    try:
        with open(pkl_path, 'rb') as f:
            network = dill.load(f)
        print(f"Successfully loaded the network from '{pkl_path}'.")
        return network
    except Exception as e:
        raise RuntimeError(f"Failed to load the network from '{pkl_path}'. Error: {e}")


def update_network_with_csv(network, lines_csv, transformers_csv):
    """
    Updates the PyPSA network by replacing lines and transformers with new data from CSV files.

    Parameters:
        network (pypsa.Network): The PyPSA network object to update.
        lines_csv (str): Path to the new lines CSV file.
        transformers_csv (str): Path to the new transformers CSV file.

    Returns:
        None
    """
    # Read the new lines CSV into a DataFrame
    try:
        new_lines_df = pd.read_csv(lines_csv, dtype={'bus0': str, 'bus1': str})
        print(f"Loaded new lines from '{lines_csv}' with {len(new_lines_df)} records.")
    except Exception as e:
        raise RuntimeError(f"Failed to read new lines from '{lines_csv}'. Error: {e}")

    # Read the new transformers CSV into a DataFrame
    try:
        new_transformers_df = pd.read_csv(transformers_csv, dtype={'bus0': str, 'bus1': str})
        print(f"Loaded new transformers from '{transformers_csv}' with {len(new_transformers_df)} records.")
    except Exception as e:
        raise RuntimeError(f"Failed to read new transformers from '{transformers_csv}'. Error: {e}")

    # Validate required columns for lines and transformers
    required_columns = ['bus0', 'bus1', 'type', 'carrier', 'x', 'r', 'g', 'b', 's_nom',
                        's_nom_extendable', 's_nom_min', 's_nom_max', 's_max_pu',
                        'build_year', 'lifetime', 'capital_cost', 'length',
                        'terrain_factor', 'num_parallel', 'v_ang_min', 'v_ang_max',
                        'geom', 's_nom_mod', 'sub_network', 'v_nom', 'x_pu',
                        'r_pu', 'b_pu', 'g_pu', 'x_pu_eff', 'r_pu_eff', 's_nom_opt']

    for df, name in zip([new_lines_df, new_transformers_df], ['Lines', 'Transformers']):
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"The {name} CSV is missing the following required columns: {missing_cols}")
        else:
            print(f"All required columns are present in the {name} CSV.")

    # Remove existing lines and transformers from the network
    if 'Line' in network.components:
        network.lines.drop(network.lines.index, inplace=True)
        print("Removed existing lines from the network.")
    else:
        print("No existing lines found in the network to remove.")

    if 'Transformer' in network.components:
        network.transformers.drop(network.transformers.index, inplace=True)
        print("Removed existing transformers from the network.")
    else:
        print("No existing transformers found in the network to remove.")

    # Add new lines to the network
    try:
        # Ensure each line has a unique name; if not, create one
        if 'name' not in new_lines_df.columns:
            new_lines_df = new_lines_df.reset_index().rename(columns={'index': 'name'})
        else:
            # Ensure names are unique
            if new_lines_df['name'].duplicated().any():
                raise ValueError(
                    "Duplicate names found in the new lines CSV. Please ensure all lines have unique names.")

        network.import_components_from_dataframe("Line", new_lines_df.set_index('name'))
        print(f"Added {len(new_lines_df)} new lines to the network.")
    except Exception as e:
        raise RuntimeError(f"Failed to add new lines to the network. Error: {e}")

    # Add new transformers to the network
    try:
        # Ensure each transformer has a unique name; if not, create one
        if 'name' not in new_transformers_df.columns:
            new_transformers_df = new_transformers_df.reset_index().rename(columns={'index': 'name'})
        else:
            # Ensure names are unique
            if new_transformers_df['name'].duplicated().any():
                raise ValueError(
                    "Duplicate names found in the new transformers CSV. Please ensure all transformers have unique names.")

        network.import_components_from_dataframe("Transformer", new_transformers_df.set_index('name'))
        print(f"Added {len(new_transformers_df)} new transformers to the network.")
    except Exception as e:
        raise RuntimeError(f"Failed to add new transformers to the network. Error: {e}")


def save_pypsa_network(network, pkl_path, backup=True):
    """
    Saves the PyPSA network object to a pickle file using dill.

    Parameters:
        network (pypsa.Network): The PyPSA network object to save.
        pkl_path (str): Path to save the network pickle file.
        backup (bool): Whether to create a backup of the original pickle file before overwriting.

    Returns:
        None
    """
    # Create a backup if required
    if backup:
        backup_pkl_path = pkl_path.replace('.pkl', '_backup.pkl')
        try:
            with open(pkl_path, 'rb') as original_pkl, open(backup_pkl_path, 'wb') as backup_pkl:
                backup_pkl.write(original_pkl.read())
            print(f"Backup of the original network created at '{backup_pkl_path}'.")
        except Exception as e:
            print(f"Warning: Failed to create backup. Proceeding without backup.\nError: {e}")

    # Save the updated network using dill
    try:
        with open(pkl_path, 'wb') as f:
            dill.dump(network, f)
        print(f"Successfully saved the updated network to '{pkl_path}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to save the updated network to '{pkl_path}'. Error: {e}")


# Example usage
if __name__ == "__main__":
    # Define file paths
    pkl_file_path = 'data/network_post_selection_status2023_8760_3_post_reactance_fix.pkl'  # Path to the existing PyPSA network pickle file
    lines_csv_path = 'lines.csv'  # Path to the new lines.csv
    transformers_csv_path = 'transformers.csv'  # Path to the new transformers.csv

    # Load the existing network
    try:
        network = load_pypsa_network(pkl_file_path)
    except Exception as e:
        print(f"An error occurred while loading the network: {e}")
        exit(1)

    # Update the network with new lines and transformers
    try:
        update_network_with_csv(network, lines_csv_path, transformers_csv_path)
    except Exception as e:
        print(f"An error occurred during the network update: {e}")
        exit(1)

    # Save the updated network
    try:
        save_pypsa_network(network, pkl_file_path)
    except Exception as e:
        print(f"An error occurred while saving the updated network: {e}")
        exit(1)
