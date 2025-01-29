import dill
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def create_comparison_plot(backup_data, updated_data, col_name, component_type):
    """
    Create a scatter plot comparing values between backup and updated data for a specific column.
    Only plots points where values are different.
    """
    # Get values that are different
    mask = backup_data[col_name] != updated_data[col_name]
    if mask.any():
        plt.figure(figsize=(8, 6))
        plt.scatter(
            backup_data[col_name][mask],
            updated_data[col_name][mask],
            alpha=0.5,
            label='Different values'
        )

        # Add diagonal line
        min_val = min(backup_data[col_name].min(), updated_data[col_name].min())
        max_val = max(backup_data[col_name].max(), updated_data[col_name].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')

        plt.xlabel(f'Backup {col_name}')
        plt.ylabel(f'Updated {col_name}')
        plt.title(f'{component_type} {col_name} Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return True
    return False


def compare_dataframes(backup_df, updated_df, component_type):
    """
    Compare two dataframes and generate statistics and plots for differences.
    """
    print(f"\n=== {component_type} Comparison ===")

    # Compare shapes
    print(f"\nShape comparison:")
    print(f"Backup: {backup_df.shape}")
    print(f"Updated: {updated_df.shape}")

    # Compare columns
    print(f"\nColumns comparison:")
    backup_cols = set(backup_df.columns)
    updated_cols = set(updated_df.columns)

    if backup_cols != updated_cols:
        print("\nColumns only in backup:", backup_cols - updated_cols)
        print("Columns only in updated:", updated_cols - backup_cols)
    else:
        print("Both datasets have the same columns")

    # Compare numerical columns
    common_cols = list(backup_cols.intersection(updated_cols))
    numeric_cols = backup_df[common_cols].select_dtypes(include=[np.number]).columns

    print(f"\nNumerical columns being compared: {len(numeric_cols)}")

    # Create plots for different values
    plots_created = 0
    for col in numeric_cols:
        if create_comparison_plot(backup_df, updated_df, col, component_type):
            plots_created += 1
            plt.show()

    if plots_created == 0:
        print(f"\nNo differences found in numerical values for {component_type}")
    else:
        print(f"\nCreated {plots_created} plots showing differences in {component_type}")

    # Calculate statistics for differences
    print(f"\nStatistical summary of differences:")
    for col in numeric_cols:
        diff = backup_df[col] - updated_df[col]
        diff_count = (diff != 0).sum()
        if diff_count > 0:
            print(f"\n{col}:")
            print(f"Number of differences: {diff_count}")
            print(f"Mean difference: {diff.mean():.6f}")
            print(f"Max absolute difference: {abs(diff).max():.6f}")
            print(f"Standard deviation of differences: {diff.std():.6f}")


def compare_networks(backup_file, updated_file):
    """
    Main function to compare two network files.
    """
    # Load both networks
    print("Loading networks...")
    with open(backup_file, 'rb') as f:
        backup_network = dill.load(f)
    with open(updated_file, 'rb') as f:
        updated_network = dill.load(f)
    print("Networks loaded successfully")

    # Compare transformers
    compare_dataframes(backup_network.transformers, updated_network.transformers, "Transformers")

    # Compare lines
    compare_dataframes(backup_network.lines, updated_network.lines, "Lines")

    # Compare buses if available
    if hasattr(backup_network, 'buses') and hasattr(updated_network, 'buses'):
        compare_dataframes(backup_network.buses, updated_network.buses, "Buses")

    # Compare generators if available
    if hasattr(backup_network, 'generators') and hasattr(updated_network, 'generators'):
        compare_dataframes(backup_network.generators, updated_network.generators, "Generators")

    # Compare loads if available
    if hasattr(backup_network, 'loads') and hasattr(updated_network, 'loads'):
        compare_dataframes(backup_network.loads, updated_network.loads, "Loads")


def main():
    # File paths
    backup_file = "lightsource/network_post_selection_status2023_8760_3_post_reactance_fix_backup_20250115_091817.pkl"
    updated_file = "lightsource/network_post_selection_status2023_8760_3_post_reactance_fix.pkl"

    try:
        compare_networks(backup_file, updated_file)
    except Exception as e:
        import traceback
        print(f"\nError occurred: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()