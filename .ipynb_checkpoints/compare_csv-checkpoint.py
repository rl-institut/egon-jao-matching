import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
DATA_DIR = "lightsource"

FILES = {
    'transformers_original': 'transformers_original.csv',
    'transformers_updated': 'transformers_updated.csv',
    'lines_original': 'lines_original.csv',
    'lines_updated': 'lines_updated.csv'
}

# Add 'b', 'x', and 'r' columns
COLUMNS_TO_ANALYZE = {
    'transformers': ['b', 'x', 'r'],
    'lines': ['b', 'x', 'r']
}

IDENTIFIER_COLUMNS = {
    'transformers': 'Transformer',
    'lines': ['bus0', 'bus1', 'type']
}

# Create separate output directories for b, x, and r analyses
OUTPUT_DIR_BASE = os.path.join(DATA_DIR, "analysis_results")
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)


def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} successfully.")
        print(f"Columns in {file_path}: {df.columns.tolist()}")
        print("-" * 50)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_column_values(df, component, column_name):
    """Extract the specified column from the DataFrame."""
    if column_name in df.columns:
        values = pd.to_numeric(df[column_name], errors='coerce').dropna()
        if values.empty:
            print(f"No valid numeric data found in column '{column_name}' for '{component}'.")
            raise ValueError(f"No valid numeric data in '{column_name}' for '{component}'.")
        print(f"Extracted '{column_name}' values for {component}. Count after cleaning: {values.count()}")
        return values
    else:
        available_cols = df.columns.tolist()
        print(f"Column '{column_name}' not found in the dataset for '{component}'. Available columns: {available_cols}")
        raise ValueError(f"Column '{column_name}' not found in the dataset for '{component}'.")


def plot_corresponding_values(original_df, updated_df, component, column_name):
    """Plot corresponding values for original and updated datasets, excluding identical values."""
    identifier = IDENTIFIER_COLUMNS.get(component)
    output_dir = os.path.join(OUTPUT_DIR_BASE, column_name)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(identifier, list):
        merged_df = pd.merge(original_df, updated_df, on=identifier, suffixes=('_original', '_updated'))
    else:
        merged_df = pd.merge(original_df, updated_df, on=identifier, suffixes=('_original', '_updated'))

    # Filter out identical values
    col_original = f'{column_name}_original'
    col_updated = f'{column_name}_updated'

    # Handle potential infinity and NaN values
    merged_df[col_original] = pd.to_numeric(merged_df[col_original], errors='coerce')
    merged_df[col_updated] = pd.to_numeric(merged_df[col_updated], errors='coerce')

    # Remove inf, -inf, and nan values
    merged_df = merged_df[~merged_df[col_original].isin([np.inf, -np.inf]) &
                          ~merged_df[col_updated].isin([np.inf, -np.inf]) &
                          ~merged_df[col_original].isna() &
                          ~merged_df[col_updated].isna()]

    different_values = merged_df[merged_df[col_original] != merged_df[col_updated]]

    if different_values.empty:
        print(f"No different values found for {component} {column_name}.")
        return

    plt.figure(figsize=(8, 8))

    # Create scatter plot with difference magnitude coloring
    difference_magnitude = abs(different_values[col_updated] - different_values[col_original])
    scatter = plt.scatter(different_values[col_original],
                          different_values[col_updated],
                          c=difference_magnitude,
                          cmap='viridis',
                          alpha=0.6)

    plt.colorbar(scatter, label='Absolute Difference')

    # Add diagonal line
    all_values = np.concatenate([different_values[col_original], different_values[col_updated]])
    min_val, max_val = all_values.min(), all_values.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

    plt.xlabel(f'Original {column_name}')
    plt.ylabel(f'Updated {column_name}')
    plt.title(f'Different {column_name} Values for {component.capitalize()}\n(Excluding identical values)')
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{component}_corresponding_{column_name}_plot.png")
    plt.savefig(plot_path)
    plt.close()

    # Save statistics to text file
    stats_path = os.path.join(output_dir, f"{component}_{column_name}_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Statistics for {component} {column_name}:\n")
        f.write(f"Number of different values: {len(different_values)}\n")
        f.write(f"Number of identical values: {len(merged_df) - len(different_values)}\n")
        f.write(f"Percentage of different values: {(len(different_values) / len(merged_df)) * 100:.2f}%\n")
        f.write(f"Mean absolute difference: {difference_magnitude.mean():.6f}\n")
        f.write(f"Max absolute difference: {difference_magnitude.max():.6f}\n")

    print(f"Corresponding values plot and statistics saved to {output_dir}")


def plot_histogram_different_values(original, updated, component, column_name):
    """Create and save overlapping histograms for different values only."""
    output_dir = os.path.join(OUTPUT_DIR_BASE, column_name)
    os.makedirs(output_dir, exist_ok=True)

    # Handle potential infinity and NaN values
    original = pd.Series(original).replace([np.inf, -np.inf], np.nan).dropna()
    updated = pd.Series(updated).replace([np.inf, -np.inf], np.nan).dropna()

    # Create a mask for different values
    mask = original != updated
    original_diff = original[mask]
    updated_diff = updated[mask]

    if len(original_diff) == 0:
        print(f"No different values found for {component} {column_name} histogram.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Regular histogram
    sns.histplot(original_diff, color='blue', label='Original', kde=True, ax=ax1, alpha=0.6)
    sns.histplot(updated_diff, color='orange', label='Updated', kde=True, ax=ax1, alpha=0.6)
    ax1.set_title(f"Distribution of Different {column_name} Values\n(Linear Scale)")
    ax1.set_xlabel(column_name)
    ax1.set_ylabel('Count')
    ax1.legend()

    # Log scale histogram for better visualization of small values
    if (original_diff > 0).all() and (updated_diff > 0).all():
        sns.histplot(np.log10(original_diff), color='blue', label='Original', kde=True, ax=ax2, alpha=0.6)
        sns.histplot(np.log10(updated_diff), color='orange', label='Updated', kde=True, ax=ax2, alpha=0.6)
        ax2.set_title(f"Distribution of Different {column_name} Values\n(Log Scale)")
        ax2.set_xlabel(f'log10({column_name})')
        ax2.set_ylabel('Count')
        ax2.legend()

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{component}_{column_name}_histogram.png")
    plt.savefig(plot_path)
    plt.close()

    # Save distribution statistics to text file
    stats_path = os.path.join(output_dir, f"{component}_{column_name}_distribution_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Distribution Statistics for {component} {column_name}:\n\n")
        f.write("Original Different Values:\n")
        f.write(f"Count: {len(original_diff)}\n")
        f.write(f"Mean: {original_diff.mean():.6f}\n")
        f.write(f"Median: {original_diff.median():.6f}\n")
        f.write(f"Std Dev: {original_diff.std():.6f}\n")
        f.write(f"Min: {original_diff.min():.6f}\n")
        f.write(f"Max: {original_diff.max():.6f}\n\n")

        f.write("Updated Different Values:\n")
        f.write(f"Count: {len(updated_diff)}\n")
        f.write(f"Mean: {updated_diff.mean():.6f}\n")
        f.write(f"Median: {updated_diff.median():.6f}\n")
        f.write(f"Std Dev: {updated_diff.std():.6f}\n")
        f.write(f"Min: {updated_diff.min():.6f}\n")
        f.write(f"Max: {updated_diff.max():.6f}\n")

    print(f"Histogram and distribution statistics saved to {output_dir}")


def identify_extreme_changes(original_df, updated_df, component, column_name):
    """Identify rows where updated value is >10x or <0.1x of original value."""
    output_dir = os.path.join(OUTPUT_DIR_BASE, column_name)
    os.makedirs(output_dir, exist_ok=True)

    identifier = IDENTIFIER_COLUMNS.get(component)

    if isinstance(identifier, list):
        merged_df = pd.merge(original_df, updated_df, on=identifier, suffixes=('_original', '_updated'))
    else:
        merged_df = pd.merge(original_df, updated_df, on=identifier, suffixes=('_original', '_updated'))

    col_original = f'{column_name}_original'
    col_updated = f'{column_name}_updated'

    # Handle potential infinity and NaN values
    merged_df[col_original] = pd.to_numeric(merged_df[col_original], errors='coerce')
    merged_df[col_updated] = pd.to_numeric(merged_df[col_updated], errors='coerce')

    # Remove inf, -inf, and nan values
    merged_df = merged_df[~merged_df[col_original].isin([np.inf, -np.inf]) &
                          ~merged_df[col_updated].isin([np.inf, -np.inf]) &
                          ~merged_df[col_original].isna() &
                          ~merged_df[col_updated].isna()]

    # Handle division by zero by filtering out rows where original value is zero
    merged_df = merged_df[merged_df[col_original] != 0]

    # Calculate the ratio
    merged_df[f'{column_name}_ratio'] = merged_df[col_updated] / merged_df[col_original]

    # Identify extreme changes
    extreme_changes = merged_df[(merged_df[f'{column_name}_ratio'] > 10) |
                                (merged_df[f'{column_name}_ratio'] < 0.1)]

    if not extreme_changes.empty:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Regular scale plot
        scatter1 = ax1.scatter(extreme_changes[col_original],
                               extreme_changes[col_updated],
                               c=extreme_changes[f'{column_name}_ratio'],
                               cmap='coolwarm',
                               norm=plt.LogNorm(),
                               alpha=0.7)
        ax1.set_xlabel(f'Original {column_name}')
        ax1.set_ylabel(f'Updated {column_name}')
        ax1.set_title(f'Extreme Changes in {column_name} (Linear Scale)')
        plt.colorbar(scatter1, ax=ax1, label='Ratio (Updated/Original)')

        # Log scale plot
        scatter2 = ax2.scatter(extreme_changes[col_original],
                               extreme_changes[col_updated],
                               c=extreme_changes[f'{column_name}_ratio'],
                               cmap='coolwarm',
                               norm=plt.LogNorm(),
                               alpha=0.7)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(f'Original {column_name}')
        ax2.set_ylabel(f'Updated {column_name}')
        ax2.set_title(f'Extreme Changes in {column_name} (Log Scale)')
        plt.colorbar(scatter2, ax=ax2, label='Ratio (Updated/Original)')

        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{component}_extreme_changes_{column_name}_plot.png")
        plt.savefig(plot_path)
        plt.close()

        # Save extreme changes to CSV
        csv_path = os.path.join(output_dir, f"{component}_extreme_changes_{column_name}.csv")
        extreme_changes.to_csv(csv_path, index=False)

        # Save extreme changes statistics
        stats_path = os.path.join(output_dir, f"{component}_extreme_changes_{column_name}_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Extreme Changes Statistics for {component} {column_name}:\n")
            f.write(f"Total number of extreme changes: {len(extreme_changes)}\n")
            f.write(f"Percentage of total values: {(len(extreme_changes) / len(merged_df)) * 100:.2f}%\n")
            f.write(f"Mean ratio: {extreme_changes[f'{column_name}_ratio'].mean():.6f}\n")
            f.write(f"Median ratio: {extreme_changes[f'{column_name}_ratio'].median():.6f}\n")
            f.write(f"Min ratio: {extreme_changes[f'{column_name}_ratio'].min():.6f}\n")
            f.write(f"Max ratio: {extreme_changes[f'{column_name}_ratio'].max():.6f}\n")

        print(f"Extreme changes analysis saved to {output_dir}")
    else:
        print(f"No extreme changes found for {component} {column_name}")


def main():
    print("Starting analysis...")

    # Load all datasets
    datasets = {}
    for key, filename in FILES.items():
        file_path = os.path.join(DATA_DIR, filename)
        df = load_csv(file_path)
        if df is not None:
            datasets[key] = df

    # Process each component and column
    for component in ['transformers', 'lines']:
        for column_name in COLUMNS_TO_ANALYZE[component]:
            print(f"\nAnalyzing {column_name} for {component}...")

            original_key = f"{component}_original"
            updated_key = f"{component}_updated"

            if original_key in datasets and updated_key in datasets:
                try:
                    # Extract values
                    original_values = extract_column_values(datasets[original_key], component, column_name)
                    updated_values = extract_column_values(datasets[updated_key], component, column_name)

                    # Create plots and analysis
                    plot_corresponding_values(datasets[original_key], datasets[updated_key], component, column_name)
                    plot_histogram_different_values(original_values, updated_values, component, column_name)
                    identify_extreme_changes(datasets[original_key], datasets[updated_key], component, column_name)

                except Exception as e:
                    print(f"Error processing {column_name} for {component}: {e}")
            else:
                print(f"Missing data for {component}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()