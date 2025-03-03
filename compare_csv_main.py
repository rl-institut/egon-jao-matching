import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy import stats

# ----------------------------
# Configuration
# ----------------------------

# Directory containing your datasets
DATA_DIR = "lightsource"

# File names
FILES = {
    "transformers_original": "transformers_original.csv",
    "transformers_updated": "transformers_updated.csv",
    "lines_original": "lines_original.csv",
    "lines_updated": "lines_updated.csv",
}

# Column names for b values
B_COLUMNS = {"transformers": "b", "lines": "b"}

# Common Identifier Columns per Component
IDENTIFIER_COLUMNS = {
    "transformers": "Transformer",  # Unique identifier for transformers
    "lines": ["bus0", "bus1", "type"],  # Composite identifier for lines
}

# Output directory for plots and results related to 'b'
OUTPUT_DIR = os.path.join(DATA_DIR, "analysis_results_b")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Functions
# ----------------------------


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


def extract_b_values(df, component):
    """Extract the b column from the DataFrame."""
    b_col = B_COLUMNS.get(component)
    if b_col and b_col in df.columns:
        # Convert to numeric, coercing errors
        b_series = pd.to_numeric(df[b_col], errors="coerce").dropna()
        if b_series.empty:
            print(f"No valid numeric data found in column '{b_col}' for '{component}'.")
            raise ValueError(f"No valid numeric data in '{b_col}' for '{component}'.")
        print(
            f"Extracted 'b' values for {component}. Count after cleaning: {b_series.count()}"
        )
        return b_series
    else:
        available_cols = df.columns.tolist()
        print(
            f"Column '{b_col}' not found in the dataset for '{component}'. Available columns: {available_cols}"
        )
        raise ValueError(
            f"Column '{b_col}' not found in the dataset for '{component}'."
        )


def descriptive_statistics(series):
    """Calculate descriptive statistics for a pandas Series."""
    return {
        "Count": series.count(),
        "Mean": series.mean(),
        "Median": series.median(),
        "Std Dev": series.std(),
        "Min": series.min(),
        "Max": series.max(),
    }


def perform_hypothesis_test(original, updated):
    """Perform statistical test between original and updated data."""
    # Check for normality using Shapiro-Wilk test
    if len(original) > 5000:
        # Shapiro-Wilk test is not suitable for large samples
        print("Sample size > 5000, skipping Shapiro-Wilk test for normality.")
        normal = False
    else:
        _, p_orig = stats.shapiro(original)
        _, p_upd = stats.shapiro(updated)
        normal = (p_orig > 0.05) and (p_upd > 0.05)

    if normal:
        # Use Independent t-test
        test_stat, p_value = stats.ttest_ind(original, updated, equal_var=False)
        test_name = "Independent t-test"
    else:
        # Use Mann-Whitney U test
        test_stat, p_value = stats.mannwhitneyu(
            original, updated, alternative="two-sided"
        )
        test_name = "Mann-Whitney U test"

    return {
        "Test": test_name,
        "Test Statistic": test_stat,
        "p-value": p_value,
        "Significant (α=0.05)": p_value < 0.05,
    }


def plot_boxplot(original, updated, component, b_label):
    """Create and save a boxplot comparing original and updated data."""
    data = [original, updated]
    labels = ["Original", "Updated"]

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, palette="Set2")
    plt.xticks([0, 1], labels)
    plt.ylabel(b_label)
    plt.title(f"Comparison of {b_label} for {component.capitalize()}")
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(OUTPUT_DIR, f"{component}_boxplot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Boxplot saved to {plot_path}.")


def plot_histogram(original, updated, component, b_label):
    """Create and save overlapping histograms with KDE for original and updated data."""
    plt.figure(figsize=(8, 6))
    sns.histplot(
        original,
        color="blue",
        label="Original",
        kde=True,
        stat="density",
        linewidth=0,
        alpha=0.6,
    )
    sns.histplot(
        updated,
        color="orange",
        label="Updated",
        kde=True,
        stat="density",
        linewidth=0,
        alpha=0.6,
    )
    plt.xlabel(b_label)
    plt.ylabel("Density")
    plt.title(f"Distribution of {b_label} for {component.capitalize()}")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(OUTPUT_DIR, f"{component}_histogram.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Histogram saved to {plot_path}.")


def identify_extreme_changes_b(original_df, updated_df, component):
    """Identify rows where updated 'b' is >10x or <0.1x of original 'b'."""
    identifier = IDENTIFIER_COLUMNS.get(component)

    if isinstance(identifier, list):
        # Use multiple columns as identifier
        merged_df = pd.merge(
            original_df, updated_df, on=identifier, suffixes=("_original", "_updated")
        )
    else:
        # Use single column as identifier
        merged_df = pd.merge(
            original_df, updated_df, on=identifier, suffixes=("_original", "_updated")
        )

    # Handle division by zero by filtering out rows where original 'b' is zero
    merged_df = merged_df[merged_df["b_original"] != 0]

    # Calculate the ratio
    merged_df["b_ratio"] = merged_df["b_updated"] / merged_df["b_original"]

    # Identify extreme changes
    extreme_changes = merged_df[
        (merged_df["b_ratio"] > 10) | (merged_df["b_ratio"] < 0.1)
    ]

    # Calculate percentage change correctly
    extreme_changes["percentage_change"] = (extreme_changes["b_ratio"] - 1) * 100

    # Save extreme changes to CSV
    extreme_output_path = os.path.join(OUTPUT_DIR, f"{component}_extreme_changes.csv")
    extreme_changes.to_csv(extreme_output_path, index=False)
    print(f"Extreme changes saved to {extreme_output_path}.")

    return extreme_changes


def plot_extreme_changes_b(extreme_df, component, b_label):
    """Plot extreme changes for a component."""
    if extreme_df.empty:
        print(f"No extreme changes to plot for {component}.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=extreme_df,
        x="b_original",
        y="b_updated",
        hue="b_ratio",
        palette="coolwarm",
        size="b_ratio",
        sizes=(50, 200),
        alpha=0.7,
    )
    plt.xlabel(f"Original {b_label}")
    plt.ylabel(f"Updated {b_label}")
    plt.title(f"Extreme Changes in {b_label} for {component.capitalize()}")
    plt.legend(title="b_ratio", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(OUTPUT_DIR, f"{component}_extreme_changes_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Extreme changes plot saved to {plot_path}.")


def plot_corresponding_b(original_df, updated_df, component, b_label):
    """Plot corresponding b values for original and updated datasets."""
    identifier = IDENTIFIER_COLUMNS.get(component)

    if isinstance(identifier, list):
        # Use multiple columns as identifier
        merged_df = pd.merge(
            original_df, updated_df, on=identifier, suffixes=("_original", "_updated")
        )
    else:
        # Use single column as identifier
        merged_df = pd.merge(
            original_df, updated_df, on=identifier, suffixes=("_original", "_updated")
        )

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=merged_df, x="b_original", y="b_updated", alpha=0.3, edgecolor=None
    )
    plt.plot(
        [merged_df["b_original"].min(), merged_df["b_original"].max()],
        [merged_df["b_original"].min(), merged_df["b_original"].max()],
        "r--",
        label="y = x",
    )
    plt.xlabel(f"Original {b_label}")
    plt.ylabel(f"Updated {b_label}")
    plt.title(f"Corresponding {b_label} Values for {component.capitalize()}")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(OUTPUT_DIR, f"{component}_corresponding_b_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Corresponding b values plot saved to {plot_path}.")


def main():
    print("Starting 'b' column analysis...")
    # Load all datasets
    datasets = {}
    for key, filename in FILES.items():
        file_path = os.path.join(DATA_DIR, filename)
        df = load_csv(file_path)
        if df is not None:
            datasets[key] = df
        else:
            print(f"Skipping {key} due to loading error.")

    # Extract b values
    b_values = {}
    for component in ["transformers", "lines"]:
        for version in ["original", "updated"]:
            key = f"{component}_{version}"
            df = datasets.get(key)
            if df is not None:
                try:
                    b = extract_b_values(df, component)
                    b_values[key] = b
                except Exception as e:
                    print(f"Error extracting 'b' values for {key}: {e}")
            else:
                print(f"Dataset for {key} is not available.")

    # Descriptive Statistics
    stats_list = []
    for key, series in b_values.items():
        stats = descriptive_statistics(series)
        stats["Dataset"] = key
        stats_list.append(stats)

    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        print("\nDescriptive Statistics:")
        print(stats_df[["Dataset", "Count", "Mean", "Median", "Std Dev", "Min", "Max"]])

        # Save to CSV
        stats_output_path = os.path.join(OUTPUT_DIR, "descriptive_statistics.csv")
        stats_df.to_csv(stats_output_path, index=False)
        print(f"Descriptive statistics saved to {stats_output_path}.")
    else:
        print(
            "\nNo descriptive statistics to display because no 'b' values were extracted."
        )

    # Hypothesis Testing
    hypothesis_results = []
    for component in ["transformers", "lines"]:
        original_key = f"{component}_original"
        updated_key = f"{component}_updated"
        if original_key in b_values and updated_key in b_values:
            result = perform_hypothesis_test(
                b_values[original_key], b_values[updated_key]
            )
            result["Component"] = component
            hypothesis_results.append(result)
            print(
                f"Performed {result['Test']} for {component}. p-value = {result['p-value']:.4f}"
            )
        else:
            print(f"Insufficient data for hypothesis testing on {component}.")

    if hypothesis_results:
        hypothesis_df = pd.DataFrame(hypothesis_results)
        print("\nHypothesis Testing Results:")
        print(
            hypothesis_df[
                [
                    "Component",
                    "Test",
                    "Test Statistic",
                    "p-value",
                    "Significant (α=0.05)",
                ]
            ]
        )

        # Save to CSV
        hypothesis_output_path = os.path.join(
            OUTPUT_DIR, "hypothesis_testing_results.csv"
        )
        hypothesis_df.to_csv(hypothesis_output_path, index=False)
        print(f"Hypothesis testing results saved to {hypothesis_output_path}.")
    else:
        print("\nNo hypothesis testing results to display.")

    # Visualization
    for component in ["transformers", "lines"]:
        original_key = f"{component}_original"
        updated_key = f"{component}_updated"
        if original_key in b_values and updated_key in b_values:
            b_label = B_COLUMNS.get(component, "b")
            plot_boxplot(
                b_values[original_key], b_values[updated_key], component, b_label
            )
            plot_histogram(
                b_values[original_key], b_values[updated_key], component, b_label
            )
            plot_corresponding_b(
                datasets[original_key], datasets[updated_key], component, b_label
            )
        else:
            print(f"Skipping plots for {component} due to insufficient data.")

    # Identify Extreme Changes
    for component in ["transformers", "lines"]:
        original_key = f"{component}_original"
        updated_key = f"{component}_updated"
        if original_key in b_values and updated_key in b_values:
            # Retrieve the original and updated DataFrames
            original_df = datasets.get(original_key)
            updated_df = datasets.get(updated_key)

            # Ensure the identifier column exists
            identifier = IDENTIFIER_COLUMNS.get(component)
            if isinstance(identifier, list):
                missing_cols = [
                    col
                    for col in identifier
                    if col not in original_df.columns or col not in updated_df.columns
                ]
                if missing_cols:
                    print(
                        f"Identifier columns {missing_cols} not found in one of the datasets for '{component}'. Skipping extreme change identification."
                    )
                    continue
            else:
                if (
                    identifier not in original_df.columns
                    or identifier not in updated_df.columns
                ):
                    print(
                        f"Identifier column '{identifier}' not found in one of the datasets for '{component}'. Skipping extreme change identification."
                    )
                    continue

            try:
                extreme_changes = identify_extreme_changes_b(
                    original_df, updated_df, component
                )

                if not extreme_changes.empty:
                    plot_extreme_changes_b(
                        extreme_changes, component, B_COLUMNS.get(component, "b")
                    )
                else:
                    print(f"No extreme changes found for {component}.")
            except Exception as e:
                print(f"Error identifying extreme changes for {component}: {e}")
        else:
            print(f"Insufficient data for identifying extreme changes on {component}.")

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
