import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths to your CSV files
lines_original_path = "lightsource/lines_original.csv"
lines_updated_path = "lightsource/lines_updated.csv"

# Read the CSVs
original_df = pd.read_csv(lines_original_path)
updated_df = pd.read_csv(lines_updated_path)

# Convert 'id' to string in both DataFrames before merging
original_df['id'] = original_df['id'].astype(str)
updated_df['id'] = updated_df['id'].astype(str)

# Print debug info
print("Original DataFrame 'id' type:", original_df['id'].dtype)
print("Updated DataFrame 'id' type:", updated_df['id'].dtype)
print("Sample of original ids:", original_df['id'].head())
print("Sample of updated ids:", updated_df['id'].head())

# Merge on the now-string 'id' column
df_merged = pd.merge(
    original_df,
    updated_df,
    on="id",
    suffixes=("_orig", "_upd")
)
print("DEBUG: Merged shape =", df_merged.shape)

# Columns we want to compare
columns_to_compare = ["x", "r", "b"]
output_dir = "lightsource/analysis_results_clamped_ratio"
os.makedirs(output_dir, exist_ok=True)

# Create a comprehensive comparison DataFrame
comparison_df = pd.DataFrame({'id': df_merged['id']})

for col in columns_to_compare:
    col_orig = f"{col}_orig"
    col_upd = f"{col}_upd"

    # Make sure these columns exist in the merged DataFrame
    if col_orig not in df_merged.columns or col_upd not in df_merged.columns:
        print(f"Skipping {col}, columns not found in merged DataFrame.")
        continue

    # Convert to numeric
    df_merged[col_orig] = pd.to_numeric(df_merged[col_orig], errors="coerce")
    df_merged[col_upd] = pd.to_numeric(df_merged[col_upd], errors="coerce")

    # Add to comparison DataFrame
    comparison_df[f'{col}_original'] = df_merged[col_orig]
    comparison_df[f'{col}_updated'] = df_merged[col_upd]

    # Add ratio where applicable
    mask = (~df_merged[col_orig].isna()) & (~df_merged[col_upd].isna()) & (df_merged[col_orig] > 0)
    comparison_df[f'{col}_ratio'] = np.nan
    comparison_df.loc[mask, f'{col}_ratio'] = df_merged.loc[mask, col_upd] / df_merged.loc[mask, col_orig]

    # Filter rows: original must be > 0 to avoid division by zero
    df_compare = df_merged.loc[mask].copy()
    if df_compare.empty:
        print(f"No valid rows to compare ratio for '{col}'.")
        continue

    # Compute the raw ratio = updated / original
    raw_ratio_col = f"{col}_raw_ratio"
    df_compare[raw_ratio_col] = df_compare[col_upd] / df_compare[col_orig]

    # ------------------------------------------------------------
    # 1) SAVE ROWS WITH ratio >= 10 BEFORE CLAMPING
    # ------------------------------------------------------------
    big_mask = df_compare[raw_ratio_col] >= 10
    if big_mask.any():
        # Save these rows to a CSV
        big_ratio_csv = os.path.join(output_dir, f"{col}_ratio_gte10.csv")
        df_compare.loc[big_mask].to_csv(big_ratio_csv, index=False)
        print(f"Saved {big_mask.sum()} rows with ratio >= 10 for '{col}' to {big_ratio_csv}")

    # ------------------------------------------------------------
    # 2) CLAMP ratio to [0, 10] for plotting
    # ------------------------------------------------------------
    col_ratio_clamped = f"{col}_ratio_clamped"
    df_compare[col_ratio_clamped] = df_compare[raw_ratio_col].clip(lower=0, upper=10)

    # Plot scatter with color = clamped ratio
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        df_compare[col_orig],
        df_compare[col_upd],
        c=df_compare[col_ratio_clamped],
        cmap="viridis",
        alpha=0.7,
        vmin=0,
        vmax=10
    )
    plt.colorbar(scatter, label="Updated / Original ratio (0..10)")

    # Diagonal line for reference
    min_val = min(df_compare[col_orig].min(), df_compare[col_upd].min())
    max_val = max(df_compare[col_orig].max(), df_compare[col_upd].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")

    plt.title(f"{col.upper()} with Clamped Ratio Color Scale [0..10]")
    plt.xlabel(f"{col} (original)")
    plt.ylabel(f"{col} (updated)")
    plt.legend()

    # Save plot
    plot_path = os.path.join(output_dir, f"{col}_clamped_ratio.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot for '{col}' saved to {plot_path}. Ratio scale clamped to [0..10].")

# Save the comprehensive comparison DataFrame
comparison_csv_path = os.path.join(output_dir, "parameter_comparison.csv")
comparison_df.to_csv(comparison_csv_path, index=False)
print(f"Comprehensive parameter comparison saved to {comparison_csv_path}")

# Save a sample of lines with the most interesting changes
if not comparison_df.empty:
    # Find lines with non-zero values in the updated columns
    non_zero_mask = (
            (comparison_df['r_updated'] > 0) |
            (comparison_df['x_updated'] > 0) |
            (comparison_df['b_updated'] > 0)
    )

    # Get 50 samples with significant changes
    interesting_samples = comparison_df[non_zero_mask].sort_values(
        by=['r_ratio', 'x_ratio', 'b_ratio'],
        ascending=False
    ).head(50)

    sample_path = os.path.join(output_dir, "interesting_changes_sample.csv")
    interesting_samples.to_csv(sample_path, index=False)
    print(f"Sample of interesting changes saved to {sample_path}")