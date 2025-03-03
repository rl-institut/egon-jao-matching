import pandas as pd

# 1. Read the CSV that includes columns: r_allocated, x_allocated, b_allocated, plus your original r, x, b
df = pd.read_csv("results/csv/matched_network_lines_with_allocated_attributes.csv")

# 2. Update r, x, b only when allocated values are nonzero.
#    If r_allocated != 0, then r <- r_allocated, else keep old r.
df.loc[df["r_allocated"] != 0, "r"] = df.loc[df["r_allocated"] != 0, "r_allocated"]
df.loc[df["x_allocated"] != 0, "x"] = df.loc[df["x_allocated"] != 0, "x_allocated"]
df.loc[df["b_allocated"] != 0, "b"] = df.loc[df["b_allocated"] != 0, "b_allocated"]

# 3. Drop the extra columns (r_allocated, x_allocated, b_allocated) if no longer needed
df.drop(
    columns=["r_allocated", "x_allocated", "b_allocated"], inplace=True, errors="ignore"
)

# 4. Keep only the specified final columns
columns_to_keep = [
    "id",
    "bus0",
    "bus1",
    "type",
    "carrier",
    "x",
    "r",
    "g",
    "b",
    "s_nom",
    "s_nom_extendable",
    "s_nom_min",
    "s_nom_max",
    "s_max_pu",
    "build_year",
    "lifetime",
    "capital_cost",
    "length",
    "terrain_factor",
    "num_parallel",
    "v_ang_min",
    "v_ang_max",
    "geom",
    "s_nom_mod",
    "sub_network",
    "v_nom",
    "x_pu",
    "r_pu",
    "b_pu",
    "g_pu",
    "x_pu_eff",
    "r_pu_eff",
    "s_nom_opt",
]

# Depending on your actual DataFrame, some columns above might not exist,
# so we can intersect them with your real columns to avoid KeyErrors:
final_columns = [col for col in columns_to_keep if col in df.columns]

df_final = df[final_columns]

# 5. Save to the final CSV
df_final.to_csv("results/csv/lines_updated.csv", index=False)

print("Done. The file 'lines_updated.csv' has been created.")
