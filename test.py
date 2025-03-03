import pandas as pd

counts_df = pd.read_csv("results/csv/dlr_substations_count.csv")
multiple_dlr = counts_df[counts_df["num_dlr_substations"] > 1]

print(f"Number of network substations with >1 DLR substations: {len(multiple_dlr)}")
print(multiple_dlr.head())
