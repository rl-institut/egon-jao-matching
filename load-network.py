import dill
import pypsa
import pypsa.descriptors

# Ensure 'Dict' exists in pypsa.descriptors
if not hasattr(pypsa.descriptors, "Dict"):
    pypsa.descriptors.Dict = dict
    print("Added 'Dict' attribute to pypsa.descriptors.")

# Path to the Pickle file
pickle_path = (
    "network/network_post_selection_status2023_8760_3_post_reactance_fix_updated.pkl"
)

try:
    with open(pickle_path, "rb") as f:
        network = dill.load(f)
    print("Network object loaded successfully.")
except Exception as e:
    print(f"Error loading network object: {e}")
    raise e

# Print network information
print("\n==== Basic Network Summary ====")
print(network)

# Inspect Lines
print("\n==== Lines Information ====")
print("Columns:", network.lines.columns.tolist())
print("Shape:", network.lines.shape)
print(network.lines.info())
print("\nSample rows (head):")
print(network.lines.head(5))
print("-" * 40)

# Inspect Transformers
print("\n==== Transformers Information ====")
print("Columns:", network.transformers.columns.tolist())
print("Shape:", network.transformers.shape)
print(network.transformers.info())
print("\nSample rows (head):")
print(network.transformers.head(5))
print("-" * 40)

# Optionally, inspect other components
print("\n==== Generators Information ====")
print("Columns:", network.generators.columns.tolist())
print("Shape:", network.generators.shape)
print(network.generators.info())
print("\nSample rows (head):")
print(network.generators.head(5))
print("-" * 40)
