import folium
import pandas as pd


def save_csv(df, path):
    """
    Save a DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): DataFrame to save.
    - path (str): Destination path for the CSV file.
    """
    try:
        df.to_csv(path, index=False)
        print(f"Saved CSV to {path}")
    except Exception as e:
        print(f"Failed to save CSV to {path}: {e}")


def save_folium_map(folium_map, path):
    """
    Save a Folium map to an HTML file.

    Parameters:
    - folium_map (folium.Map): The Folium map object.
    - path (str): Destination path for the HTML file.
    """
    try:
        folium_map.save(path)
        print(f"Saved map to {path}")
    except Exception as e:
        print(f"Failed to save map to {path}: {e}")
