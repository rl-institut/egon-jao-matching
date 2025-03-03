import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def plot_lines_comparison(lines_in_germany_original, lines_in_germany):
    """
    Create a Cartopy plot comparing original and straightened network lines.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(20, 10),
        subplot_kw={
            "projection": ccrs.LambertConformal(
                central_longitude=10, central_latitude=52, standard_parallels=(50, 60)
            )
        },
    )

    # Original Lines Plot
    axes[0].set_title("Original Curved Network Lines")
    lines_in_germany_original.plot(
        ax=axes[0], color="green", linewidth=1, transform=ccrs.Geodetic()
    )
    axes[0].add_feature(cfeature.LAND, facecolor="white")
    axes[0].add_feature(cfeature.OCEAN, facecolor="lightblue")
    axes[0].add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
    axes[0].add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.5)

    # Straightened Lines Plot
    axes[1].set_title("Straightened Network Lines")
    lines_in_germany.plot(
        ax=axes[1], color="green", linewidth=1, transform=ccrs.Geodetic()
    )
    axes[1].add_feature(cfeature.LAND, facecolor="white")
    axes[1].add_feature(cfeature.OCEAN, facecolor="lightblue")
    axes[1].add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
    axes[1].add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.5)

    # Show the plot
    plt.show()
