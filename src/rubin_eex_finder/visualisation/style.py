import matplotlib.pyplot as plt

# Custom colour palette (colourblind-friendly)
COLOUR_SN_LIKE = "#0072B2"  # blue
COLOUR_NON_SN_LIKE = "#D55E00"  # orange
COLOUR_PHOTOMETRIC = "#7F8C8D"  # grey
COLOUR_CUT_LINES = "#E94E77"  # pink

# Default Figure Sizes
FIGSIZE_DISTRIBUTION = (8, 5)
FIGSIZE_SCATTER = (7, 6)


def set_publication_style():
    """Applying standard styling for thesis/paper figures."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "figure.autolayout": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
