import io

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests as re
from astropy.time import Time

band_colours = {
    "u": "darkblue",
    "g": "blue",
    "r": "red",
    "i": "darkred",
    "z": "orange",
    "y": "black",
}


def collect_light_curve_data(diaObjectId: str) -> pl.DataFrame | None:
    r = re.post(
        "https://api.lsst.fink-portal.org/api/v1/sources",
        json={
            "diaObjectId": diaObjectId,
            "columns": "r:diaObjectId, r:diaSourceId,r:midpointMjdTai,r:psfFlux,r:psfFluxErr, r:band",
            "output-format": "json",
        },
    )
    if r.status_code != 200:
        print(
            f"Failed to fetch the light curve info for {diaObjectId}: {r.status_code}"
        )
        return None
    return pl.read_json(io.BytesIO(r.content))


def flux_to_mag():
    """Returns Polars expressions for nanojansky flux to AB magnitude conversion with uncertainties.

    Returns:
        pl.DataFrame: returns a Polars dataframe with two new columns for AB magnitude and magnitude error.
    """
    mag = -2.5 * pl.col("r:psfFlux").log10() + 31.4
    mag_err = (2.5 / np.log(10)) * (pl.col("r:psfFluxErr") / pl.col("r:psfFlux"))
    return mag.alias("mag"), mag_err.alias("mag_err")


def lightcurve_plotting(
    light_curve_data: pl.DataFrame, diaObjectId: str, output_dir: str
):
    """The plotting of the objects lightcurve this function is what produces the plot

    Args:
        light_curve_data (pl.DataFrame): The dataframe of all the object's light curve data.
        diaObjectId (str): This is just the string that is the name of the Rubin object.
        output_dir (str): The directory where the plots are saved to.
    """
    light_curve_data = light_curve_data.filter(pl.col("r:psfFlux") > 0).sort(
        "r:midpointMjdTai"
    )

    dates = Time(
        light_curve_data["r:midpointMjdTai"].to_numpy(), format="mjd"
    ).to_datetime()
    light_curve_data = light_curve_data.with_columns(date=pl.Series("date", dates))

    fig, ax = plt.subplots(figsize=(10, 6))

    for (band,), group in light_curve_data.group_by("r:band"):
        colour = band_colours.get(band, "black")
        ax.errorbar(
            group["date"].to_numpy(),
            group["r:psfFlux"].to_numpy(),
            yerr=group["r:psfFluxErr"].to_numpy(),
            fmt="o",
            color=colour,
            label=f"{band}-band",
            capsize=3,
            markersize=5,
        )

    ax.set_xlabel("Observation Date")
    ax.set_ylabel("Flux (nanojanskys)")
    ax.set_title(f"Light curve - {diaObjectId}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_dir / f"{diaObjectId}.png")
    plt.close()
