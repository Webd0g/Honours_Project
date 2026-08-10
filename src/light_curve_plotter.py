# plots the light curve of the objects that haven't pass through all three classifiers that we run.
import io
import requests as re
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

band_colours = {"u": "darkblue", "g":"blue", "r": "red", "i":"darkred", "z":"orange", "y":"black"}


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
        print(f"Failed to fetch the light curve info for {diaObjectId}: {r.status_code}")
        return None
    return pl.read_json(io.BytesIO(r.content))


def flux_to_mag():
    """Returns Polars expressions for nanojansky flux to AB magnitude conversion with uncertainties."""
    mag = -2.5 * pl.col("r:psfFlux").log10() + 31.4
    mag_err = (2.5 / np.log(10)) * (pl.col("r:psfFluxErr") / pl.col("r:psfFlux"))
    return mag.alias("mag"), mag_err.alias("mag_err")


def lightcurve_plotting(df: pl.DataFrame, diaObjectId: str, output_dir: str):
    # Filter non-positive flux and sort chronologically
    df = df.filter(pl.col("r:psfFlux") > 0).sort("r:midpointMjdTai")

    # Convert MJD to datetime objects
    dates = Time(df["r:midpointMjdTai"].to_numpy(), format="mjd").to_datetime()
    df = df.with_columns(date=pl.Series("date", dates))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over groups using Polars group_by tuple unpacking
    for (band,), group in df.group_by("r:band"):
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