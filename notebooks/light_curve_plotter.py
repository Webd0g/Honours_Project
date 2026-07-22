# plots the light curve of the objects that haven't pass through all three classifiers that we run.
import io
import requests as re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

band_colours = {"u": "darkblue", "g":"blue", "r": "gray", "i":"pink", "z":"orange"}   

def collect_light_curve_data(diaObjectId):
    r = re.post(
        "https://api.lsst.fink-portal.org/api/v1/sources",
        json={
            "diaObjectId": diaObjectId,
            "columns": "r:diaObjectId, r:diaSourceId,r:midpointMjdTai,r:psfFlux,r:psfFluxErr, r:band",
            "output-format": "json",
        },
    )
    if r.status_code != 200:
        print (f"Failed to fetch the light curve info for {diaObjectId}: {r.status_code}")
        return None
    return pd.read_json(io.BytesIO(r.content))

def flux_to_mag(flux, flux_err):
    """Converting nanojansky flux to AB magnitude with uncertainties"""
    AB_mag = -2.5*np.log10(flux) + 31.4
    AB_mag_err = (2.5 / np.log(10)) * (flux_err / flux)
    return AB_mag, AB_mag_err

def lightcurve_plotting(df: pd.DataFrame, diaObjectId: str, source_file: str, output_dir: str):
    df = df[df["r:psfFlux"] > 0].copy()
    df = df.sort_values("r:midpointMjdTai")
    df["mag"], df["mag_err"] = flux_to_mag(df["r:psfFlux"], df["r:psfFluxErr"])
    df["date"] = Time(df["r:midpointMjdTai"].values, format="mjd").to_datetime()

    fig, ax = plt.subplots(figsize = (10, 6))

    for band, group in df.groupby("r:band"):
        colour = band_colours.get(band, "black")
        ax.errorbar(
            group["date"],
            group["mag"],
            yerr = group["mag_err"],
            fmt = "o",
            color = colour,
            label = f"{band}-band",
            capsize = 3,
            markersize = 5)
        
    ax.invert_yaxis()
    ax.set_xlabel("Observation Date")
    ax.set_ylabel("AB magnitude")
    ax.set_title(f"Light curve - {diaObjectId}\n({source_file})")
    ax.legend()
    ax.grid(True, alpha = 0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close(fig)