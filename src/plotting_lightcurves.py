import io
import requests as re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from pathlib import Path
import polars as pl
from rate_of_rise_cut import one_point_per_night

SLOPES_OUTPUT_PATH = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "slope_fitting_results" / "slope_values.csv"

TEST_DATA_DIR = Path.home() / "Documents" / "University"/"Honours"/"Honours_Project"/"data"/"test_data"
PASSED_OBJECTS =  TEST_DATA_DIR / "test_passed.csv"
FAILED_OBJECTS = TEST_DATA_DIR / "test_failed.csv"

PASSED_OBJECTS_DF = pl.read_csv(PASSED_OBJECTS, schema_overrides={"r:diaObjectId": pl.String}, columns=["r:diaObjectId"])["r:diaObjectId"].to_list()
FAILED_OBJECTS_DF = pl.read_csv(FAILED_OBJECTS, schema_overrides={"r:diaObjectId": pl.String}, columns=["r:diaObjectId"])["r:diaObjectId"].to_list()

def append_fitted_slopes(rows: list[dict], output_path: Path = SLOPES_OUTPUT_PATH) -> None:
    """
    Appends per-band fitted slope rows to a CSV file, writing a header only if the
    file doesn't already exist. Each row is expected to contain diaObjectId, band,
    slope, and n_pre_peak_points.
    """
    if not rows:
        return

    new_rows_df = pd.DataFrame(rows)

    new_rows_df.to_csv(output_path, mode="a", header=False, index=False)

band_colours = {"u": "darkblue", "g":"blue", "r": "red", "i":"darkred", "z":"orange", "y":"black"}   

def collect_light_curve_data(diaObjectId):
    r = re.post(
        "https://api.lsst.fink-portal.org/api/v1/sources",
        json={
            "diaObjectId": diaObjectId,
            "columns": "r:diaObjectId, r:diaSourceId, r:midpointMjdTai, r:psfFlux, r:psfFluxErr, r:band",
            "output-format": "json",
        },
    )
    if r.status_code != 200:
        print (f"Failed to fetch the light curve info for {diaObjectId}: {r.status_code}")
        return None
    return pd.read_json(io.BytesIO(r.content))

def lightcurve_plotting(df: pd.DataFrame, diaObjectId: str, source_file: str, output_dir: str, min_pre_peak_points: int = 4, min_pre_peak_time_span: float = 5, slopes_output_path: Path = SLOPES_OUTPUT_PATH, min_peak_SNR: int = 10):
    """
    Plots the light curve for one object directly in flux space (no magnitude
    conversion), with the pre-peak linear fit for each band overlaid as a dashed
    line. Also prints the fitted slope and number of pre-peak points used, per band,
    and appends the same values to slopes_output_path so they can be inspected after
    a batch run rather than just read off the console.

    Working in flux rather than magnitude avoids fitting a straight line to a
    logarithmic quantity, which was flagged as a potential source of distortion in
    the rate-of-rise filter's output.
    """
    df = df[df["r:psfFlux"] > 0].copy()
    df = df.sort_values("r:midpointMjdTai")
    df["date"] = Time(df["r:midpointMjdTai"].values, format="mjd").to_datetime()

    fig, ax = plt.subplots(figsize=(10, 6))
    fitted_slope_rows = []

    for band, group in df.groupby("r:band"):
        colour = band_colours.get(band, "black")

        ax.errorbar(
            group["date"],
            group["r:psfFlux"],
            yerr=group["r:psfFluxErr"],
            fmt="o",
            color=colour,
            label=f"{band}-band",
            capsize=3,
            markersize=5,
        )
        
        # Fit and overlay the pre-peak slope for this band
        max_idx = group["r:psfFlux"].idxmax()
        peak_flux = group.loc[max_idx, "r:psfFlux"]
        peak_flux_err = group.loc[max_idx, "r:psfFluxErr"]
        peak_mjd = group.loc[max_idx, "r:midpointMjdTai"]
        pre_peak = group[group["r:midpointMjdTai"] < peak_mjd]
        peak_SNR = peak_flux/peak_flux_err

        if (peak_SNR) < min_peak_SNR:
            print (f"Object {diaObjectId}'s peak point's SNR is {peak_SNR} which doesn't pass the cut-off of {min_peak_SNR} ")
            continue

        print (pre_peak)
        pre_peak = one_point_per_night(pre_peak)
        print (pre_peak)

        if len(pre_peak) < min_pre_peak_points:
            print(f"  [{diaObjectId}] {band}-band: only {len(pre_peak)} pre-peak point(s), skipping fit.")
            continue

        time_span = pre_peak["r:midpointMjdTai"].max() - pre_peak["r:midpointMjdTai"].min()
        if time_span < min_pre_peak_time_span:
            print(f"  [{diaObjectId}] {band}-band: pre-peak points only span {time_span:.2f} days, skipping fit.")
            continue

        normalised_flux = pre_peak["r:psfFlux"] / peak_flux
        normalised_err = (pre_peak["r:psfFluxErr"] / peak_flux).clip(lower=1e-6)
        weights = 1 / normalised_err

        slope, intercept = np.polyfit(pre_peak["r:midpointMjdTai"], normalised_flux, 1, w=weights)

        print(f"  [{diaObjectId}] {band}-band: slope={slope:.5f}, n_pre_peak={len(pre_peak)}")

        fitted_slope_rows.append({
            "diaObjectId": diaObjectId,
            "band": band,
            "slope": slope,
            "n_pre_peak_points": len(pre_peak),
        })

        fit_mjd = pre_peak["r:midpointMjdTai"].sort_values()
        fit_flux = (slope * fit_mjd + intercept) * peak_flux
        fit_dates = Time(fit_mjd.values, format="mjd").to_datetime()

        ax.plot(
            fit_dates,
            fit_flux,
            linestyle="--",
            color=colour,
            alpha=0.7,
            label=f"{band} fit (slope={slope:.4f})",
        )

    ax.set_xlabel("Observation Date")
    ax.set_ylabel("Flux (nanojansky)")
    ax.set_title(f"Light curve - {diaObjectId}\n({source_file})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{output_dir}.png")
    plt.close(fig)

    append_fitted_slopes(fitted_slope_rows, slopes_output_path)

def batch_plotter(object_ids: list[str], chunk_size: int = 500, output_path: Path = Path.home() / "Pictures"):
    """Batch Plotter takes a list of object IDs for which in this case are the reuslts 
    from the slope fitting filter and plots the objects lightcurve in chunks."""

    for diaObjectId in range(0, len(object_ids), chunk_size):
        chunk = ",".join(object_ids[diaObjectId:diaObjectId + chunk_size])
        light_curve_info = collect_light_curve_data(chunk)
        if light_curve_info is None or light_curve_info.empty:
            print(f"No light curve data found for chunk starting at index {diaObjectId}.")
            continue
        for object_id, group_df in light_curve_info.groupby("r:diaObjectId"):
            object_id = str(object_id)
            lightcurve_plotting(group_df, object_id, object_id, output_path / f"{object_id}")

if __name__ == "__main__":
    batch_plotter(["170235219334922248"], 500, Path.home() / "Pictures")