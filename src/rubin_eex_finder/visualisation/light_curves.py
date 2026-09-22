import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests as re
from astropy.time import Time
from utils.light_curve_utils import add_night_and_SNR_columns

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


def batch_plotter(object_path: Path, light_curve_data_path: Path, output_path: Path):
    """_summary_

    Args:
        object_ids (_type_): _description_
        output_path (Path): _description_
        chunk_size (int, optional): _description_. Defaults to 500.
    """
    object_ids = pl.read_csv(
        object_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    )["r:diaObjectId"].to_list()

    light_curve_data = pl.read_parquet(light_curve_data_path).with_columns(
        pl.col("r:diaObjectId").cast(pl.String)
    )
    light_curve_data_filtered = light_curve_data.filter(
        pl.col("r:diaObjectId").is_in(object_ids)
    )

    groups = {
        key[0]: group_df
        for key, group_df in light_curve_data_filtered.group_by(
            "r:diaObjectId", maintain_order=True
        )
    }

    for object_id in object_ids:
        group_df = groups.get(object_id).filter(pl.col("r:psfFlux") > 0)

        if group_df is None:
            print(f"Object {object_id} does not have any light curve data.")
            continue

        lightcurve_plotting(group_df, object_id, output_path)


band_colours = {
    "u": "darkblue",
    "g": "blue",
    "r": "red",
    "i": "darkred",
    "z": "orange",
    "y": "black",
}


def append_fitted_slopes(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Appends per-band fitted slope rows to a CSV file, writing a header only if the

    file doesn't already exist. Each row is expected to contain diaObjectId, band,
    slope, and n_pre_peak_points.
    """
    if not rows:
        return

    new_rows_df = pl.DataFrame(rows)
    if not output_path.exists():
        new_rows_df.write_csv(output_path)
    else:
        with open(output_path, "ab") as f:
            new_rows_df.write_csv(f, include_header=False)


def collect_light_curve_data(diaObjectId: str) -> pl.DataFrame | None:
    r = re.post(
        "https://api.lsst.fink-portal.org/api/v1/sources",
        json={
            "diaObjectId": diaObjectId,
            "columns": (
                "r:diaObjectId, r:diaSourceId, r:midpointMjdTai, r:psfFlux,"
                " r:psfFluxErr, r:band"
            ),
            "output-format": "json",
        },
    )
    if r.status_code != 200:
        print(
            f"Failed to fetch the light curve info for {diaObjectId}: {r.status_code}"
        )
        return None
    return pl.read_json(io.BytesIO(r.content))


def slope_fit_plotting(
    df: pl.DataFrame,
    diaObjectId: str,
    source_file: str,
    output_dir: str,
    min_pre_peak_points: int = 3,
    min_pre_peak_time_span: float = 5,
    slopes_output_path: Path | None = None,
    min_peak_SNR: int = 10,
):
    """Plots the light curve for one object directly in flux space (no magnitude

    conversion), with the pre-peak linear fit for each band overlaid as a dashed
    line. Also prints the fitted slope and number of pre-peak points used, per band,
    and appends the same values to slopes_output_path so they can be inspected after
    a batch run rather than just read off the console.

    Working in flux rather than magnitude avoids fitting a straight line to a
    logarithmic quantity, which was flagged as a potential source of distortion in
    the rate-of-rise filter's output.
    """
    df = df.filter(pl.col("r:psfFlux") > 0).sort("r:midpointMjdTai")

    # Convert MJD to datetime
    dates = Time(df["r:midpointMjdTai"].to_numpy(), format="mjd").to_datetime()
    df = df.with_columns(pl.Series("date", dates))

    fig, ax = plt.subplots(figsize=(10, 6))
    fitted_slope_rows = []

    for (band,), group in df.group_by("r:band"):
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

        # Extract peak flux details using max row sorting
        peak_row = group.sort("r:psfFlux", descending=True).row(0, named=True)
        peak_flux = peak_row["r:psfFlux"]
        peak_flux_err = peak_row["r:psfFluxErr"]
        peak_mjd = peak_row["r:midpointMjdTai"]

        peak_SNR = peak_flux / peak_flux_err
        if peak_SNR < min_peak_SNR:
            print(
                f"Object {diaObjectId}'s peak point's SNR is {peak_SNR:.2f},"
                f" which doesn't pass the cut-off of {min_peak_SNR}"
            )
            continue

        pre_peak = group.filter(pl.col("r:midpointMjdTai") <= peak_mjd)
        pre_peak = add_night_and_SNR_columns(pre_peak)

        # Retain highest SNR point per night
        pre_peak_best_points = (
            pre_peak.sort("SNR", descending=True)
            .group_by(["r:diaObjectId", "r:band", "night"])
            .first()
        )

        if len(pre_peak_best_points) < min_pre_peak_points:
            print(
                f"  [{diaObjectId}] {band}-band: only"
                f" {len(pre_peak_best_points)} pre-peak point(s), skipping"
                " fit."
            )
            continue

        mjd_series = pre_peak_best_points["r:midpointMjdTai"]
        time_span = mjd_series.max() - mjd_series.min()

        if time_span < min_pre_peak_time_span:
            print(
                f"  [{diaObjectId}] {band}-band: pre-peak points only span"
                f" {time_span:.2f} days, skipping fit."
            )
            continue

        normalised_flux = (pre_peak_best_points["r:psfFlux"] / peak_flux).to_numpy()
        normalised_err = (
            (pre_peak_best_points["r:psfFluxErr"] / peak_flux)
            .clip(lower_bound=1e-6)
            .to_numpy()
        )
        weights = 1.0 / normalised_err

        mjd_vals = mjd_series.to_numpy()
        slope, intercept = np.polyfit(mjd_vals, normalised_flux, 1, w=weights)

        print(
            f"  [{diaObjectId}] {band}-band: slope={slope:.5f},"
            f" n_pre_peak={len(pre_peak_best_points)},"
            f" pre-peak_time_span={time_span:.2f}"
        )

        fitted_slope_rows.append(
            {
                "diaObjectId": diaObjectId,
                "band": band,
                "slope": slope,
                "n_pre_peak_points": len(pre_peak_best_points),
            }
        )

        fit_mjd = np.sort(mjd_vals)
        fit_flux = (slope * fit_mjd + intercept) * peak_flux
        fit_dates = Time(fit_mjd, format="mjd").to_datetime()

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


def batch_plotter(
    object_path: Path,
    light_curve_data_path: Path,
    output_path: Path,
):
    """Batch Plotter takes a list of object IDs for which in this case are the results

    from the slope fitting filter and plots the objects lightcurve in chunks.
    """
    objects = pl.read_csv(
        object_path,
        columns=["r:diaObjectId"],
        schema_overrides={"r:diaObjectId": pl.String},
    )["r:diaObjectId"].to_list()

    light_curve_data = pl.read_parquet(light_curve_data_path)
    light_curve_data_filtered = light_curve_data.filter(
        pl.col("r:diaObjectId").cast(pl.String).is_in(objects)
    )

    for (object_id,), group_df in light_curve_data_filtered.group_by("r:diaObjectId"):
        slope_fit_plotting(group_df, object_id, object_id, output_path / f"{object_id}")
