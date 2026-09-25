import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests as re
from astropy.time import Time

from rubin_eex_finder.utils.light_curve_utils import add_night_and_SNR_columns
from rubin_eex_finder.visualisation.style import (
    set_publication_style,
)

# Set global thesis plotting styles
set_publication_style()

BAND_COLOURS = {
    "u": "darkblue",
    "g": "blue",
    "r": "red",
    "i": "darkred",
    "z": "orange",
    "y": "black",
}


def collect_light_curve_data(diaObjectId: str) -> pl.DataFrame | None:
    """Fetch light curve data for a given diaObjectId directly from the Fink API."""
    response = re.post(
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
    if response.status_code != 200:
        print(
            f"Failed to fetch light curve info for {diaObjectId}: {response.status_code}"
        )
        return None
    return pl.read_json(io.BytesIO(response.content))


def flux_to_mag(
    flux_col: str = "r:psfFlux", err_col: str = "r:psfFluxErr"
) -> tuple[pl.Expr, pl.Expr]:
    """Returns Polars expressions for nanojansky flux to AB magnitude conversion."""
    mag = -2.5 * pl.col(flux_col).log10() + 31.4
    mag_err = (2.5 / np.log(10)) * (pl.col(err_col) / pl.col(flux_col))
    return mag.alias("mag"), mag_err.alias("mag_err")


def append_fitted_slopes(rows: list[dict], output_path: Path) -> None:
    """Appends per-band fitted slope rows to a CSV file."""
    if not rows:
        return

    new_rows_df = pl.DataFrame(rows)
    if not output_path.exists():
        new_rows_df.write_csv(output_path)
    else:
        with open(output_path, "ab") as f:
            new_rows_df.write_csv(f, include_header=False)


def lightcurve_plotting(
    light_curve_data: pl.DataFrame, diaObjectId: str, output_dir: Path
) -> None:
    """Plots a standard flux-space time-series light curve for a single object."""
    light_curve_data = light_curve_data.filter(pl.col("r:psfFlux") > 0).sort(
        "r:midpointMjdTai"
    )

    dates = Time(
        light_curve_data["r:midpointMjdTai"].to_numpy(), format="mjd"
    ).to_datetime()
    light_curve_data = light_curve_data.with_columns(date=pl.Series("date", dates))

    fig, ax = plt.subplots(figsize=(10, 6))

    for (band,), group in light_curve_data.group_by("r:band"):
        colour = BAND_COLOURS.get(band, "black")
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
    ax.set_ylabel("Flux (nJy)")
    ax.set_title(f"Light Curve - {diaObjectId}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{diaObjectId}.png")
    plt.close(fig)


def slope_fit_plotting(
    df: pl.DataFrame,
    diaObjectId: str,
    source_file: str,
    output_dir: Path,
    min_pre_peak_points: int = 3,
    min_pre_peak_time_span: float = 5.0,
    slopes_output_path: Path | None = None,
    min_peak_SNR: float = 10.0,
) -> None:
    """Plots flux light curve with an overlaid pre-peak linear rate-of-rise fit."""
    df = df.filter(pl.col("r:psfFlux") > 0).sort("r:midpointMjdTai")

    dates = Time(df["r:midpointMjdTai"].to_numpy(), format="mjd").to_datetime()
    df = df.with_columns(pl.Series("date", dates))

    fig, ax = plt.subplots(figsize=(10, 6))
    fitted_slope_rows = []

    for (band,), group in df.group_by("r:band"):
        colour = BAND_COLOURS.get(band, "black")

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

        # Extract peak flux details
        peak_row = group.sort("r:psfFlux", descending=True)[0]
        peak_flux = peak_row["r:psfFlux"][0]
        peak_flux_err = peak_row["r:psfFluxErr"][0]
        peak_mjd = peak_row["r:midpointMjdTai"][0]

        peak_SNR = peak_flux / peak_flux_err
        if peak_SNR < min_peak_SNR:
            print(
                f"[{diaObjectId}] {band}-band peak SNR ({peak_SNR:.2f}) < cutoff ({min_peak_SNR})"
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
                f"  [{diaObjectId}] {band}-band: insufficient pre-peak points ({len(pre_peak_best_points)})"
            )
            continue

        mjd_series = pre_peak_best_points["r:midpointMjdTai"]
        time_span = mjd_series.max() - mjd_series.min()

        if time_span < min_pre_peak_time_span:
            print(
                f"  [{diaObjectId}] {band}-band: pre-peak time span ({time_span:.2f}d) too short"
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
    ax.set_ylabel("Flux (nJy)")
    ax.set_title(f"Light Curve - {diaObjectId}\n({source_file})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{diaObjectId}.png")
    plt.close(fig)

    if slopes_output_path is not None:
        append_fitted_slopes(fitted_slope_rows, slopes_output_path)


def batch_plotter(
    object_path: Path,
    light_curve_data_path: Path,
    output_path: Path,
    plot_slope_fit: bool = False,
    slopes_output_path: Path | None = None,
) -> None:
    """Executes batch light curve plotting for a collection of diaObjectIds."""
    object_ids = pl.read_csv(
        object_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    )["r:diaObjectId"].to_list()

    light_curve_data = pl.read_parquet(light_curve_data_path).with_columns(
        pl.col("r:diaObjectId").cast(pl.String)
    )
    filtered_data = light_curve_data.filter(pl.col("r:diaObjectId").is_in(object_ids))

    for (object_id,), group_df in filtered_data.group_by("r:diaObjectId"):
        if plot_slope_fit:
            slope_fit_plotting(
                df=group_df,
                diaObjectId=object_id,
                source_file=object_path.name,
                output_dir=output_path,
                slopes_output_path=slopes_output_path,
            )
        else:
            lightcurve_plotting(
                light_curve_data=group_df,
                diaObjectId=object_id,
                output_dir=output_path,
            )


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    LIGHT_CURVE_DATA_DIR = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    OBJECTS = (
        DATA_DIR / "constant_minus_shape_results" / "constant_minus_shape_passed.csv"
    )

    BAZIN_FIT_DATA_PATH = (
        DATA_DIR / "bazin_fit_results" / "bazin_fits_i_band_results.csv"
    )

    PASS = DATA_DIR / "constant_minus_shape_result" / "passed.csv"

    FAIL = DATA_DIR / "constant_minus_shape_result" / "failed.csv"

    VALUES = DATA_DIR / "constant_minus_shape_result" / "values.csv"

    PLOT_OUTPUT_PATH = IMAGE_DIR / "constant_minus_shape_plots" / "pass_2"

    batch_plotter(
        object_path=OBJECTS,
        light_curve_data_path=LIGHT_CURVE_DATA_DIR,
        output_path=PLOT_OUTPUT_PATH,
    )
