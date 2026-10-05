import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests as re
import sncosmo
from astropy.time import Time

from rubin_eex_finder.utils.light_curve_utils import (
    add_night_and_SNR_columns,
    compute_nightly_g_minus_r,
    nightly_weighted_average,
)
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


def colour_v_time_plot(
    colour_dataframe: pl.DataFrame,
    colour_v_time_output_path: Path | None = None,
) -> None:
    """Makes a plot of the magnitude colour as a function of the objects time, to understand the colour evolution over time.

    Args:
        colour_dataframe (pl.DataFrame): The dataframe with the colour in AB mag in whatever bands the user wants as well as with time data.
    """

    object_id = colour_dataframe["r:diaObjectId"].first()

    t_fd_rest = colour_dataframe["t_fd_rest_frame"].first()

    phases = colour_dataframe["rest_frame_phase"].to_numpy()

    g_minus_r_colour = colour_dataframe["g_minus_r"].to_numpy()

    g_minus_r_colour_err = colour_dataframe["g_minus_r_err"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        x=phases,
        y=g_minus_r_colour,
        yerr=g_minus_r_colour_err,
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        mec="white",
        mew=0.8,
        markersize=6,
        capsize=3,
        label=r"observed $g-r$",
        zorder=3,
    )

    ax.axvspan(
        xmin=t_fd_rest,
        xmax=t_fd_rest + 4.0,
        label="Early Excess Window",
        zorder=1,
        color="#E94E77",
        alpha=0.15,
        lw=2,
        ec="#E31A50",
        ls="--",
    )

    # model = sncosmo.Model(source="salt3")
    # model.set(
    #     z=(colour_dataframe["z_used"].first()),
    #     t0=(colour_dataframe["t0"].first()),
    #     x0=(colour_dataframe["x0"].first()),
    #     x1=(colour_dataframe["x1"].first()),
    #     c=(colour_dataframe["c"].first()),
    # )

    # phase_grid = np.linspace(min(phases) - 2, max(phases) + 5, 200)
    # t_grid = colour_dataframe["t0"].first() + phase_grid * (1.0 + colour_dataframe["z_used"].first())

    # g_flux = np.array([model.bandflux("lsstg", t, zp=31.4, zpsys="ab") for t in t_grid])
    # r_flux = np.array([model.bandflux("lsstr", t, zp=31.4, zpsys="ab") for t in t_grid])

    # valid = (g_flux > 0) & (r_flux > 0)
    # salt_color = (-2.5 * np.log10(g_flux[valid])) - (-2.5 * np.log10(r_flux[valid]))

    # ax.plot(
    #     phase_grid[valid],
    #     salt_color,
    #     "--",
    #     color="#D55E00",
    #     lw=1.8,
    #     label="SALT3 Model",
    #     zorder=2,
    # )

    ax.set_xlabel(xlabel="Rest frame days before B-band peak")
    ax.set_ylabel(ylabel=r"($g-r$) AB Mag")
    ax.legend(loc="best")
    ax.set_title(
        label=f"{object_id}", fontsize=11, pad=10
    )
    plt.tight_layout()

    if colour_v_time_output_path is not None:
        colour_v_time_output_path.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(fname=colour_v_time_output_path / f"{object_id}.png")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    LIGHT_CURVE_DATA_PATH = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    OBJECT_PATH = (
        DATA_DIR
        / "eex_results"
        / "salt_method"
        / "eex_cut_results"
        / "3_day_window"
        / "pass.csv"
    )

    PLOT_OUTPUT_PATH = IMAGE_DIR / "results_plots"

    LIGHT_CURVE_DATA = pl.read_parquet(source=LIGHT_CURVE_DATA_PATH)
    OBJECTS_DF = pl.read_csv(
        source=OBJECT_PATH,
        columns=[
            "r:diaObjectId",
            "z_used",
            "t0",
            "x0",
            "x1",
            "c",
            "t_fd",
        ],
        schema_overrides={"r:diaObjectId": pl.String},
    )

    OBJECTS_DF = OBJECTS_DF.with_columns(
        t_fd_rest_frame=(pl.col("t_fd") - pl.col("t0")) / (1 + pl.col("z_used"))
    )

    OBJECT_LIGHT_CURVE_DATA = OBJECTS_DF.join(
        other=LIGHT_CURVE_DATA,
        on="r:diaObjectId",
        how="inner",
    )

    nightly_df = nightly_weighted_average(light_curve_data=OBJECT_LIGHT_CURVE_DATA)

    g_r_colour_df = compute_nightly_g_minus_r(
        nightly_df=nightly_df,
    ).filter(pl.col("r:diaObjectId") == "170019716323082348")

    colour_v_time_plot(
        colour_dataframe=g_r_colour_df,
        colour_v_time_output_path=PLOT_OUTPUT_PATH,
    )
