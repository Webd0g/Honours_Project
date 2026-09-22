from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


def bazin_curve(
    t: np.ndarray,
    amplitude: float,
    t0: float,
    tau_rise: float,
    tau_fall: float,
    baseline: float,
) -> np.ndarray:
    """The Bazin et al. (2009) model, used here just to draw the fitted curve
    for visual inspection (feets returns the fitted parameters, not the curve
    itself)."""
    return (
        amplitude * np.exp(-(t - t0) / tau_fall) / (1 + np.exp(-(t - t0) / tau_rise))
        + baseline
    )


def plot_bazin_fit_single_object(
    group_df: pd.DataFrame,
    band: str,
    fit_result: dict,
    diaObjectId: str,
    output_path: Path,
) -> None:
    """
    Quick visual check: plots the band's actual flux data points with the fitted
    Bazin curve overlaid, so the fit can be sanity-checked by eye.
    """

    band_df = group_df.filter(
        (pl.col("r:band") == band) & (pl.col("r:psfFlux") > 0)
    ).sort("r:midpointMjdTai")

    t = band_df["r:midpointMjdTai"].to_numpy().astype(np.float64)
    flux = band_df["r:psfFlux"].to_numpy().astype(np.float64)
    flux_err = band_df["r:psfFluxErr"].to_numpy().astype(np.float64)

    t_fine = np.linspace(t.min(), t.max(), 300)
    fit_flux = bazin_curve(
        t_fine,
        fit_result["amplitude"],
        fit_result["reference_time"],
        fit_result["rise_time"],
        fit_result["fall_time"],
        fit_result["baseline"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        t,
        flux,
        yerr=flux_err,
        fmt="o",
        color="darkred",
        label=f"{band}-band data",
        capsize=3,
        markersize=5,
    )
    ax.plot(
        t_fine,
        fit_flux,
        "--",
        color="purple",
        label=f"Bazin fit (chi2={fit_result['reduced_chi2']:.2f})",
    )
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux (nanojansky)")
    ax.set_title(f"Bazin fit - {diaObjectId} ({band}-band)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f"{diaObjectId}_{band}_band.png")
    print(f"saved the plot of object {diaObjectId}")
    plt.close(fig)


def plot_bazin_fit_batch(
    dict_path: Path,
    csv_path: Path,
    light_curve_data_path: Path,
    band: str,
    output_path: Path,
) -> None:
    """
    Plots the light curve of the objects data in the band requested and overlays the fitted Bazin
    line on the plot.

    This is to check that the Bazin fit is both doing what we want it to do and gives up data for visual
    inspection which is both valuable and neccessary.
    """

    objects = pl.read_csv(
        csv_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    )["r:diaObjectId"].to_list()
    light_curve_data = pl.read_parquet(light_curve_data_path)
    light_curve_data_filtered = light_curve_data.filter(
        pl.col("r:diaObjectId").cast(pl.Utf8).is_in(objects)
    )

    bazin_df = pl.read_csv(dict_path, schema_overrides={"r:diaObjectId": pl.String})
    bazin_fit_dict = (
        bazin_df.to_pandas().set_index("r:diaObjectId").to_dict(orient="index")
    )

    groups = {
        key[0]: group_df
        for key, group_df in light_curve_data_filtered.group_by(
            "r:diaObjectId", maintain_order=True
        )
    }

    for object_id in objects:
        group_df = groups.get(object_id)
        if group_df is None:
            continue
        fit_result = bazin_fit_dict.get(object_id)
        print(fit_result)
        if fit_result is None:
            print(
                f"Object {object_id} did not have any data in that band and couldn't be fitted to and thus won't be plotted."
            )
            continue
        if plot_bazin_fit_single_object(
            group_df, band, fit_result, object_id, output_path
        ):
            print(f"{object_id} has been plotted.")
