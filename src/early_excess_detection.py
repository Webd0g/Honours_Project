"""This file is for all the functions that go to finding the parameters used for early excess detection and the computation of early excess detection itself."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import sncosmo
from astropy.table import Table

from salt_selection_cuts import get_salt3_fitted_bands


def loading_light_curve_data_and_objects(
    object_path: Path, light_curve_data_path: Path
) -> pl.DataFrame:
    """This function loads the dataframes for the objects from a CSV file and the light curve data for those objects in the CSV file.

    Args:
        object_path (Path): The Path-type object that points to the CSV file for the objects.
        light_curve_data_path (Path): The Path-type object that points to the light curve data for the objects used.

    Returns:
        pl.DataFrame: The returned polars dataframe that contains the light curve data for the objects and the list of objects themselves.
    """

    objects = pl.read_csv(
        object_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
            "z_used": pl.Float64,
            "t0": pl.Float64,
            "x0": pl.Float64,
            "x1": pl.Float64,
            "c": pl.Float64,
        },
    )

    light_curve_data = (
        pl.read_parquet(
            light_curve_data_path,
            columns=[
                "r:diaObjectId",
                "r:band",
                "r:psfFlux",
                "r:psfFluxErr",
                "r:midpointMjdTai",
                "r:diaSourceId",
            ],
        )
        .with_columns(pl.col("r:diaObjectId").cast(pl.String))
        .filter(
            pl.col("r:diaObjectId").is_in(objects["r:diaObjectId"])
            & pl.col("r:psfFlux").is_finite()
            & pl.col("r:psfFluxErr").is_finite()
            & pl.col("r:midpointMjdTai").is_finite()
            & (pl.col("r:psfFluxErr") > 0)
        )
    )

    light_curve_data = light_curve_data.join(
        objects.select("r:diaObjectId", "z_used", "t0", "x0", "x1", "c"),
        on="r:diaObjectId",
        how="inner",
    )

    return light_curve_data


def rest_frame_observation_conversion(light_curve_data: pl.DataFrame) -> pl.DataFrame:
    """This function converts the data points to their rest-frame and returns them as a polars dataframe.

    Args:
        light_curve_data (pl.DataFrame): The polars dataframe of the combined light curve data and the nessecary info from the phase cuts SALT fits.

    Returns:
        pl.DataFrame: The dataframe containing all the light curve data with each epoch converted to their rest-frame.
    """

    return light_curve_data.with_columns(
        ((pl.col("r:midpointMjdTai") - pl.col("t0")) / (1.0 + pl.col("z_used"))).alias(
            "rest_frame_phase"
        )
    )


def remove_sporadic_detections(light_curve_data: pl.DataFrame) -> pl.DataFrame:
    """This function removes the sporadic detections if any for each object as they would not be part of the true supernova light curve and are errors and may influence the detection of true early excess points.

    Args:
        light_curve_data (pl.DataFrame): The polars dataframe of the combined light curve data and the nesseccary info from the phase cuts SALT fits converted to the rest-frame-phase.

    Returns:
        pl.DataFrame: The polars dataframe including the converted res-frame-phase points without any sporadic points that are not part of the main light curve.
    """

    return light_curve_data.filter(pl.col("rest_frame_phase") >= -25)


def nightly_weighted_average(light_curve_data: pl.DataFrame) -> pl.DataFrame:
    """Combine multiple observations from the same night and band using the inverse variance weighting.

    Args:
        light_curve_data (pl.DataFrame): Light curve dataframe after the conversion to rest-frame-phase and removal of epochs earlier than 25 days.

    Returns:
        pl.DataFrame: One row per object, band and observing night containing the weighted neightly flux, uncertainty, representative MJD, rest-frame phase and S/N.
    """

    nightly_data = (
        light_curve_data.with_columns(
            (1.0 / pl.col("r:psfFluxErr").pow(2)).alias("weight"),
            pl.col("r:midpointMjdTai").floor().cast(pl.Int64).alias("night"),
        )
        .group_by(
            [
                "r:diaObjectId",
                "r:band",
                "night",
            ]
        )
        .agg(
            (
                (pl.col("r:psfFlux") * pl.col("weight")).sum() / pl.col("weight").sum()
            ).alias("nightly_flux"),
            (1.0 / pl.col("weight").sum().sqrt()).alias("nightly_flux_err"),
            (
                (pl.col("r:midpointMjdTai") * pl.col("weight")).sum()
                / pl.col("weight").sum()
            ).alias("nightly_mjd"),
            pl.col("z_used").first(),
            pl.col("t0").first(),
            pl.len().alias("n_observations"),
        )
        .with_columns(
            ((pl.col("nightly_mjd") - pl.col("t0")) / (1.0 + pl.col("z_used"))).alias(
                "rest_frame_phase"
            ),
            (pl.col("nightly_flux") / pl.col("nightly_flux_err")).alias("snr"),
        )
        .sort(
            [
                "r:diaObjectId",
                "r:band",
                "nightly_mjd",
            ]
        )
    )

    return nightly_data


def first_detection_determination(nightly_data: pl.DataFrame) -> pl.DataFrame:
    """Filter for the qualities that pass as a first detection.

    Args:
        nightly_data (pl.DataFrame): Light curve dataframe after the conversion to rest frame, removal of sporadic epochs and the weighted snr for each epoch.

    Returns:
        pl.DataFrame: another column per night where it is counted as a detection if the SNR is greater than 3 sigma.
    """

    nightly_data_filtered = nightly_data.with_columns(
        (pl.col("snr") >= 3).alias("is_detection")
    )

    nightly_data_filtered = nightly_data_filtered.with_columns(
        pl.col("is_detection")
        .shift(-2)
        .over(["r:diaObjectId", "r:band"])
        .alias("two_epochs_before"),
        pl.col("is_detection")
        .shift(-1)
        .over(["r:diaObjectId", "r:band"])
        .alias("one_epoch_before"),
        pl.col("is_detection")
        .shift(1)
        .over(["r:diaObjectId", "r:band"])
        .alias("one_epoch_after"),
    )

    candidates = nightly_data_filtered.filter(
        (~pl.col("two_epochs_before"))
        & (~pl.col("one_epoch_before"))
        & pl.col("is_detection")
        & pl.col("one_epoch_after")
    )

    first_detections = (
        candidates.sort(["r:diaObjectId"])
        .group_by(
            "r:diaObjectId",
            maintain_order=True,
        )
        .first()
        .select(
            "r:diaObjectId",
            pl.col("nightly_mjd").alias("t_fd"),
            pl.col("rest_frame_phase").alias("t_fd_phase"),
            pl.col("r:band").alias("t_fd_band"),
            pl.col("snr").alias("t_fd_snr"),
            "z_used",
            "t0",
        )
        .with_columns(pl.lit("method_1").alias("t_fd_method"))
    )

    return first_detections


def find_first_detection(nightly_data: pl.DataFrame) -> pl.DataFrame:
    """Here we find the first detection of each object followed by a second detection.

    Args:
        nightly_data (pl.DataFrame): Light curve dataframe consisting of the rest-frame points and removal of sporadic epochs.

    Returns:
        pl.DataFrame: Dataframe of objects whose first detection is followed by a consecutive detection.
    """

    data = nightly_data.with_columns(
        (pl.col("nightly_mjd"))
        .shift(-1)
        .over(["r:diaObjectId", "r:band"])
        .alias("next_nightly_mjd")
    )

    candidates = data.filter(pl.col("next_nightly_mjd").is_not_null())

    first_detections = (
        candidates.sort(["r:diaObjectId", "nightly_mjd"])
        .group_by("r:diaObjectId", maintain_order=True)
        .first()
        .select(
            "r:diaObjectId",
            pl.col("nightly_mjd").alias("t_fd"),
            pl.col("rest_frame_phase").alias("t_fd_phase"),
            pl.col("r:band").alias("t_fd_band"),
            "z_used",
            "t0",
        )
    )

    return first_detections


def eex_salt_fits_with_mask(
    object_df: pl.DataFrame,
    salt_fit_output_path: Path | None = None,
    pull_output_path: Path | None = None,
    plot_output_path: Path | None = None,
) -> pl.DataFrame:
    """Refit SALT3 after masking the potential early-excess window."""

    band_map = {
        "u": "lsstu",
        "g": "lsstg",
        "r": "lsstr",
        "i": "lssti",
        "z": "lsstz",
        "y": "lssty",
    }

    excess_mask = (pl.col("r:midpointMjdTai") >= pl.col("t_fd")) & (
        pl.col("r:midpointMjdTai") <= pl.col("t_fd") + 4.0 * (1.0 + pl.col("z_used"))
    )

    withheld_df = object_df.filter(excess_mask)

    if plot_output_path is not None:
        plot_output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    groups = {
        key[0]: group_df.sort("r:midpointMjdTai")
        for key, group_df in object_df.group_by(
            "r:diaObjectId",
            maintain_order=True,
        )
    }

    results = []
    pull_results = []

    for object_id, group_df in groups.items():
        z_used = group_df["z_used"].item(0)
        t0 = group_df["t0"].item(0)
        x0 = group_df["x0"].item(0)
        x1 = group_df["x1"].item(0)
        c = group_df["c"].item(0)

        model = sncosmo.Model(source="salt3")

        fitted_bands = get_salt3_fitted_bands(
            redshift=z_used,
            model=model,
        )

        object_fit_df = group_df.filter(
            pl.col("r:band").is_in(fitted_bands)
        ).with_columns(pl.col("r:band").replace(band_map).alias("sncosmo_band"))

        if object_fit_df.height == 0:
            results.append(
                {
                    "r:diaObjectId": object_id,
                    "z_used": z_used,
                    "t0": None,
                    "t0_err": None,
                    "x0": None,
                    "x0_err": None,
                    "x1": None,
                    "x1_err": None,
                    "c": None,
                    "c_err": None,
                    "chi2": None,
                    "ndof": None,
                    "reduced_chi2": None,
                    "n_fit_data": 0,
                    "n_used": 0,
                    "fit_success": False,
                    "fit_message": "No supported SALT3 bands",
                    "n_withheld_points": 0,
                    "n_excess_points": 0,
                    "n_excess_nights": 0,
                    "early_excess": False,
                }
            )
            continue

        data = Table(
            {
                "time": object_fit_df["r:midpointMjdTai"].to_numpy(),
                "band": object_fit_df["sncosmo_band"].to_numpy(),
                "flux": object_fit_df["r:psfFlux"].to_numpy(),
                "fluxerr": object_fit_df["r:psfFluxErr"].to_numpy(),
                "zp": np.full(object_fit_df.height, 31.4),
                "zpsys": np.full(object_fit_df.height, "ab"),
            }
        )

        try:
            model.set(z=z_used, t0=t0, x0=x0, x1=x1, c=c)

            fitted_model = model

            withheld_object_df = withheld_df.filter(
                pl.col("r:diaObjectId") == object_id
            )

            withheld_object_df = withheld_object_df.filter(
                pl.col("r:band").is_in(fitted_bands)
            ).with_columns(pl.col("r:band").replace(band_map).alias("sncosmo_band"))

            model_fluxes = []

            for row in withheld_object_df.iter_rows(named=True):
                model_flux = fitted_model.bandflux(
                    row["sncosmo_band"], row["r:midpointMjdTai"], zp=31.4, zpsys="ab"
                )
                model_fluxes.append(model_flux)

            withheld_object_df = withheld_object_df.with_columns(
                pl.Series("model_flux", model_fluxes)
            )

            withheld_object_df = withheld_object_df.with_columns(
                (
                    (pl.col("r:psfFlux") - pl.col("model_flux"))
                    / pl.col("r:psfFluxErr")
                ).alias("pull")
            )

            withheld_object_df = withheld_object_df.with_columns(
                (pl.col("pull") > 5).alias("significant_excess")
            )

            pull_results.extend(
                withheld_object_df.select(
                    "r:diaObjectId",
                    "r:band",
                    "r:midpointMjdTai",
                    "rest_frame_phase",
                    "r:psfFlux",
                    "r:psfFluxErr",
                    "model_flux",
                    "pull",
                    "significant_excess",
                ).to_dicts()
            )

            excess_nights = (
                withheld_object_df.filter(pl.col("significant_excess"))
                .with_columns(
                    pl.col("r:midpointMjdTai").floor().cast(pl.Int64).alias("night")
                )
                .select("night")
                .n_unique()
            )

            is_early_excess = excess_nights >= 2

            reduced_chi2 = model.chisq / model.ndof if model.ndof > 0 else None

            n_used = int(np.sum(model.data_mask))

            results.append(
                {
                    "r:diaObjectId": object_id,
                    "z_used": z_used,
                    "t0": fitted_model["t0"],
                    "t0_err": model.errors.get("t0"),
                    "x0": fitted_model["x0"],
                    "x0_err": model.errors.get("x0"),
                    "x1": fitted_model["x1"],
                    "x1_err": model.errors.get("x1"),
                    "c": fitted_model["c"],
                    "c_err": model.errors.get("c"),
                    "chi2": model.chisq,
                    "ndof": model.ndof,
                    "reduced_chi2": reduced_chi2,
                    "n_fit_data": len(data),
                    "n_used": n_used,
                    "fit_success": model.success,
                    "fit_message": str(model.message),
                    "n_withheld_points": withheld_object_df.height,
                    "n_excess_points": (
                        withheld_object_df.filter(pl.col("significant_excess")).height
                    ),
                    "n_excess_nights": excess_nights,
                    "early_excess": is_early_excess,
                }
            )

            if plot_output_path is not None:
                sncosmo.plot_lc(
                    data,
                    model=fitted_model,
                    errors=model.errors,
                )

                plt.savefig(
                    plot_output_path / f"{object_id}_eex_SALT3.png",
                    bbox_inches="tight",
                )

                plt.close()

        except Exception as error:
            print(f"{object_id}: {type(error).__name__}: {error}")
            results.append(
                {
                    "r:diaObjectId": object_id,
                    "z_used": z_used,
                    "t0": None,
                    "t0_err": None,
                    "x0": None,
                    "x0_err": None,
                    "x1": None,
                    "x1_err": None,
                    "c": None,
                    "c_err": None,
                    "chi2": None,
                    "ndof": None,
                    "reduced_chi2": None,
                    "n_fit_data": len(data),
                    "n_used": 0,
                    "fit_success": False,
                    "fit_message": str(error),
                    "n_withheld_points": 0,
                    "n_excess_points": 0,
                    "n_excess_nights": 0,
                    "early_excess": False,
                }
            )

    results_df = pl.DataFrame(results)
    pull_results_df = pl.DataFrame(pull_results)

    if salt_fit_output_path is not None:
        results_df.write_csv(salt_fit_output_path)

    if pull_output_path is not None:
        pull_results_df.write_csv(pull_output_path)

    return results_df


if __name__ == "__main__":
    OBJECTS_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "with_z_results"
        / "colour_stretch_results"
        / "pass.csv"
    )

    LIGHT_CURVE_DATA_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "light_curve_data"
        / "combined_light_curve_data.parquet"
    )

    EEX_SALT_RESULTS_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "eex_results"
        / "salt_method"
        / "colour_stretch_results_test"
        / "salt3_fits.csv"
    )

    EEX_PULL_RESULTS_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "eex_results"
        / "salt_method"
        / "colour_stretch_results_test"
        / "pull_results.csv"
    )

    EEX_SALT_PLOTS_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "images"
        / "early_excess_plots"
        / "salt_plots"
        / "colour_stretch_results_test"
    )

    light_curve_data = loading_light_curve_data_and_objects(
        OBJECTS_PATH, LIGHT_CURVE_DATA_PATH
    )

    light_curve_data = rest_frame_observation_conversion(light_curve_data)

    light_curve_data = remove_sporadic_detections(light_curve_data)

    nightly_data = nightly_weighted_average(light_curve_data)

    first_detections = find_first_detection(nightly_data)

    light_curve_data = light_curve_data.join(
        first_detections.select(
            "r:diaObjectId",
            "t_fd",
        ),
        on="r:diaObjectId",
        how="inner",
    )
    results = eex_salt_fits_with_mask(
        light_curve_data,
        salt_fit_output_path=EEX_SALT_RESULTS_PATH,
        pull_output_path=EEX_PULL_RESULTS_PATH,
        plot_output_path=EEX_SALT_PLOTS_PATH,
    )
