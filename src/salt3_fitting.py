from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests as re
import sncosmo
from astropy.table import Table

FINK_OBJECTS_URL = "https://api.lsst.fink-portal.org/api/v1/sources"


def fetch_host_photoz_with_fallback(
    object_ids: list[str],
) -> list[dict]:
    """
    Fetch host photo-z information from Fink.

    If a request times out or returns a 502/504 response, recursively
    split the list of diaObjectIds in half and retry each half.

    Individual objects that still fail are skipped.
    """

    if not object_ids:
        return []

    try:
        response = re.post(
            FINK_OBJECTS_URL,
            json={
                "diaObjectId": ",".join(object_ids),
                "columns": (
                    "r:diaObjectId,f:xm_legacydr8_zphot,f:xm_legacydr8_e_zphot"
                ),
                "output-format": "json",
            },
            timeout=120,
        )

        # Large Fink requests may fail at the gateway.
        if response.status_code in (502, 504):
            raise re.exceptions.HTTPError(f"Fink returned {response.status_code}")

        # Other errors, e.g. 400, should not be hidden by the fallback.
        response.raise_for_status()

        return response.json()

    except (
        re.exceptions.Timeout,
        re.exceptions.ConnectionError,
        re.exceptions.HTTPError,
    ) as error:
        # Only recursively recover from transient/server-side failures.
        if (
            isinstance(error, re.exceptions.HTTPError)
            and "502" not in str(error)
            and "504" not in str(error)
        ):
            raise

        if len(object_ids) <= 1:
            print(f"Could not retrieve photo-z for {object_ids[0]}: {error}")
            return []

        mid = len(object_ids) // 2

        print(
            f"Request failed for {len(object_ids)} objects. "
            f"Splitting into {mid} and {len(object_ids) - mid}."
        )

        left = fetch_host_photoz_with_fallback(object_ids[:mid])

        right = fetch_host_photoz_with_fallback(object_ids[mid:])

        return left + right


def get_host_photoz_from_fink(
    objects_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    """
    Retrieve host-galaxy photometric redshift information from Fink
    for a set of Rubin diaObjectIds.

    Large requests are recursively split when Fink returns a
    502/504 error or times out.

    Parameters
    ----------
    objects_path : Path
        CSV containing ``r:diaObjectId``.

    output_path : Path
        Output CSV path.

    Returns
    -------
    pl.DataFrame
        Columns:
        - r:diaObjectId
        - host_photoz
        - host_photoz_err
    """

    objects = (
        pl.read_csv(
            objects_path,
            schema_overrides={
                "r:diaObjectId": pl.String,
            },
        )
        .select("r:diaObjectId")
        .unique(maintain_order=True)
    )

    object_ids = objects["r:diaObjectId"].to_list()

    records = fetch_host_photoz_with_fallback(object_ids)

    if len(records) == 0:
        results_df = objects.with_columns(
            pl.lit(
                None,
                dtype=pl.Float64,
            ).alias("host_photoz"),
            pl.lit(
                None,
                dtype=pl.Float64,
            ).alias("host_photoz_err"),
        )

    else:
        fink_df = (
            pl.DataFrame(records)
            .select(
                pl.col("r:diaObjectId").cast(pl.String),
                pl.col("f:xm_legacydr8_zphot")
                .cast(
                    pl.Float64,
                    strict=False,
                )
                .alias("host_photoz"),
                pl.col("f:xm_legacydr8_e_zphot")
                .cast(
                    pl.Float64,
                    strict=False,
                )
                .alias("host_photoz_err"),
            )
            .unique(
                subset="r:diaObjectId",
                keep="first",
            )
        )

        # Left join ensures objects with no returned photo-z
        # remain in the sample as nulls.
        results_df = objects.join(
            fink_df,
            on="r:diaObjectId",
            how="left",
            maintain_order="left",
        )

    results_df.write_csv(output_path)

    print(f"Retrieved records for {len(records)} / {len(object_ids)} objects.")

    return results_df


def single_object_salt3_fit_test(
    object_id: str, light_curve_data_path: Path, plot_output_path: Path
):
    """Test how the SALT 3 fit works against a single object to verify that it works how expected.

    Args:
        object_id (str): The ID of the object as a string.
        light_curve_data_path (Path): The path to the light curve data as a parquet file.
    """

    df = (
        pl.read_parquet(light_curve_data_path)
        .with_columns(pl.col("r:diaObjectId").cast(pl.String))
        .filter(
            (pl.col("r:diaObjectId") == object_id)
            & pl.col("r:psfFlux").is_finite()
            & pl.col("r:psfFluxErr").is_finite()
            & (pl.col("r:psfFluxErr") > 0)
        )
        .sort("r:midpointMjdTai")
    )

    peak_rows = df.sort("r:psfFlux", descending=True).unique(
        subset=["r:band"], keep="first"
    )

    peak_mjd = (
        peak_rows.filter(pl.col("r:band") == "i").select("r:midpointMjdTai").item()
    )

    band_map = {
        "u": "lsstu",
        "g": "lsstg",
        "r": "lsstr",
        "i": "lssti",
        "z": "lsstz",
        "y": "lssty",
    }
    df = df.with_columns(pl.col("r:band").replace(band_map).alias("sncosmo_band"))

    data = Table(
        {
            "time": df["r:midpointMjdTai"].to_numpy(),
            "band": df["sncosmo_band"].to_numpy(),
            "flux": df["r:psfFlux"].to_numpy(),
            "fluxerr": df["r:psfFluxErr"].to_numpy(),
            "zp": np.full(df.height, 31.4),
            "zpsys": np.full(df.height, "ab"),
        }
    )

    model = sncosmo.Model(source="salt3")

    zphot = 0.152
    zphot_err = 0.05

    z_min = max(0.001, zphot - 2 * zphot_err)
    z_max = zphot + 2 * zphot_err

    model.set(z=0.15)

    result, fitted_model = sncosmo.fit_lc(
        data,
        model,
        ["t0", "x0", "x1", "c"],
        bounds={
            "z": (z_min, z_max),
            "t0": (peak_mjd - 30, peak_mjd + 30),
            "x1": (-5, 5),
            "c": (-0.5, 0.5),
        },
        modelcov=True,
    )

    sncosmo.plot_lc(
        data,
        model=fitted_model,
        errors=result.errors,
    )

    plt.savefig(plot_output_path / f"{object_id}.png")
    plt.show()

    return result, fitted_model


def many_object_salt3_fits(
    objects_path: Path,
    light_curve_data_path: Path,
    output_path: Path,
    plot_output_path: Path | None = None,
) -> pl.DataFrame:
    """
    Fit SALT3 to multiple SN-like objects using host photometric redshifts.

    The host photometric redshift is used as the initial redshift estimate,
    while redshift is allowed to vary within +/- 2 host photo-z uncertainties.
    """

    objects_df = pl.read_csv(
        objects_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
            "host_photoz": pl.Float64,
            "host_photoz_err": pl.Float64,
        },
    ).filter(
        pl.col("host_photoz").is_not_null() & pl.col("host_photoz_err").is_not_null()
    )

    light_curve_data = (
        pl.read_parquet(light_curve_data_path)
        .with_columns(pl.col("r:diaObjectId").cast(pl.String))
        .filter(
            pl.col("r:diaObjectId").is_in(objects_df["r:diaObjectId"])
            & pl.col("r:psfFlux").is_finite()
            & pl.col("r:psfFluxErr").is_finite()
            & (pl.col("r:psfFluxErr") > 0)
        )
    )

    groups = {
        key[0]: group_df.sort("r:midpointMjdTai")
        for key, group_df in light_curve_data.group_by(
            "r:diaObjectId",
            maintain_order=True,
        )
    }

    band_map = {
        "u": "lsstu",
        "g": "lsstg",
        "r": "lsstr",
        "i": "lssti",
        "z": "lsstz",
        "y": "lssty",
    }

    results = []

    for row in objects_df.iter_rows(named=True):
        object_id = row["r:diaObjectId"]
        redshift = row["host_photoz"]
        redshift_err = row["host_photoz_err"]

        # ----------------------------------------------------------
        # Require host photo-z information
        # ----------------------------------------------------------

        group_df = groups.get(object_id)

        if group_df is None or group_df.height == 0:
            results.append(
                {
                    "r:diaObjectId": object_id,
                    "host_photoz": redshift,
                    "host_photoz_err": redshift_err,
                    "fitted_z": None,
                    "fitted_z_err": None,
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
                    "n_data": 0,
                    "n_used": 0,
                    "fit_success": False,
                    "fit_message": "No light-curve data",
                }
            )
            continue

        # ----------------------------------------------------------
        # Initialise t0 using observed i-band maximum
        # ----------------------------------------------------------

        i_band = group_df.filter(pl.col("r:band") == "i")

        if i_band.height == 0:
            results.append(
                {
                    "r:diaObjectId": object_id,
                    "host_photoz": redshift,
                    "host_photoz_err": redshift_err,
                    "fitted_z": None,
                    "fitted_z_err": None,
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
                    "n_data": group_df.height,
                    "n_used": 0,
                    "fit_success": False,
                    "fit_message": "No i-band data",
                }
            )
            continue

        peak_mjd = i_band.sort("r:psfFlux", descending=True)["r:midpointMjdTai"].item(0)

        # ----------------------------------------------------------
        # Convert Rubin bands to SNCosmo bands
        # ----------------------------------------------------------

        model = sncosmo.Model(source="salt3")

        fitted_bands = {
            band
            for band, sncosmo_band in band_map.items()
            if model.bandoverlap(sncosmo_band, z=redshift)
        }

        fit_df = group_df.filter(pl.col("r:band").is_in(fitted_bands)).with_columns(
            pl.col("r:band").replace(band_map).alias("sncosmo_band")
        )

        if fit_df.height == 0:
            results.append(
                {
                    "r:diaObjectId": object_id,
                    # Host-galaxy information
                    "host_photoz": redshift,
                    "host_photoz_err": redshift_err,
                    "fitted_z": None,
                    "fitted_z_err": None,
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
                    "n_data": group_df.height,
                    "n_used": 0,
                    "fit_success": False,
                    "fit_message": "No data after removing unsupported bands",
                }
            )
            continue
        data = Table(
            {
                "time": fit_df["r:midpointMjdTai"].to_numpy(),
                "band": fit_df["sncosmo_band"].to_numpy(),
                "flux": fit_df["r:psfFlux"].to_numpy(),
                "fluxerr": fit_df["r:psfFluxErr"].to_numpy(),
                "zp": np.full(fit_df.height, 31.4),
                "zpsys": np.full(fit_df.height, "ab"),
            }
        )

        # ----------------------------------------------------------
        # Photometric-redshift bounds
        # ----------------------------------------------------------
        if redshift and redshift_err != None:
            z_min = max(
                0.001,
                redshift - 2.0 * redshift_err,
            )

            z_max = redshift + 2.0 * redshift_err
        else:
            continue
            # z_min = 0.001
            # z_max = 2.00

        try:
            # Initial guess is the host photo-z.
            # model.set(z=redshift)

            result, fitted_model = sncosmo.fit_lc(
                data,
                model,
                ["z", "t0", "x0", "x1", "c"],
                bounds={
                    "z": (z_min, z_max),
                    "t0": (
                        peak_mjd - 30,
                        peak_mjd + 30,
                    ),
                    "x1": (-5, 5),
                    "c": (-0.5, 0.5),
                },
                modelcov=True,
            )

            reduced_chi2 = result.chisq / result.ndof if result.ndof > 0 else None

            n_used = int(np.sum(result.data_mask))

            results.append(
                {
                    "r:diaObjectId": object_id,
                    # Host-galaxy information
                    "host_photoz": redshift,
                    "host_photoz_err": redshift_err,
                    # SALT fitted redshift
                    "fitted_z": fitted_model["z"],
                    "fitted_z_err": result.errors.get("z"),
                    # SALT parameters
                    "t0": fitted_model["t0"],
                    "t0_err": result.errors.get("t0"),
                    "x0": fitted_model["x0"],
                    "x0_err": result.errors.get("x0"),
                    "x1": fitted_model["x1"],
                    "x1_err": result.errors.get("x1"),
                    "c": fitted_model["c"],
                    "c_err": result.errors.get("c"),
                    # Fit statistics
                    "chi2": result.chisq,
                    "ndof": result.ndof,
                    "reduced_chi2": reduced_chi2,
                    "n_total_data": group_df.height,
                    "n_fit_data": len(data),
                    "n_used": n_used,
                    "fit_success": result.success,
                    "fitted_bands": ",".join(sorted(fitted_bands)),
                    "u_fitted": "u" in fitted_bands,
                    "g_fitted": "g" in fitted_bands,
                    "r_fitted": "r" in fitted_bands,
                    "i_fitted": "i" in fitted_bands,
                    "z_fitted": "z" in fitted_bands,
                    "y_fitted": "y" in fitted_bands,
                    "fit_message": str(result.message),
                }
            )

            if plot_output_path is not None:
                sncosmo.plot_lc(
                    data,
                    model=fitted_model,
                    errors=result.errors,
                )

                plt.savefig(
                    plot_output_path / f"{object_id}_SALT3.png",
                    bbox_inches="tight",
                )

                plt.close()

        except Exception as error:
            results.append(
                {
                    "r:diaObjectId": object_id,
                    "host_photoz": redshift,
                    "host_photoz_err": redshift_err,
                    "fitted_z": None,
                    "fitted_z_err": None,
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
                    "n_data": len(data),
                    "n_used": 0,
                    "fit_success": False,
                    "fit_message": str(error),
                }
            )

    results_df = pl.DataFrame(results)


    results_df = results_df.with_columns(
        (pl.col("fitted_z") - pl.col("host_photoz")).alias("delta_z"),
        (
            (pl.col("fitted_z") - pl.col("host_photoz"))
            / (1 + pl.col("host_photoz"))
        ).alias("delta_z_normalised"),
    )

    results_df.write_csv(output_path)

    return results_df


if __name__ == "__main__":
    TEST_OBJECT = "170032917163016282"

    TEST_OBJECT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "test_data"
        / "salt_fit_tests"
        / "50_object_test.csv"
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

    PLOT_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "images"
        / "salt_plots"
    )

    TEST_PLOT_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "images"
        / "salt_plots"
        / "50_object_test"
    )

    CONFIRMED_IA_CSV = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "confirmed_ia_tests"
        / "confirmed_ia.csv"
    )

    TEST_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "50_object_test"
        / "fit_values"
        / "50_object_test_values.csv"
    )

    OBJECT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "objects_with_host_z.csv"
    )

    WITH_PHOTO_Z_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "with_hostz"
        / "salt_fit_values.csv"
    )

    WITH_PHOTO_Z_PLOTS_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "images"
        / "salt_plots"
        / "with_photo_z_plots"
    )

    # result, fitted_model = single_object_salt3_fit_test(
    #     TEST_OBJECT, LIGHT_CURVE_DATA_PATH, PLOT_OUTPUT_PATH
    # )

    # get_host_photoz_from_fink(TEST_OBJECT_PATH, TEST_OBJECT_PATH)

    results = many_object_salt3_fits(
        OBJECT_PATH,
        LIGHT_CURVE_DATA_PATH,
        WITH_PHOTO_Z_PATH,
        WITH_PHOTO_Z_PLOTS_PATH,
    )

    # pl.Config.set_tbl_cols(-1)
    # print(confirmed_ia_results.describe([0.25, 0.5, 0.75]))
