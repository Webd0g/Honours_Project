from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests as re
import sncosmo
from astropy.table import Table

from rubin_eex_finder.cuts.salt_selection_cuts import get_salt3_fitted_bands

FINK_OBJECTS_URL = "https://api.lsst.fink-portal.org/api/v1/sources"


def fetch_object_z_with_fallback(
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
                    "r:diaObjectId,f:xm_legacydr8_zphot,f:xm_legacydr8_e_zphot,f:xm_tns_redshift"
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
            print(f"Could not retrieve z for {object_ids[0]}: {error}")
            return []

        mid = len(object_ids) // 2

        print(
            f"Request failed for {len(object_ids)} objects. "
            f"Splitting into {mid} and {len(object_ids) - mid}."
        )

        left = fetch_object_z_with_fallback(object_ids[:mid])

        right = fetch_object_z_with_fallback(object_ids[mid:])

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

    records = fetch_object_z_with_fallback(object_ids)

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
            pl.lit(
                None,
                dtype=pl.Float64,
            ).alias("redshift"),
        )

    else:
        raw_df = pl.DataFrame(
            records,
            infer_schema_length=None,
        ).with_columns(
            pl.col("r:diaObjectId").cast(pl.String),
        )
        host_df = (
            raw_df.select(
                pl.col("r:diaObjectId"),
                pl.col("f:xm_legacydr8_zphot")
                .cast(pl.Float64, strict=False)
                .alias("host_photoz"),
                pl.col("f:xm_legacydr8_e_zphot")
                .cast(pl.Float64, strict=False)
                .alias("host_photoz_err"),
            )
            .group_by("r:diaObjectId")
            .agg(
                pl.col("host_photoz").drop_nulls().first().alias("host_photoz"),
                pl.col("host_photoz_err").drop_nulls().first().alias("host_photoz_err"),
            )
        )
        tns_df = (
            raw_df.select(
                pl.col("r:diaObjectId"),
                pl.col("f:xm_tns_redshift")
                .cast(pl.Float64, strict=False)
                .alias("redshift"),
            )
            .group_by("r:diaObjectId")
            .agg(pl.col("redshift").drop_nulls().first().alias("redshift"))
        )

        results_df = objects.join(
            host_df,
            on="r:diaObjectId",
            how="left",
            maintain_order="left",
        ).join(
            tns_df,
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
            "z_used": pl.Float64,
            "host_photoz": pl.Float64,
            "host_photoz_err": pl.Float64,
            "spec_z": pl.Float64,
            "t0": pl.Float64,
        },
    ).filter(
        pl.col("fit_success")
        & pl.col("z_used").is_not_null()
        & pl.col("z_used").is_finite()
        & (pl.col("z_used") > 0)
        & pl.col("t0").is_not_null()
        & pl.col("t0").is_finite()
    )
    light_curve_data = (
        pl.read_parquet(light_curve_data_path)
        .with_columns(pl.col("r:diaObjectId").cast(pl.String))
        .filter(
            pl.col("r:diaObjectId").is_in(objects_df["r:diaObjectId"])
            & pl.col("r:midpointMjdTai").is_finite()
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

    if plot_output_path is not None:
        plot_output_path.mkdir(
            parents=True,
            exist_ok=True,
        )
    results = []

    for row in objects_df.iter_rows(named=True):
        object_id = row["r:diaObjectId"]
        redshift = row["z_used"]
        z_source = row["z_source"]
        old_t0 = row["t0"]

        # ----------------------------------------------------------
        # Require host photo-z information
        # ----------------------------------------------------------

        group_df = groups.get(object_id)

        if group_df is None or group_df.height == 0:
            results.append(
                {
                    "r:diaObjectId": object_id,
                    "z_used": redshift,
                    "z_source": z_source,
                    "host_photoz": row["host_photoz"],
                    "host_photoz_err": row["host_photoz_err"],
                    "spec_z": row["spec_z"],
                    "old_t0": old_t0,
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
                    "n_total_data": 0,
                    "n_phase_data": 0,
                    "n_fit_data": 0,
                    "n_used": 0,
                    "fit_success": False,
                    "fit_message": "No light-curve data",
                }
            )
            continue

        n_total_data = group_df.height

        # ----------------------------------------------------------
        # Initialise t0 using observed i-band maximum
        # ----------------------------------------------------------

        group_df = group_df.with_columns(
            ((pl.col("r:midpointMjdTai") - old_t0) / (1.0 + redshift)).alias(
                "rest_frame_phase"
            )
        )

        group_df = group_df.filter(
            (pl.col("rest_frame_phase") >= -10) & (pl.col("rest_frame_phase") <= 40)
        )

        n_phase_data = group_df.height

        if n_phase_data == 0:
            results.append(
                {
                    "r:diaObjectId": object_id,
                    "z_used": redshift,
                    "z_source": z_source,
                    "host_photoz": row["host_photoz"],
                    "host_photoz_err": row["host_photoz_err"],
                    "spec_z": row["spec_z"],
                    "old_t0": old_t0,
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
                    "n_total_data": 0,
                    "n_phase_data": 0,
                    "n_fit_data": 0,
                    "n_used": 0,
                    "fit_success": False,
                    "fit_message": "No light-curve data",
                }
            )
            continue

        # ----------------------------------------------------------
        # Convert Rubin bands to SNCosmo bands
        # ----------------------------------------------------------

        model = sncosmo.Model(source="salt3")

        fitted_bands = get_salt3_fitted_bands(
            redshift=redshift,
            model=model,
        )

        fit_df = group_df.filter(pl.col("r:band").is_in(fitted_bands)).with_columns(
            pl.col("r:band").replace(band_map).alias("sncosmo_band")
        )

        if fit_df.height == 0:
            results.append(
                {
                    "r:diaObjectId": object_id,
                    "z_used": redshift,
                    "z_source": z_source,
                    "host_photoz": row["host_photoz"],
                    "host_photoz_err": row["host_photoz_err"],
                    "spec_z": row["spec_z"],
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
                    "n_total_data": n_total_data,
                    "n_phase_data": n_phase_data,
                    "n_fit_data": 0,
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
        # if redshift and redshift_err != None:
        #     z_min = max(
        #         0.001,
        #         redshift - 2.0 * redshift_err,
        #     )

        #     z_max = redshift + 2.0 * redshift_err
        # else:
        #     continue
        #     # z_min = 0.001
        #     # z_max = 2.00
        try:
            # Initial guess is the host photo-z.
            model.set(
                z=redshift,
                t0=old_t0,
            )

            result, fitted_model = sncosmo.fit_lc(
                data,
                model,
                ["t0", "x0", "x1", "c"],
                bounds={
                    "t0": (
                        old_t0 - 30,
                        old_t0 + 30,
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
                    "z_used": redshift,
                    "z_source": z_source,
                    "host_photoz": row["host_photoz"],
                    "host_photoz_err": row["host_photoz_err"],
                    "spec_z": row["spec_z"],
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
                    "n_total_data": n_total_data,
                    "n_phase_data": n_phase_data,
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
                    "z_used": redshift,
                    "z_source": z_source,
                    "host_photoz": row["host_photoz"],
                    "host_photoz_err": row["host_photoz_err"],
                    "spec_z": row["spec_z"],
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
                    "n_total_data": n_total_data,
                    "n_phase_data": n_phase_data,
                    "n_fit_data": 0,
                    "n_used": 0,
                    "fit_success": False,
                    "fit_message": str(error),
                }
            )

    results_df = pl.DataFrame(results)

    results_df.write_csv(output_path)

    return results_df


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data" / "forced_photometry_test"

    IMAGE_DIR = PROJECT_DIR / "images"

    LIGHT_CURVE_DATA_DIR = (
        DATA_DIR / "light_curve_data" / "forced_photometry_light_curve_data.parquet"
    )

    I_BAND_BAZIN_FITS_RESULTS = (
        DATA_DIR / "bazin_fit_results" / "bazin_fits_i_band_results.csv"
    )

    R_BAND_BAZIN_FITS_RESULTS = (
        DATA_DIR / "bazin_fit_results" / "bazin_fits_r_band_results.csv"
    )

    OBJECTS = DATA_DIR / "fitted_peak_timing_results" / "pass.csv"

    PASS = DATA_DIR / "fitted_peak_timing_results" / "pass.csv"

    FAIL = DATA_DIR / "fitted_peak_timing_results" / "fail.csv"

    VALUES = DATA_DIR / "fitted_peak_timing_results" / "values.csv"

    results = many_object_salt3_fits(
        objects_path=OBJECTS_PATH,
        light_curve_data_path=LIGHT_CURVE_DATA_PATH,
        output_path=EEX_SALT_RESULTS_PATH,
        plot_output_path=EEX_SALT_PLOTS_PATH,
    )
