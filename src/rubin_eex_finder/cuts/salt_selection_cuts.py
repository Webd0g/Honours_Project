from pathlib import Path

import numpy as np
import polars as pl
import sncosmo
from astropy.table import Table

SALT3_FIT_MIN_WAVE = 2800.0  # Angstrom
SALT3_FIT_MAX_WAVE = 8700.0  # Angstrom

BAND_MAP = {
    "u": "lsstu",
    "g": "lsstg",
    "r": "lsstr",
    "i": "lssti",
    "z": "lsstz",
    "y": "lssty",
}


def bazin_salt_phase_cut(
    salt_fits_path: Path,
    light_curve_data_path: Path,
    pass_output_path: Path | None = None,
    fail_output_path: Path | None = None,
    values_output_path: Path | None = None,
) -> pl.DataFrame:

    salt_fits = pl.read_csv(
        salt_fits_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
            "z_used": pl.Float64,
            "t0": pl.Float64,
            "fit_success": pl.Boolean,
        },
    )

    light_curve_data = (
        pl.read_parquet(light_curve_data_path)
        .with_columns(pl.col("r:diaObjectId").cast(pl.String))
        .filter(
            pl.col("r:diaObjectId").is_in(salt_fits["r:diaObjectId"])
            & pl.col("r:midpointMjdTai").is_finite()
            & pl.col("r:psfFlux").is_finite()
            & pl.col("r:psfFluxErr").is_finite()
            & (pl.col("r:psfFluxErr") > 0)
        )
    )

    phase_data = light_curve_data.join(
        salt_fits.select(
            "r:diaObjectId",
            "z_used",
            "t0",
            "fit_success",
        ),
        on="r:diaObjectId",
        how="left",
    )

    phase_data = phase_data.with_columns(
        ((pl.col("r:midpointMjdTai") - pl.col("t0")) / (1.0 + pl.col("z_used"))).alias(
            "tau"
        )
    )

    phase_data = phase_data.with_columns(
        ((pl.col("tau") > -10) & (pl.col("tau") < 5)).alias("in_peak_phase"),
        ((pl.col("tau") > 5) & (pl.col("tau") < 20)).alias("in_shape_phase"),
        ((pl.col("tau") > -10) & (pl.col("tau") < 35)).alias("in_colour_phase"),
    )

    quality_counts = phase_data.group_by("r:diaObjectId").agg(
        pl.col("in_peak_phase").sum().alias("n_peak_phase"),
        pl.col("in_shape_phase").sum().alias("n_shape_phase"),
        ((pl.col("in_colour_phase")) & (pl.col("r:band") == "g"))
        .sum()
        .alias("n_g_colour"),
        ((pl.col("in_colour_phase")) & (pl.col("r:band") == "r"))
        .sum()
        .alias("n_r_colour"),
        ((pl.col("in_colour_phase")) & (pl.col("r:band") == "i"))
        .sum()
        .alias("n_i_colour"),
        ((pl.col("in_colour_phase")) & (pl.col("r:band") == "z"))
        .sum()
        .alias("n_z_colour"),
    )

    results = salt_fits.join(
        quality_counts,
        on="r:diaObjectId",
        how="left",
        maintain_order="left",
    ).with_columns(
        pl.col(
            "n_peak_phase",
            "n_shape_phase",
            "n_g_colour",
            "n_r_colour",
            "n_i_colour",
            "n_z_colour",
        ).fill_null(0)
    )

    results = results.with_columns(
        (
            pl.col("fit_success").fill_null(False)
            & pl.col("t0").is_not_null()
            & pl.col("t0").is_finite()
            & pl.col("z_used").is_not_null()
            & pl.col("z_used").is_finite()
            & (pl.col("z_used") > 0)
        ).alias("has_quality_inputs"),
        (pl.col("n_peak_phase") >= 1).alias("pass_peak_phase"),
        (pl.col("n_shape_phase") >= 1).alias("pass_shape_phase"),
        ((pl.col("n_g_colour") >= 1) & (pl.col("n_i_colour") >= 1)).alias(
            "pass_gi_colour"
        ),
        ((pl.col("n_r_colour") >= 1) & (pl.col("n_z_colour") >= 1)).alias(
            "pass_rz_colour"
        ),
        ((pl.col("n_i_colour") >= 1) & (pl.col("n_z_colour") >= 1)).alias(
            "pass_iz_colour"
        ),
    )

    results = results.with_columns(
        (
            pl.col("pass_gi_colour")
            | pl.col("pass_rz_colour")
            | pl.col("pass_iz_colour")
        ).alias("pass_colour")
    ).with_columns(
        (
            pl.col("has_quality_inputs")
            & pl.col("pass_peak_phase")
            & pl.col("pass_shape_phase")
            & pl.col("pass_colour")
        ).alias("pass_salt_quality")
    )

    results = results.with_columns(
        (
            (~pl.col("pass_peak_phase")).cast(pl.Int8)
            + (~pl.col("pass_shape_phase")).cast(pl.Int8)
            + (~pl.col("pass_colour")).cast(pl.Int8)
        ).alias("number_quality_failed")
    )

    if values_output_path is not None:
        values_output_path.parent.mkdir(parents=True, exist_ok=True)
        results.write_csv(values_output_path)

    if pass_output_path is not None:
        pass_output_path.parent.mkdir(parents=True, exist_ok=True)
        results.filter(pl.col("pass_salt_quality")).write_csv(pass_output_path)

    if fail_output_path is not None:
        fail_output_path.parent.mkdir(parents=True, exist_ok=True)
        results.filter(~pl.col("pass_salt_quality")).write_csv(fail_output_path)

    return results


def get_salt3_fitted_bands(
    redshift: float,
    model: sncosmo.Model,
    band_map: dict[str, str] = BAND_MAP,
) -> set[str]:

    fitted_bands = set()

    for rubin_band, sncosmo_band in band_map.items():
        bandpass = sncosmo.get_bandpass(sncosmo_band)

        rest_wave_eff = bandpass.wave_eff / (1.0 + redshift)

        in_trusted_range = SALT3_FIT_MIN_WAVE <= rest_wave_eff <= SALT3_FIT_MAX_WAVE

        model_can_evaluate = bool(
            model.bandoverlap(
                sncosmo_band,
                z=redshift,
            )
        )

        if in_trusted_range and model_can_evaluate:
            fitted_bands.add(rubin_band)

    return fitted_bands


def unfitted_band_chi2(
    group_df: pl.DataFrame,
    fitted_model: sncosmo.Model,
    band: str,
    band_map: dict[str, str] = BAND_MAP,
) -> dict:

    sncosmo_band = band_map[band]

    # Can the final fitted SALT3 model physically evaluate this band?
    if not fitted_model.bandoverlap(sncosmo_band):
        return {
            "chi2": None,
            "n_points": 0,
            "chi2_per_point": None,
            "model_can_evaluate": False,
        }

    band_df = group_df.filter(pl.col("r:band") == band)

    if band_df.height == 0:
        return {
            "chi2": None,
            "n_points": 0,
            "chi2_per_point": None,
            "model_can_evaluate": True,
        }

    data = Table(
        {
            "time": band_df["r:midpointMjdTai"].to_numpy(),
            "band": np.full(
                band_df.height,
                sncosmo_band,
            ),
            "flux": band_df["r:psfFlux"].to_numpy(),
            "fluxerr": band_df["r:psfFluxErr"].to_numpy(),
            "zp": np.full(
                band_df.height,
                31.4,
            ),
            "zpsys": np.full(
                band_df.height,
                "ab",
            ),
        }
    )

    chi2 = float(
        sncosmo.chisq(
            data,
            fitted_model,
            modelcov=True,
        )
    )

    n_points = len(data)

    return {
        "chi2": chi2,
        "n_points": n_points,
        "chi2_per_point": chi2 / n_points,
        "model_can_evaluate": True,
    }


def bazin_salt_chi2_cut(
    phase_cuts_path: Path,
    light_curve_data_path: Path,
    values_output_path: Path,
    pass_output_path: Path,
    fail_output_path: Path,
) -> pl.DataFrame:

    objects_df = pl.read_csv(
        phase_cuts_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
        },
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

    results = []

    for row in objects_df.iter_rows(named=True):
        object_id = row["r:diaObjectId"]

        group_df = groups.get(object_id)

        if group_df is None or group_df.height == 0:
            continue

        model = sncosmo.Model(source="salt3")

        model.set(
            z=row["fitted_z"],
            t0=row["t0"],
            x0=row["x0"],
            x1=row["x1"],
            c=row["c"],
        )

        band_results = {}

        fitted_band_flags = {
            "g": row["g_fitted"],
            "r": row["r_fitted"],
            "i": row["i_fitted"],
            "z": row["z_fitted"],
        }

        for rubin_band, was_fitted in fitted_band_flags.items():
            sncosmo_band = BAND_MAP[rubin_band]

            if not was_fitted:
                band_results[rubin_band] = {
                    "chi2": None,
                    "n": 0,
                    "chi2_per_point": None,
                }
                continue

            band_df = group_df.filter(pl.col("r:band") == rubin_band)

            if band_df.height == 0:
                band_results[rubin_band] = {
                    "chi2": None,
                    "n": 0,
                    "chi2_per_point": None,
                }
                continue

            band_data = Table(
                {
                    "time": band_df["r:midpointMjdTai"].to_numpy(),
                    "band": np.full(
                        band_df.height,
                        sncosmo_band,
                    ),
                    "flux": band_df["r:psfFlux"].to_numpy(),
                    "fluxerr": band_df["r:psfFluxErr"].to_numpy(),
                    "zp": np.full(
                        band_df.height,
                        31.4,
                    ),
                    "zpsys": np.full(
                        band_df.height,
                        "ab",
                    ),
                }
            )

            # It is possible for the final fitted z to move far enough
            # that SALT3 can no longer physically evaluate a band which
            # was valid at host_photoz.
            if not model.bandoverlap(sncosmo_band):
                band_results[rubin_band] = {
                    "chi2": None,
                    "n": 0,
                    "chi2_per_point": None,
                }
                continue

            chi2 = float(
                sncosmo.chisq(
                    band_data,
                    model,
                    modelcov=True,
                )
            )

            n_band = len(band_data)

            band_results[rubin_band] = {
                "chi2": chi2,
                "n": n_band,
                "chi2_per_point": chi2 / n_band,
            }

        g_band = band_results["g"]
        r_band = band_results["r"]
        i_band = band_results["i"]
        z_band = band_results["z"]

        # --------------------------------------------------------
        # Calculate unfitted band chi squared
        # --------------------------------------------------------

        g_unfitted = (
            unfitted_band_chi2(group_df, model, "g")
            if not row["g_fitted"]
            else {
                "chi2": None,
                "n_points": 0,
                "chi2_per_point": None,
                "model_can_evaluate": None,
            }
        )

        z_unfitted = (
            unfitted_band_chi2(group_df, model, "z")
            if not row["z_fitted"]
            else {
                "chi2": None,
                "n_points": 0,
                "chi2_per_point": None,
                "model_can_evaluate": None,
            }
        )

        results.append(
            {
                "r:diaObjectId": object_id,
                "g_chi2": g_band["chi2"],
                "g_n_used": g_band["n"],
                "g_chi2_per_point": g_band["chi2_per_point"],
                "r_chi2": r_band["chi2"],
                "r_n_used": r_band["n"],
                "r_chi2_per_point": r_band["chi2_per_point"],
                "i_chi2": i_band["chi2"],
                "i_n_used": i_band["n"],
                "i_chi2_per_point": i_band["chi2_per_point"],
                "z_chi2": z_band["chi2"],
                "z_n_used": z_band["n"],
                "z_chi2_per_point": z_band["chi2_per_point"],
                "g_unfitted_chi2": g_unfitted["chi2"],
                "g_unfitted_n": g_unfitted["n_points"],
                "g_unfitted_chi2_per_point": g_unfitted["chi2_per_point"],
                "g_unfitted_model_can_evaluate": g_unfitted["model_can_evaluate"],
                "z_unfitted_chi2": z_unfitted["chi2"],
                "z_unfitted_n": z_unfitted["n_points"],
                "z_unfitted_chi2_per_point": z_unfitted["chi2_per_point"],
                "z_unfitted_model_can_evaluate": z_unfitted["model_can_evaluate"],
            }
        )

    chi2_df = pl.DataFrame(results)

    results_df = objects_df.join(
        chi2_df,
        on="r:diaObjectId",
        how="left",
        maintain_order="left",
    )

    results_df = results_df.with_columns(
        (pl.col("g_chi2_per_point") < 10).alias("pass_g_fitted_chi2"),
        (pl.col("r_chi2_per_point") < 8).alias("pass_r_fitted_chi2"),
        (pl.col("i_chi2_per_point") < 8).alias("pass_i_fitted_chi2"),
        (pl.col("z_chi2_per_point") < 8).alias("pass_z_fitted_chi2"),
        (pl.col("g_unfitted_chi2_per_point") < 6).alias("pass_g_unfitted_chi2"),
        (pl.col("z_unfitted_chi2_per_point") < 3).alias("pass_z_unfitted_chi2"),
        (pl.col("reduced_chi2") < 6).alias("pass_overall_chi2"),
    )

    results_df = results_df.with_columns(
        (
            pl.col("pass_g_fitted_chi2").fill_null(True)
            & pl.col("pass_r_fitted_chi2").fill_null(True)
            & pl.col("pass_i_fitted_chi2").fill_null(True)
            & pl.col("pass_z_fitted_chi2").fill_null(True)
            & pl.col("pass_g_unfitted_chi2").fill_null(True)
            & pl.col("pass_z_unfitted_chi2").fill_null(True)
            & pl.col("pass_overall_chi2").fill_null(False)
        ).alias("pass_chi2_quality")
    )

    values_output_path.parent.mkdir(parents=True, exist_ok=True)
    pass_output_path.parent.mkdir(parents=True, exist_ok=True)
    fail_output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.write_csv(values_output_path)
    results_df.filter(pl.col("pass_chi2_quality")).write_csv(pass_output_path)
    results_df.filter(~pl.col("pass_chi2_quality")).write_csv(fail_output_path)

    return results_df


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    SN_IA_SALT_PHASE_PATH = (
        DATA_DIR / "sn_ia_test_results" / "phase_results" / "pass.csv"
    )

    SN_IA_SALT_PASS_OUTPUT_PATH = (
        DATA_DIR / "sn_ia_test_results" / "salt_chi2_results" / "pass.csv"
    )

    SN_IA_SALT_FAIL_OUTPUT_PATH = (
        DATA_DIR / "sn_ia_test_results" / "salt_chi2_results" / "fail.csv"
    )

    SN_IA_SALT_VALUES_OUTPUT_PATH = (
        DATA_DIR / "sn_ia_test_results" / "salt_chi2_results" / "values.csv"
    )

    LIGHT_CURVE_DATA_PATH = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    SN_IA_SALT_PHASE_PLOT_OUTPUT_PATH = (
        IMAGE_DIR / "sn_ia_light_curves" / "salt_plots" / "phase_cut"
    )

if SN_IA_SALT_PHASE_PATH.exists():
        bazin_salt_chi2_cut(
            phase_cuts_path=SN_IA_SALT_PHASE_PATH,
            light_curve_data_path=LIGHT_CURVE_DATA_PATH,
            values_output_path=SN_IA_SALT_VALUES_OUTPUT_PATH,
            pass_output_path=SN_IA_SALT_PASS_OUTPUT_PATH,
            fail_output_path=SN_IA_SALT_FAIL_OUTPUT_PATH,
        )