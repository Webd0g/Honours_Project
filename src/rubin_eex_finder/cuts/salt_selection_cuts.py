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
    """
    Apply the Bazin et al. SALT quality cuts.

    The rest-frame phase is defined as

        tau = (t_obs - t0) / (1 + z_host)

    where:
        t0     = fitted SALT time of maximum
        z_host = host-galaxy photometric redshift

    Quality requirements
    --------------------
    1. At least one measurement with -10 < tau < +5 days.
    2. At least one measurement with  +5 < tau < +20 days.
    3. At least one colour pair among:
            (g, i), (r, z), (i, z)
        where each band has at least one measurement with
            -10 < tau < +35 days.

    Parameters
    ----------
    salt_fits_path : Path
        CSV containing the SALT fit results. Must contain:
        - r:diaObjectId
        - host_photoz
        - t0
        - fit_success

    light_curve_data_path : Path
        Rubin light-curve parquet file.

    output_path : Path
        Path to save the quality-cut results.

    Returns
    -------
    pl.DataFrame
        SALT fit results with phase-coverage counts and quality flags.
    """

    # --------------------------------------------------------------
    # Load SALT fit results
    # --------------------------------------------------------------

    salt_fits = pl.read_csv(
        salt_fits_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
            "z_used": pl.Float64,
            "t0": pl.Float64,
            "fit_success": pl.Boolean,
        },
    )

    # --------------------------------------------------------------
    # Load only photometry belonging to these objects
    #
    # Use the same basic measurement validity requirements as the
    # SALT fitting stage. Negative fluxes are retained.
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Attach SALT t0 and host photo-z to every photometric point
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Calculate rest-frame phase
    #
    # IMPORTANT:
    #   host_photoz is used here, NOT fitted_z.
    # --------------------------------------------------------------

    phase_data = phase_data.with_columns(
        ((pl.col("r:midpointMjdTai") - pl.col("t0")) / (1.0 + pl.col("z_used"))).alias(
            "tau"
        )
    )

    # --------------------------------------------------------------
    # Define phase-window membership for every observation
    # --------------------------------------------------------------

    phase_data = phase_data.with_columns(
        ((pl.col("tau") > -10) & (pl.col("tau") < 5)).alias("in_peak_phase"),
        ((pl.col("tau") > 5) & (pl.col("tau") < 20)).alias("in_shape_phase"),
        ((pl.col("tau") > -10) & (pl.col("tau") < 35)).alias("in_colour_phase"),
    )

    # --------------------------------------------------------------
    # Count measurements in each required region
    # --------------------------------------------------------------

    quality_counts = phase_data.group_by("r:diaObjectId").agg(
        # Requirement 1
        pl.col("in_peak_phase").sum().alias("n_peak_phase"),
        # Requirement 2
        pl.col("in_shape_phase").sum().alias("n_shape_phase"),
        # Requirement 3:
        # number of observations in each band in the
        # -10 < tau < +35 colour window
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

    # --------------------------------------------------------------
    # Join counts back to SALT results
    #
    # Left join means objects without usable photometry remain in
    # the output rather than disappearing.
    # --------------------------------------------------------------

    results = salt_fits.join(
        quality_counts,
        on="r:diaObjectId",
        how="left",
        maintain_order="left",
    )

    # Objects without valid phase information get zero counts.
    results = results.with_columns(
        pl.col(
            "n_peak_phase",
            "n_shape_phase",
            "n_g_colour",
            "n_r_colour",
            "n_i_colour",
            "n_z_colour",
        ).fill_null(0)
    )

    # --------------------------------------------------------------
    # Individual Bazin quality flags
    # --------------------------------------------------------------

    results = results.with_columns(
        # Do we actually have everything required to calculate tau?
        (
            pl.col("fit_success").fill_null(False)
            & pl.col("t0").is_not_null()
            & pl.col("t0").is_finite()
            & pl.col("z_used").is_not_null()
            & pl.col("z_used").is_finite()
            & (pl.col("z_used") > 0)
        ).alias("has_quality_inputs"),
        # Criterion (i)
        (pl.col("n_peak_phase") >= 1).alias("pass_peak_phase"),
        # Criterion (ii)
        (pl.col("n_shape_phase") >= 1).alias("pass_shape_phase"),
        # Criterion (iii): individual colour combinations
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

    # --------------------------------------------------------------
    # Overall colour and SALT quality flags
    # --------------------------------------------------------------

    results = results.with_columns(
        (
            pl.col("pass_gi_colour")
            | pl.col("pass_rz_colour")
            | pl.col("pass_iz_colour")
        ).alias("pass_colour")
    )

    results = results.with_columns(
        (
            pl.col("has_quality_inputs")
            & pl.col("pass_peak_phase")
            & pl.col("pass_shape_phase")
            & pl.col("pass_colour")
        ).alias("pass_salt_quality")
    )

    # --------------------------------------------------------------
    # Number of Bazin quality requirements failed
    # --------------------------------------------------------------

    results = results.with_columns(
        (
            (~pl.col("pass_peak_phase")).cast(pl.Int8)
            + (~pl.col("pass_shape_phase")).cast(pl.Int8)
            + (~pl.col("pass_colour")).cast(pl.Int8)
        ).alias("number_quality_failed")
    )

    results.write_csv(values_output_path)
    results.filter(pl.col("pass_salt_quality")).write_csv(pass_output_path)
    results.filter(~pl.col("pass_salt_quality")).write_csv(fail_output_path)

    return results


def get_salt3_fitted_bands(
    redshift: float,
    model: sncosmo.Model,
    band_map: dict[str, str] = BAND_MAP,
) -> set[str]:
    """
    Determine which Rubin bands should enter the SALT3 fit.

    A band is included when:

    1. Its rest-frame effective wavelength is within the adopted
        SALT3 fitting range of 2800--8700 Angstrom.

    2. The full observed bandpass is physically covered by the
        SALT3 model at the supplied redshift.

    The supplied redshift should be the host-galaxy photo-z for
    the host-photo-z branch of the analysis.
    """

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


def bazin_salt_chi2_cut(
    phase_cuts_path: Path,
    light_curve_data_path: Path,
    values_output_path: Path,
    pass_output_path: Path,
    fail_output_path: Path,
) -> pl.DataFrame:
    """
    Calculate SALT chi-square statistics per fitted band for objects
    that passed the Bazin SALT phase-quality cut.

    Uses the best-fit SALT3 parameters already stored in the input CSV.

    Bazin fitted-band requirements:
        g : chi2 / N < 10
        r : chi2 / N < 8
        i : chi2 / N < 8
        z : chi2 / N < 8

    Overall SALT requirement:
        chi2 / ndof < 6

    Notes
    -----
    SNCosmo provides a global ndof, but not a separate ndof for each band.
    Therefore the per-band statistic is stored explicitly as chi2 / N_band
    rather than being labelled a formal reduced chi-square.
    """

    # ------------------------------------------------------------
    # Read SALT / phase-quality results
    # ------------------------------------------------------------

    objects_df = pl.read_csv(
        phase_cuts_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
        },
    )

    # ------------------------------------------------------------
    # Read only photometry for objects entering this stage
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Process each phase-quality-passing object
    # ------------------------------------------------------------

    for row in objects_df.iter_rows(named=True):
        object_id = row["r:diaObjectId"]

        group_df = groups.get(object_id)

        if group_df is None or group_df.height == 0:
            continue

        # --------------------------------------------------------
        # Reconstruct the fitted SALT model
        # --------------------------------------------------------

        model = sncosmo.Model(source="salt3")

        model.set(
            z=row["fitted_z"],
            t0=row["t0"],
            x0=row["x0"],
            x1=row["x1"],
            c=row["c"],
        )

        # --------------------------------------------------------
        # Calculate chi2 in bands that actually entered the fit
        # --------------------------------------------------------

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

        g_unfitted = {
            "chi2": None,
            "n_points": 0,
            "chi2_per_point": None,
            "model_can_evaluate": None,
        }

        z_unfitted = {
            "chi2": None,
            "n_points": 0,
            "chi2_per_point": None,
            "model_can_evaluate": None,
        }

        if row["g_fitted"] is False:
            g_unfitted = unfitted_band_chi2(
                group_df=group_df,
                fitted_model=model,
                band="g",
            )

        if row["z_fitted"] is False:
            z_unfitted = unfitted_band_chi2(
                group_df=group_df,
                fitted_model=model,
                band="z",
            )
        # --------------------------------------------------------
        # Extract results
        # --------------------------------------------------------

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
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    # ------------------------------------------------------------
    # Join per-band statistics back onto the phase-quality table
    # ------------------------------------------------------------

    results_df = objects_df.join(
        chi2_df,
        on="r:diaObjectId",
        how="left",
        maintain_order="left",
    )

    # ------------------------------------------------------------
    # Bazin fitted-band chi2 flags
    #
    # A null means that band was not part of the SALT fit.
    # We keep the individual flags for now and do not yet combine
    # them into the final chi2 selection, because the next stage is
    # the fitted-vs-unfitted band logic.
    # ------------------------------------------------------------

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

    results_df.write_csv(values_output_path)
    results_df.filter(pl.col("pass_chi2_quality")).write_csv(pass_output_path)
    results_df.filter(~pl.col("pass_chi2_quality")).write_csv(fail_output_path)

    return results_df


def unfitted_band_chi2(
    group_df: pl.DataFrame,
    fitted_model: sncosmo.Model,
    band: str,
    band_map: dict[str, str] = BAND_MAP,
) -> dict:
    """
    Calculate chi-square for a band that did not participate in
    the SALT3 fit.

    The SALT parameters remain fixed. If SALT3 cannot physically
    evaluate the band at the fitted redshift, the statistic is
    returned as null rather than extrapolating the model.
    """

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

    bazin_salt_chi2_cut(
        phase_cuts_path=SN_IA_SALT_PHASE_PATH,
        light_curve_data_path=LIGHT_CURVE_DATA_PATH,
        values_output_path=SN_IA_SALT_VALUES_OUTPUT_PATH,
        pass_output_path=SN_IA_SALT_PASS_OUTPUT_PATH,
        fail_output_path=SN_IA_SALT_FAIL_OUTPUT_PATH,
    )
