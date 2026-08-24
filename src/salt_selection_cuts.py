from pathlib import Path

import numpy as np
import polars as pl
import sncosmo
from astropy.table import Table


def bazin_salt_quality_cut(
    salt_fits_path: Path,
    light_curve_data_path: Path,
    output_path: Path | None = None,
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
            "host_photoz": pl.Float64,
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
            "host_photoz",
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
        (
            (pl.col("r:midpointMjdTai") - pl.col("t0")) / (1.0 + pl.col("host_photoz"))
        ).alias("tau")
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
            & pl.col("host_photoz").is_not_null()
            & pl.col("host_photoz").is_finite()
            & (pl.col("host_photoz") > 0)
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

    results.write_csv(output_path)

    return results


def calculate_band_chi2(
    data: Table,
    fitted_model: sncosmo.Model,
    data_mask: np.ndarray,
) -> dict:
    """
    Calculate chi-square statistics in each filter for a fitted SALT model.

    Only observations that were actually used by sncosmo.fit_lc are included.

    Returns
    -------
    dict
        Dictionary keyed by SNCosmo band name. Each band contains:
        - chi2
        - n_points
        - chi2_per_point
    """

    used_data = sncosmo.select_data(
        data,
        data_mask,
    )

    results = {}

    bands = np.unique(np.asarray(used_data["band"]))

    for band in bands:
        band_mask = np.asarray(used_data["band"]) == band

        band_data = sncosmo.select_data(
            used_data,
            band_mask,
        )

        n_points = len(band_data)

        chi2 = sncosmo.chisq(
            band_data,
            fitted_model,
            modelcov=True,
        )

        results[band] = {
            "chi2": float(chi2),
            "n_points": n_points,
            "chi2_per_point": (float(chi2) / n_points if n_points > 0 else None),
        }

    return results


from pathlib import Path

import numpy as np
import polars as pl
import sncosmo
from astropy.table import Table


def bazin_salt_chi2_cut(
    salt_quality_path: Path,
    light_curve_data_path: Path,
    output_path: Path,
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
        salt_quality_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
        },
    ).filter(pl.col("pass_salt_quality"))

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

    band_map = {
        "u": "lsstu",
        "g": "lsstg",
        "r": "lsstr",
        "i": "lssti",
        "z": "lsstz",
        "y": "lssty",
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
        # Convert Rubin bands to SNCosmo band names
        # --------------------------------------------------------

        fit_df = group_df.with_columns(
            pl.col("r:band").replace(band_map).alias("sncosmo_band")
        )

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

        # --------------------------------------------------------
        # Determine which bands are actually supported by the
        # fitted SALT model at this redshift.
        #
        # bandoverlap() returns True when the model wavelength
        # range fully overlaps that observed band.
        # --------------------------------------------------------

        used_band_mask = np.array(
            [model.bandoverlap(band) for band in data["band"]],
            dtype=bool,
        )

        used_data = sncosmo.select_data(
            data,
            used_band_mask,
        )

        # --------------------------------------------------------
        # Calculate per-band chi2
        # --------------------------------------------------------

        band_results = {}

        for band in ["lsstg", "lsstr", "lssti", "lsstz"]:
            band_mask = np.asarray(used_data["band"]) == band

            if not np.any(band_mask):
                band_results[band] = {
                    "chi2": None,
                    "n": 0,
                    "chi2_per_point": None,
                }
                continue

            band_data = sncosmo.select_data(
                used_data,
                band_mask,
            )

            chi2 = float(
                sncosmo.chisq(
                    band_data,
                    model,
                    modelcov=True,
                )
            )

            n_band = len(band_data)

            band_results[band] = {
                "chi2": chi2,
                "n": n_band,
                "chi2_per_point": chi2 / n_band,
            }

        # --------------------------------------------------------
        # Extract results
        # --------------------------------------------------------

        g = band_results["lsstg"]
        r = band_results["lsstr"]
        i = band_results["lssti"]
        z = band_results["lsstz"]

        results.append(
            {
                "r:diaObjectId": object_id,
                "g_chi2": g["chi2"],
                "g_n_used": g["n"],
                "g_chi2_per_point": g["chi2_per_point"],
                "r_chi2": r["chi2"],
                "r_n_used": r["n"],
                "r_chi2_per_point": r["chi2_per_point"],
                "i_chi2": i["chi2"],
                "i_n_used": i["n"],
                "i_chi2_per_point": i["chi2_per_point"],
                "z_chi2": z["chi2"],
                "z_n_used": z["n"],
                "z_chi2_per_point": z["chi2_per_point"],
            }
        )

    chi2_df = pl.DataFrame(results)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    print (chi2_df)
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
        (pl.col("reduced_chi2") < 6).alias("pass_overall_chi2"),
    )

    results_df = results_df.select([pl.col("r:diaObjectId"), pl.col("pass_g_fitted_chi2"), pl.col("pass_r_fitted_chi2"), pl.col("pass_i_fitted_chi2"), pl.col("pass_z_fitted_chi2"), pl.col("pass_overall_chi2")])

    results_df.write_csv(output_path)

    return results_df




if __name__ == "__main__":
    WITH_PHOTOZ_TEST_OBJECTS = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "50_object_test"
        / "with_photo_z.csv"
    )

    LIGHT_CURVE_DATA = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "light_curve_data"
        / "combined_light_curve_data.parquet"
    )

    TEST_OUTPUT = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "50_object_test"
        / "light_curve_sampling_results"
        / "salt_selection_cuts_results.csv"
    )

    CHI2_OUTPUT_DIR = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "50_object_test"
        / "chi2_requirement_results"
        / "chi2_requirement_results.csv"
    )

    bazin_salt_quality_cut(WITH_PHOTOZ_TEST_OBJECTS, LIGHT_CURVE_DATA, TEST_OUTPUT)
    result = bazin_salt_chi2_cut(TEST_OUTPUT, LIGHT_CURVE_DATA, CHI2_OUTPUT_DIR)
    print (result)