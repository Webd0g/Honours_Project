"""
Calculates the peak observation times derived from Bazin function fits across $r$ and $i$ bands.
Applies a threshold cut on the absolute difference between fitted peak times.
"""

from pathlib import Path

import polars as pl


def fitted_peak_times(
    objects_path: Path,
    i_band_bazin_fits_path: Path,
    r_band_bazin_fits_path: Path,
) -> pl.DataFrame:
    """Finds the time of the peak points from the Bazin fit results.

    Args:
        objects_path (Path): Path to the CSV containing the list of objects.
        i_band_bazin_fits_path (Path): Bazin fit results for the fits to the object's i band data.
        r_band_bazin_fits_path (Path): Bazin fit results for the fits to the object's r band data.

    Returns:
        pl.DataFrame: Returns the dataframe that consists of the new columns for the fitted peak times.
    """

    object_ids_df = pl.read_csv(
        objects_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    )

    def calc_peak_time(col_name: str) -> pl.Expr:
        """Expression used to calculate t_peak from the Bazin fit results

        Args:
            col_name (str): Name of the to return as the peak time calculation.

        Returns:
            pl.Expr: Returns the expression to calculate t_peak from the Bazin fits.
        """
        fall_over_rise = pl.col("fall_time") / pl.col("rise_time")
        return (
            pl.when(fall_over_rise > 1.0)
            .then(
                pl.col("reference_time")
                + pl.col("rise_time") * (fall_over_rise - 1.0).log()
            )
            .otherwise(None)
            .alias(col_name)
        )

    i_band_bazin_data_df = (
        pl.read_csv(
            i_band_bazin_fits_path,
            schema_overrides={"r:diaObjectId": pl.String},
            columns=["r:diaObjectId", "reference_time", "rise_time", "fall_time"],
        )
        .with_columns(calc_peak_time("i_peak_time"))
        .select("r:diaObjectId", "i_peak_time")
    )

    r_band_bazin_data_df = (
        pl.read_csv(
            r_band_bazin_fits_path,
            schema_overrides={"r:diaObjectId": pl.String},
            columns=["r:diaObjectId", "reference_time", "rise_time", "fall_time"],
        )
        .with_columns(calc_peak_time("r_peak_time"))
        .select("r:diaObjectId", "r_peak_time")
    )

    return (
        object_ids_df.join(
            i_band_bazin_data_df, on="r:diaObjectId", how="left", maintain_order="left"
        )
        .join(
            r_band_bazin_data_df, on="r:diaObjectId", how="left", maintain_order="left"
        )
        .with_columns(
            pl.col("i_peak_time").is_finite().alias("valid_i_peak"),
            pl.col("r_peak_time").is_finite().alias("valid_r_peak"),
        )
        .with_columns(
            pl.when(pl.col("valid_i_peak") & pl.col("valid_r_peak"))
            .then((pl.col("i_peak_time") - pl.col("r_peak_time")).abs())
            .otherwise(None)
            .alias("peak_time_difference")
        )
    )


def peak_timing_cut(
    objects_path: Path,
    i_band_bazin_fits_path: Path,
    r_band_bazin_fits_path: Path,
    peak_time_difference_threshold: float,
    peak_time_difference_output_path: Path | None = None,
    pass_output_path: Path | None = None,
    fail_output_path: Path | None = None,
) -> pl.DataFrame:
    """Makes a cut based on the difference between the fitted peak in the i and r bands from the Bazin fits.

    Args:
        objects_path (Path): Path to the CSV file that stores the list of objects to be used in the cut.
        i_band_bazin_fits_path (Path): Bazin fit results for the fits to the object's i band data.
        r_band_bazin_fits_path (Path): Bazin fit results for the fits to the object's r band data.
        peak_time_difference_threshold (float): Set the threshold for the difference between the peak times in the two bands.
        peak_time_difference_output_path (Path | None, optional): The output path for the CSV that is outputted of the values for the cut before the cut is made. Defaults to None.
        pass_output_path (Path | None, optional): The path to where the passing objects get outputted to after the cut is made. Defaults to None.
        fail_output_path (Path | None, optional): THe path to where the failing objects get outputted to as a CSV file after the cuts are made. Defaults to None.

    Returns:
        pl.DataFrame: Returns that is the dataframe that stores all of the objects after the cut has been made.
    """

    fitted_peak_times_df = fitted_peak_times(
        objects_path, i_band_bazin_fits_path, r_band_bazin_fits_path
    ).with_columns(
        passess_peak_timing=pl.when(pl.col("valid_i_peak") & pl.col("valid_r_peak"))
        .then(pl.col("peak_time_difference") <= peak_time_difference_threshold)
        .otherwise(True),
        fitted_peak_testable=(pl.col("valid_i_peak") & pl.col("valid_r_peak")),
    )

    if pass_output_path is not None:
        pass_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fitted_peak_times_df.filter(pl.col("passes_peak_timing")).write_csv(
            pass_output_path
        )

    if fail_output_path is not None:
        fail_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fitted_peak_times_df.filter(~pl.col("passes_peak_timing")).write_csv(
            fail_output_path
        )

    if peak_time_difference_output_path is not None:
        peak_time_difference_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fitted_peak_times_df.write_csv(peak_time_difference_output_path)

    return fitted_peak_times_df


def x_match(
    test_csv_path: Path,
    fitted_peak_times_df: pl.DataFrame,
) -> None:
    """X-match between known supernovae and contaminants to calculate pass fractions."""
    sn_like_df = pl.read_csv(
        test_csv_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    ).head(50)

    cont_like_df = pl.read_csv(
        test_csv_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    ).tail(50)

    cont_timing_df = fitted_peak_times_df.join(
        cont_like_df, on="r:diaObjectId", how="inner"
    )
    sn_timing_df = fitted_peak_times_df.join(
        sn_like_df, on="r:diaObjectId", how="inner"
    )

    pl.Config.set_tbl_rows(-1)

    # Evaluate contaminants
    if cont_timing_df.height > 0:
        cont_pass = cont_timing_df.filter(pl.col("passes_peak_timing"))
        cont_fail = cont_timing_df.filter(~pl.col("passes_peak_timing"))
        print(f"Known cont objects tested: {cont_timing_df.height}")
        print(f"Passed: {cont_pass.height}/{cont_timing_df.height}")
        print(f"Failed: {cont_fail.height}/{cont_timing_df.height}")
        print(f"Cont retention: {cont_pass.height / cont_timing_df.height:.2%}")
        print(
            cont_timing_df.select("peak_time_difference").describe(
                percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
            )
        )

    # Evaluate SN-like
    if sn_timing_df.height > 0:
        sn_pass = sn_timing_df.filter(pl.col("passes_peak_timing"))
        sn_fail = sn_timing_df.filter(~pl.col("passes_peak_timing"))
        print(f"\nKnown SN-like objects tested: {sn_timing_df.height}")
        print(f"Passed: {sn_pass.height}/{sn_timing_df.height}")
        print(f"Failed: {sn_fail.height}/{sn_timing_df.height}")
        print(f"SN-like retention: {sn_pass.height / sn_timing_df.height:.2%}")
        print(
            sn_timing_df.select("peak_time_difference").describe(
                percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
            )
        )


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    SN_IA_OBJECTS = DATA_DIR / "sn_ia_test_results" / "sampling_results" / "pass.csv"

    SN_IA_I_BAND_BAZIN_FITS_RESULTS = (
        DATA_DIR
        / "sn_ia_test_results"
        / "bazin_fits_results"
        / "bazin_fits_i_band_results.csv"
    )

    SN_IA_R_BAND_BAZIN_FITS_RESULTS = (
        DATA_DIR
        / "sn_ia_test_results"
        / "bazin_fits_results"
        / "bazin_fits_r_band_results.csv"
    )

    SN_II_I_BAND_BAZIN_FITS_RESULTS = (
        DATA_DIR
        / "sn_ii_test_results"
        / "bazin_fits_results"
        / "bazin_fits_i_band_results.csv"
    )

    SN_II_R_BAND_BAZIN_FITS_RESULTS = (
        DATA_DIR
        / "sn_ii_test_results"
        / "bazin_fits_results"
        / "bazin_fits_r_band_results.csv"
    )

    SN_II_OBJECTS = DATA_DIR / "sn_ii_test_results" / "sampling_results" / "pass.csv"

    SN_IA_PASS_OUTPUT_PATH = (
        DATA_DIR / "sn_ia_test_results" / "fitted_peak_timing_results" / "pass.csv"
    )

    SN_IA_FAIL_OUTPUT_PATH = (
        DATA_DIR / "sn_ia_test_results" / "fitted_peak_timing_results" / "fail.csv"
    )

    SN_IA_VALUES_OUTPUT_PATH = (
        DATA_DIR / "sn_ia_test_results" / "fitted_peak_timing_results" / "values.csv"
    )

    SN_II_PASS_OUTPUT_PATH = (
        DATA_DIR / "sn_ii_test_results" / "fitted_peak_timing_results" / "pass.csv"
    )

    SN_II_FAIL_OUTPUT_PATH = (
        DATA_DIR / "sn_ii_test_results" / "fitted_peak_timing_results" / "fail.csv"
    )

    SN_II_VALUES_OUTPUT_PATH = (
        DATA_DIR / "sn_ii_test_results" / "fitted_peak_timing_results" / "values.csv"
    )

    SPECTRO_STRIP_PLOT_OUTPUT_PATH = IMAGE_DIR / "fitted_peak_timing_plots" / "spectro"

    df = peak_timing_cut(
        objects_path=SN_IA_OBJECTS,
        i_band_bazin_fits_path=SN_IA_I_BAND_BAZIN_FITS_RESULTS,
        r_band_bazin_fits_path=SN_IA_R_BAND_BAZIN_FITS_RESULTS,
        peak_time_difference_threshold=10.0,
        # peak_time_difference_output_path=SN_IA_VALUES_OUTPUT_PATH,
        # pass_output_path=SN_IA_PASS_OUTPUT_PATH,
        # fail_output_path=SN_IA_FAIL_OUTPUT_PATH,
    )

    print(df.head())
