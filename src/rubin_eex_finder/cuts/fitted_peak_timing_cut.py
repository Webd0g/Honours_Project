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


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data" / "forced_photometry_test"

    IMAGE_DIR = PROJECT_DIR / "images"

    LIGHT_CURVE_DATA_DIR = (
        DATA_DIR / "light_curve_data" / "forced_photometry_light_curve_data.parquet"
    )

    OBJECTS = DATA_DIR / "sampling_results" / "pass.csv"

    PASS = DATA_DIR / "fitted_peak_timing_results" / "pass.csv"

    FAIL = DATA_DIR / "fitted_peak_timing_results" / "fail.csv"

    VALUES = DATA_DIR / "fitted_peak_timing_results" / "values.csv"

    df = peak_timing_cut(
        objects_path=OBJECTS,
        i_band_bazin_fits_path=SN_IA_I_BAND_BAZIN_FITS_RESULTS,
        r_band_bazin_fits_path=SN_IA_R_BAND_BAZIN_FITS_RESULTS,
        peak_time_difference_threshold=10.0,
        # peak_time_difference_output_path=SN_IA_VALUES_OUTPUT_PATH,
        # pass_output_path=SN_IA_PASS_OUTPUT_PATH,
        # fail_output_path=SN_IA_FAIL_OUTPUT_PATH,
    )

    print(df.head())
