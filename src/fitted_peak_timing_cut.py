from pathlib import Path

import polars as pl


def fitted_peak_times(
    objects_path: Path, i_band_bazin_fits_path: Path, r_band_bazin_fits_path: Path
) -> pl.DataFrame:
    """Finds the peak times from the bazin fit parameters and
    returns them as a column in a df.

    Args:
        object_path (Path): Path to the CSV of objects.
        bazin_fits_path (Path): Path to the CSV of the Bazin fit results.

    Returns:
        pl.DataFrame: Returns a polars dataframe with the new column of the peak time derived from the Bazin fits.
    """

    object_ids_df = pl.read_csv(
        objects_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    )

    i_band_bazin_data_df = (
        pl.read_csv(
            i_band_bazin_fits_path,
            schema_overrides={"r:diaObjectId": pl.String},
            columns=["r:diaObjectId", "reference_time", "rise_time", "fall_time"],
        )
        .with_columns(
            (
                pl.col("reference_time")
                + pl.col("rise_time")
                * (pl.col("fall_time") / pl.col("rise_time") - 1).log()
            ).alias("i_peak_time")
        )
        .select("r:diaObjectId", "i_peak_time")
    )

    r_band_bazin_data_df = (
        pl.read_csv(
            r_band_bazin_fits_path,
            schema_overrides={"r:diaObjectId": pl.String},
            columns=["r:diaObjectId", "reference_time", "rise_time", "fall_time"],
        )
        .with_columns(
            (
                pl.col("reference_time")
                + pl.col("rise_time")
                * (pl.col("fall_time") / pl.col("rise_time") - 1).log()
            ).alias("r_peak_time")
        )
        .select("r:diaObjectId", "r_peak_time")
    )

    fitted_peak_time_df = (
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
    return fitted_peak_time_df


def peak_timing_cut(
    objects_path: Path,
    i_band_bazin_fits_path: Path,
    r_band_bazin_fits_path: Path,
    peak_time_difference_threshold: int,
    pass_output_path: Path,
    fail_output_path: Path,
) -> None:
    """The threshold for how great the difference between the peak of the i band bazin fit and the r band bazin fit are to pass the cut.

    Args:
        objects_path (Path): Path to the CSV of objects
        i_band_bazin_fits_path (Path): Path to the i band Bazin fits CSV
        r_band_bazin_fits_path (Path): Path to the r band Bazin fits CSV
        peak_time_difference_threshold (int): Number of days the fitted peaks can be apart.
    """

    fitted_peak_times_df = fitted_peak_times(
        objects_path, i_band_bazin_fits_path, r_band_bazin_fits_path
    ).with_columns(
        (
            pl.col("peak_time_difference").is_finite()
            & (pl.col("peak_time_difference") <= peak_time_difference_threshold)
        ).alias("passes_peak_timing")
    )

    fitted_peak_times_df.filter(pl.col("passes_peak_timing") == "true").write_csv(
        pass_output_path
    )

    fitted_peak_times_df.filter(pl.col("passes_peak_timing") == "false").write_csv(
        fail_output_path
    )

    return fitted_peak_times_df


def x_match(test_csv_path: Path, fitted_peak_times_df: pl.DataFrame) -> None:
    """X-match between the known sn and contaminants in the testing sample CSV.

    Args:
        test_csv_path (Path): Path to the test object CSV.
    """

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
        sn_like_df,
        on="r:diaObjectId",
        how="inner",
    )

    pl.Config.set_tbl_rows(-1)

    cont_pass = cont_timing_df.filter(pl.col("passes_peak_timing"))

    cont_fail = cont_timing_df.filter(~pl.col("passes_peak_timing"))

    print(f"Known cont objects tested: {cont_timing_df.height}")
    print(f"Passed: {cont_pass.height}/{cont_timing_df.height}")
    print(f"Failed: {cont_fail.height}/{cont_timing_df.height}")

    pass_fraction = cont_pass.height / cont_timing_df.height

    print(f"Cont retention: {pass_fraction:.2%}")
    print(
        cont_timing_df.select("peak_time_difference").describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
        )
    )
    print(
        cont_timing_df.sort("peak_time_difference", descending=True).select(
            "r:diaObjectId",
            "i_peak_time",
            "r_peak_time",
            "peak_time_difference",
            "passes_peak_timing",
        )
    )

    sn_pass = sn_timing_df.filter(pl.col("passes_peak_timing"))

    sn_fail = sn_timing_df.filter(~pl.col("passes_peak_timing"))
    pl.Config.set_tbl_rows(-1)
    print(f"Known SN-like objects tested: {sn_timing_df.height}")
    print(f"Passed: {sn_pass.height}/{sn_timing_df.height}")
    print(f"Failed: {sn_fail.height}/{sn_timing_df.height}")

    pass_fraction = sn_pass.height / sn_timing_df.height

    print(f"SN-like retention: {pass_fraction:.2%}")
    print(
        sn_timing_df.select("peak_time_difference").describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
        )
    )
    print(
        sn_timing_df.sort("peak_time_difference", descending=True).select(
            "r:diaObjectId",
            "i_peak_time",
            "r_peak_time",
            "peak_time_difference",
            "passes_peak_timing",
        )
    )


if __name__ == "__main__":

    TEST_OBJECTS_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "test_data"
        / "bazin_fit_peak_time_tests"
        / "100_object_test.csv"
    )

    OBJECT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "sampling_results"
        / "sampling_pass.csv"
    )

    BAZIN_FITS_DATA_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "bazin_fits_results"
    )

    TEST_PASS_DIRECTORY = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "fitted_peak_timing_results"
        / "100_object_test"
        / "pass.csv"
    )

    TEST_FAIL_DIRECTORY = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "fitted_peak_timing_results"
        / "100_object_test"
        / "fail.csv"
    )

    PASS_DIRECTORY = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "fitted_peak_timing_results"
        / "pass.csv"
    )

    FAIL_DIRECTORY = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "fitted_peak_timing_results"
        / "fail.csv"
    )

    # pl.Config.set_tbl_rows(-1)
    # print(
    #     fitted_peak_times(
    #         TEST_OBJECTS_PATH,
    #         BAZIN_FITS_DATA_PATH / "bazin_fits_i_band_results.csv",
    #         BAZIN_FITS_DATA_PATH / "bazin_fits_r_band_results.csv",
    #     )
    #     .head(50)
    #     .describe()
    # )

    # print(
    #     fitted_peak_times(
    #         TEST_OBJECTS_PATH,
    #         BAZIN_FITS_DATA_PATH / "bazin_fits_i_band_results.csv",
    #         BAZIN_FITS_DATA_PATH / "bazin_fits_r_band_results.csv",
    #     )
    #     .tail(50)
    #     .describe()
    # )

    pl.Config.set_tbl_rows(-1)
    fitted_peak_times_df = peak_timing_cut(
        OBJECT_PATH,
        BAZIN_FITS_DATA_PATH / "bazin_fits_i_band_results.csv",
        BAZIN_FITS_DATA_PATH / "bazin_fits_r_band_results.csv",
        50,
        PASS_DIRECTORY,
        FAIL_DIRECTORY,
    )

    # x_match(TEST_OBJECTS_PATH, fitted_peak_times_df)
