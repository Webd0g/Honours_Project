from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def fitted_peak_times(
    objects_path: Path,
    i_band_bazin_fits_path: Path,
    r_band_bazin_fits_path: Path,
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
    peak_time_difference_output_path: Path,
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

    fitted_peak_times_df = (
        fitted_peak_times(objects_path, i_band_bazin_fits_path, r_band_bazin_fits_path)
        .with_columns(
            pl.when(pl.col("valid_i_peak") & pl.col("valid_r_peak"))
            .then(pl.col("peak_time_difference") <= peak_time_difference_threshold)
            .otherwise(True)
            .alias("passes_peak_timing")
        )
        .with_columns(
            (pl.col("valid_i_peak") & pl.col("valid_r_peak")).alias(
                "fitted_peak_testable"
            )
        )
    )

    fitted_peak_times_df.filter(pl.col("passes_peak_timing")).write_csv(
        pass_output_path
    )

    fitted_peak_times_df.filter(pl.col("passes_peak_timing") == False).write_csv(
        fail_output_path
    )

    fitted_peak_times_df.write_csv(peak_time_difference_output_path)

    return fitted_peak_times_df


def x_match(
    test_csv_path: Path,
    fitted_peak_times_df: pl.DataFrame,
) -> None:
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


def spectro_strip_plots(
    sn_ii_spectro_fitted_peak_timing_results: Path,
    sn_ia_spectro_fitted_peak_timing_results: Path,
    output_path: Path | None = None,
) -> None:
    """Makes a strip plot of the results for the spectroscopic classes of SN that we have and what their delta fitted peak times were in the i and r bands.

    Args:
        spectro_fitted_peak_timing_results (Path): Path object to the results for the spectro objects after they have passed through the fitted peak timing function.
        output_path (Path): The Path object where the plot will be outputted to.
    """

    # Set the params of the strip plot
    sns.set_theme(
        style="whitegrid",
    )

    # Load in the results from the fitted peak timing csv file
    sn_ia_spectro_results = pl.read_csv(
        source=sn_ia_spectro_fitted_peak_timing_results,
        columns=[
            "r:diaObjectId",
            "peak_time_difference",
        ],
        schema_overrides={
            "r:diaObjectId": pl.String,
            "peak_time_difference": pl.Float64,
        },
    ).with_columns(
        pl.lit("SN Ia").alias("spec_class"),
    )
    sn_ii_spectro_results = pl.read_csv(
        source=sn_ii_spectro_fitted_peak_timing_results,
        columns=[
            "r:diaObjectId",
            "peak_time_difference",
        ],
        schema_overrides={
            "r:diaObjectId": pl.String,
            "peak_time_difference": pl.Float64,
        },
    ).with_columns(
        pl.lit("SN II").alias("spec_class"),
    )

    spectro_results = pl.concat(
        [
            sn_ia_spectro_results,
            sn_ii_spectro_results,
        ],
    )

    colours = {
        "SN Ia": "#0072B2",
        "SN II": "#D55E00",
    }

    sns.stripplot(
        data=spectro_results,
        x="peak_time_difference",
        y="spec_class",
        hue="spec_class",
        order=["SN Ia", "SN II"],
        hue_order=["SN Ia", "SN II"],
        jitter=0.2,
        dodge=False,
        palette=colours,
    )
    plt.axvline(
        x=50,
        ls="--",
        label=r"$|t^i_{max} - t^r_{max}|$ = 50 d",
    )

    plt.legend()
    plt.ylabel("")
    plt.xlabel(r"$|t^i_{max} - t^r_{max}|$ (days)")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path / "spectro_fitted_peak_delta_times.png")
    plt.show()


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

    spectro_strip_plots(
        sn_ii_spectro_fitted_peak_timing_results=SN_IA_VALUES_OUTPUT_PATH,
        sn_ia_spectro_fitted_peak_timing_results=SN_IA_VALUES_OUTPUT_PATH,
        output_path=SPECTRO_STRIP_PLOT_OUTPUT_PATH,
    )
