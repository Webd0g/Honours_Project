"""Here we will fit the objects with a constant and determine that object's reduced chi square with the constant fit to the i band data.
Then the bazin fit reduced chi squared in the i band will be subtracted from that constant fit. The values obtained for each object from that subtraction
will then be scatter plotted as a function of the reduced chi squared score of the Bazin fit in the i band.

This is the most significant cut that can be done for photmetric selection and shoudl hopefully result in SN-like obejcts being greatly separated from
objects that are more flat than SN-like."""

from pathlib import Path

import numpy as np
import polars as pl


def constant_fit(
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> tuple:
    """Finds the weighted average flux and uses that as the constant for the constant fit.

    Args:
        flux (np.ndarray): The flux of the object as a numpy n dimensional array.
        flux_err (np.ndarray): The flux error of the object as a numpy n dimensional array.

    Returns:
        tuple: Returns a tuple of the constant and reduced chi squared value for that fit.
    """

    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)

    weights = 1.0 / flux_err**2

    constant = float(np.sum(weights * flux)) / np.sum(weights)

    chi2 = float(np.sum(((flux - constant) / flux_err) ** 2))

    dof = len(flux) - 1

    return constant, chi2 / dof if dof > 0 else 0.0


def constant_batch_fits(
    object_path: Path,
    light_curve_data_path: Path,
    band: str,
) -> pl.DataFrame:
    """Performs the constant fits over a batch of objects instead of just one and returns the polars dataframe of the results.

    Args:
        object_path (Path): The path to the CSV file of the objects that will be fitted with a constant.
        light_curve_data_path (Path): Path to the parquet file of the light curve objects.
        band (str): The band the constant batch fits is performed on.

    Returns:
        pl.DataFrame: Returns a dataframe of the results of the constant batch fits.
    """

    objects = pl.read_csv(
        object_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    )["r:diaObjectId"].to_list()

    return (
        pl.read_parquet(
            light_curve_data_path,
            columns=[
                "r:diaObjectId",
                "r:band",
                "r:psfFlux",
                "r:psfFluxErr",
                "r:midpointMjdTai",
            ],
        )
        .with_columns(pl.col("r:diaObjectId").cast(pl.String))
        .filter(
            pl.col("r:diaObjectId").is_in(objects)
            & (pl.col("r:band") == band)
            & (pl.col("r:psfFlux") > 0)
            & pl.col("r:psfFlux").is_finite()
            & pl.col("r:psfFluxErr").is_finite()
            & (pl.col("r:psfFluxErr") > 0)
        )
        .unique(subset=["r:diaObjectId", "r:midpointMjdTai"], keep="first")
        .with_columns(weight=1.0 / (pl.col("r:psfFluxErr") ** 2))
        .group_by("r:diaObjectId")
        .agg(
            [
                pl.len().alias("n_obs"),
                (
                    (pl.col("weight") * pl.col("r:psfFlux")).sum()
                    / pl.col("weight").sum()
                ).alias("constant"),
                pl.col("r:psfFlux"),
                pl.col("r:psfFluxErr"),
            ]
        )
        .filter(pl.col("n_obs") >= 2)
        .with_columns(
            reduced_chi2=(
                (
                    (pl.col("r:psfFlux") - pl.col("constant")) / pl.col("r:psfFluxErr")
                ).list.eval(pl.element() ** 2)
            ).list.sum()
            / (pl.col("n_obs") - 1)
        )
        .select(["r:diaObjectId", "constant", "reduced_chi2"])
    )


def compute_delta_chi(
    object_path: Path,
    light_curve_data_path: Path,
    band: str,
    bazin_fit_data_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    """Computes the difference between the reduced chi squared of the Bazin fit and the constant fit.

    Args:
        object_path (Path): Path to the objects that will be fit with the constant.
        light_curve_data_path (Path): Path to the parquet file for the light curve data of the objects.
        band (str): The band for the objects that will be fitted to and compared.
        bazin_fit_data_path (Path): The path to where the Bazin fits results are for the objects.
        output_path (Path): The output path for the CSV that will host the new delta chi column.

    Returns:
        pl.DataFrame: Returns a polars dataframe that has the new delta chi parameter for each object.
    """

    # Fit with a constant for the objects
    constant_fit_result = constant_batch_fits(
        object_path, light_curve_data_path, band
    ).rename({"reduced_chi2": "constant_reduced_chi2"})

    bazin_fit_chi_squared = (
        pl.read_csv(bazin_fit_data_path)
        .with_columns(pl.col("r:diaObjectId").cast(pl.String))
        .rename({"reduced_chi2": "bazin_reduced_chi2"})
    )

    # Joining the two data sources into one df making it easy to compare the columns of the data and add a new column for the delta
    delta_chi_df = constant_fit_result.join(
        bazin_fit_chi_squared, on="r:diaObjectId", how="inner", suffix="_bazin"
    ).with_columns(
        (pl.col("constant_reduced_chi2") - pl.col("bazin_reduced_chi2")).alias(
            "delta_reduced_chi2"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    delta_chi_df.write_csv(output_path)

    return delta_chi_df


def delta_chi_check(
    path_to_delta_chi_results: Path,
) -> pl.DataFrame:
    """Checks if objects have passed or failed based on the delta chi squared compared to the chi squared of the constant fit.

    Args:
        path_to_delta_chi_results (Path): Path to the values saved by the compute delta chi function.

    Returns:
        pl.DataFrame: Returns the objects that passed or failed as a polars dataframe.
    """

    delta_chi_df = pl.read_csv(path_to_delta_chi_results)

    return delta_chi_df.with_columns(
        passes_delta_chi_cut=(
            pl.col("delta_reduced_chi2") > pl.col("bazin_reduced_chi2")
        ),
        failed_delta_chi_cut=(
            pl.col("delta_reduced_chi2") <= pl.col("bazin_reduced_chi2")
        ),
    )


def save_pass_fail_objects(
    path_to_delta_chi_results: Path,
    output_path_pass: Path,
    output_path_fail: Path,
) -> None:
    """Saves the passing objects and failing objects to their own CSV files.

    Args:
        path_to_delta_chi_results (Path): Path to where the delta chi compute results are stored as a CSV file.
        output_path_pass (Path): Where the passing objects should be outputted to, as a CSV file.
        output_path_fail (Path): Where the failing objects will be saved to as a CSV file.
    """
    delta_chi_df = delta_chi_check(path_to_delta_chi_results)

    output_path_pass.parent.mkdir(parents=True, exist_ok=True)
    output_path_fail.parent.mkdir(parents=True, exist_ok=True)

    delta_chi_df.filter(pl.col("passes_delta_chi_cut")).write_csv(output_path_pass)
    delta_chi_df.filter(pl.col("failed_delta_chi_cut")).write_csv(output_path_fail)


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    LIGHT_CURVE_DATA = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    TWENTY_OBJECT_DELTA_CHI2_RESULTS = (
        DATA_DIR
        / "constant_minus_shape_results"
        / "20_object_test"
        / "20_object_test.csv"
    )

    TWENTY_OBJECT_DELTA_CHI2_PLOT_OUTPUT_PATH = (
        IMAGE_DIR / "constant_minus_shape_plots" / "delta_chi_plots" / "20_object_test"
    )

    TWENTY_OBJECT_TIMING_CUTS_RESULTS = (
        DATA_DIR
        / "peak_timing_cuts_results"
        / "20_object_test"
        / "20_object_test_pass.csv"
    )

    TWENTY_OBJECT_I_BAND_BAZIN_FITS_RESULTS = (
        DATA_DIR
        / "bazin_fits_results"
        / "20_object_test"
        / "bazin_fits_i_band_results.csv"
    )

    HUNDRED_OBJECT_TIMING_CUTS_RESULTS = (
        DATA_DIR
        / "peak_timing_cuts_results"
        / "100_object_test"
        / "100_object_test_pass.csv"
    )

    HUNDRED_OBJECT_I_BAND_BAZIN_FITS_RESULTS = (
        DATA_DIR
        / "bazin_fits_results"
        / "100_object_test"
        / "bazin_fits_i_band_results.csv"
    )

    HUNDRED_OBJECT_DELTA_CHI2_RESULTS = (
        DATA_DIR
        / "constant_minus_shape_results"
        / "100_object_test"
        / "100_object_test.csv"
    )

    HUNDRED_OBJECT_DELTA_CHI2_PLOT_OUTPUT_PATH = (
        IMAGE_DIR / "constant_minus_shape_plots" / "delta_chi_plots" / "100_object_test"
    )

    SN_IA_TIMING_CUTS_RESULTS = (
        DATA_DIR / "sn_ia_test_results" / "timing_cuts_results" / "pass.csv"
    )

    SN_II_TIMING_CUTS_RESULTS = (
        DATA_DIR / "sn_ii_test_results" / "timing_cuts_results" / "pass.csv"
    )

    SN_IA_I_BAND_BAZIN_FITS_RESULTS = (
        DATA_DIR
        / "sn_ia_test_results"
        / "bazin_fits_results"
        / "bazin_fits_i_band_results.csv"
    )

    SN_II_I_BAND_BAZIN_FITS_RESULTS = (
        DATA_DIR
        / "sn_ii_test_results"
        / "bazin_fits_results"
        / "bazin_fits_i_band_results.csv"
    )

    SN_IA_CHI2_CUT_PASS_PATH = (
        DATA_DIR / "sn_ia_test_results" / "constant_minus_shape_results" / "pass.csv"
    )

    SN_IA_CHI2_CUT_FAIL_PATH = (
        DATA_DIR / "sn_ia_test_results" / "constant_minus_shape_results" / "fail.csv"
    )

    SN_IA_CHI2_CUT_VALUES_PATH = (
        DATA_DIR / "sn_ia_test_results" / "constant_minus_shape_results" / "values.csv"
    )

    SN_II_CHI2_CUT_PASS_PATH = (
        DATA_DIR / "sn_ii_test_results" / "constant_minus_shape_results" / "pass.csv"
    )

    SN_II_CHI2_CUT_FAIL_PATH = (
        DATA_DIR / "sn_ii_test_results" / "constant_minus_shape_results" / "fail.csv"
    )

    SN_II_CHI2_CUT_VALUES_PATH = (
        DATA_DIR / "sn_ii_test_results" / "constant_minus_shape_results" / "values.csv"
    )

    print(
        constant_batch_fits(
            object_path=SN_IA_TIMING_CUTS_RESULTS,
            light_curve_data_path=LIGHT_CURVE_DATA,
            band="i",
        )
    )
