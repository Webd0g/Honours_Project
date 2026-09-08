from pathlib import Path

import polars as pl

from early_excess_salt_detection import loading_light_curve_data_and_objects


def fifty_percent_flux_calculation(
    objects_path: Path, light_curve_data_path: Path
) -> pl.DataFrame:
    """This function finds the time when the observed flux across each band was 50 per cent of the g band flux.

    Args:
        object_path (Path): The path to the CSV where all the objects are.
        light_curve_data_path (Path): The path to the light curve data of all objects.

    Returns:
        pl.DataFrame: Returns a dataframe with an new column that is the mjd time where the observed flux is 50 per cent of the peak in the g band.
    """
    # Reading and loading the light curve data and finding the light curve data that matches the objects that we have loaded in.
    light_curve_data = loading_light_curve_data_and_objects(
        objects_path=objects_path, light_curve_data_path=light_curve_data_path
    )

    # Getting only the g band detections and finding the peak g band flux for each object.
    g_band_df = light_curve_data.filter(pl.col("r:band") == "i").with_columns(
        pl.col("r:psfFlux").max().over("r:diaObjectId").alias("peak_i_band_flux")
    )

    g_band_df = g_band_df.with_columns(
        (pl.col("peak_i_band_flux") * 0.5).alias("50_per_cent_peak_i_flux")
    )

    return g_band_df


def find_fifty_per_cent_time(
    fifty_per_cent_i_band_df: pl.DataFrame,
) -> pl.DataFrame:
    """Here we find the midpointMjdTai that aligns closest with the quantity of the 50 % flux column.

    Args:
        fifty_per_cent_g_band_df (pl.DataFrame): The polars dataframe that has all the data from the fifty per cent g band flux calcualtion.
    Returns:
        pl.DataFrame: Returns the dataframe that was passed to this function with a new column identifying the epoch that is closest to the 50 per cent peak flux calculation.
    """
    # Finding the peak row so we can track the MJD for that peak.
    peak_df = fifty_per_cent_i_band_df.filter(
        pl.col("r:psfFlux") == pl.col("peak_i_band_flux")
    ).with_columns(pl.col("r:midpointMjdTai").alias("peak_mjd"))

    # Join the df that contains the peak info and the df with all the objects in it.
    peak_df = fifty_per_cent_i_band_df.join(
        peak_df.select("r:diaObjectId", "peak_mjd"), on="r:diaObjectId", how="left"
    )

    # Finding the difference between the peak in the g band for the objects and all the fluxes for the other measurements for the objects.
    difference_df = peak_df.with_columns(
        (pl.col("r:psfFlux") - pl.col("50_per_cent_peak_i_flux"))
        .abs()
        .alias("delta_flux")
    ).sort(["r:diaObjectId", "r:midpointMjdTai"])

    # Adding a column that is the previous flux measurement for each row.
    difference_df = difference_df.with_columns(
        pl.col("r:psfFlux").shift(1).over("r:diaObjectId").alias("previous_flux")
    )

    # Adding a column that is the next flux measurement for each row.
    difference_df = difference_df.with_columns(
        pl.col("r:psfFlux").shift(-1).over("r:diaObjectId").alias("next_flux")
    )
    print(difference_df.select("r:diaObjectId").n_unique())

    # Filter to retain only the points for the objects that are before the peak flux in g band.
    rise_df = difference_df.filter(pl.col("r:midpointMjdTai") <= pl.col("peak_mjd"))

    print(rise_df.select("r:diaObjectId").n_unique())

    # Adding a column that is the minimum delta_flux measurement for each object to find the point that is the closest to the 50 per cent flux calculation.
    fifty_per_cent_point_df = rise_df.with_columns(
        pl.col("delta_flux").min().over("r:diaObjectId").alias("min_abs_delta_flux")
    )
    print(fifty_per_cent_point_df.select("r:diaObjectId").n_unique())

    # Filtering to points that are apart of the rise component where the previous measurement was less than the middle measurement and the next measurement is higher flux.
    filtered_df = fifty_per_cent_point_df.filter(
        (pl.col("previous_flux") < pl.col("r:psfFlux"))
        & (pl.col("next_flux") > pl.col("r:psfFlux"))
    )

    print(
        (
            fifty_per_cent_point_df.join(
                filtered_df, on=pl.col("r:diaObjectId"), how="anti"
            )
        )
        .select("r:diaObjectId")
        .unique()
    )

    # Here we find where the delta flux is the smallest value over each diaObjectId, however it only finds 14 rows?
    filtered_df = filtered_df.filter(
        (pl.col("delta_flux") == pl.col("min_abs_delta_flux")).over("r:diaObjectId")
    )
    print(filtered_df.select("r:diaObjectId").n_unique())


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    OBJECT_DIR = (
        DATA_DIR
        / "salt_fitting"
        / "ia_search"
        / "with_z_results"
        / "colour_stretch_results"
        / "pass.csv"
    )

    LIGHT_CURVE_DATA_DIR = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    result = fifty_percent_flux_calculation(
        objects_path=OBJECT_DIR, light_curve_data_path=LIGHT_CURVE_DATA_DIR
    )

    find_fifty_per_cent_time(result)
