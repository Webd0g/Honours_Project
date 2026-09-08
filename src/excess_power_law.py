from pathlib import Path

import polars as pl

from early_excess_salt_detection import loading_light_curve_data_and_objects


def fifty_percent_flux_calculation(
    objects_path: Path,
    light_curve_data_path: Path,
    band: str,
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
    band_df = light_curve_data.filter(pl.col("r:band") == band).with_columns(
        pl.col("r:psfFlux").max().over("r:diaObjectId").alias("peak_band_flux")
    )

    band_df = band_df.with_columns(
        (pl.col("peak_band_flux") * 0.5).alias("50_per_cent_peak_flux")
    )

    return band_df


def find_fifty_per_cent_time(
    band_df: pl.DataFrame,
    band: str,
) -> pl.DataFrame:
    """Here we find the midpointMjdTai that aligns closest with the quantity of the 50 % flux column.

    Args:
        fifty_per_cent_band_df (pl.DataFrame): The polars dataframe that has all the data from the fifty per cent band flux calcualtion.
    Returns:
        pl.DataFrame: Returns the dataframe that was passed to this function with a new column identifying the epoch that is closest to the 50 per cent peak flux calculation.
    """
    # Finding the peak row so we can track the MJD for that peak.
    peak_df = band_df.filter(
        pl.col("r:psfFlux") == pl.col("peak_band_flux")
    ).with_columns(pl.col("r:midpointMjdTai").alias("peak_mjd"))

    # Join the df that contains the peak info and the df with all the objects in it.
    peak_df = band_df.join(
        peak_df.select("r:diaObjectId", "peak_mjd"), on="r:diaObjectId", how="left"
    )

    # Finding the difference between the peak in the g band for the objects and all the fluxes for the other measurements for the objects.
    difference_df = peak_df.with_columns(
        (pl.col("r:psfFlux") - pl.col("50_per_cent_peak_flux"))
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

    # Filter to retain only the points for the objects that are before the peak flux in g band.
    rise_df = difference_df.filter(pl.col("r:midpointMjdTai") <= pl.col("peak_mjd"))

    # Filtering to points that are apart of the rise component where the previous measurement was less than the middle measurement and the next measurement is higher flux.
    filtered_df = rise_df.filter(
        (pl.col("previous_flux") < pl.col("r:psfFlux"))
        & (pl.col("next_flux") > pl.col("r:psfFlux"))
    )

    # Adding a column that is the minimum delta_flux measurement for each object to find the point that is the closest to the 50 per cent flux calculation.
    fifty_per_cent_point_df = filtered_df.with_columns(
        pl.col("delta_flux").min().over("r:diaObjectId").alias("min_abs_delta_flux")
    )

    # Here we find where the delta flux is the smallest value over each diaObjectId, however it only finds 14 rows?
    filtered_df = fifty_per_cent_point_df.filter(
        (pl.col("delta_flux").abs()) == pl.col("min_abs_delta_flux")
    )

    return filtered_df


def iterate_over_bands(
    objects_dir: Path,
    light_curve_data_dir: Path,
    bands: list[str],
) -> pl.DataFrame:
    """Iterates over the bands that you feed as a list of strings, this is so that if an object isn't well sampled in one band but you would like to keep it you can try other bands and they weill get appended.

    Args:
        band_df (pl.DataFrame): The dataframe that is the result from the 50 per cent peak flux calcuation.
        bands (list[str]): The list of the bands as string entries

    Returns:
        pl.DataFrame: The dataframe with the row that is the 50 per cent flux calculatio which is how we find the time to fit to for the power law.
    """

    # Iterates through the bands that have been entered in the list
    for band in bands:
        result = fifty_percent_flux_calculation(
            objects_path=objects_dir,
            light_curve_data_path=light_curve_data_dir,
            band=band,
        )

        filtered_result = find_fifty_per_cent_time(
            band_df=result,
            band=band,
        )
        if band == "g":
            g_result = filtered_result
        elif band == "i":
            i_result = filtered_result

    i_only = i_result.join(
        g_result.select("r:diaObjectId"),
        on="r:diaObjectId",
        how="anti",
    )

    final_reference_points = pl.concat([g_result, i_only])

    return final_reference_points.rename({"r:midpointMjdTai": "t50_mjd"}).select(
        [
            "r:diaObjectId",
            "r:band",
            "r:psfFlux",
            "r:psfFluxErr",
            "t50_mjd",
            "peak_band_flux",
            "50_per_cent_peak_flux",
            "peak_mjd",
            "delta_flux",
            "previous_flux",
            "next_flux",
        ]
    )


def keep_observations_upto_t50(
    objects: Path,
    light_curve_data_path: Path,
    reference_df: pl.DataFrame,
) -> pl.DataFrame:
    """_summary_

    Args:
        objects (Path): _description_
        light_curve_data (Path): _description_
        reference_df (pl.DataFrame): _description_

    Returns:
        pl.DataFrame: _description_
    """

    light_curve_data = loading_light_curve_data_and_objects(
        objects_path=objects,
        light_curve_data_path=light_curve_data_path,
    )

    t50_df = reference_df

    light_curve_data = light_curve_data.join(
        t50_df, on="r:diaObjectId", how="inner"
    ).filter(
        pl.col("r:midpointMjdTai") <= pl.col("t50_mjd"),
    ).select(["r:diaObjectId", "r:band", "r:midpointMjdTai", "r:psfFlux", "r:psfFluxErr", "t50_mjd"])

    print(light_curve_data.select("r:diaObjectId").n_unique())


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

    BANDS = [
        "g",
        "i",
    ]

    result = iterate_over_bands(
        objects_dir=OBJECT_DIR,
        light_curve_data_dir=LIGHT_CURVE_DATA_DIR,
        bands=BANDS,
    )

    keep_observations_upto_t50(
        objects=OBJECT_DIR,
        light_curve_data_path=LIGHT_CURVE_DATA_DIR,
        reference_df=result,
    )
