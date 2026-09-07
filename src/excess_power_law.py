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
    g_band_df = light_curve_data.filter(pl.col("r:band") == "g").with_columns(
        pl.col("r:psfFlux").max().over("r:diaObjectId").alias("peak_g_band_flux")
    )

    g_band_df = g_band_df.with_columns((pl.col("peak_g_band_flux") * 0.5).alias("50_per_cent_peak_g_flux"))

    print (g_band_df)

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

    fifty_percent_flux_calculation(
        objects_path=OBJECT_DIR, light_curve_data_path=LIGHT_CURVE_DATA_DIR
    )
