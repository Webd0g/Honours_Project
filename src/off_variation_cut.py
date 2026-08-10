"""This code computes the off main-variation chi squared result which will help remove those objects that have variable fluxes that might pass the Bazin fit. 
Bazin 2009 state this was a complemantary cut to the cut from the Bazin chi square and constant chi square fit as this can remove objects that will pass that 
cut."""

from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from light_curve_plotter import lightcurve_plotting

# Path to the directory where all the Bazin fits CSV are
BAZIN_FITS_DATA_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "bazin_fits"

LIGHT_CURVE_DATA_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "light_curve_data"

VARIABILITY_WINDOW_OUTPUT_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "images" / "bazin_fits" / "off_variability_tests"

def variability_window(bazin_fits_data_path: Path, light_curve_data_path: Path) -> pl.DataFrame:
    """Here is where we find the main variability window for each objects Bazin fit.
    
    We do this so that we can seperate the points of main variation from the points that 
    aren't in the main varaition of the Bazin fit. """

    objects = pl.read_csv(bazin_fits_data_path, columns=["r:diaObjectId"], schema_overrides={"r:diaObjectId": pl.String})["r:diaObjectId"].to_list()
    bazin_data_df = pl.read_csv(bazin_fits_data_path, schema_overrides={"reference_time": pl.Float64, "rise_time": pl.Float64, "fall_time": pl.Float64, "baseline": pl.Float64}, columns=["r:diaObjectId", "reference_time", "rise_time", "fall_time", "baseline"])
    light_curve_data = pl.read_parquet(light_curve_data_path)
    light_curve_data_filtered = light_curve_data.filter(pl.col("r:diaObjectId").cast(pl.Utf8).is_in(objects))

    bazin_data_df = bazin_data_df.with_columns([(pl.col("reference_time") - (2.7*pl.col("rise_time"))).alias("var_start"), 
                                                (pl.col("reference_time") + (2.7*pl.col("fall_time"))).alias("var_end")])

    bazin_light_curve_data = light_curve_data_filtered.join(bazin_data_df, on = ["r:diaObjectId"], how="inner")
    off_var_data = bazin_light_curve_data.filter((pl.col("r:midpointMjdTai") < pl.col("var_start")) |
                                                (pl.col("r:midpointMjdTai") > pl.col("var_end")))
    return off_var_data

def plotting_off_var_data(bazin_fits_data_path: Path, light_curve_data_path: Path):
    """Here we plot the data points 
    that are off the main variation. 
    This is done iteratively to ensure that we aren't cutting
    off any points that are in the main variation."""

    off_var_data = variability_window(bazin_fits_data_path, light_curve_data_path)
    for (diaObjectId,), off_var_data in off_var_data.group_by("r:diaObjectId"):
        diaObjectId = str(diaObjectId)
        lightcurve_plotting(off_var_data, diaObjectId, VARIABILITY_WINDOW_OUTPUT_DIR)

def off_var_chi_squared(bazin_fits_data_path: Path, light_curve_data_path: Path):
    """Here we compute the chi squared for the data points that are considered 
    off the main variation of the SN
    
    We do this so we can make a cut based on this parameter empirically using the SN-like
    objects as our ground truth."""

    off_var_window_df = variability_window(bazin_fits_data_path, light_curve_data_path)
    result = (
    off_var_window_df
    .with_columns((((pl.col("r:psfFlux") - pl.col("baseline")) / pl.col("r:psfFluxErr")) ** 2).alias("term"))
    .group_by("r:diaObjectId")
    .agg([pl.col("term").sum().alias("chi2_sum"), pl.col("term").count().alias("n_points")])
    .with_columns((pl.col("chi2_sum") / pl.col("n_points")).alias("chi2_off"))
)
    return result


if __name__ == "__main__":
    # print(variability_window(BAZIN_FITS_DATA_DIR / "20_object_test" / "bazin_fits_i_band_results.csv", LIGHT_CURVE_DATA_DIR / "combined_light_curve_data.parquet"))
    # plotting_off_var_data(BAZIN_FITS_DATA_DIR / "20_object_test" / "bazin_fits_i_band_results.csv", LIGHT_CURVE_DATA_DIR / "combined_light_curve_data.parquet")
    print(off_var_chi_squared(BAZIN_FITS_DATA_DIR / "20_object_test" / "bazin_fits_i_band_results.csv", LIGHT_CURVE_DATA_DIR / "combined_light_curve_data.parquet"))