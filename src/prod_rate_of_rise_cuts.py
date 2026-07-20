"""
This performs a rate of rise cuts for the g and r bands of the candidates in sn_near_galaxy_candidate.csv, 
extragalactic_lt20mag_candidates.csv and candidates_in_TNS.csv. It pulls the diaObjectIds from the CSV file, 
fetch the light curve data from the api of the objects, apply the 4 minimum detections cut before peak, 
normalise the fluxes to the current brightest values, compute the linear slope, and then 
output the candidates that passed and those that didn't into their respective CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from light_curve_plotter import collect_light_curve_data

# Directories for the objects that passed the first filtering done through the tags, second directory is for the output of the rate of rise cut.
CANDIDATE_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "extragalactic_object_ids"
RATE_OF_RISE_DIR = Path.home() / "Documents"/"University"/"Honours"/"Honours_Project"/ "data" / "Rate_of_rise_cuts"

# Reading the csv for the objects that passed the Fink tag filtering and creating the CSVs where data will be stored. 
SN_NEAR_GALAXY_CANDIDATES = list(pd.read_csv(CANDIDATE_DIR / "sn_near_galaxy_candidate.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"])
EXTRAGALACTIC_CANDIDATES = list(pd.read_csv(CANDIDATE_DIR / "extragalactic_lt20mag_candidate.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"])
IN_TNS_CANDIDATES = list(pd.read_csv(CANDIDATE_DIR / "candidates_in_TNS.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"])

# Creating the passed and failed CSVs that store the passing and failing objects from the filter.
SN_NEAR_GALAXY_CANDIDATES_OUTPUT_PASSED = RATE_OF_RISE_DIR / "sn_near_galaxy_candidates_passed.csv"
SN_NEAR_GALAXY_CANDIDATES_OUTPUT_FAILED = RATE_OF_RISE_DIR / "sn_near_galaxy_candidates_failed.csv"
EXTRAGALACTIC_CANDIDATES_OUTPUT_PASSED = RATE_OF_RISE_DIR / "extragalactic_candidates_passed.csv"
EXTRAGALACTIC_CANDIDATES_OUTPUT_FAILED = RATE_OF_RISE_DIR / "extragalactic_candidates_failed.csv"
IN_TNS_CANDIDATES_OUTPUT_PASSED = RATE_OF_RISE_DIR / "in_tns_candidates_passed.csv"
IN_TNS_CANDIDATES_OUTPUT_FAILED = RATE_OF_RISE_DIR / "in_tns_candidates_failed.csv"

# Specifying how many objects go into each query to the API and the slope thresholds for the rate of rise in the g and r bands for the rate of rise filtering.
CHUNK_SIZE = 500
RED_SLOPE_CUT_THRESHOLD = 0.0053
GREEN_SLOPE_CUT_THRESHOLD = 0.0223

def chunked_light_curve_data(candidates : list , chunk_size = 500 ) -> pd.DataFrame:
    """
    This function scrapes the Fink
    API for the light curve info nessacary
    to make cuts absed on object qualities and 
    produce plots of the light curves.
    """

    candidate_results =[]

    for diaObjectId in range(0, len(candidates), chunk_size):
        candidates_light_curve_data = collect_light_curve_data(','.join(candidates[diaObjectId:diaObjectId + chunk_size]))
        if candidates_light_curve_data is not None:
            candidate_results.append(candidates_light_curve_data)

    if not candidate_results:
        raise RuntimeError(f"No light curve data returned for candidate batch of size {len(candidates)}")

    return pd.concat(candidate_results, ignore_index=True)

def slope_calculation(group_df: pd.DataFrame, band: str) -> tuple[float, int] | None:
    """
    This function finds the slope of the band of the lightcurve
    prepeak by fitting a polynomial of degree 1 to the points of the light
    curve before the brightest point of the light curve. This also incorporates
    the minimum four detections before the peak check before the slope is calculated.

    This is done to remove the flat objects whose brightness doesn't change that much over time.
    We make it a function of bands so that we can easily use different bandpasses when neccesary for 
    higher redshift SN.
    """
    band_df = group_df[group_df["r:band"] == band]

    if band_df.empty:
        return None

    peak_flux = band_df["r:psfFlux"].max()
    peak_mjd = band_df.loc[band_df["r:psfFlux"].idxmax(), "r:midpointMjdTai"]
    pre_peak_mjd = band_df[band_df["r:midpointMjdTai"] < peak_mjd]

    if len(pre_peak_mjd) >= 4:
        slope, _ = np.polyfit(pre_peak_mjd["r:midpointMjdTai"], (pre_peak_mjd["r:psfFlux"]/peak_flux), 1)

        return slope, len(pre_peak_mjd)
    
    else:
        return None


def check_rate_of_rise(group_df: pd.DataFrame, red_threshold: float, green_threshold: float) -> bool:
    """
    This function calls the slope calculation of the object in the specified band and checks if 
    it passes the specifed threshold for to be considered potentially SN. 
    
    This function is nessecary incase we desire to easily change what bands we use and change the
    slope thresholds. 
    """
    red_slope = slope_calculation(group_df, "r")
    green_slope = slope_calculation(group_df, "g")
    
    if red_slope is None or green_slope is None:
        return False
    
    return red_slope[0] >= red_threshold and green_slope[0] >= green_threshold

def applying_the_rate_of_rise(objects: list[str], chunk_size: int, red_threshold: float, green_threshold: float, output_path_passed: Path, output_path_failed: Path) -> tuple[list[str], list[str]]:
    """
    This function pulls the light curve data for the specified list of object ids and the chunk size can be customised if you have a large list of objects or small. The function then checks against 
    the rate of rise threshold and directs the objects to their respective lists. Finally this function then produces two CSV files of the obejcts that passed and ones that failed.
    """

    passed, failed = [], []
    
    light_curve_info = chunked_light_curve_data(objects, chunk_size)

    for object_id, group_df in light_curve_info.groupby("r:diaObjectId"):
        if check_rate_of_rise(group_df, red_threshold, green_threshold):
            passed.append(object_id)
        else:
            failed.append(object_id)
    
    pd.DataFrame(passed, columns=["r:diaObjectId"]).to_csv(output_path_passed, index=False)
    pd.DataFrame(failed, columns=["r:diaObjectId"]).to_csv(output_path_failed, index=False)

    return passed, failed

if __name__ ==  "__main__":
    applying_the_rate_of_rise(SN_NEAR_GALAXY_CANDIDATES, CHUNK_SIZE, RED_SLOPE_CUT_THRESHOLD, GREEN_SLOPE_CUT_THRESHOLD, SN_NEAR_GALAXY_CANDIDATES_OUTPUT_PASSED, SN_NEAR_GALAXY_CANDIDATES_OUTPUT_FAILED)
    applying_the_rate_of_rise(EXTRAGALACTIC_CANDIDATES, CHUNK_SIZE, RED_SLOPE_CUT_THRESHOLD, GREEN_SLOPE_CUT_THRESHOLD, EXTRAGALACTIC_CANDIDATES_OUTPUT_PASSED, EXTRAGALACTIC_CANDIDATES_OUTPUT_FAILED)
    applying_the_rate_of_rise(IN_TNS_CANDIDATES, CHUNK_SIZE, RED_SLOPE_CUT_THRESHOLD, GREEN_SLOPE_CUT_THRESHOLD, IN_TNS_CANDIDATES_OUTPUT_PASSED, IN_TNS_CANDIDATES_OUTPUT_FAILED)

# in_tns_test_candidates = ["170094456400773135", "170094456357257301", "170226393328648251", "170235219334922248", "170257199649521669", "170411112000913487"]

# applying_the_rate_of_rise(in_tns_test_candidates, 500, RED_SLOPE_CUT_THRESHOLD, GREEN_SLOPE_CUT_THRESHOLD, r"C:\Users\riley\Downloads\in_tns_test_passed.csv", r"C:\Users\riley\Downloads\in_tns_test_failed.csv")