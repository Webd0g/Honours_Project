"""This performs a rate of rise cuts for the g and r bands of the candidates in sn_near_galaxy_candidate.csv, 
    extragalactic_lt20mag_candidates.csv and candidates_in_TNS.csv. It pulls the diaObjectIds from the CSV file, 
    fetch the light curve data from the api of the objects, apply the 4 minimum detections cut before peak, 
    normalise the fluxes to the current brightest values, compute the linear slope, and then 
    output the candidates that passed and those that didn't into their respective CSV files"""

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


def applying_rate_of_rise_filter(candidates, chunk_size, red_slope_cut_threshold, green_slope_cut_threshold, output_path_passed, output_path_failed):
    """This function applies the rate of rise filter to a CSV file of diaObjectIds acquired from the Rubin telescope using Fink. It uses thresholds 
    for the slope of the g and r bandpasses derived from the 6 confirmed in_tns objects. This function also employs a number of data points pre-peak
    cut off which is 4 to prevent two point detection cases from pasing which are not useful cases.
    
    This function is nessacary to cut out any flat and non rising objects out of the large data base as the tag filtering doesn't cut out flat light curves
    which are obviously not SN candidates."""

    passed, failed = [], []

    candidate_results =[]

    for diaObjectId in range(0, len(candidates), chunk_size):
        candidates_light_curve_data = collect_light_curve_data(','.join(candidates[diaObjectId:diaObjectId + chunk_size]))
        if candidates_light_curve_data is not None:
            candidate_results.append(candidates_light_curve_data)
    
    if not candidate_results:
        raise RuntimeError(f"No light curve data returned for candidate batch of size {len(candidates)}")

    candidate_light_curve_info = pd.concat(candidate_results, ignore_index=True)

    for object_id, group_df in candidate_light_curve_info.groupby("r:diaObjectId"):
        red_group_df = group_df[group_df["r:band"].isin(["r"])]
        green_group_df = group_df[group_df["r:band"].isin(["g"])]
        if red_group_df.empty or green_group_df.empty:
            failed.append(object_id)
            continue
        red_peak_flux = red_group_df["r:psfFlux"].max()
        green_peak_flux = green_group_df["r:psfFlux"].max()
        peak_red_mjd = red_group_df.loc[red_group_df["r:psfFlux"].idxmax(), "r:midpointMjdTai"]
        peak_green_mjd = green_group_df.loc[green_group_df["r:psfFlux"].idxmax(), "r:midpointMjdTai"]
        pre_peak_red_mjd = red_group_df[red_group_df["r:midpointMjdTai"] < peak_red_mjd]
        pre_peak_green_mjd = green_group_df[green_group_df["r:midpointMjdTai"] < peak_green_mjd]
        if len(pre_peak_red_mjd) >= 4 and len(pre_peak_green_mjd) >= 4:
            red_slope, red_intercept = np.polyfit(pre_peak_red_mjd["r:midpointMjdTai"], (pre_peak_red_mjd["r:psfFlux"]/red_peak_flux), 1)
            green_slope, green_intercept = np.polyfit(pre_peak_green_mjd["r:midpointMjdTai"], (pre_peak_green_mjd["r:psfFlux"]/green_peak_flux), 1)
            if red_slope >= red_slope_cut_threshold and green_slope >= green_slope_cut_threshold:
                # candidate passes the cut
                passed.append(object_id)
            else: 
                failed.append(object_id)
        else:
            failed.append(object_id)

    # writing the passed candidates to their respective CSV files
    passed_df = pd.DataFrame(passed, columns=["r:diaObjectId"])
    passed_df.to_csv(output_path_passed, index=False)
    failed_df = pd.DataFrame(failed, columns=["r:diaObjectId"])
    failed_df.to_csv(output_path_failed, index=False)

    return passed, failed

sn_near_galaxy_candidates_passed, sn_near_galaxy_candidates_failed = applying_rate_of_rise_filter(SN_NEAR_GALAXY_CANDIDATES, CHUNK_SIZE, RED_SLOPE_CUT_THRESHOLD, GREEN_SLOPE_CUT_THRESHOLD, SN_NEAR_GALAXY_CANDIDATES_OUTPUT_PASSED, SN_NEAR_GALAXY_CANDIDATES_OUTPUT_FAILED)
extragalactic_candidates_passed, extragalactic_candidates_failed = applying_rate_of_rise_filter(EXTRAGALACTIC_CANDIDATES, CHUNK_SIZE, RED_SLOPE_CUT_THRESHOLD, GREEN_SLOPE_CUT_THRESHOLD, EXTRAGALACTIC_CANDIDATES_OUTPUT_PASSED, EXTRAGALACTIC_CANDIDATES_OUTPUT_FAILED)
in_tns_candidates_passed, in_tns_candidates_failed = applying_rate_of_rise_filter(IN_TNS_CANDIDATES, CHUNK_SIZE, RED_SLOPE_CUT_THRESHOLD, GREEN_SLOPE_CUT_THRESHOLD, IN_TNS_CANDIDATES_OUTPUT_PASSED, IN_TNS_CANDIDATES_OUTPUT_FAILED)
