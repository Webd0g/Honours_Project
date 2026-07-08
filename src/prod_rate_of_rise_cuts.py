# refactoring the rate of rise and points pre-peak filter into a function
import pandas as pd
import numpy as np
from pathlib import Path
from light_curve_plotter import collect_light_curve_data

"""This is the production level script of the rate of rise cuts for the g and r bands of the candidates in sn_near_galaxy_candidate.csv and extragalactic_lt20mag_candidates.csv.
    it will pull object id from the CSV file, fetch the light curve data from the api of the objects,
    apply the 4 minimum detections cut before peak, normalise the fluxes to the current brightest values, 
    compute the linear slope, then apply the 1 sigma cut to the slopes across both bands and then 
    output the candidates that passed and those that didn't into their respective CSV files"""

base_candidate_paths = Path("/home/riley/Documents/University/Honours/Honours_Project/data/extragalactic_object_ids")
sn_near_galaxy_candidates = list(pd.read_csv(base_candidate_paths / "sn_near_galaxy_candidate.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"])
extragalactic_candidates = list(pd.read_csv(base_candidate_paths / "extragalactic_lt20mag_candidate.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"])
in_tns_candidates = list(pd.read_csv(base_candidate_paths / "candidates_in_TNS.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"])
sn_near_galaxy_candidates_output_passed = "/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/sn_near_galaxy_candidates_passed.csv"
sn_near_galaxy_candidates_output_failed = "/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/sn_near_galaxy_candidates_failed.csv"
extragalactic_candidates_output_passed = "/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/extragalactic_candidates_passed.csv"
extragalactic_candidates_output_failed = "/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/extragalactic_candidates_failed.csv"
in_tns_candidates_output_passed = "/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/in_tns_candidates_passed.csv"
in_tns_candidates_output_failed = "/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/in_tns_candidates_failed.csv"

chunk_size = 500
red_slope_cut_threshold = 0.0053
green_slope_cut_threshold = 0.0223


def applying_rate_of_rise_filter(candidates, chunk_size, red_slope_cut_threshold, green_slope_cut_threshold, output_path_passed, output_path_failed):
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

sn_near_galaxy_candidates_passed, sn_near_galaxy_candidates_failed = applying_rate_of_rise_filter(sn_near_galaxy_candidates, chunk_size, red_slope_cut_threshold, green_slope_cut_threshold, sn_near_galaxy_candidates_output_passed, sn_near_galaxy_candidates_output_failed)
extragalactic_candidates_passed, extragalactic_candidates_failed = applying_rate_of_rise_filter(extragalactic_candidates, chunk_size, red_slope_cut_threshold, green_slope_cut_threshold, extragalactic_candidates_output_passed, extragalactic_candidates_output_failed)
in_tns_candidates_passed, in_tns_candidates_failed = applying_rate_of_rise_filter(in_tns_candidates, chunk_size, red_slope_cut_threshold, green_slope_cut_threshold, in_tns_candidates_output_passed, in_tns_candidates_output_failed)
