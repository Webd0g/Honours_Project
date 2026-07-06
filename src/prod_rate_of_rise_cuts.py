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

chunk_size = 500
sn_near_galaxy_candidate_results = []
extragalactic_candidate_results = []
sn_near_galaxy_candidates_passed = []
extragalactic_candidates_passed = []
sn_near_galaxy_candidates_failed = []
extragalactic_candidates_failed = []
red_slope_cut_threshold = 0.0053
green_slope_cut_threshold = 0.0223


# fetch light curve data for each object in the candidate lists with chunking to avoid timeout issues
for diaObjectId in range(0, len(sn_near_galaxy_candidates), chunk_size):
    sn_near_galaxy_candidates_light_curve_data = collect_light_curve_data(','.join(sn_near_galaxy_candidates[diaObjectId:diaObjectId + chunk_size]))
    if sn_near_galaxy_candidates_light_curve_data is not None:
        sn_near_galaxy_candidate_results.append(sn_near_galaxy_candidates_light_curve_data)

for diaObjectId in range(0, len(extragalactic_candidates), chunk_size):
    extragalactic_candidates_light_curve_data = collect_light_curve_data(','.join(extragalactic_candidates[diaObjectId:diaObjectId + chunk_size]))
    if extragalactic_candidates_light_curve_data is not None:
        extragalactic_candidate_results.append(extragalactic_candidates_light_curve_data)

if not sn_near_galaxy_candidate_results:
    raise RuntimeError("No light curve data returned for sn_near_galaxy candidates")

if not extragalactic_candidate_results:
    raise RuntimeError("No light curve data returned for extragalactic candidates")

sn_near_galaxy_light_curve_info = pd.concat(sn_near_galaxy_candidate_results, ignore_index=True)
extragalactic_light_curve_info = pd.concat(extragalactic_candidate_results, ignore_index=True)

# applying the 4 detections before peak cut and the 1 sigma cut to the slopes in both bands for each candidate
for object_id, group_df in sn_near_galaxy_light_curve_info.groupby("r:diaObjectId"):
    red_group_df = group_df[group_df["r:band"].isin(["r"])]
    green_group_df = group_df[group_df["r:band"].isin(["g"])]
    if red_group_df.empty or green_group_df.empty:
        sn_near_galaxy_candidates_failed.append(object_id)
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
            sn_near_galaxy_candidates_passed.append(object_id)
        else: 
            sn_near_galaxy_candidates_failed.append(object_id)
    else:
        sn_near_galaxy_candidates_failed.append(object_id)


for object_id, group_df in extragalactic_light_curve_info.groupby("r:diaObjectId"):
    red_group_df = group_df[group_df["r:band"].isin(["r"])]
    green_group_df = group_df[group_df["r:band"].isin(["g"])]
    if red_group_df.empty or green_group_df.empty:
        extragalactic_candidates_failed.append(object_id)
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
            extragalactic_candidates_passed.append(object_id)
        else:
            extragalactic_candidates_failed.append(object_id)
    else:
        extragalactic_candidates_failed.append(object_id)

# writing the passed candidates to their respective CSV files
sn_near_galaxy_candidates_passed_df = pd.DataFrame(sn_near_galaxy_candidates_passed, columns=["r:diaObjectId"])
sn_near_galaxy_candidates_passed_df.to_csv("/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/sn_near_galaxy_candidates_passed.csv", index=False)
extragalactic_candidates_passed_df = pd.DataFrame(extragalactic_candidates_passed, columns=["r:diaObjectId"])
extragalactic_candidates_passed_df.to_csv("/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/extragalactic_candidates_passed.csv", index=False)

# writing objects that didn't pass the rate of rise cuts and detections before peak to their own CSV files
sn_near_galaxy_candidates_failed_df = pd.DataFrame(sn_near_galaxy_candidates_failed, columns=["r:diaObjectId"])
sn_near_galaxy_candidates_failed_df.to_csv("/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/sn_near_galaxy_candidates_failed.csv", index=False)
extragalactic_candidates_failed_df = pd.DataFrame(extragalactic_candidates_failed, columns=["r:diaObjectId"])
extragalactic_candidates_failed_df.to_csv("/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/extragalactic_candidates_failed.csv", index=False)