"""
This performs a rate of rise cut across all available bands for the candidates in
sn_near_galaxy_candidate.csv, extragalactic_lt20mag_candidates.csv and
candidates_in_TNS.csv. It pulls the diaObjectIds from the CSV file, fetches the light
curve data from the Fink API, computes a peak-normalised pre-peak linear slope for
every band present in each object's light curve, and passes any object with at least
one band showing a nonzero pre-peak rise. This replaces the earlier g/r-only, fixed
empirical threshold version, since that was found to miss objects whose only clear
pre-peak rise was in a band other than g or r (e.g. i-band for higher redshift SNe).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from light_curve_plotter import collect_light_curve_data

# Directories for the objects that passed the first filtering done through the tags, second directory is for the output of the rate of rise cut.
TEST_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "test_data"
PLOTS_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "images" / "test_data_images"
TEST_PASSED_DIR = TEST_DIR / "test_passed.csv"
TEST_FAILED_DIR = TEST_DIR / "test_failed.csv"
# Reading the csv for the objects that passed the Fink tag filtering and creating the CSVs where data will be stored.

# Specifying how many objects go into each query to the API.
CHUNK_SIZE = 500

# Minimum number of pre-peak points needed in a band before a slope is even attempted.
# np.polyfit needs at least 2 points to fit a line; kept low deliberately while we test
# the loosened nonzero-slope criterion, per today's discussion with Anais.
MIN_PRE_PEAK_POINTS = 4

# Minimum time baseline (in days) spanned by the pre-peak points used in a fit.
# Points clustered within a very short window can produce wildly unstable slopes,
# since the denominator of the fit (variance in time) collapses toward zero even
# with many points - this was seen directly in some objects showing slopes of 10-100+
# despite having 10-20+ pre-peak detections, all crammed into a single night or two.
MIN_PRE_PEAK_TIME_SPAN_DAYS = 5

# This is so we remove spurious and noisey data that skews the results of the fit process too much
MIN_PEAK_SNR = 10

def collect_light_curve_info_with_fallback(candidates: list[str]) -> pd.DataFrame | None:
    """This function collects the lightcurve info of an object
    and factors in a recursive self repair incase the script timesout
    while running on large requests."""

    data = collect_light_curve_data(",".join(candidates))
    if data is not None:
        return data

    if len(candidates) <= 1:
        print(f"Could not fetch data for object {candidates}, skipping.")
        return None

    mid = len(candidates) // 2
    left = collect_light_curve_info_with_fallback(candidates[:mid])
    right = collect_light_curve_info_with_fallback(candidates[mid:])

    candidates_parts = [df for df in (left, right) if df is not None]
    if not candidates_parts:
        return None
    return pd.concat(candidates_parts)


def chunked_light_curve_data(candidates: list, chunk_size: int = 500) -> pd.DataFrame:
    """
    This function scrapes the Fink API for the light curve info necessary
    to make cuts based on object qualities and produce plots of the light curves.
    """
    candidate_results = []

    for diaObjectId in range(0, len(candidates), chunk_size):
        chunked_ids = candidates[diaObjectId:diaObjectId + chunk_size]
        candidates_light_curve_data = collect_light_curve_info_with_fallback(chunked_ids)
        if candidates_light_curve_data is not None:
            candidate_results.append(candidates_light_curve_data)

    if not candidate_results:
        raise RuntimeError(f"No light curve data returned for candidate batch of size {len(candidates)}")

    return pd.concat(candidate_results, ignore_index=True)

def one_point_per_night(band_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the dataframe of all the bands data and filters it further to 
    one point per night of observation, with that point being the 
    point with the highest SNR. 

    This is so that when the slope fitting happens it doesn't fit to 
    points that have been clustered and ensures that the points
    that are being fit to are actually changing in time.
    """

    band_df["night"] = band_df["r:midpointMjdTai"].astype(int)
    band_df["SNR"] = band_df["r:psfFlux"]/band_df["r:psfFluxErr"]

    return band_df

    
def slope_calculation(group_df: pd.DataFrame, band: str, min_points: int = MIN_PRE_PEAK_POINTS, min_time_span: int = MIN_PRE_PEAK_TIME_SPAN_DAYS, min_peak_snr: int = MIN_PEAK_SNR) -> tuple[float, int] | None:
    """
    This function finds the slope of the band of the lightcurve prepeak by fitting a
    polynomial of degree 1 to the points of the light curve before the brightest point
    of the light curve. Returns None if there isn't enough pre-peak data in this band
    to attempt a fit.

    This is done to remove the flat objects whose brightness doesn't change that much
    over time. We make it a function of bands so that we can easily use different
    bandpasses when necessary for higher redshift SNe.
    """
    band_df = group_df[(group_df["r:band"] == band) & (group_df["r:psfFlux"] > 0)]

    if band_df.empty:
        return None

    significant_df = band_df[band_df["r:psfFlux"] / band_df["r:psfFluxErr"] > MIN_PEAK_SNR]
    if significant_df.empty:
        return None

    peak_flux = significant_df["r:psfFlux"].max()
    peak_mjd = significant_df.loc[significant_df["r:psfFlux"].idxmax(), "r:midpointMjdTai"]
    pre_peak = significant_df[significant_df["r:midpointMjdTai"] < peak_mjd]

    if len(pre_peak) < min_points:
        return None

    time_span = pre_peak["r:midpointMjdTai"].max() - pre_peak["r:midpointMjdTai"].min()
    if time_span < min_time_span:
        return None

    normalised_flux = pre_peak["r:psfFlux"] / peak_flux
    normalised_err = pre_peak["r:psfFluxErr"] / peak_flux

    # Guard against zero/negative errors, which would otherwise produce an
    # infinite or undefined weight.
    normalised_err = normalised_err.clip(lower=1e-6)
    weights = 1 / normalised_err

    slope, _ = np.polyfit(pre_peak["r:midpointMjdTai"], normalised_flux, 1, w=weights)


    return slope, len(pre_peak)


def compute_all_band_slopes(group_df: pd.DataFrame, min_points: int = MIN_PRE_PEAK_POINTS) -> dict[str, tuple[float, int]]:
    """
    Computes the pre-peak slope for every band present in this object's light curve,
    not just a fixed pair. Returns a dict of band -> (slope, n_points) for every band
    where a slope could be calculated; bands with insufficient pre-peak data are
    omitted rather than included as None, so downstream code doesn't need to filter.
    """
    bands_present = group_df["r:band"].unique()
    slopes = {}

    for band in bands_present:
        result = slope_calculation(group_df, band, min_points)
        if result is not None:
            slopes[band] = result

    return slopes


def check_rate_of_rise(group_df: pd.DataFrame, min_points: int = MIN_PRE_PEAK_POINTS) -> bool:
    """
    Computes the pre-peak slope across every available band for this object, and
    passes the object if at least one band shows a nonzero (positive) pre-peak rise.

    This is deliberately loose for now: rather than requiring specific bands (as the
    old g/r-only version did) or a calibrated threshold, this just checks whether the
    object is doing *anything* pre-peak in at least one band. Tighter criteria
    (uncertainty weighting, max time-gap between points, a proper threshold) are
    intended to be layered on top of this once it's validated on a small batch.
    """
    slopes = compute_all_band_slopes(group_df, min_points)

    if not slopes:
        return False

    return any(slope > 0.001 for slope, _ in slopes.values())


def applying_the_rate_of_rise(objects: list[str], chunk_size: int, output_path_passed: Path = None, output_path_failed: Path = None) -> tuple[list[str], list[str]]:
    """
    This function pulls the light curve data for the specified list of object ids and
    the chunk size can be customised if you have a large list of objects or small. The
    function then checks whether the object shows a nonzero pre-peak rise in at least
    one band and directs the objects to their respective lists. Finally this function
    produces two CSV files of the objects that passed and ones that failed.
    """
    passed, failed = [], []

    light_curve_info = chunked_light_curve_data(objects, chunk_size)
    
    light_curve_info = one_point_per_night(light_curve_info)

    sorted_light_curve_info = light_curve_info.sort_values(by="SNR", ascending=False)
    best_points = sorted_light_curve_info.groupby(["r:diaObjectId", "r:band", "night"]).first().reset_index()

    for object_id, group_df in light_curve_info.groupby("r:diaObjectId"):
        if check_rate_of_rise(group_df):
            passed.append(object_id)
        else:
            failed.append(object_id)

    pd.DataFrame(passed, columns=["r:diaObjectId"]).to_csv(output_path_passed, index=False)
    pd.DataFrame(failed, columns=["r:diaObjectId"]).to_csv(output_path_failed, index=False)

    return passed, failed


if __name__ == "__main__":

    # candidates = pd.read_csv(Path.home() / "Documents"/"University"/"Honours"/"Honours_Project"/"data"/"test_data"/"20_test_objects.csv", dtype={"r:diaObjectId":str})["r:diaObjectId"].to_list()
    # applying_the_rate_of_rise(candidates, 500, TEST_PASSED_DIR, TEST_FAILED_DIR)
    candidate = ["170235219334922248"]
    applying_the_rate_of_rise(candidate, 500)