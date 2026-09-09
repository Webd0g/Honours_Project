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

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from light_curve_plotter import collect_light_curve_data
from light_curve_utils import add_night_and_SNR_columns
from plotting_lightcurves import batch_plotter

# Directories for the objects that passed the first filtering done through the tags, second directory is for the output of the rate of rise cut.
TEST_DIR = (
    Path.home()
    / "Documents"
    / "University"
    / "Honours"
    / "Honours_Project"
    / "data"
    / "test_data"
)
PLOTS_DIR = (
    Path.home()
    / "Documents"
    / "University"
    / "Honours"
    / "Honours_Project"
    / "images"
    / "test_data_images"
)
TEST_PASSED_DIR = TEST_DIR / "test_passed.csv"
TEST_FAILED_DIR = TEST_DIR / "test_failed.csv"
FAILED_DIR = (
    Path.home()
    / "Documents"
    / "University"
    / "Honours"
    / "Honours_Project"
    / "data"
    / "slope_fitting_results"
    / "objects_failing_slope_fit.csv"
)
PASSED_DIR = (
    Path.home()
    / "Documents"
    / "University"
    / "Honours"
    / "Honours_Project"
    / "data"
    / "slope_fitting_results"
    / "objects_passing_slope_fit.csv"
)
# Reading the csv for the objects that passed the Fink tag filtering and creating the CSVs where data will be stored.

# Specifying how many objects go into each query to the API.
CHUNK_SIZE = 500

# Minimum number of pre-peak points needed in a band before a slope is even attempted.
# np.polyfit needs at least 2 points to fit a line; kept low deliberately while we test
# the loosened nonzero-slope criterion, per today's discussion with Anais.
MIN_PRE_PEAK_POINTS = 3

# Minimum time baseline (in days) spanned by the pre-peak points used in a fit.
# Points clustered within a very short window can produce wildly unstable slopes,
# since the denominator of the fit (variance in time) collapses toward zero even
# with many points - this was seen directly in some objects showing slopes of 10-100+
# despite having 10-20+ pre-peak detections, all crammed into a single night or two.
MIN_PRE_PEAK_TIME_SPAN_DAYS = 5

# This is so we remove spurious and noisey data that skews the results of the fit process too much
MIN_PEAK_SNR = 10


def collect_light_curve_info_with_fallback(
    candidates: list[str],
) -> pd.DataFrame | None:
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
        chunked_ids = candidates[diaObjectId : diaObjectId + chunk_size]
        candidates_light_curve_data = collect_light_curve_info_with_fallback(
            chunked_ids
        )
        if candidates_light_curve_data is not None:
            candidate_results.append(candidates_light_curve_data)

    if not candidate_results:
        raise RuntimeError(
            f"No light curve data returned for candidate batch of size {len(candidates)}"
        )

    return pd.concat(candidate_results, ignore_index=True)


def slope_calculation(
    group_df: pl.DataFrame,
    band: str,
    min_points: int = MIN_PRE_PEAK_POINTS,
    min_time_span: int = MIN_PRE_PEAK_TIME_SPAN_DAYS,
    min_peak_snr: int = MIN_PEAK_SNR,
) -> tuple[float, int] | None:
    """This function finds the slope of the band of the lightcurve prepeak by fitting a

    polynomial of degree 1 to the points of the light curve before the brightest point
    of the light curve. Returns None if there isn't enough pre-peak data in this band
    to attempt a fit.
    """
    band_df = group_df.filter((pl.col("r:band") == band) & (pl.col("r:psfFlux") > 0))
    if band_df.is_empty():
        return None

    peak_row = band_df.filter(pl.col("r:psfFlux") == pl.col("r:psfFlux").max()).row(
        0, named=True
    )

    peak_flux = peak_row["r:psfFlux"]
    peak_flux_err = peak_row["r:psfFluxErr"]
    peak_mjd = peak_row["r:midpointMjdTai"]

    peak_snr = peak_flux / peak_flux_err
    if peak_snr < min_peak_snr:
        return None

    pre_peak = band_df.filter(pl.col("r:midpointMjdTai") <= peak_mjd)

    pre_peak = add_night_and_SNR_columns(pre_peak)

    pre_peak_best_points = (
        pre_peak.sort(by="SNR", descending=True)
        .group_by(["r:diaObjectId", "r:band", "night"])
        .first()
    )

    if len(pre_peak_best_points) < min_points:
        return None

    min_mjd = pre_peak_best_points["r:midpointMjdTai"].min()
    time_span = peak_mjd - min_mjd
    if time_span < min_time_span:
        return None

    mjd_vec = pre_peak_best_points["r:midpointMjdTai"].to_numpy()
    norm_flux = (pre_peak_best_points["r:psfFlux"] / peak_flux).to_numpy()
    norm_err = (pre_peak_best_points["r:psfFluxErr"] / peak_flux).to_numpy()

    norm_err = np.clip(norm_err, a_min=1e-6, a_max=None)
    weights = 1 / norm_err

    slope, _ = np.polyfit(mjd_vec, norm_flux, 1, w=weights)

    return float(slope), len(pre_peak_best_points)


def compute_all_band_slopes(
    group_df: pd.DataFrame, min_points: int = MIN_PRE_PEAK_POINTS
) -> dict[str, tuple[float, int]]:
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


def check_rate_of_rise(
    group_df: pd.DataFrame, min_points: int = MIN_PRE_PEAK_POINTS
) -> bool:
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


def applying_the_rate_of_rise(
    object_csv_path: Path,
    light_curve_data_path: Path,
    output_path_passed: Path | None = None,
    output_path_failed: Path | None = None,
) -> tuple[list[str], list[str]]:
    """
    This function pulls the light curve data for the specified list of object ids and
    the chunk size can be customised if you have a large list of objects or small. The
    function then checks whether the object shows a nonzero pre-peak rise in at least
    one band and directs the objects to their respective lists. Finally this function
    produces two CSV files of the objects that passed and ones that failed.
    """
    passed, failed = [], []

    objects = pl.read_csv(
        object_csv_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    )["r:diaObjectId"].to_list()
    light_curve_data = pl.read_parquet(light_curve_data_path)
    light_curve_data_filtered = light_curve_data.filter(
        pl.col("r:diaObjectId").cast(pl.String).is_in(objects)
    )

    for object_id, group_df in light_curve_data_filtered.group_by("r:diaObjectId"):
        if check_rate_of_rise(group_df):
            passed.append(object_id)
        else:
            failed.append(object_id)

    pl.DataFrame(passed, schema=["r:diaObjectId"]).write_csv(output_path_passed)
    pl.DataFrame(failed, schema=["r:diaObjectId"]).write_csv(output_path_failed)

    return passed, failed

def bar_chart_pass_fractions()

if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    LIGHT_CURVE_DATA_DIR = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    IA_OBJECT_PATH = (
        DATA_DIR / "salt_fitting" / "confirmed_ia_tests" / "confirmed_ia.csv"
    )

    II_OBJECT_PATH = (
        DATA_DIR / "salt_fitting" / "core_collapse_results" / "confirmed_CC_SN_tns.csv"
    )

    II_PASS_OUTPUT_PATH = (
        DATA_DIR / "sn_ii_test_results" / "rate_of_rise_results" / "pass.csv"
    )

    II_FAIL_OUTPUT_PATH = (
        DATA_DIR / "sn_ii_test_results" / "rate_of_rise_results" / "fail.csv"
    )

    II_PASS_PLOT_OUTPUT = (
        IMAGE_DIR / "sn_ii_light_curves" / "rate_of_rise_results" / "pass"
    )

    II_FAIL_PLOT_OUTPUT = (
        IMAGE_DIR / "sn_ii_light_curves" / "rate_of_rise_results" / "fail"
    )

    # applying_the_rate_of_rise(
    #     object_csv_path=II_OBJECT_PATH,
    #     light_curve_data_path=LIGHT_CURVE_DATA_DIR,
    #     output_path_passed=II_PASS_OUTPUT_PATH,
    #     output_path_failed=II_FAIL_OUTPUT_PATH,
    # )

    batch_plotter(
        object_path=II_FAIL_OUTPUT_PATH,
        light_curve_data_path=LIGHT_CURVE_DATA_DIR,
        output_path=II_FAIL_PLOT_OUTPUT,
    )
