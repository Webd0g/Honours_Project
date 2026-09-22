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
import polars as pl

from rubin_eex_finder.utils.light_curve_utils import add_night_and_SNR_columns

MIN_PRE_PEAK_POINTS = 3
MIN_PRE_PEAK_TIME_SPAN_DAYS = 5
MIN_PEAK_SNR = 10


def slope_calculation(
    group_df: pl.DataFrame,
    band: str,
    min_points: int = MIN_PRE_PEAK_POINTS,
    min_time_span: float = MIN_PRE_PEAK_TIME_SPAN_DAYS,
    min_peak_snr: float = MIN_PEAK_SNR,
) -> tuple[float, int] | None:
    """This function finds the slope of the band of the lightcurve prepeak by fitting a

    polynomial of degree 1 to the points of the light curve before the brightest point
    of the light curve. Returns None if there isn't enough pre-peak data in this band
    to attempt a fit.
    """
    band_df = group_df.filter((pl.col("r:band") == band) & (pl.col("r:psfFlux") > 0))
    if band_df.is_empty():
        return None

    peak_row = band_df.sort(by="r:psfFlux", descending=True)[0]
    peak_flux = peak_row["r:psfFlux"][0]
    peak_flux_err = peak_row["r:psfFluxErr"][0]
    peak_mjd = peak_row["r:midpointMjdTai"][0]

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

    time_span = peak_mjd - pre_peak_best_points["r:midpointMjdTai"].min()
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
    group_df: pl.DataFrame,
    min_points: int = MIN_PRE_PEAK_POINTS,
) -> dict[str, tuple[float, int]]:
    """
    Computes the pre-peak slope for every band present in this object's light curve,
    not just a fixed pair. Returns a dict of band -> (slope, n_points) for every band
    where a slope could be calculated; bands with insufficient pre-peak data are
    omitted rather than included as None, so downstream code doesn't need to filter.
    """
    bands_present = group_df["r:band"].unique().to_list()
    slopes = {}

    for band in bands_present:
        result = slope_calculation(group_df, band, min_points=min_points)
        if result is not None:
            slopes[band] = result

    return slopes


def check_rate_of_rise(
    group_df: pl.DataFrame,
    min_points: int = MIN_PRE_PEAK_POINTS,
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

    passed_df = pl.DataFrame(passed, schema=["r:diaObjectId"]).write_csv(
        output_path_passed
    )
    failed_df = pl.DataFrame(failed, schema=["r:diaObjectId"]).write_csv(
        output_path_failed
    )

    if output_path_passed is not None:
        output_path_passed.parent.mkdir(parents=True, exist_ok=True)
        passed_df.write_csv(output_path_passed)
    if output_path_failed is not None:
        output_path_failed.parent.mkdir(parents=True, exist=True)
        failed_df.write_csv(output_path_failed)

    return passed, failed


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    LIGHT_CURVE_DATA_DIR = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    TEST_OBJECTS = DATA_DIR / "object_id_collection_results" / "all_objects.csv"

    # Example Run
    if TEST_OBJECTS.exists():
        passed_ids, failed_ids = applying_the_rate_of_rise(
            object_csv_path=TEST_OBJECTS,
            light_curve_data_path=LIGHT_CURVE_DATA_DIR,
            # output_path_passed=PASS_OUT,
            # output_path_failed=FAIL_OUT,
        )
        print(f"Passed: {len(passed_ids)} | Failed: {len(failed_ids)}")
