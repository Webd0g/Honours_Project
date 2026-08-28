"""Here we make the cut based on teh consistent timing between the peak flux points for the objects.
The cut is made in Bazin et al., 2009 and after checking for significant flux variation in the object, the r and i band peaks
should be reletivly close to one another."""

from pathlib import Path

import polars as pl


def loading_objects_and_light_data(
    object_path: Path, light_curve_data_path: Path, output_path: Path
) -> pl.DataFrame:
    """Here we load the object list from the rate_of_rise fit script"""

    objects = pl.read_csv(
        object_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    )["r:diaObjectId"].to_list()

    light_curve_data = pl.read_parquet(light_curve_data_path).with_columns(
        pl.col("r:diaObjectId").cast(pl.String)
    )

    light_curve_data = light_curve_data.filter(pl.col("r:diaObjectId").is_in(objects))

    groups = {
        key[0]: group_df
        for key, group_df in light_curve_data.group_by(
            "r:diaObjectId", maintain_order=True
        )
    }

    results = {
        "r:diaObjectId": [],
        "i_peak_mjd": [],
        "r_peak_mjd": [],
        "delta_peak_mjd": [],
        "passes_peak_cut": [],
    }

    for object_id in objects:
        group_df = groups.get(object_id)

        if group_df is None:
            continue

        band_df = group_df.filter(
            (pl.col("r:band").is_in(["i", "r"])) & (pl.col("r:psfFlux") > 0)
        )

        peak_rows = (
            band_df.filter(pl.col("r:band").is_in(["i", "r"]))
            .sort("r:psfFlux", descending=True)
            .unique(subset=["r:diaObjectId", "r:band"], keep="first")
        )

        i_peak_rows = peak_rows.filter(pl.col("r:band") == "i")
        r_peak_rows = peak_rows.filter(pl.col("r:band") == "r")

        if i_peak_rows.height == 0 or r_peak_rows.height == 0:
            results["r:diaObjectId"].append(object_id)
            results["i_peak_mjd"].append("null")
            results["r_peak_mjd"].append("null")
            results["delta_peak_mjd"].append("null")
            results["passes_peak_cut"].append("false")
            continue

        i_peak_mjd = i_peak_rows["r:midpointMjdTai"].item()
        r_peak_mjd = r_peak_rows["r:midpointMjdTai"].item()

        delta_peak_mjd = abs(r_peak_mjd - i_peak_mjd)

        results["r:diaObjectId"].append(object_id)
        results["i_peak_mjd"].append(i_peak_mjd)
        results["r_peak_mjd"].append(r_peak_mjd)
        results["delta_peak_mjd"].append(delta_peak_mjd)

        if delta_peak_mjd < 50:
            results["passes_peak_cut"].append("true")

        else:
            results["passes_peak_cut"].append("false")

    results = pl.DataFrame(results, strict=False)

    results.write_csv(output_path)

    return results


def timing_pass_check(
    timing_cuts_values_path: Path, output_path_pass: Path, output_path_fail: Path
) -> None:
    """Here we check the objects that passed or failed the timing check and sort them into
    their respective CSV files."""

    timing_cuts_values = pl.read_csv(timing_cuts_values_path)

    timing_cuts_values_pass = timing_cuts_values.filter(
        pl.col("passes_peak_cut") == "true"
    )

    timing_cuts_values_fail = timing_cuts_values.filter(
        pl.col("passes_peak_cut") == "false"
    )

    timing_cuts_values_pass.write_csv(output_path_pass)

    timing_cuts_values_fail.write_csv(output_path_fail)


if __name__ == "__main__":
    TEST_DATA = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "test_data"
        / "20_test_objects.csv"
    )

    LIGHT_CURVE_DATA = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "light_curve_data"
        / "combined_light_curve_data.parquet"
    )

    OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "timing_cuts_results"
        / "timing_cuts_values.csv"
    )

    OUTPUT_PATH_PASS = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "timing_cuts_results"
        / "timing_cuts_passed.csv"
    )

    OUTPUT_PATH_FAIL = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "timing_cuts_results"
        / "timing_cuts_failed.csv"
    )

    CANDIDATES = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "sn_ia_test_results"
        / "confirmed_ia.csv"
    )

    loading_objects_and_light_data(CANDIDATES, LIGHT_CURVE_DATA, "/home/riley/Documents/University/Honours/Honours_Project/data/sn_ia_test_results/timing_cuts_results/values.csv")

    timing_pass_check("/home/riley/Documents/University/Honours/Honours_Project/data/sn_ia_test_results/timing_cuts_results/values.csv", "/home/riley/Documents/University/Honours/Honours_Project/data/sn_ia_test_results/timing_cuts_results/pass.csv", "/home/riley/Documents/University/Honours/Honours_Project/data/sn_ia_test_results/timing_cuts_results/fail.csv")
