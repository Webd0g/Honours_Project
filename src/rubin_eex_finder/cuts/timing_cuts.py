"""Here we make the cut based on teh consistent timing between the peak flux points for the objects.
The cut is made in Bazin et al., 10009 and after checking for significant flux variation in the object, the r and i band peaks
should be reletivly close to one another."""

from pathlib import Path

import polars as pl


def loading_objects_and_light_data(
    object_path: Path, light_curve_data_path: Path, output_path: Path
) -> pl.DataFrame:
    """Loads the objects and the light curve data for the objects, then finds the peak rows in the i and r bands and finds the difference between the two in mjd and checks if they are less then 50 days apart or more than 50 days apart

    Args:
        object_path (Path): Path to the CSV of objects to test.
        light_curve_data_path (Path): Path to where the light curve data for all objects is stored as a parquet file.
        output_path (Path): Output path.

    Returns:
        pl.DataFrame: Returns a polars dataframe with the new delta peak column and passes or not column.
    """

    objects = pl.read_csv(
        object_path,
        columns=[
            "r:diaObjectId",
        ],
        schema_overrides={
            "r:diaObjectId": pl.String,
        },
    )
    pl.Config.set_tbl_rows(-1)

    light_curve_data = pl.read_parquet(light_curve_data_path).with_columns(
        pl.col("r:diaObjectId").cast(pl.String),
    )

    light_curve_data = light_curve_data.join(
        objects,
        on="r:diaObjectId",
        how="inner",
    )

    groups = {
        key[0]: group_df.sort("r:midpointMjdTai")
        for key, group_df in light_curve_data.group_by(
            "r:diaObjectId",
            maintain_order=True,
        )
    }

    results = {
        "r:diaObjectId": [],
        "i_peak_mjd": [],
        "r_peak_mjd": [],
        "delta_peak_mjd": [],
        "passes_peak_cut": [],
    }

    for object_id, group_df in groups.items():
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
        pl.col("passes_peak_cut"),
    )

    timing_cuts_values_fail = timing_cuts_values.filter(
        ~pl.col("passes_peak_cut"),
    )

    timing_cuts_values_pass.write_csv(output_path_pass)

    timing_cuts_values_fail.write_csv(output_path_fail)


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data" / "forced_photometry_test"

    IMAGE_DIR = PROJECT_DIR / "images"

    LIGHT_CURVE_DATA_DIR = (
        DATA_DIR / "light_curve_data" / "forced_photometry_light_curve_data.parquet"
    )

    OBJECTS = DATA_DIR / "pre_maximum_slope_fit_results" / "pass.csv"

    PASS = DATA_DIR / "peak_timing_cut_results"/"pass.csv"

    FAIL = DATA_DIR / "peak_timing_cut_results" / "fail.csv"

    VALUES = DATA_DIR / "peak_timing_cut_results" / "values.csv"

    loading_objects_and_light_data(
        object_path=OBJECTS,
        light_curve_data_path=LIGHT_CURVE_DATA_DIR,
        output_path=VALUES,
    )

    timing_pass_check(
        timing_cuts_values_path=VALUES,
        output_path_pass=PASS,
        output_path_fail=FAIL,
    )