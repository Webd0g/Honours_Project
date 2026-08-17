"""Here we have our sampling cut where we first will test the number of epochs across each band our SN-like objects have. 
then check the number of epochs the contaminants have and find a sampling value to separate those objects in both the pre-peak phase and the post-peak phase."""

import polars as pl
from pathlib import Path
import numpy as np

def numb_pre_peak_post_peak(objects_path: Path, light_curve_data_path: Path) -> pl.DataFrame:
    """This function finds the number of data points that pre-peak 
    and post-peak within 30 days before the peak and 60 days after the peak point."""

    results = {"r:diaObjectId": [], 
            "u pre" : [], 
            "u post": [], 
            "g pre" : [],
            "g post": [],
            "r pre": [],
            "r post": [],
            "i pre": [],
            "i post": [],
            "z pre": [],
            "z post": [],
            "y pre": [],
            "y post": []}

    object_ids = pl.read_csv(objects_path, schema_overrides={"r:diaObjectId": pl.String}, columns=["r:diaObjectId"])["r:diaObjectId"].to_list()

    light_curve_data = (pl.read_parquet(light_curve_data_path).with_columns(pl.col("r:diaObjectId").cast(pl.String)))
    light_curve_data_filtered = light_curve_data.filter(pl.col("r:diaObjectId").is_in(object_ids))

    groups = {key[0]: group_df for key, group_df in light_curve_data_filtered.group_by("r:diaObjectId",maintain_order=True)}

    for object_id in object_ids:
    
        group_df = groups.get(object_id)

        if group_df is None:
            continue

        peak_rows = (group_df.filter(pl.col("r:psfFlux") > 0).sort("r:psfFlux", descending=True).unique(subset=["r:band"], keep="first"))

        u_peak_row = peak_rows.filter(pl.col("r:band") == "u")
        g_peak_row = peak_rows.filter(pl.col("r:band") == "g")
        r_peak_row = peak_rows.filter(pl.col("r:band") == "r")
        i_peak_row = peak_rows.filter(pl.col("r:band") == "i")
        z_peak_row = peak_rows.filter(pl.col("r:band") == "z")
        y_peak_row = peak_rows.filter(pl.col("r:band") == "y")

        if u_peak_row.height == 0:
            print (f"Object {object_id} doesn't have any u band data")
            u_pre_peak = 0
            u_post_peak = 0
        else:
            u_peak_time = peak_rows.filter(pl.col("r:band") == "u")["r:midpointMjdTai"].item()
            u_pre_peak = group_df.filter((pl.col("r:band") == "u") & (pl.col("r:midpointMjdTai") >= u_peak_time - 30)  &  (pl.col("r:midpointMjdTai") < u_peak_time)).height
            u_post_peak = group_df.filter((pl.col("r:midpointMjdTai") <= u_peak_time + 60)  &  (pl.col("r:midpointMjdTai") > u_peak_time)).height

        if g_peak_row.height == 0:
            print (f"Object {object_id} doesn't have any g band data")
            g_pre_peak = 0
            g_post_peak = 0
        else:
            g_peak_time = peak_rows.filter(pl.col("r:band") == "g")["r:midpointMjdTai"].item()
            g_pre_peak = group_df.filter((pl.col("r:band") == "g") & (pl.col("r:midpointMjdTai") >= g_peak_time - 30)  &  (pl.col("r:midpointMjdTai") < g_peak_time)).height
            g_post_peak = group_df.filter((pl.col("r:midpointMjdTai") <= g_peak_time + 60)  &  (pl.col("r:midpointMjdTai") > g_peak_time)).height

        if r_peak_row.height == 0:
            print (f"Object {object_id} doesn't have any r band data")
            r_pre_peak = 0
            r_post_peak = 0
        else:
            r_peak_time = peak_rows.filter(pl.col("r:band") == "r")["r:midpointMjdTai"].item()
            r_pre_peak = group_df.filter((pl.col("r:band") == "r") & (pl.col("r:midpointMjdTai") >= r_peak_time - 30)  &  (pl.col("r:midpointMjdTai") < r_peak_time)).height
            r_post_peak = group_df.filter((pl.col("r:band") == "r") & (pl.col("r:midpointMjdTai") <= r_peak_time + 60)  &  (pl.col("r:midpointMjdTai") > r_peak_time)).height

        if i_peak_row.height == 0:
            print (f"Object {object_id} doesn't have any i band data")
            i_pre_peak = 0 
            i_post_peak = 0
        else:
            i_peak_time = peak_rows.filter(pl.col("r:band") == "i")["r:midpointMjdTai"].item()
            i_pre_peak = group_df.filter((pl.col("r:band") == "i") & (pl.col("r:midpointMjdTai") >= i_peak_time - 30)  &  (pl.col("r:midpointMjdTai") < i_peak_time)).height
            i_post_peak = group_df.filter((pl.col("r:band") == "i") & (pl.col("r:midpointMjdTai") <= i_peak_time + 60)  &  (pl.col("r:midpointMjdTai") > i_peak_time)).height

        if z_peak_row.height == 0:
            print (f"Object {object_id} doesn't have any z band data")
            z_pre_peak = 0
            z_post_peak = 0
        else:
            z_peak_time = peak_rows.filter(pl.col("r:band") == "z")["r:midpointMjdTai"].item()
            z_pre_peak = group_df.filter((pl.col("r:band") == "z") & (pl.col("r:midpointMjdTai") >= z_peak_time - 30)  &  (pl.col("r:midpointMjdTai") < z_peak_time)).height
            z_post_peak = group_df.filter((pl.col("r:band") == "z") & (pl.col("r:midpointMjdTai") <= z_peak_time + 60)  &  (pl.col("r:midpointMjdTai") > z_peak_time)).height

        if y_peak_row.height == 0:
            print (f"Object {object_id} doesn't have any y band data")
            y_pre_peak = 0
            y_post_peak = 0
        else:
            y_peak_time = peak_rows.filter(pl.col("r:band") == "y")["r:midpointMjdTai"].item()
            y_pre_peak = group_df.filter((pl.col("r:band") == "y") & (pl.col("r:midpointMjdTai") >= y_peak_time - 30)  &  (pl.col("r:midpointMjdTai") < y_peak_time)).height
            y_post_peak = group_df.filter((pl.col("r:band") == "y") & (pl.col("r:midpointMjdTai") <= y_peak_time + 60)  &  (pl.col("r:midpointMjdTai") > y_peak_time)).height

        results["r:diaObjectId"].append(object_id)
        results["u pre"].append(u_pre_peak)
        results["u post"].append(u_post_peak)
        results["g pre"].append(g_pre_peak)
        results["g post"].append(g_post_peak)
        results["r pre"].append(r_pre_peak)
        results["r post"].append(r_post_peak)
        results["i pre"].append(i_pre_peak)
        results["i post"].append(i_post_peak)
        results["z pre"].append(z_pre_peak)
        results["z post"].append(z_post_peak)
        results["y pre"].append(y_pre_peak)
        results["y post"].append(y_post_peak)

    return pl.DataFrame(results)

if __name__ == "__main__":

    TEST_CANDIDATES = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "test_data" / "sampling_tests" / "20_object_sampling_test.csv"

    LIGHT_CURVE_DATA = Path.home() / "Documents"/"University"/"Honours"/"Honours_Project"/"data"/ "light_curve_data" / "combined_light_curve_data.parquet"

    print(numb_pre_peak_post_peak(TEST_CANDIDATES, LIGHT_CURVE_DATA))