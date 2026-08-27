#!/usr/bin/env python3

from pathlib import Path

import polars as pl

from light_curve_plotter import lightcurve_plotting


def batch_plotter(object_path: Path, light_curve_data_path: Path, output_path: Path):
    """_summary_

    Args:
        object_ids (_type_): _description_
        output_path (Path): _description_
        chunk_size (int, optional): _description_. Defaults to 500.
    """
    object_ids = pl.read_csv(
        object_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    )["r:diaObjectId"].to_list()

    light_curve_data = pl.read_parquet(light_curve_data_path).with_columns(
        pl.col("r:diaObjectId").cast(pl.String)
    )
    light_curve_data_filtered = light_curve_data.filter(
        pl.col("r:diaObjectId").is_in(object_ids)
    )

    groups = {
        key[0]: group_df
        for key, group_df in light_curve_data_filtered.group_by(
            "r:diaObjectId", maintain_order=True
        )
    }

    for object_id in object_ids:
        group_df = groups.get(object_id).filter(pl.col("r:psfFlux") > 0)

        if group_df is None:
            print(f"Object {object_id} does not have any light curve data.")
            continue

        lightcurve_plotting(group_df, object_id, output_path)


if __name__ == "__main__":
    CANDIDATES_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "sampling_results"
        / "sampling_pass.csv"
    )

    LIGHT_CURVE_DATA_PATH = (
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
        / "images"
        / "sampling_plots"
        / "pass"
    )

    batch_plotter(CANDIDATES_PATH, LIGHT_CURVE_DATA_PATH, OUTPUT_PATH)
