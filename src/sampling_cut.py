"""Here we have our sampling cut where we first will test the number of epochs across each band our SN-like objects have.
then check the number of epochs the contaminants have and find a sampling value to separate those objects in both the pre-peak phase and the post-peak phase.
"""

from pathlib import Path

import polars as pl


def numb_pre_peak_post_peak(
    objects_path: Path, light_curve_data_path: Path, output_path: Path
) -> pl.DataFrame:
    """This function finds the number of data points that pre-peak
    and post-peak within 30 days before the peak and 60 days after the peak point."""

    results = {
        "r:diaObjectId": [],
        "u pre": [],
        "u post": [],
        "g total": [],
        "r pre": [],
        "r post": [],
        "i pre": [],
        "i post": [],
        "z total": [],
        "y pre": [],
        "y post": [],
    }

    object_ids = pl.read_csv(
        objects_path,
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

        group_df = groups.get(object_id)

        if group_df is None:
            continue

        group_df = group_df.filter(pl.col("r:psfFlux") > 0)

        peak_rows = (
            group_df.filter(pl.col("r:psfFlux") > 0)
            .sort("r:psfFlux", descending=True)
            .unique(subset=["r:band"], keep="first")
        )

        u_peak_row = peak_rows.filter(pl.col("r:band") == "u")
        g_peak_row = peak_rows.filter(pl.col("r:band") == "g")
        r_peak_row = peak_rows.filter(pl.col("r:band") == "r")
        i_peak_row = peak_rows.filter(pl.col("r:band") == "i")
        z_peak_row = peak_rows.filter(pl.col("r:band") == "z")
        y_peak_row = peak_rows.filter(pl.col("r:band") == "y")

        if u_peak_row.height == 0:
            print(f"Object {object_id} doesn't have any u band data")
            u_pre_peak = 0
            u_post_peak = 0
        else:
            u_peak_time = peak_rows.filter(pl.col("r:band") == "u")[
                "r:midpointMjdTai"
            ].item()
            u_pre_peak = group_df.filter(
                (pl.col("r:band") == "u")
                & (pl.col("r:midpointMjdTai") >= u_peak_time - 30)
                & (pl.col("r:midpointMjdTai") < u_peak_time)
            ).height
            u_post_peak = group_df.filter(
                (pl.col("r:band") == "u")
                & (pl.col("r:midpointMjdTai") <= u_peak_time + 60)
                & (pl.col("r:midpointMjdTai") > u_peak_time)
            ).height

        if g_peak_row.height == 0:
            print(f"Object {object_id} doesn't have any g band data")
            g_pre_peak = 0
            g_post_peak = 0
        else:
            i_peak_time = peak_rows.filter(pl.col("r:band") == "i")[
                "r:midpointMjdTai"
            ].item()
            g_pre_peak = group_df.filter(
                (pl.col("r:band") == "g")
                & (pl.col("r:midpointMjdTai") >= i_peak_time - 30)
                & (pl.col("r:midpointMjdTai") < i_peak_time)
            ).height
            g_post_peak = group_df.filter(
                (pl.col("r:band") == "g")
                & (pl.col("r:midpointMjdTai") <= i_peak_time + 60)
                & (pl.col("r:midpointMjdTai") > i_peak_time)
            ).height

        if r_peak_row.height == 0:
            print(f"Object {object_id} doesn't have any r band data")
            r_pre_peak = 0
            r_post_peak = 0
        else:
            r_peak_time = peak_rows.filter(pl.col("r:band") == "r")[
                "r:midpointMjdTai"
            ].item()
            r_pre_peak = group_df.filter(
                (pl.col("r:band") == "r")
                & (pl.col("r:midpointMjdTai") >= r_peak_time - 30)
                & (pl.col("r:midpointMjdTai") < r_peak_time)
            ).height
            r_post_peak = group_df.filter(
                (pl.col("r:band") == "r")
                & (pl.col("r:midpointMjdTai") <= r_peak_time + 60)
                & (pl.col("r:midpointMjdTai") > r_peak_time)
            ).height

        if i_peak_row.height == 0:
            print(f"Object {object_id} doesn't have any i band data")
            i_pre_peak = 0
            i_post_peak = 0
        else:
            i_peak_time = peak_rows.filter(pl.col("r:band") == "i")[
                "r:midpointMjdTai"
            ].item()
            i_pre_peak = group_df.filter(
                (pl.col("r:band") == "i")
                & (pl.col("r:midpointMjdTai") >= i_peak_time - 30)
                & (pl.col("r:midpointMjdTai") < i_peak_time)
            ).n_unique(subset=["r:diaSourceId"])
            i_post_peak = group_df.filter(
                (pl.col("r:band") == "i")
                & (pl.col("r:midpointMjdTai") <= i_peak_time + 60)
                & (pl.col("r:midpointMjdTai") > i_peak_time)
            ).n_unique(subset=["r:diaSourceId"])
            # print (i_pre_peak)
            # print(i_post_peak)

        if z_peak_row.height == 0:
            print(f"Object {object_id} doesn't have any z band data")
            z_pre_peak = 0
            z_post_peak = 0
        else:
            i_peak_time = peak_rows.filter(pl.col("r:band") == "i")[
                "r:midpointMjdTai"
            ].item()
            z_pre_peak = group_df.filter(
                (pl.col("r:band") == "z")
                & (pl.col("r:midpointMjdTai") >= i_peak_time - 30)
                & (pl.col("r:midpointMjdTai") < i_peak_time)
            ).height
            z_post_peak = group_df.filter(
                (pl.col("r:band") == "z")
                & (pl.col("r:midpointMjdTai") <= i_peak_time + 60)
                & (pl.col("r:midpointMjdTai") > i_peak_time)
            ).height

        if y_peak_row.height == 0:
            print(f"Object {object_id} doesn't have any y band data")
            y_pre_peak = 0
            y_post_peak = 0
        else:
            y_peak_time = peak_rows.filter(pl.col("r:band") == "y")[
                "r:midpointMjdTai"
            ].item()
            y_pre_peak = group_df.filter(
                (pl.col("r:band") == "y")
                & (pl.col("r:midpointMjdTai") >= y_peak_time - 30)
                & (pl.col("r:midpointMjdTai") < y_peak_time)
            ).height
            y_post_peak = group_df.filter(
                (pl.col("r:band") == "y")
                & (pl.col("r:midpointMjdTai") <= y_peak_time + 60)
                & (pl.col("r:midpointMjdTai") > y_peak_time)
            ).height

        results["r:diaObjectId"].append(object_id)
        results["u pre"].append(u_pre_peak)
        results["u post"].append(u_post_peak)
        results["g total"].append(g_pre_peak + g_post_peak)
        results["r pre"].append(r_pre_peak)
        results["r post"].append(r_post_peak)
        results["i pre"].append(i_pre_peak)
        results["i post"].append(i_post_peak)
        results["z total"].append(z_pre_peak + z_post_peak)
        results["y pre"].append(y_pre_peak)
        results["y post"].append(y_post_peak)

    results_df = pl.DataFrame(results)
    results_df.write_csv(output_path)
    return results_df


def sampling_statistics(counts_path: Path) -> None:
    """Here we calculate the statistics of the sample upon each test.
    Specifically we find the mean and quartiles for the bands."""

    counts_df = pl.read_csv(counts_path)

    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)

    sn_counts_df = counts_df.head(10)
    contaminants_counts_df = counts_df.tail(10)

    print(sn_counts_df.drop("r:diaObjectId").describe([0.25, 0.5, 0.75]))

    print(contaminants_counts_df.drop("r:diaObjectId").describe([0.25, 0.5, 0.75]))


def pass_fractions(counts_path: Path) -> None:
    """_summary_

    Args:
        counts_path (Path): _description_
    """

    counts_df = pl.read_csv(counts_path)

    sn_counts_df = counts_df.head(50)
    contaminants_counts_df = counts_df.tail(50)

    first_criterion = (
        (pl.col("i pre") >= 1)
        & (pl.col("i post") >= 1)
        & (pl.col("r pre") >= 1)
        & (pl.col("r post") >= 1)
    )

    sn_pass_1 = sn_counts_df.filter(first_criterion)
    cont_pass_1 = contaminants_counts_df.filter(first_criterion)

    print(f"SN-like: {sn_pass_1.height}/{sn_counts_df.height}")
    print(f"Contaminants: {cont_pass_1.height}/{contaminants_counts_df.height}")

    second_criterion = (pl.col("g total") >= 2) & (pl.col("z total") >= 2)

    sn_pass_2 = sn_pass_1.filter(second_criterion)
    cont_pass_2 = cont_pass_1.filter(second_criterion)

    # print(sn_pass_1)
    # print(sn_pass_2)
    print(f"SN-like: {sn_pass_2.height}/{sn_pass_1.height}")
    print(f"Contaminants: {cont_pass_2.height}/{cont_pass_1.height}")


def sampling_cut(
    candidate_path: Path,
    light_curve_data_path: Path,
    count_output_path: Path,
    pass_output_path: Path,
    fail_output_path: Path,
) -> None:
    """_summary_

    Args:
        candidate_path (Path): _description_
        light_curve_data_path (Path): _description_
        count_output_path (Path): _description_
        cut_output_path (Path): _description_
    """
    sampling_counts_df = numb_pre_peak_post_peak(
        candidate_path, light_curve_data_path, count_output_path
    )

    sampling_counts_df = sampling_counts_df.with_columns(
        (pl.col("i pre") >= 1).alias("pass_i_pre"),
        (pl.col("i post") >= 1).alias("pass_i_post"),
        (pl.col("r pre") >= 1).alias("pass_r_pre"),
        (pl.col("r post") >= 1).alias("pass_r_post"),
        (pl.col("g total") >= 0).alias("pass_g"),
        (pl.col("z total") >= 0).alias("pass_z"),
    )

    sampling_counts_df = sampling_counts_df.with_columns(
        (
            pl.col("pass_i_pre")
            & pl.col("pass_i_post")
            & pl.col("pass_r_pre")
            & pl.col("pass_r_post")
            & pl.col("pass_g")
            & pl.col("pass_z")
        ).alias("sampling_pass")
    )

    sampling_counts_df = sampling_counts_df.with_columns(
        pl.sum_horizontal(
            ~pl.col("pass_i_pre"),
            ~pl.col("pass_i_post"),
            ~pl.col("pass_r_pre"),
            ~pl.col("pass_r_post"),
            ~pl.col("pass_g"),
            ~pl.col("pass_z"),
        ).alias("number_failed")
    )

    for column in [
        "pass_i_pre",
        "pass_i_post",
        "pass_r_pre",
        "pass_r_post",
        "pass_g",
        "pass_z",
    ]:

        failures = sampling_counts_df.filter(~pl.col(column)).height
        print(f"{column}: {failures} failures")

    passing_objects = sampling_counts_df.filter(pl.col("sampling_pass"))

    failing_objects = sampling_counts_df.filter(~pl.col("sampling_pass"))

    passing_objects.write_csv(pass_output_path)

    failing_objects.write_csv(fail_output_path)

    return sampling_counts_df


if __name__ == "__main__":

    TEST_CANDIDATES = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "test_data"
        / "sampling_tests"
        / "100_object_sampling_test.csv"
    )

    CANDIDATES = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "constant_minus_shape_passed.csv"
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

    OBJECT_TEST_OUTPUT = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "sampling_results"
        / "100_object_test"
        / "100_object_test_results.csv"
    )

    COUNT_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "sampling_results"
        / "sampling_values.csv"
    )

    PASSING_SAMPLING_CUT_OUTPUT = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "sampling_results"
        / "sampling_pass.csv"
    )

    FAILING_SAMPLING_CUT_OUTPUT = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "sampling_results"
        / "sampling_fail.csv"
    )

    # numb_pre_peak_post_peak(TEST_CANDIDATES, LIGHT_CURVE_DATA, OBJECT_TEST_OUTPUT)

    # sampling_statistics(OBJECT_TEST_OUTPUT)

    # pass_fractions(OBJECT_TEST_OUTPUT)

    sampling_cut_df = sampling_cut(
        CANDIDATES,
        LIGHT_CURVE_DATA,
        COUNT_OUTPUT_PATH,
        PASSING_SAMPLING_CUT_OUTPUT,
        FAILING_SAMPLING_CUT_OUTPUT,
    ).filter((pl.col("sampling_pass") == "false") & (pl.col("number_failed") <= 2))

    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    print(sampling_cut_df)
