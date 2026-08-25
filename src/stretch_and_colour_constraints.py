from pathlib import Path

import polars as pl


def colour_stretch_cut(
    object_path: Path,
    pass_output_path: Path,
    fail_output_path: Path,
    values_output_path: Path,
) -> pl.DataFrame:
    """This function reads the list of objects that have passed the phase cuts and the chi squared cuts and then finds the objects that have colour and srtetch parameters within the standard colour and stretch ranges for type Ia SN.

    Args:
        object_path (Path): Path object of the path to the CSV file that hosts the diaObjectId and the SALT fit results for each object.
        output_path (Path): Path object that is the path the user wants the CSV output to be.

    Returns:
        pl.DataFrame: returns a dataframe of the CSV that is saved.
    """

    objects = pl.read_csv(object_path, schema_overrides={"r:diaObjectId": pl.String})

    # Making standard colour cut for normal Ias
    colour_cut_df = objects.with_columns(
        pl.col("c").is_between(-0.3, 0.3).alias("pass_colour_cut")
    )

    # Making the standard stretch cut for normal Ias
    stretch_cut_df = colour_cut_df.with_columns(
        pl.col("x1").is_between(-3.0, 3.0).alias("pass_stretch_cut")
    )

    # Separating the passing objects and failing objects and saving them.
    stretch_cut_df.filter(
        pl.col("pass_colour_cut") & pl.col("pass_stretch_cut")
    ).write_csv(pass_output_path)

    stretch_cut_df.filter(
        ~pl.col("pass_colour_cut") | ~pl.col("pass_stretch_cut")
    ).write_csv(fail_output_path)

    stretch_cut_df.write_csv(values_output_path)

    return stretch_cut_df


if __name__ == "__main__":
    TEST_OBJECT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "50_object_test"
        / "chi2_requirements_results"
        / "pass.csv"
    )

    TEST_PASS_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "50_object_test"
        / "colour_stretch_results"
        / "pass.csv"
    )

    TEST_FAIL_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "50_object_test"
        / "colour_stretch_results"
        / "fail.csv"
    )

    TEST_VALUES_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "50_object_test"
        / "colour_stretch_results"
        / "values.csv"
    )

    colour_stretch_cut(
        TEST_OBJECT_PATH,
        TEST_PASS_OUTPUT_PATH,
        TEST_FAIL_OUTPUT_PATH,
        TEST_VALUES_OUTPUT_PATH,
    )
