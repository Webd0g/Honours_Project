from pathlib import Path

import polars as pl


def colour_stretch_cut(object_path: Path, output_path: Path) -> pl.DataFrame:
    """This function reads the list of objects that have passed the phase cuts and the chi squared cuts and then finds the objects that have colour and srtetch parameters within the standard colour and stretch ranges for type Ia SN.

    Args:
        object_path (Path): Path object of the path to the CSV file that hosts the diaObjectId and the SALT fit results for each object.
        output_path (Path): Path object that is the path the user wants the CSV output to be.

    Returns:
        pl.DataFrame: returns a dataframe of the CSV that is saved.
    """

    objects = pl.read_csv(object_path, schema_overrides={"r:diaObjectId": pl.String})



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
        / "chi2_requirement_results.csv"
    )

    TEST_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "salt_fitting"
        / "50_object_test"
        / "colour_stretch_results"
    )

    colour_stretch_cut(TEST_OBJECT_PATH, TEST_OUTPUT_PATH)
