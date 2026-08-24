from pathlib import Path

import polars as pl


def colour_stretch_cut(object_path: Path, output_path: Path) -> pl.DataFrame:
    """This function reads the list of objects that have passed the 

    Args:
        object_path (Path): _description_
        output_path (Path): _description_

    Returns:
        pl.DataFrame: _description_
    """