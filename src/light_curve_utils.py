import polars as pl


def add_night_and_SNR_columns(band_df: pl.DataFrame) -> pl.DataFrame:
    """
    Takes the dataframe of all the bands data and filters it further to
    one point per night of observation, with that point being the
    point with the highest SNR.

    This is so that when the slope fitting happens it doesn't fit to
    points that have been clustered and ensures that the points
    that are being fit to are actually changing in time.
    """

    band_df = band_df.with_columns(
        (pl.col("r:midpointMjdTai").cast(int)).alias("night")
    )
    band_df = band_df.with_columns(
        (pl.col("r:psfFlux") / pl.col("r:psfFluxErr")).alias("SNR")
    )

    return band_df