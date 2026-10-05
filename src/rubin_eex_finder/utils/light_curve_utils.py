import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests as re
from astropy.time import Time


def nightly_weighted_average(light_curve_data: pl.DataFrame) -> pl.DataFrame:
    """Combine multiple observations from the same night and band using the inverse variance weighting.

    Args:
        light_curve_data (pl.DataFrame): Light curve dataframe after the conversion to rest-frame-phase and removal of epochs earlier than 25 days.

    Returns:
        pl.DataFrame: One row per object, band and observing night containing the weighted neightly flux, uncertainty, representative MJD, rest-frame phase and S/N.
    """

    nightly_data = (
        light_curve_data.with_columns(
            (1.0 / pl.col("r:psfFluxErr").pow(2)).alias("weight"),
            pl.col("r:midpointMjdTai").floor().cast(pl.Int64).alias("night"),
        )
        .group_by(
            [
                "r:diaObjectId",
                "r:band",
                "night",
            ]
        )
        .agg(
            (
                (pl.col("r:psfFlux") * pl.col("weight")).sum() / pl.col("weight").sum()
            ).alias("nightly_flux"),
            (1.0 / pl.col("weight").sum().sqrt()).alias("nightly_flux_err"),
            (
                (pl.col("r:midpointMjdTai") * pl.col("weight")).sum()
                / pl.col("weight").sum()
            ).alias("nightly_mjd"),
            pl.col("z_used").first(),
            pl.col("t0").first(),
            pl.col("t_fd_rest_frame").first(),
            pl.col("x0").first(),
            pl.col("x1").first(),
            pl.col("c").first(),
            pl.len().alias("n_observations"),
        )
        .with_columns(
            ((pl.col("nightly_mjd") - pl.col("t0")) / (1.0 + pl.col("z_used"))).alias(
                "rest_frame_phase"
            ),
            (pl.col("nightly_flux") / pl.col("nightly_flux_err")).alias("snr"),
        )
        .sort(
            [
                "r:diaObjectId",
                "r:band",
                "nightly_mjd",
            ]
        )
    )

    return nightly_data


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


def pull_data_from_fink(
    objects: list[str] | None,
    api_url: str,
    columns: str,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Queries the Fink API for the data for the list of objects and the columns that are requested in the columns input parameter.

    Args:
        objects (list[str] | None): Pass a list of objects to the function to grab data just for those objects.
        api_url (str): The API endpoint that you want to query.
        columns (str): The columns of data wanted for each object.

    Returns:
        pl.DataFrame: Returns a dataframe with all the data for each object.
    """

    r = re.post(
        api_url,
        json={
            "diaObjectId": ",".join(objects),
            "columns": columns,
            "output-format": "json",
        },
    )

    # Format output in a DataFrame
    if r.status_code == 200:
        pdf = pl.read_json(io.BytesIO(r.content))
        if output_path is not None:
            pdf.write_parquet(output_path)
        return pdf


def query_fink_chunk(
    chunk_objects: list[str],
    api_url: str = "https://api.lsst.fink-portal.org/api/v1/fp",
    columns: str = "r:diaObjectId,r:band,nightly_mjd,fink_flux,fink_flux_err,snr,rest_frame_phase,z_used,t0",
    timeout: int = 120,
) -> pl.DataFrame:
    """
    Queries Fink API for a chunk of diaObjectIds.
    Recursively splits the chunk in half if timeouts, 500, 502, 504, or 429 errors occur.
    """
    if not chunk_objects:
        return pl.DataFrame()

    try:
        r = re.post(
            api_url,
            json={
                "diaObjectId": ",".join(chunk_objects),
                "columns": columns,
                "output-format": "json",
            },
            timeout=timeout,
        )

        # Explicit check for server errors before calling raise_for_status()
        if r.status_code in (500, 502, 503, 504, 429):
            raise re.exceptions.HTTPError(f"Server error {r.status_code}", response=r)

        r.raise_for_status()

        if r.status_code == 200 and len(r.content) > 0:
            return pl.read_json(io.BytesIO(r.content))
        return pl.DataFrame()

    except (
        re.exceptions.HTTPError,
        re.exceptions.Timeout,
        re.exceptions.ConnectionError,
    ) as e:
        status_code = (
            getattr(e.response, "status_code", None) if hasattr(e, "response") else None
        )

        # Target transient server/gateway failures and timeouts for splitting
        if isinstance(
            e, (re.exceptions.Timeout, re.exceptions.ConnectionError)
        ) or status_code in (500, 502, 503, 504, 429):
            if len(chunk_objects) <= 1:
                print(f"  [Warning] Skipping unqueryable object ID: {chunk_objects}")
                return pl.DataFrame()

            mid = len(chunk_objects) // 2
            print(
                f"  [{status_code or 'Timeout/ConnError'}] Splitting chunk of {len(chunk_objects)} into {mid} and {len(chunk_objects) - mid}..."
            )

            left_df = query_fink_chunk(chunk_objects[:mid], api_url, columns, timeout)
            right_df = query_fink_chunk(chunk_objects[mid:], api_url, columns, timeout)

            dfs = [df for df in [left_df, right_df] if df.height > 0]
            return pl.concat(dfs, how="diagonal") if dfs else pl.DataFrame()
        else:
            raise


band_colours = {
    "u": "darkblue",
    "g": "blue",
    "r": "red",
    "i": "darkred",
    "z": "orange",
    "y": "black",
}


def flux_to_mag():
    """Returns Polars expressions for nanojansky flux to AB magnitude conversion with uncertainties.

    Returns:
        pl.DataFrame: returns a Polars dataframe with two new columns for AB magnitude and magnitude error.
    """
    mag = -2.5 * pl.col("r:psfFlux").log10() + 31.4
    mag_err = (2.5 / np.log(10)) * (pl.col("r:psfFluxErr") / pl.col("r:psfFlux"))
    return mag.alias("mag"), mag_err.alias("mag_err")


def lightcurve_plotting(
    light_curve_data: pl.DataFrame, diaObjectId: str, output_dir: str
):
    """The plotting of the objects lightcurve this function is what produces the plot

    Args:
        light_curve_data (pl.DataFrame): The dataframe of all the object's light curve data.
        diaObjectId (str): This is just the string that is the name of the Rubin object.
        output_dir (str): The directory where the plots are saved to.
    """
    light_curve_data = light_curve_data.filter(pl.col("r:psfFlux") > 0).sort(
        "r:midpointMjdTai"
    )

    dates = Time(
        light_curve_data["r:midpointMjdTai"].to_numpy(), format="mjd"
    ).to_datetime()
    light_curve_data = light_curve_data.with_columns(date=pl.Series("date", dates))

    fig, ax = plt.subplots(figsize=(10, 6))

    for (band,), group in light_curve_data.group_by("r:band"):
        colour = band_colours.get(band, "black")
        ax.errorbar(
            group["date"].to_numpy(),
            group["r:psfFlux"].to_numpy(),
            yerr=group["r:psfFluxErr"].to_numpy(),
            fmt="o",
            color=colour,
            label=f"{band}-band",
            capsize=3,
            markersize=5,
        )

    ax.set_xlabel("Observation Date")
    ax.set_ylabel("Flux (nanojanskys)")
    ax.set_title(f"Light curve - {diaObjectId}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / f"{diaObjectId}.png")
        plt.close()
    plt.show()


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
        group_df = groups.get(object_id)
        # group_df = groups.get(object_id).filter(pl.col("r:psfFlux") > 0)

        if group_df is None:
            print(f"Object {object_id} does not have any light curve data.")
            continue

        lightcurve_plotting(group_df, object_id, output_path)


def colour_calculation(
    salt_eex_candidates_object_path: Path,
    power_law_eex_candidate_objects_path: Path,
    light_curve_data_path: Path,
    colour_calculation_output_path: Path,
) -> pl.DataFrame:
    """Calcualtes the colour in magnitude space between two different bands in the early excess region of the candidate.

    Args:
        eex_candidate_objects (Path): Path to the CSV file for the eex candidate objects
        light_curve_data (Path): Path to the light curve data for all the objects.

    Returns:
        pl.DataFrame: The dataframe with the g-r colour for the eex candidate objects.
    """

    power_law_eex_candidates = pl.read_csv(
        source=power_law_eex_candidate_objects_path,
        schema_overrides={"r:diaObjectId": pl.String},
    )

    salt_eex_candidates = pl.read_csv(
        source=salt_eex_candidates_object_path,
        schema_overrides={"r:diaObjectId": pl.String},
    )

    eex_candidates_df = pl.concat(
        [power_law_eex_candidates, salt_eex_candidates], how="diagonal"
    )

    eex_candidates_list = eex_candidates_df["r:diaObjectId"].to_list()

    light_curve_data = pl.read_parquet(source=light_curve_data_path)

    light_curve_data_filtered = light_curve_data.filter(
        pl.col("r:diaObjectId").is_in(eex_candidates_list),
    ).sort(by=["r:diaObjectId", "r:midpointMjdTai", "r:band"], descending=False)

    # Extract data points within the excess window which is three days for both salt and pl methods

    eex_data_points = (
        light_curve_data_filtered.join(
            eex_candidates_df,
            on="r:diaObjectId",
            how="inner",
        )
        .filter(
            pl.col("r:midpointMjdTai").is_between(
                lower_bound=pl.col("t_fd"),
                upper_bound=pl.col("t_fd") + 3.0,
                closed="both",
            )
            & (pl.col("r:psfFlux") > 0)
        )
        .sort(by=["r:diaObjectId", "r:midpointMjdTai"])
    )

    eex_data_points = eex_data_points.with_columns(
        (-2.5 * pl.col("r:psfFlux").log10() + 31.4).alias("mag"),
        ((2.5 / np.log(10)) * (pl.col("r:psfFluxErr") / pl.col("r:psfFlux"))).alias(
            "mag_err"
        ),
    )

    # 1. Floor midpoint MJD to get observing night
    nightly_mags = (
        eex_data_points.with_columns(
            pl.col("r:midpointMjdTai").floor().cast(pl.Int64).alias("night")
        )
        .group_by(["r:diaObjectId", "night", "r:band"])
        .agg(
            pl.col("mag").median().alias("median_mag"),
            pl.col("mag_err").mean().alias("mean_mag_err"),
        )
    )

    # 2. Pivot bands into separate columns (g, r, i)
    pivoted = nightly_mags.pivot(
        on="r:band",
        index=["r:diaObjectId", "night"],
        values=["median_mag", "mean_mag_err"],
    )

    # 3. Calculate g - r color (and magnitude error propagation in quadrature)
    # Check if 'g' and 'r' exist in the pivoted columns
    pivoted_cols = pivoted.columns

    colour_exprs = []
    if "median_mag_g" in pivoted_cols and "median_mag_r" in pivoted_cols:
        colour_exprs.append(
            (pl.col("median_mag_g") - pl.col("median_mag_r")).alias("g_minus_r")
        )
        if "mean_mag_err_g" in pivoted_cols and "mean_mag_err_r" in pivoted_cols:
            colour_exprs.append(
                (pl.col("mean_mag_err_g").pow(2) + pl.col("mean_mag_err_r").pow(2))
                .sqrt()
                .alias("g_minus_r_err")
            )

    if "median_mag_r" in pivoted_cols and "median_mag_i" in pivoted_cols:
        colour_exprs.append(
            (pl.col("median_mag_r") - pl.col("median_mag_i")).alias("r_minus_i")
        )

    if "median_mag_g" in pivoted_cols and "median_mag_i" in pivoted_cols:
        colour_exprs.append(
            (pl.col("median_mag_g") - pl.col("median_mag_i")).alias("g_minus_i")
        )

    if "median_mag_u" in pivoted_cols and "median_mag_g" in pivoted_cols:
        colour_exprs.append(
            (pl.col("median_mag_u") - pl.col("median_mag_g")).alias("u_minus_g")
        )

    if colour_exprs:
        colours_df = pivoted.with_columns(colour_exprs)
    else:
        colours_df = pivoted

    colours_df = colours_df.filter(
        pl.col("g_minus_r").is_not_null()
        | pl.col("r_minus_i").is_not_null()
        | pl.col("g_minus_i").is_not_null()
        | pl.col("u_minus_g").is_not_null()
    )

    if colour_calculation_output_path is not None:
        colours_df.write_csv(file=colour_calculation_output_path)

    # Filter for nights where same-night multi-band observations actually occurred
    return colours_df


def compute_nightly_g_minus_r(
    nightly_df: pl.DataFrame,
) -> pl.DataFrame:
    """Computes the g-r colour throughout time for observations in the same night.

    Args:
        nightly_df (pl.DataFrame): The nightly weighted average for points in the same night and other properties of the object in a polars dataframe.

    Returns:
        pl.DataFrame: Returns the polars dataframe with the g-r colour added.
    """

    df_mag = nightly_df.with_columns(
        [
            (-2.5 * pl.col("nightly_flux").log10() + 31.4).alias("mag"),
            (
                (2.5 / np.log(10))
                * (pl.col("nightly_flux_err") / pl.col("nightly_flux"))
            ).alias("mag_err"),
        ]
    )

    # Seperate the g and r bands
    g_band = df_mag.filter(pl.col("r:band") == "g").select(
        [
            "r:diaObjectId",
            "night",
            "rest_frame_phase",
            "t_fd_rest_frame",
            "t0",
            "x0",
            "x1",
            "c",
            "z_used",
            pl.col("mag").alias("mag_g"),
            pl.col("mag_err").alias("mag_err_g"),
        ]
    )

    r_band = df_mag.filter(pl.col("r:band") == "r").select(
        [
            "r:diaObjectId",
            "night",
            "rest_frame_phase",
            "t_fd_rest_frame",
            "t0",
            "x0",
            "x1",
            "c",
            "z_used",
            pl.col("mag").alias("mag_r"),
            pl.col("mag_err").alias("mag_err_r"),
        ]
    )

    colour_df = g_band.join(
        other=r_band,
        on=["r:diaObjectId", "night"],
        how="inner",
    )

    colour_df = colour_df.with_columns(
        [
            (pl.col("mag_g") - pl.col("mag_r")).alias("g_minus_r"),
            ((pl.col("mag_err_g").pow(2) + pl.col("mag_err_r").pow(2)).sqrt()).alias(
                "g_minus_r_err"
            ),
        ]
    )

    return colour_df


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    OBJECT_PATH = (
        DATA_DIR
        / "eex_results"
        / "salt_method"
        / "eex_cut_results"
        / "3_day_window"
        / "pass.csv"
    )

    LIGHT_CURVE_DATA_PATH = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    LIGHT_CURVE_DATA = pl.read_parquet(source=LIGHT_CURVE_DATA_PATH)
    OBJECTS_DF = pl.read_csv(
        source=OBJECT_PATH,
        columns=[
            "r:diaObjectId",
            "z_used",
            "t0",
        ],
        schema_overrides={"r:diaObjectId": pl.String},
    )

    OBJECT_LIGHT_CURVE_DATA = OBJECTS_DF.join(
        other=LIGHT_CURVE_DATA,
        on="r:diaObjectId",
        how="inner",
    )

    nightly_df = nightly_weighted_average(light_curve_data=OBJECT_LIGHT_CURVE_DATA)

    g_r_colour_df = compute_nightly_g_minus_r(
        nightly_df=nightly_df,
    ).filter(pl.col("r:diaObjectId") == "170019716323082348")

    print(g_r_colour_df)
