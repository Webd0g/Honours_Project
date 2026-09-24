import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests as re
from astropy.time import Time
from tqdm import tqdm


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

    except (re.exceptions.HTTPError, re.exceptions.Timeout, re.exceptions.ConnectionError) as e:
        status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
        
        # Target transient server/gateway failures and timeouts for splitting
        if isinstance(e, (re.exceptions.Timeout, re.exceptions.ConnectionError)) or status_code in (500, 502, 503, 504, 429):
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


if __name__ == "__main__":
    API_URL = "https://api.lsst.fink-portal.org/api/v1/fp"

    COLUMNS = (
        "r:diaObjectId,r:diaSourceId,r:midpointMjdTai,r:band,r:psfFlux,r:psfFluxErr"
    )

    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data" / "forced_photometry_test"

    IMAGES_DIR = PROJECT_DIR / "images"

    OBJECTS_PATH = DATA_DIR / "object_collection_result" / "all_objects.csv"

    OBJECTS = pl.read_csv(
        source=OBJECTS_PATH, schema_overrides={"r:diaObjectId": pl.String}
    )["r:diaObjectId"].to_list()

    # Batch calling helper to prevent initial massive payloads
    BATCH_SIZE = 20
    all_dfs = []

    for i in tqdm(range(0, len(OBJECTS), BATCH_SIZE), desc="Fetching Fink forced photometry"):
        batch = OBJECTS[i : i + BATCH_SIZE]
        df_chunk = query_fink_chunk(
            chunk_objects=batch,
            api_url=API_URL,
            columns=COLUMNS,
        )
        if df_chunk.height > 0:
            all_dfs.append(df_chunk)

    forced_light_curve_data = pl.concat(all_dfs, how="diagonal") if all_dfs else pl.DataFrame()

    forced_light_curve_data.write_parquet(file="/home/riley/Documents/University/Honours/Honours_Project/data/forced_photometry_test/light_curve_data/forced_photometry_light_curve_data.parquet")