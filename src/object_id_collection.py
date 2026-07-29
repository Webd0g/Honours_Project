"""
This script scrapes the Fink broker and finds alerts that pass the extragalactic_lt20mag, sn_near_galaxy and in_tns Fink filters.
it then pulls the diaObjectId from the alerts and appends it to a csv file for each object that passes the distinct filters
"""

import pandas as pd
import requests as re
from datetime import datetime, timedelta, timezone
from pathlib import Path

# We use these tags so that we can Xmatch objects with galaxies as we want objects to be able to be associated with a host galaxy, we also use in_tns to find SN that have been confirmed before hand so we can expand groun truth SN before photometric classification.
API_URL_TAGS = "https://api.lsst.fink-portal.org/api/v1/tags"
PROJECT_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
TAGS = [("sn_near_galaxy_candidate",        PROJECT_DIR / "data" / "extragalactic_object_ids" /  "sn_near_galaxy_candidate.csv"),
        ("extragalactic_lt20mag_candidate",  PROJECT_DIR / "data" / "extragalactic_object_ids" / "extragalactic_lt20mag_candidate.csv"),
        ("in_tns",                           PROJECT_DIR / "data" / "extragalactic_object_ids" / "candidates_in_TNS.csv")]

# To query jsut the latest alerts from the API
STARTDATE = (datetime.now(tz=timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
STOPDATE  = (datetime.now(tz=timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

# To limit the number of alerts we query for so that we can get full alert volume from the tags and not be limited.
N_ALERTS = 100000

def fetch_tag_with_fallback(tag: str, startdate: str, stopdate: str) -> list:
    """
    Fetch alert object IDs for a given Fink tag within a date range.

    Queries the Fink /api/v1/tags endpoint for objects matching `tag` between
    `startdate` and `stopdate`. On a 502/504 Bad Gateway response, recursively
    splits the date range in half and retries each half, to handle large
    historical backfills without timing out. For routine daily 24hr queries
    this fallback is never triggered.
    """

    try:
        r = re.post(
            API_URL_TAGS,
            json={
                "tag":           tag,
                "startdate":     startdate,
                "stopdate":      stopdate,
                "n":             N_ALERTS,
                "columns":       "r:diaObjectId",
                "output-format": "json",
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    except re.exceptions.HTTPError as e:
        if e.response.status_code in (502, 504):
            start_dt = datetime.strptime(startdate, "%Y-%m-%d")
            stop_dt  = datetime.strptime(stopdate,  "%Y-%m-%d")
            mid_dt   = start_dt + (stop_dt - start_dt) / 2

            # Avoid infinite recursion if range collapses to a single day
            if mid_dt == start_dt or mid_dt == stop_dt:
                print(f"  Could not split further for {startdate} -> {stopdate}, skipping.")
                return []

            mid = mid_dt.strftime("%Y-%m-%d")
            print(f"  502 on {startdate} -> {stopdate}, splitting at {mid}")
            left  = fetch_tag_with_fallback(tag, startdate, mid)
            right = fetch_tag_with_fallback(tag, mid, stopdate)
            return left + right
        else:
            raise

def parse_and_merge_alerts(alerts: list, output_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts raw json alert package into a de-duplicated dataframe of the diaObjectIds
    that is then merged with an existing catalogue at the output_path.
    
    This is done so that alerts are transformed into a readble and easy to work with
    format that can be transformed into different data storage formats.
    """

    new_extragalactic_ids = sorted(set([str(val["r:diaObjectId"]) for val in alerts]))
    df_new = pd.DataFrame({"r:diaObjectId": new_extragalactic_ids})

    if output_path.exists():
        existing_df = pd.read_csv(output_path, dtype={"r:diaObjectId": str})
        combined_df = pd.concat([existing_df, df_new], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset="r:diaObjectId")
    else:
        combined_df = df_new

    return combined_df, df_new

def data_ingestion(startdate: str, stopdate: str) -> None:
    """
    Iterates through the tags and calls fetch_with_fallback, 
    removes duplicates and writes the collected object IDs to a csv.
    
    This is nessacary so that object IDs that are cross matched can be used 
    later in the pipeline, this also indicates how many new alerts and 
    objects are being passed through which allows the user to determine the 
    ratio of new objects to total alerts produced by Rubin.
    """

    for tag, output_path in TAGS:

        print(f"\nIngesting tag: {tag}")
        alerts = fetch_tag_with_fallback(tag, startdate, stopdate)

        if len(alerts) >= N_ALERTS:
            print(f"WARNING: n ceiling hit ({N_ALERTS}) for {tag} — increase N_ALERTS")
        print(f"Raw alerts returned: {len(alerts)}")

        combined_df, df_new = parse_and_merge_alerts(alerts, output_path)

        if len(df_new) == 0:
            print(f"No new candidates for tag: {tag}")
            continue

        combined_df.to_csv(output_path, index=False)
        print(f"The number of new candidates found is {len(df_new)}")
        print(f"The total number of candidates in the catalogue is {len(combined_df)}")

if __name__ == "__main__":
    data_ingestion(STARTDATE, STOPDATE)