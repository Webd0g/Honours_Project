import requests as re
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

# Using tags to filter through alerts, the tag is the string at [0] in each tuple with [1] being the path of the csv file being saved.
# sn_near_galaxy_candidate: Written by Anais Moller (Supervisor) - near galaxy extragalactic base with absolute magnitude cuts consistent with SNe.
# extragalactic_lt20mag_candidate: bright (mag < 20), rising extragalactic candidates - complementary to sn_near_galaxy as it has a rising component, doesn't have the abs mag cuts tho.
# in_tns: objects already reported to TNS - used as confirmed ground truth for pipeline validation.
# extragalactic_new_candidate removed: loose extragalactic flavor, no galaxy association, high contamination, single/double epoch objects only.
api_url_tags = "https://api.lsst.fink-portal.org/api/v1/tags"
base_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\extragalactic_object_ids")
tags = [("sn_near_galaxy_candidate",        base_path / "sn_near_galaxy_candidate.csv"),
        ("extragalactic_lt20mag_candidate",  base_path / "extragalactic_lt20mag_candidate.csv"),
        ("in_tns",                           base_path / "candidates_in_TNS.csv")]

# Will run everyday and scan the last 24hrs of alerts.
# NOTE: Need to check what time Rubin is programmed to start the scan of the sky as I don't want to set the code going and miss when alerts are being sent out.
startdate = (datetime.now(tz=ZoneInfo("Australia/Sydney")) - timedelta(days=1)).strftime("%Y-%m-%d")
stopdate  = (datetime.now(tz=ZoneInfo("Australia/Sydney"))).strftime("%Y-%m-%d")

N_ALERTS = 100000

# fetch_tag_with_fallback was written by Claude Sonnet 4.6 with the prompt:
# "The 502 error is a one-time problem for the historical backfill. Add a minimal
# recursive date-splitting fallback that only activates on 502, leaving the daily
# pipeline logic unchanged."
def fetch_tag_with_fallback(tag: str, startdate: str, stopdate: str) -> list:
    """Fetch alerts for a tag between two dates.
    On 502 Bad Gateway, automatically splits the date range in half and retries.
    For daily 24hr runs this fallback will never trigger.
    """
    try:
        r = re.post(
            api_url_tags,
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

def data_ingestion(startdate, stopdate):
    for tag, output_path in tags:
        print(f"\nIngesting tag: {tag}")

        raw_alerts = fetch_tag_with_fallback(tag, startdate, stopdate)

        # Warn if the alert ceiling was hit - raw alert count may be incomplete
        if len(raw_alerts) >= N_ALERTS:
            print(f"WARNING: n ceiling hit ({N_ALERTS}) for {tag} — increase N_ALERTS")

        print(f"Raw alerts returned: {len(raw_alerts)}")

        new_extragalactic_ids = sorted(set([str(val["r:diaObjectId"]) for val in raw_alerts]))
        df_new = pd.DataFrame({"r:diaObjectId": new_extragalactic_ids})

        if len(df_new) == 0:
            print(f"No new candidates for tag: {tag}")
            continue

        if output_path.exists():
            existing_df = pd.read_csv(output_path, dtype={"r:diaObjectId": str})
            combined_df = pd.concat([existing_df, df_new], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset="r:diaObjectId")
        else:
            combined_df = df_new

        combined_df.to_csv(output_path, index=False)
        print(f"The number of new candidates found is {len(df_new)}")
        print(f"The total number of candidates in the catalogue is {len(combined_df)}")

data_ingestion(startdate, stopdate)