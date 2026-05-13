import requests as re
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

# Using tags to filter through alerts, the tag is the string at [0] in each tuple with [1] being the path of the parquet file being saved.
# We want to use these filters so that we get a mix of bright extragalactic candidates, near galaxy fast rising objects which align themselves with SNe
# and finally a broad extragalactic candidates filter. 
# NOTE: Most likely will need to make my own filter with the blocks made as none of these filters use the galactic latitude cut off block which is important to identify extragalactic candidates. 
api_url_tags = "https://api.lsst.fink-portal.org/api/v1/tags"
base_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\extragalactic_object_ids")
tags = [("extragalactic_lt20mag_candidate", base_path / "extragalactic_lt20mag_candidate.csv"), 
        ("sn_near_galaxy_candidate", base_path / "sn_near_galaxy_candidate.csv"), 
        ("extragalactic_new_candidate", base_path / "extragalactic_new_candidate.csv")]

# will run everyday and scan the last 24hrs of alerts.
# NOTE: Need to check what time Rubin is programmed to start the scan of the sky as I don't want to set the code going and miss when alerts are being sent out. 
startdate = (datetime.now(tz=ZoneInfo("Australia/Sydney")) - timedelta(days=1)).strftime("%Y-%m-%d")
stopdate = (datetime.now(tz=ZoneInfo("Australia/Sydney"))).strftime("%Y-%m-%d")

# data ingestion function, iterates through each tag and output path in the tags list-tuple.
def data_ingestion(startdate, stopdate):
    for tag, output_path in tags:
        print (f"\nIngesting tag: {tag}")
        
        # pulls object id from the alerts
        r0 = re.post(api_url_tags,
            json={
                "tag": tag,
                "startdate" : startdate,
                "stopdate" : stopdate,  
                "columns": "r:diaObjectId",
                "output-format": "json"  
            },
        )
        r0.raise_for_status()

        # new extragalactic ids set created from the most recent alerts poll, pushes into a df
        new_extragalactic_ids = sorted(set([str(val["r:diaObjectId"]) for val in r0.json()]))
        df_new = pd.DataFrame({"diaObjectId": new_extragalactic_ids})
        
        if len(df_new) == 0:
            print(f"No new candidates for tag: {tag}")
            continue

        # if path exists we append the new alerts if a new object ID passes the filters based on new alerts, if path don't exist we make the combined df = new df
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            combined_df = pd.concat([existing_df, df_new], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset="diaObjectId")
        else:
            combined_df = df_new

        # then pushed to a parquet file type to be used in next function in the pipeline
        combined_df.to_csv(output_path, index = False)
        print (f"The number of new candidates found is {len(df_new)}")
        print (f"The total number of candidates in the catalogue is {len(combined_df)}")

data_ingestion(startdate, stopdate)