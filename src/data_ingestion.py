import requests as re
import pandas as pd
import io

api_url_tags = "https://api.lsst.fink-portal.org/api/v1/tags"
tag = "sn_near_galaxy_candidate"
startdate = "2025-10-25"
stopdate = "2026-04-30"
output_path = r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\interim\extragalactic_candidates_using_SNGC.parquet"

def data_ingestion(tag, startdate, stopdate, output_path):
    
    r0 = re.post(api_url_tags,
        json={
            "tag": tag,
            "columns": "r:diaObjectId",
            "output-format": "json"  
        },
    )

    extra_galactic_candidates = sorted(set([str(val["r:diaObjectId"]) for val in r0.json()]))
    df = pd.DataFrame({"diaObjectId": extra_galactic_candidates})
    #df.to_parquet(output_path, index=False)
    print (f"The list of unique extra galactic candidates is:{extra_galactic_candidates}")
    print (f"The number of unique extra galactic candidates is: {len(extra_galactic_candidates)}")

# For latest run to encapsulate all alerts to get the most possible objects do startdate = 2025-10-25 and stopdate = 2026-04-30
data_ingestion(tag, startdate, stopdate, output_path)