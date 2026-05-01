import requests as re
import pandas as pd
import io
from pathlib import Path

# Pulling from the sources api to get the classifier scores for the ObjectIds Obtained from data_ingestion
api_url_sources = "https://api.lsst.fink-portal.org/api/v1/sources"
input_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\raw\fink_alerts\extragalactic_candidates_using_ELMC.parquet")
output_path = input_path.parent / "extragalactic_candidates_using_ELMC_classified.parquet"

object_ids = pd.read_parquet(input_path)["diaObjectId"].tolist()

# grabbing the classifier scores from the parquet files object ID 
def collect_classifier_scores(diaObjectId: str) -> pd.DataFrame | None:
    r0 = re.post(api_url_sources, 
    json = {
        "diaObjectId": diaObjectId,
        "columns": "r:diaObjectId, f:clf_cats_class, f:clf_cats_score, f:clf_earlySNIa_score",
        "output-format": "json"
        })
    
    # check for if the request passed through nicely for that specific object
    if r0.status_code != 200:
        print (f"Collecting classification score failed for {diaObjectId}: {r0.status_code}")
        return None
    
    return pd.read_json(io.BytesIO(r0.content))

# concatenation of the objects to retrieve the classifier scores 
def classification_concatenation(object_ids: list) -> pd.DataFrame:
    results =[]

    for i, oid in enumerate(object_ids):
        print (f"Processing {i+1}/{len(object_ids)}: {oid}")
        df = collect_classifier_scores(oid)
        if df is not None:
            results.append(df)
    
    combined = pd.concat(results, ignore_index= True)
    combined.to_parquet(output_path, index = False)

    print (f"\nSaved {len(combined)} records for {len(results)} objects to {output_path}")
    return combined

df = classification_concatenation(object_ids)
print (df.head())

# Left off last time with the parquet file of object ids being passed to the classifier concatenator and being able to retrieve the classifier scores for the object ID, next will be working on how to aggregate the scores even if the data is currently not humungous
# Also to note, the extragalactic candidates filters appears to not be filtering using the galactic latitude cut offs. Might have to make my own filter which should be cool and chill.