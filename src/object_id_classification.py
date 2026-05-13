import requests as re
import pandas as pd
import io
from pathlib import Path

# Pulling from the sources api to get the classifier scores for the ObjectIds Obtained from data_ingestion
api_url_sources = "https://api.lsst.fink-portal.org/api/v1/sources"
#input_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\extragalactic_object_ids\extragalactic_lt20mag_candidate.parquet")
#output_path = input_path.parent / "extragalactic_candidates_using_ELMC_classified.parquet"

base_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\extragalactic_object_ids")
csv_files = [base_path / "extragalactic_lt20mag_candidate.csv", 
                base_path / "sn_near_galaxy_candidate.csv",
                base_path / "extragalactic_new_candidate.csv"]

base_output_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\extragalactic_object_classified")
output_paths = [base_output_path / "extragalactic_lt20mag_candidates_classified.csv",
                base_output_path / "sn_near_galaxy_candidate_classified.csv",
                base_output_path / "extragalactic_new_candidate_classified.csv"]


# grabbing the classifier scores from the parquet files object ID 
def collect_classifier_scores(diaObjectId: str) -> pd.DataFrame | None:
    r0 = re.post(api_url_sources, 
        json = {
            "diaObjectId": diaObjectId,
            "columns": "r:diaObjectId, f:clf_cats_class, f:clf_cats_score , f:clf_earlySNIa_score, f:clf_snnSnVsOthers_score",
            "output-format": "json"
            })
        
        # check for if the request passed through nicely for that specific object
    if r0.status_code != 200:
        print (f"Collecting classification score failed for {diaObjectId}: {r0.status_code}")
        return None
        
    return pd.read_json(io.BytesIO(r0.content))

# concatenation of the objects to retrieve the classifier scores 
def classification_concatenation(object_ids: list) -> pd.DataFrame:
    
    for csv_path, output_path in zip(csv_files, output_paths):
        results =[]
        print (f"\nProcessing file: {csv_path.name}")
        object_ids = pd.read_csv(csv_path)["diaObjectId"].tolist()

        for i, oid in enumerate(object_ids):
            print (f"Processing {i+1}/{len(object_ids)}: {oid}")
            df = collect_classifier_scores(oid)
            if df is not None:
                results.append(df)
    
        if len(results) == 0:
            print (f"No results returned for {csv_path.name}")
            continue

        combined = pd.concat(results, ignore_index= True)
        combined_agg = combined.groupby("r:diaObjectId").agg(
            clf_cats_class_mode  = ("f:clf_cats_class",       lambda x: x.mode()[0]),
            clf_cats_score_max   = ("f:clf_cats_score",        "max"),
            clf_cats_score_mean  = ("f:clf_cats_score",        "mean"),
            clf_earlySNIa_max    = ("f:clf_earlySNIa_score",   "max"),
            clf_earlySNIa_mean   = ("f:clf_earlySNIa_score",   "mean"),
            clf_snn_max          = ("f:clf_snnSnVsOthers_score","max"),
            clf_snn_mean         = ("f:clf_snnSnVsOthers_score","mean"),
            n_detections         = ("r:diaObjectId",            "count"),
        ).reset_index()
        if len(combined_agg) > 0:
            combined_agg.to_csv(output_path, index = False)
            print (f"\nSaved {len(combined)} records for {len(results)} objects to {output_path}")
        else:
            print(f"Skipping write for {output_path.name} — no results returned")

classification_concatenation (csv_files)
# Single object test
# ---------------------------------------------------------------
# test_df = collect_classifier_scores("170235219334922248")
# print (test_df.head())
# ---------------------------------------------------------------

# Multiple object test
# -------------------------------------------------------------------------
# Slice just the first 5 IDs from one parquet file
#test_ids = pd.read_parquet(parquet_files[0])["diaObjectId"].tolist()[:5]

#for oid in test_ids:
#    df = collect_classifier_scores(oid)
#    print(df)
# -------------------------------------------------------------------------

# Left off last time with the parquet file of object ids being passed to the classifier concatenator and being able to retrieve the classifier scores for the object ID, next will be working on how to aggregate the scores even if the data is currently not humungous
# Also to note, the extragalactic candidates filters appears to not be filtering using the galactic latitude cut offs. Might have to make my own filter which should be cool and chill.