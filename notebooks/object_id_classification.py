import requests as re
import pandas as pd
import io
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Pulling from the sources api to get the classifier scores for the ObjectIds Obtained from data_ingestion
api_url_sources = "https://api.lsst.fink-portal.org/api/v1/sources"
#input_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\extragalactic_object_ids\extragalactic_lt20mag_candidate.parquet")
#output_path = input_path.parent / "extragalactic_candidates_using_ELMC_classified.parquet"

base_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\extragalactic_object_ids")
csv_files = [base_path / "extragalactic_lt20mag_candidate.csv", 
                base_path / "sn_near_galaxy_candidate.csv",
                base_path / "candidates_in_TNS.csv"]

base_output_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\extragalactic_object_classified")
output_paths = [base_output_path / "extragalactic_lt20mag_candidates_classified.csv",
                base_output_path / "sn_near_galaxy_candidate_classified.csv",
                base_output_path / "candidates_in_TNS_classified.csv"]

# NOTE: The batch refactor of collect_classifier_scores and classification_concatenation
# (replacing the per-object loop with a single comma-separated API call) was 
# written by Claude Sonnet 4.6 (Anthropic) on 03/06/2026, analogous to the approach 
# used in the TNS cross-match script. The aggregation logic and overall pipeline structure 
# are unchanged from the original.

# grabbing the classifier scores from the csv files object ID 
def collect_classifier_scores_batch(object_ids: list) -> pd.DataFrame | None:
    """Collects classifier scores for all the filtered object ids
    but even further filtered to only retrieve those that had alerts 
    over the last 24 hours, this prevents error 504 when requesting """
    
    id_string = ",".join(object_ids)
    r0 = re.post(api_url_sources,
        json={
            "diaObjectId": id_string,
            "columns": "r:diaObjectId, f:clf_cats_class, f:clf_cats_score, f:clf_earlySNIa_score, f:clf_snnSnVsOthers_score",
            "output-format": "json"
        })
    
    if r0.status_code != 200:
        print(f"Batch request failed: {r0.status_code}")
        return None
    
    return pd.read_json(io.BytesIO(r0.content))


def classification_concatenation(object_ids: list) -> pd.DataFrame:
    for csv_path, output_path in zip(csv_files, output_paths):
        print(f"\nProcessing file: {csv_path.name}")
        object_ids = pd.read_csv(csv_path)["r:diaObjectId"].astype(str).tolist()

        combined = collect_classifier_scores_batch(object_ids)
        if combined is None or len(combined) == 0:
            print(f"No results returned for {csv_path.name}")
            continue

        combined_agg = combined.groupby("r:diaObjectId").agg(
            clf_cats_class_mode  = ("f:clf_cats_class",             lambda x: x.mode()[0]),
            clf_cats_score_max   = ("f:clf_cats_score",             lambda x: x[x >= 0].max()),
            clf_cats_score_mean  = ("f:clf_cats_score",             lambda x: x[x >= 0].mean()),
            clf_earlySNIa_max    = ("f:clf_earlySNIa_score",        lambda x: x[x >= 0].max()),
            clf_earlySNIa_mean   = ("f:clf_earlySNIa_score",        lambda x: x[x >= 0].mean()),
            clf_snn_max          = ("f:clf_snnSnVsOthers_score",    lambda x: x[x >= 0].max()),
            clf_snn_mean         = ("f:clf_snnSnVsOthers_score",    lambda x: x[x >= 0].mean()),
            n_detections         = ("r:diaObjectId",                "count"),
        ).reset_index()

        combined_agg.to_csv(output_path, index=False)
        print(f"Saved {len(combined_agg)} objects to {output_path}")

classification_concatenation (csv_files)            