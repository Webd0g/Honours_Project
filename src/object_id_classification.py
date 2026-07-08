import requests as re
import pandas as pd
import io
from pathlib import Path

# Pulling from the sources api to get the classifier scores for the ObjectIds Obtained from data_ingestion
api_url_sources = "https://api.lsst.fink-portal.org/api/v1/sources"

base_path = Path("/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts")
csv_files = [base_path / "extragalactic_candidates_passed.csv", 
                base_path / "sn_near_galaxy_candidates_passed.csv",
                base_path / "in_tns_candidates_passed.csv"]

base_output_path = Path("/home/riley/Documents/University/Honours/Honours_Project/data/extragalactic_object_classified")
output_paths = [base_output_path / "extragalactic_lt20mag_candidates_passed_classified.csv",
                base_output_path / "sn_near_galaxy_candidate_passed_classified.csv",
                base_output_path / "candidates_in_TNS_passed_classified.csv"]

# NOTE: The batch refactor of collect_classifier_scores and classification_concatenation
# (replacing the per-object loop with a single comma-separated API call) was 
# written by Claude Sonnet 4.6 (Anthropic) on 03/06/2026, analogous to the approach 
# used in the TNS cross-match script. The aggregation logic and overall pipeline structure 
# are unchanged from the original.

# grabbing the classifier scores from the csv files object ID 
def collect_classifier_scores_batch(object_ids: list) -> pd.DataFrame | None:
    """Collects classifier scores for all the filtered object ids"""
    
    r0 = re.post(api_url_sources,
        json={
            "diaObjectId": object_ids,
            "columns": "r:diaObjectId, f:clf_cats_class, f:clf_cats_score, f:clf_earlySNIa_score, f:clf_snnSnVsOthers_score",
            "output-format": "json"
        })
    
    if r0.status_code != 200:
        print(f"Batch request failed: {r0.status_code}")
        return None
    
    return pd.read_json(io.BytesIO(r0.content))


def classification_concatenation():
    for csv_file, output_path in zip(csv_files, output_paths):
        print(f"\nProcessing objects from {csv_file}")
        object_ids = pd.read_csv(csv_file, dtype={"r:diaObjectId" : str})["r:diaObjectId"].tolist()


        chunk_size = 500
        combined = []
        for diaObjectId in range(0, len(object_ids), chunk_size):
            object_classifier_scores = collect_classifier_scores_batch(",".join(object_ids[diaObjectId:diaObjectId + chunk_size]))
            if object_classifier_scores is not None:
                combined.append(object_classifier_scores)
    if not combined:
        raise RuntimeError(f"No classifier scores returned for {csv_file}")
    
    combined = pd.concat(combined, ignore_index=True)

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

classification_concatenation ()            