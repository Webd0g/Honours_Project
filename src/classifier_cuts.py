import pandas as pd
from pathlib import Path

base_classified_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\extragalactic_object_classified")
classified_csv_files = [base_classified_path / "extragalactic_lt20mag_candidates_classified.csv",
                            base_classified_path / "extragalactic_new_candidate_classified.csv",
                            base_classified_path / "sn_near_galaxy_candidate_classified.csv"]
SN_candidates_output_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\Potential_SN_candidates")

rejection_list_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\rejection_list.csv")

def load_rejection_list(path: Path) -> set:
    if not path.exists():
        print("No rejection list found, skipping filter.")
        return set()
    rejected = pd.read_csv(path, dtype={"r:diaObjectId": str})["r:diaObjectId"].tolist()
    print(f"Loaded {len(rejected)} rejected objects")
    return set(rejected)

# classifier_cuts function was generated with the Claude Sonnet 4.6 adaptive with the prompt "Before I get to the filtering, I would like to actually use the classifier score to then cut objects based on that metric. 
# However, what classifier do I use as my the metric to make cuts or include in the SN candidates catalogue and what is the cut-off score that I could use. 
# Anais suggested I start with a P(0.5) on the classifier confidence score but that is quite broad but it's what we could start with anyway with objects coming close to the cut-off requiring spot-checks on the light curve."
def classifier_cuts(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Selects SN candidates based on classifier scores.
    The object passes for further inspection if it passes any of the three 
    classifiers: CATS, EarlySNIa, and SNvsOthers.
    """
    # Filter out previously rejected objects
    rejected_ids = load_rejection_list(rejection_list_path)
    df = df[~df["r:diaObjectId"].isin(rejected_ids)].copy()

    mask_CATS = (df["clf_cats_class_mode"] == 11) & (df['clf_cats_score_max'] > threshold)
    mask_EarlySNIa = (df['clf_earlySNIa_max'] > threshold)
    mask_snn = (df["clf_snn_max"] > threshold)

    df_candidates = df[mask_CATS | mask_EarlySNIa | mask_snn].copy()
    
    df_candidates["n_classifiers_agree"] = (
        (df_candidates["clf_cats_score_max"] > threshold).astype(int) +
        (df_candidates["clf_earlySNIa_max"]  > threshold).astype(int) +
        (df_candidates["clf_snn_max"]        > threshold).astype(int)
    )

    print(f"Total candidates passing P({threshold}) cut: {len(df_candidates)}")
    print(f"\nClassifier agreement breakdown:")
    print(df_candidates["n_classifiers_agree"].value_counts().sort_index())

    return df_candidates.sort_values("n_classifiers_agree", ascending=False)

SN_candidates_output_path.mkdir(parents=True, exist_ok=True)

for classified_csv in classified_csv_files:
    df = pd.read_csv(classified_csv)
    df_results = classifier_cuts(df)
    
    high_confidence_objects = df_results[df_results["n_classifiers_agree"] == 3]
    output_file_high_confidence = SN_candidates_output_path / f"{classified_csv.stem}_high_confidence.csv"
    if len(high_confidence_objects) > 0:
        high_confidence_objects.to_csv(output_file_high_confidence, index=False)

    spot_check_objects_n_2 = df_results[df_results["n_classifiers_agree"] == 2]
    output_file_spot_check_n_2 = SN_candidates_output_path / f"{classified_csv.stem}_spot_check_required_n_2.csv"
    if len(spot_check_objects_n_2) > 0:
        spot_check_objects_n_2.to_csv(output_file_spot_check_n_2, index=False)

    spot_check_objects_n_1 = df_results[df_results["n_classifiers_agree"] == 1]
    output_file_spot_check_n_1 = SN_candidates_output_path / f"{classified_csv.stem}_spot_check_required_n_1.csv"
    if len(spot_check_objects_n_1) > 0:
        spot_check_objects_n_1.to_csv(output_file_spot_check_n_1, index=False)

    print(f"\n{classified_csv.stem}:")
    print(f"High confidence (n=3): {len(high_confidence_objects)}")
    print(f"Spot check required (n=2): {len(spot_check_objects_n_2)}")
    print(f"Spot check required (n=1): {len(spot_check_objects_n_1)}")

# Need to write in comments that document my understanding of this code and what it does for next time. If I do end up making my own filter these will need adjustments, but the bones of the final pipeline functions are all here and it works well.