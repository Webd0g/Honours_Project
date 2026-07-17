import pandas as pd
import os
import light_curve_plotter as lcp
from pathlib import Path
import generic_light_curve_batch_plotter as glcbp

rate_of_rise_dir = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "Rate_of_rise_cuts"

# reads the csvs that have passed the rate of rise cut that were made and combines them into one list that consists of the objects that passed the cuts.
object_ids_passed_sn_near_galaxy = pd.read_csv(rate_of_rise_dir / "sn_near_galaxy_candidates_passed.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"].tolist()
object_ids_passed_extragalactic_lt20mag = pd.read_csv(rate_of_rise_dir / "extragalactic_candidates_passed.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"].tolist()
object_ids_passed_in_tns = pd.read_csv(rate_of_rise_dir / "in_tns_candidates_passed.csv", dtype={"r:diaObjectId":str})["r:diaObjectId"].tolist()
object_ids_passed = object_ids_passed_sn_near_galaxy + object_ids_passed_extragalactic_lt20mag + object_ids_passed_in_tns

# finds the object ids of the objects that already have plots of them and combines the strings into a list.
object_ids_with_plots_false_positives = ([f.split(".")[0] for f in os.listdir(Path.home() / "Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results/False_positives") if f.endswith(".png")])
object_ids_with_plots_probable_SN = ([f.split(".")[0] for f in os.listdir(Path.home() / "Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results/Probable_Sn_Ia") if f.endswith(".png")])
object_ids_with_plots = object_ids_with_plots_false_positives + object_ids_with_plots_probable_SN

# checks in the list of objects that have passed if they have plots and if they don't
# the object id gets appended to a list of objects that don't have plots for them.  
objects_not_with_plots = []
for object_id in object_ids_passed:
    if object_id not in object_ids_with_plots:
        objects_not_with_plots.append(object_id)
print (f"{(len(objects_not_with_plots))} passed the rate of rise cuts but doesn't have a plot generated for it.")


chunk_size = 500
output_path = Path.home() / "Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results"

glcbp.batch_plotter(objects_not_with_plots, chunk_size, output_path)
