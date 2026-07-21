"""
This script is to plot the new objects that pass the rate of rise filter
so that the objects can be visually inspected and confirmed to look like SN.

The rate of rise filter is not perfect so visual inspection is necessary step to 
confirm the population of the objects at least look like SN after the filter is 
applied.
"""
import pandas as pd
import os
from pathlib import Path
import generic_light_curve_batch_plotter as glcbp

# Directory that the rate of rise data is hosted in.
RATE_OF_RISE_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "Rate_of_rise_cuts"

# reads the csvs that have passed the rate of rise cut that were made and combines them into one list that consists of the objects that passed the cuts.
OBJECT_IDS_PASSED_SN_NEAR_GALAXY = pd.read_csv(RATE_OF_RISE_DIR / "sn_near_galaxy_candidates_passed.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"].tolist()
OBJECT_IDS_PASSED_EXTRAGALACTIC_LT20MAG = pd.read_csv(RATE_OF_RISE_DIR / "extragalactic_candidates_passed.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"].tolist()
OBJECT_IDS_PASSED_IN_TNS = pd.read_csv(RATE_OF_RISE_DIR / "in_tns_candidates_passed.csv", dtype={"r:diaObjectId":str})["r:diaObjectId"].tolist()
OBJECT_IDS_PASSED = OBJECT_IDS_PASSED_SN_NEAR_GALAXY + OBJECT_IDS_PASSED_EXTRAGALACTIC_LT20MAG + OBJECT_IDS_PASSED_IN_TNS
print (OBJECT_IDS_PASSED)

# finds the object ids of the objects that already have plots of them and combines the strings into a list.
object_ids_with_plots_false_positives = ([f.split(".")[0] for f in os.listdir(Path.home() / "Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results/False_positives") if f.endswith(".png")])
object_ids_with_plots_probable_SN = ([f.split(".")[0] for f in os.listdir(Path.home() / "Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results/Probable_Sn_Ia") if f.endswith(".png")])
object_ids_with_plots = object_ids_with_plots_false_positives + object_ids_with_plots_probable_SN

# checks in the list of objects that have passed if they have plots and if they don't
# the object id gets appended to a list of objects that don't have plots for them.  
objects_not_with_plots = []
for object_id in OBJECT_IDS_PASSED:
    if object_id not in object_ids_with_plots:
        objects_not_with_plots.append(object_id)
print (f"{(len(objects_not_with_plots))} passed the rate of rise cuts but doesn't have a plot generated for it.")


chunk_size = 500
output_path = Path.home() / "Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results"

glcbp.batch_plotter(objects_not_with_plots, chunk_size, output_path)
