"""
This script plots the new objects that pass the rate of rise filter
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

PLOT_RESULTS_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "images" / "SN_lightcurves" / "passed_rate_of_rise_cuts_results"

# Sub-directories where the object plots are stored 
SUBDIRECTORIES = [PLOT_RESULTS_DIR / "anomalies", 
                PLOT_RESULTS_DIR / "declining_objects",
                PLOT_RESULTS_DIR / "flat_curves_with_noise",
                PLOT_RESULTS_DIR / "genuine_candidates",
                PLOT_RESULTS_DIR / "single_band_detections"]

# Reads the CSVs that have passed the rate of rise cut that were made and combines them into one list that consists of the objects that passed the cuts.
OBJECT_IDS_PASSED_SN_NEAR_GALAXY = pd.read_csv(RATE_OF_RISE_DIR / "sn_near_galaxy_candidates_passed.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"].tolist()
OBJECT_IDS_PASSED_EXTRAGALACTIC_LT20MAG = pd.read_csv(RATE_OF_RISE_DIR / "extragalactic_candidates_passed.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"].tolist()
OBJECT_IDS_PASSED_IN_TNS = pd.read_csv(RATE_OF_RISE_DIR / "in_tns_candidates_passed.csv", dtype={"r:diaObjectId":str})["r:diaObjectId"].tolist()
OBJECT_IDS_PASSED = OBJECT_IDS_PASSED_SN_NEAR_GALAXY + OBJECT_IDS_PASSED_EXTRAGALACTIC_LT20MAG + OBJECT_IDS_PASSED_IN_TNS

def find_objects_with_plots(list_of_subdirectories: list[Path], extension: str) -> list[str]:
    """
    Finds the stems of the files that have the specific file extension 
    in a directory the user specifies and combines the stems into a list of strings.

    This used to find objects that already have plots of them so that only new objects have plots produced. 
    """
    objects_with_plots = []
    for directory in list_of_subdirectories:
        objects_with_plots.extend([f.split(".")[0] for f in os.listdir(directory) if f.endswith(extension)])
    return objects_with_plots

def find_objects_without_plots(object_ids: list, list_of_subdirectories: list[Path], extension: str) -> list[str]:
    """
    This function takes a list of object ids that have passed the rate of rise cut, list of subdirectories of 
    different object types and a file extension. find_objects_without_plots uses list_of_subdirectories and extension
    to call find_objects_with_plots to find the objects that haven't been plotted yet and then appends them to a list.

    find_objects_with_plots was neccesary so that the use doesn't have to manually compare objects when you have thousands 
    of SN candidate objects. 
    """
    plotted_objects = set(find_objects_with_plots(list_of_subdirectories, extension))
    objects_without_plots = [oid for oid in object_ids if oid not in plotted_objects]
    print (f"{(len(objects_without_plots))} passed the rate of rise cuts but doesn't have a plot generated for it.")
    return objects_without_plots

# Finally the batch plotter receives the objects without plots and thus plots them. 
glcbp.batch_plotter(find_objects_without_plots(OBJECT_IDS_PASSED, SUBDIRECTORIES, ".png"), 500, Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "images" / "SN_lightcurves")