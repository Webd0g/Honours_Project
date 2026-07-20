# Batch plotter for the labelled in_tns objects. This was done so that I can understand what the light curves of a SN Ia looks like. Builds a validation dataset.
import pandas as pd
from pathlib import Path
from light_curve_plotter import lightcurve_plotting
from light_curve_plotter import collect_light_curve_data

# output_path = Path(r"/home/riley/Documents/University/Honours/Honours_Project/images/SN_Ia_lightcurves")

# object_ids = list((pd.read_csv("/home/riley/Documents/University/Honours/Honours_Project/images/SN_Ia_lightcurves/passed_rate_of_rise_cuts_results/unplotted_objects.csv", dtype= {"r:diaObjectId": str}))["r:diaObjectId"])

# chunk_size = 500

# need lightcurve data first retrived from the strings of diaobject ID from the csv files. 
def batch_plotter(object_ids, chunk_size=500, output_path=Path.home() / "/home/riley/Documents/University/Honours/Honours_Project/images/SN_Ia_lightcurves/passed_rate_of_rise_cuts_results"):
    for diaObjectId in range(0, len(object_ids), chunk_size):
        chunk = ",".join(object_ids[diaObjectId:diaObjectId + chunk_size])
        light_curve_info = collect_light_curve_data(chunk)
        if light_curve_info is None or light_curve_info.empty:
            print(f"No light curve data found for chunk starting at index {diaObjectId}.")
            continue
        for object_id, group_df in light_curve_info.groupby("r:diaObjectId"):
            object_id = str(object_id)
            # lightcurve_plotting takes a df, diaObjectId and a source file string as arguments
            lightcurve_plotting(group_df, object_id, object_id, output_path / f"{object_id}")