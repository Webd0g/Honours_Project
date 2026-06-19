# Batch plotter for the labelled in_tns objects. This was done so that I can understand what the light curves of a SN Ia looks like. Builds a validation dataset.
import pandas as pd
from pathlib import Path
from light_curve_plotter import lightcurve_plotting
from light_curve_plotter import collect_light_curve_data

output_path = Path(r"C:\Users\riley\Documents\University\Honours\Honours_Project\images\SN_Ia_lightcurves\in_tns")

in_tns_object_ids = list((pd.read_csv(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\Confirmed_SN\confirmed_SN_tns.csv", dtype= {"r:diaObjectId": str}))["r:diaObjectId"])
in_tns_object_names = list((pd.read_csv(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\Confirmed_SN\confirmed_SN_tns.csv", dtype= {"f:xm_tns_fullname": str}))["f:xm_tns_fullname"])
tns_lookup = dict(zip(in_tns_object_ids, in_tns_object_names))
in_tns_object_ids = ",".join(in_tns_object_ids)


# need lightcurve data first retrived from the strings of diaobject ID from the csv files. 
light_curve_info = collect_light_curve_data(in_tns_object_ids)
for object_id, group_df in light_curve_info.groupby("r:diaObjectId"):
    object_id = str(object_id)
    # lightcurve_plotting takes a df, diaObjectId and a source file string as arguments
    lightcurve_plotting(group_df, object_id, tns_lookup[object_id], output_path / f"{tns_lookup[object_id]}")