# finding the rate of rise in the r band of the in_tns SN candidates so that we have an empirical value to build the filter around
from light_curve_plotter import collect_light_curve_data, lightcurve_plotting
import pandas as pd
import numpy as np

tns_fullname = ["SN 2026ezw", "SN 2026fbm", "SN 2026ikn", "SN 2026ivl", "SN 2026kfw", "SN 2026mza"]
df = pd.read_csv(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\Confirmed_SN\confirmed_SN_tns.csv")
df = df[df["f:xm_tns_fullname"].isin(tns_fullname)]
object_ids = (df["r:diaObjectId"].tolist())
object_ids = list(map(str, object_ids))
object_ids = ','.join(object_ids)

red_band = ["r"]
green_band =["g"]

green_slopes = []
red_slopes = []
red_counts_per_object = []
green_counts_per_object = []

light_curve_info = collect_light_curve_data(object_ids)
for object_id, group_df in light_curve_info.groupby("r:diaObjectId"):
    red_group_df = group_df[group_df["r:band"].isin(red_band)]
    green_group_df = group_df[group_df["r:band"].isin(green_band)]
    red_peak_flux = red_group_df["r:psfFlux"].max()
    green_peak_flux = green_group_df["r:psfFlux"].max()
    peak_red_mjd = red_group_df.loc[red_group_df["r:psfFlux"].idxmax(), "r:midpointMjdTai"]
    peak_green_mjd = green_group_df.loc[green_group_df["r:psfFlux"].idxmax(), "r:midpointMjdTai"]
    pre_peak_red_mjd = red_group_df[red_group_df["r:midpointMjdTai"] < peak_red_mjd]
    pre_peak_green_mjd = green_group_df[green_group_df["r:midpointMjdTai"] < peak_green_mjd]
    red_counts_per_object.append(len(pre_peak_red_mjd))
    green_counts_per_object.append(len(pre_peak_green_mjd))
    red_slope = np.polyfit(pre_peak_red_mjd["r:midpointMjdTai"], (pre_peak_red_mjd["r:psfFlux"]/red_peak_flux), 1)
    green_slope = np.polyfit(pre_peak_green_mjd["r:midpointMjdTai"], (pre_peak_green_mjd["r:psfFlux"]/green_peak_flux), 1)
    green_slopes.append(green_slope)
    red_slopes.append(red_slope)

print (np.average(red_counts_per_object), np.average(green_counts_per_object))
print (np.average(red_slopes), np.average(green_slopes))
# print (f"These are the slopes in the green band before peak\n{green_slopes}")
# print (f"These are the slopes in the red band before peak\n{red_slopes}")