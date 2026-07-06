# finding the rate of rise in the r band of the in_tns SN candidates so that we have an empirical value to build the filter around
from light_curve_plotter import collect_light_curve_data, lightcurve_plotting
import pandas as pd
import numpy as np

tns_fullname = ["SN 2026ezw", "SN 2026fbm", "SN 2026ikn", "SN 2026ivl", "SN 2026kfw", "SN 2026mza"]
df = pd.read_csv(r"/home/riley/Documents/University/Honours/Honours_Project/data/Confirmed_SN/confirmed_SN_tns.csv")
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
green_std_devs = []
red_std_devs = []

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
    red_slope, red_intercept = np.polyfit(pre_peak_red_mjd["r:midpointMjdTai"], (pre_peak_red_mjd["r:psfFlux"]/red_peak_flux), 1)
    green_slope, green_intercept = np.polyfit(pre_peak_green_mjd["r:midpointMjdTai"], (pre_peak_green_mjd["r:psfFlux"]/green_peak_flux), 1)
    green_slopes.append(green_slope)
    red_slopes.append(red_slope)
    green_std_dev = np.std(pre_peak_green_mjd["r:psfFlux"]/green_peak_flux)
    red_std_dev = np.std(pre_peak_red_mjd["r:psfFlux"]/red_peak_flux)
    green_std_devs.append(green_std_dev)
    red_std_devs.append(red_std_dev)
    green_std_devs.append(green_std_dev)
    red_std_devs.append(red_std_dev)

# print (f"Average counts - Red: {np.average(red_counts_per_object):.4f}, Green: {np.average(green_counts_per_object):.4f}")
# print (f"Average slopes - Red: {np.average(red_slopes):.4f}, Green: {np.average(green_slopes):.4f}")
# print (f"Average standard deviations - Red: {np.average(red_std_devs):.4f}, Green: {np.average(green_std_devs):.4f}")
print (f"These are the slopes in the green band before peak\n{sorted(green_slopes)}")
print (f"These are the slopes in the red band before peak\n{sorted(red_slopes)}")