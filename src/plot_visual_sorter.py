# quick xcheck of the object IDs that have plots of them and the ones that don't have plots of them to see if any of the objects that passed the rate of rise cuts didn't have plots generated for them.
import pandas as pd
import os
import light_curve_plotter as lcp
from pathlib import Path
import generic_light_curve_batch_plotter as glcbp

object_ids_passed_sn_near_galaxy = pd.read_csv("/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/sn_near_galaxy_candidates_passed.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"].tolist()
object_ids_passed_extragalactic_lt20mag = pd.read_csv("/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/extragalactic_candidates_passed.csv", dtype={"r:diaObjectId": str})["r:diaObjectId"].tolist()
object_ids_passed_in_tns = pd.read_csv("/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/in_tns_candidates_passed.csv", dtype={"r:diaObjectId":str})["r:diaObjectId"].tolist()
object_ids_passed = object_ids_passed_sn_near_galaxy + object_ids_passed_extragalactic_lt20mag + object_ids_passed_in_tns
object_ids_with_plots_false_positives = ([f.split(".")[0] for f in os.listdir("/home/riley/Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results/False_positives") if f.endswith(".png")])
object_ids_with_plots_probable_SN = ([f.split(".")[0] for f in os.listdir("/home/riley/Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results/Probable_Sn_Ia") if f.endswith(".png")])
object_ids_with_plots = object_ids_with_plots_false_positives + object_ids_with_plots_probable_SN

objects_not_with_plots = []
for object_id in object_ids_passed:
    if object_id not in object_ids_with_plots:
        objects_not_with_plots.append(object_id)
print (f"{(objects_not_with_plots)} passed the rate of rise cuts but doesn't have a plot generated for it.")

# fetching light curve info for the new objects that passed

chunk_size = 500
objects_not_with_plots_results =[]
output_path = "/home/riley/Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results"

if not objects_not_with_plots:
    print("No new objects to plot.")
else:
    for diaObjectId in range(0, len(objects_not_with_plots), chunk_size):
            objects_not_with_plots_light_curve_data = lcp.collect_light_curve_data(','.join(objects_not_with_plots[diaObjectId:diaObjectId + chunk_size]))
            if objects_not_with_plots_light_curve_data is not None:
                objects_not_with_plots_results.append(objects_not_with_plots_light_curve_data)

    if not objects_not_with_plots_results:
        raise RuntimeError("No light curve data returned for in_tns candidates")

    objects_not_with_plots_light_curve_info = pd.concat(objects_not_with_plots_results, ignore_index=True)
    objects_not_with_plots_light_curve_info.to_csv("/home/riley/Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results/unplotted_objects.csv", index=False)

    glcbp.batch_plotter("/home/riley/Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results/unplotted_objects.csv")

    # Written by Claude Sonnet 4.6 (Anthropic) with the prompt:
    # "write a script to sort light curve images into genuine_candidates and false_positives
    # subfolders based on a markdown tally file of visual inspection results"

    image_dir = Path("/home/riley/Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results")
    genuine_dir = image_dir / "Probable_Sn_Ia"
    false_positive_dir = image_dir / "False_positives"
    genuine_dir.mkdir(exist_ok=True)
    false_positive_dir.mkdir(exist_ok=True)

    markdown_path = Path("/home/riley/Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/Honours Project/SN Ia Rise Rate Cut Results Tally.md")

    # Parse genuine rising transient IDs from the markdown file
    genuine_ids = set()
    in_genuine_section = False
    with open(markdown_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "## Genuine Rising transients":
                in_genuine_section = True
                continue
            if line.startswith("##"):
                in_genuine_section = False
            if in_genuine_section and line.startswith("-"):
                # Extract just the ID (first token after the dash, ignore parenthetical notes)
                tokens = line.lstrip("- ").split()
                if not tokens:
                    continue
                object_id = tokens[0]
                if not object_id.isdigit():
                    continue
                genuine_ids.add(object_id)
                
    print(f"Found {len(genuine_ids)} genuine candidate IDs")

    moved_genuine = 0
    moved_false = 0

    for image_path in image_dir.glob("*.png"):
        object_id = image_path.stem
        if object_id in genuine_ids:
            image_path.rename(genuine_dir / image_path.name)
            moved_genuine += 1
        else:
            image_path.rename(false_positive_dir / image_path.name)
            moved_false += 1

    print(f"Moved {moved_genuine} images to genuine_candidates/")
    print(f"Moved {moved_false} images to false_positives/")