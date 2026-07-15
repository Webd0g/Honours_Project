from pathlib import Path
import os

# Define directories
image_dir = Path(Path.home() / "Documents/University/Honours/Honours_Project/images/SN_lightcurves/passed_rate_of_rise_cuts_results")
genuine_dir = Path(image_dir / "genuine_candidates")
flat_curve_dir = Path(image_dir / "flat_curves_with_noise")
declining_dir = Path(image_dir / "declining_objects")
single_band_dir = Path(image_dir / "single_band_detections")
anomaly_dir = Path(image_dir / "anomalies")

# Ensure directories exist
genuine_dir.mkdir(parents=True, exist_ok=True)
flat_curve_dir.mkdir(parents=True, exist_ok=True)
declining_dir.mkdir(parents=True, exist_ok=True)
single_band_dir.mkdir(parents=True, exist_ok=True)
anomaly_dir.mkdir(parents=True, exist_ok=True)

def parse_markdown(markdown_ids):
    categories = {
        "genuine_candidates": [],
        "flat_curves_with_noise": [],
        "declining_objects": [],
        "single_band_detections": [],
        "anomalies": []
    }

    lines = markdown_ids.split("\n")
    current_category = None

    for line_number, line in enumerate(lines, start=1):
        if line.startswith("##"):
            category_name = line[3:].strip()
            current_category = category_name.lower().replace(" ", "_").replace("-", "_")

            # Adjust category names based on the structure
            if category_name == "Genuine Rising transients":
                current_category = "genuine_candidates"
            elif category_name == "Declining Objects":
                current_category = "declining_objects"
            elif category_name == "Single Band Detections":
                current_category = "single_band_detections"
            elif category_name == "Anomaly":
                current_category = "anomalies"
        else:
            line = (line.strip().lower().replace('-', ''))[:19]
            if len(line) == 19:
                categories[current_category].append(line.strip())
            else:
                print (f"Line {line_number} does not fit the format for a Rubin diaObjectId")

    return categories

# Read the markdown_ids of the markdown file
with open(Path.home() / "Documents/University/Honours/Honours_Project/data/Rate_of_rise_cuts/Honours Project/SN Ia Rise Rate Cut Results Tally.md", "r") as file:
    markdown_ids = file.read()

categories = parse_markdown(markdown_ids)

def move_images_not_in_markdown(image_dir, categories):
    all_ids = set()
    items = os.listdir(image_dir)
    images = []
    images_not_in_markdown = []

    for id_list in categories.values():
        all_ids.update(id_list)

    for item in items:
        if item.endswith(".png"):
            images.append(item)
    
    for item in images:
        object_id = Path(item).stem
        if object_id not in all_ids:
            images_not_in_markdown.append(item)
    
    for image in images_not_in_markdown:
        image_path = image_dir / f"{image}"
        if image_path.exists():
            image_path.rename(flat_curve_dir / f"{image}")

def move_images_in_markdown(image_dir, target_dir ,ids):
    for id_str in ids:
        image_path = image_dir / f"{id_str}.png"
        if image_path.exists():
            image_path.rename(target_dir / f"{id_str}.png")

# Move images to respective directories
move_images_in_markdown(image_dir, genuine_dir, categories["genuine_candidates"])
move_images_in_markdown(image_dir, declining_dir, categories["declining_objects"])
move_images_in_markdown(image_dir, single_band_dir, categories["single_band_detections"])
move_images_in_markdown(image_dir, anomaly_dir, categories["anomalies"])
move_images_not_in_markdown(image_dir, categories)

print("Images have been moved to respective directories.")