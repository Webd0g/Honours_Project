"""
This script cross-checks the visual inspection CSV against the current location of
each object's light curve plot, and moves any plot that isn't already sitting in the
directory matching its assigned category.
"""

# Written by Claude Sonnet 4.6 (Anthropic) with the prompt:
# "Here is what I have so far as a draft and I am kind of stuck with how to find the
# file that currently has that object ID... [draft script pasted]... could you make
# the edits and I will supervise please?" followed by "Now can you refactor it so
# each different step is its own function please?"

import pandas as pd
from pathlib import Path

# Define passed directories
PLOT_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "images" / "SN_lightcurves"
IMAGE_DIR_PASSED = PLOT_DIR / "passed_rate_of_rise_cuts_results"
GENUINE_DIR_PASSED = IMAGE_DIR_PASSED / "genuine_candidates"
FLAT_CURVE_DIR_PASSED = IMAGE_DIR_PASSED / "flat_curves_with_noise"
DECLINING_DIR_PASSED = IMAGE_DIR_PASSED / "declining_objects"
SINGLE_BAND_DIR_PASSED = IMAGE_DIR_PASSED / "single_band_detections"
ANOMALY_DIR_PASSED = IMAGE_DIR_PASSED / "anomalies"

# Define failed directories
IMAGE_DIR_FAILED = PLOT_DIR / "failed_rate_of_rise_cuts_results"
GENUINE_DIR_FAILED = IMAGE_DIR_FAILED / "genuine_candidates"
FLAT_CURVE_DIR_FAILED = IMAGE_DIR_FAILED / "flat_curves_with_noise"
DECLINING_DIR_FAILED = IMAGE_DIR_FAILED / "declining_objects"
SINGLE_BAND_DIR_FAILED = IMAGE_DIR_FAILED / "single_band_detections"
ANOMALY_DIR_FAILED = IMAGE_DIR_FAILED / "anomalies"

CSV_RESULTS_DIR = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "visual_inspection"

CATEGORIES = [
    'anomalies',
    'declining_objects',
    'flat_curves_with_noise',
    'genuine_candidates',
    'single_band_detections'
]


def build_object_id_lookup(image_dir: Path) -> dict[str, Path]:
    """
    Searches recursively across image_dir for every .png file and builds a
    lookup of diaObjectId (the file stem) -> its current Path.

    This is done once upfront so that finding an object's current location is a
    single dict lookup rather than a fresh filesystem search per object.
    """
    return {image.stem: image for image in image_dir.rglob("*.png")}


def build_expected_paths(csv_object_df: pd.DataFrame, image_dir: Path, categories: list[str]) -> dict[str, list[tuple[str, Path]]]:
    """
    Groups (diaObjectId, expected_path) pairs by category, based on the
    visual inspection CSV. expected_path is where that object's plot should
    live if it has already been sorted correctly.
    """
    constructed_dirs = {category: [] for category in categories}

    for _, row in csv_object_df.iterrows():
        if row["Category"] in constructed_dirs:
            expected_path = image_dir / row["Category"] / (row["diaObjectId"] + ".png")
            constructed_dirs[row["Category"]].append((row["diaObjectId"], expected_path))

    return constructed_dirs


def move_misplaced_plots(constructed_dirs: dict[str, list[tuple[str, Path]]], object_id_lookup: dict[str, Path]) -> tuple[int, int]:
    """
    For each expected (diaObjectId, expected_path) pair, checks whether the
    plot is already at expected_path. If not, looks up its current location
    and moves it there. Returns (moved_count, missing_count).
    """
    moved = 0
    missing = 0

    for category, entries in constructed_dirs.items():
        for object_id, expected_path in entries:
            if expected_path.exists():
                continue

            current_path = object_id_lookup.get(object_id)
            if current_path is None:
                print(f"No plot found on disk for {object_id} (expected category: {category})")
                missing += 1
                continue

            expected_path.parent.mkdir(parents=True, exist_ok=True)
            current_path.rename(expected_path)
            object_id_lookup[object_id] = expected_path
            moved += 1
            print(f"Moved {object_id}: {current_path} -> {expected_path}")

    return moved, missing


def sort_plots_from_csv(csv_path: Path, image_dir: Path, categories: list[str]) -> None:
    """
    Orchestrates the full sorting process: reads the visual inspection CSV,
    builds the current-location lookup, works out where each plot should be,
    and moves any plots that are in the wrong place.
    """
    csv_object_df = pd.read_csv(csv_path, dtype={"diaObjectId": str})

    object_id_lookup = build_object_id_lookup(image_dir)
    constructed_dirs = build_expected_paths(csv_object_df, image_dir, categories)
    moved, missing = move_misplaced_plots(constructed_dirs, object_id_lookup)

    print(f"\nDone. Moved {moved} plots, {missing} objects had no plot found on disk.")


if __name__ == "__main__":
    sort_plots_from_csv(CSV_RESULTS_DIR / "failed_visual_inspection_results.csv", IMAGE_DIR_FAILED, CATEGORIES)