"""
Fits the Bazin et al. (2009) light curve model to a single band of an object's
light curve, using the feets library's BazinFit extractor rather than a hand-rolled
scipy.optimize.curve_fit implementation. feets wraps the `light-curve` (Rust) package
and supports MCMC-based fitting, which is more robust than plain least-squares for
this 5-parameter, highly nonlinear model - particularly given how sensitive Bazin
fits are known to be to initial guesses.

Install with: pip install feets
"""

# Written by Claude Sonnet 4.6 (Anthropic) with the prompt: "Yeah we will do the
# adoption of this method." (adopting feets.extractors.BazinFit rather than a
# custom curve_fit implementation), following review of feets' documented
# BazinFit extractor and its MCMC-based fitting algorithm.

import pandas as pd
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import light_curve as licu
from rate_of_rise_cut import chunked_light_curve_data


def fit_bazin_single_object(csv_path: Path, band: str, mcmc_niter: int, algorithm: str = "mcmc") -> dict | None:
    """Fits a Bazin fit to the single band of an objects light curve and returns the amplitude, baseline, reference time, 
    rise time, fall time and reduced chi squared values. 
    
    We are using this fit so that we can filter out objects that pass the linear slope fit but do not posses SN-like 
    shapes in their light curve. This should remove objects that have a rise but this function also uses the decline 
    of the object as well."""

    # initial_params = [
    #     peak_flux, # Amplitude
    #     np.median(flux) * 0.1, # Baseline
    #     peak_time, # Reference time
    #     (peak_time - time.min()) / 2,       # rise_time, rough guess
    #     (time.max() - peak_time) / 2,       # fall_time, rough guess
    # ]

    # object = pl.read_csv(csv_path, schema_overrides={"r:diaObjectId": pl.String}, columns=["r:diaObjectId"],)["r:diaObjectId"].to_list()
    light_curve_info = chunked_light_curve_data(object, 500)
    
    band_df = light_curve_info[light_curve_info["r:band"] == band].sort_values("r:midpointMjdTai")

        # As the bazin fit needs to fit 5 params we should have at least a decent number of points to get an okay fit with the minimum for the function to be 6.
    if len(band_df) < 6:
        print (f"Band {band} has {len(band_df)} which doesn't pass the 6 point cut-off")
        return None

    flux = band_df["r:psfFlux"].values.astype(np.float64)
    flux_err = band_df["r:psfFluxErr"].values.astype(np.float64)
    time = band_df["r:midpointMjdTai"].values.astype(np.float64)

    bazin = licu.BazinFit(algorithm, mcmc_niter=mcmc_niter)
    result = bazin(time, flux, flux_err)
    bazin_param_dict = {name: val for name, val in zip(bazin.names, result)}
    return bazin_param_dict

def bazin_batch_fits(csv_path: Path, light_curve_data_path: Path, band: str, algorithm: str, mcmc_niter: int, output_path: Path) -> pl.DataFrame:
    """Batch Bazin fits for batch testing and processing of the Bazin fitting to many objects.
    
    This is to be used for batch processing of light curves so that many objects can be fit to at once
    and it just goes really fast. """

    bazin = licu.BazinFit(algorithm=algorithm, mcmc_niter=mcmc_niter)

    lightcurves = []
    object_ids_in_order = []

    bazin_results = {"r:diaObjectId":[],
                    "amplitude":[], 
                    "baseline":[],
                    "reference_time":[],
                    "rise_time":[],
                    "fall_time":[],
                    "reduced_chi2":[]}

    # This is so when an object cannot be fitted by the Bazin fit the exception doesn't cause a crash and instead logs a Nan for the object
    fill_value = np.nan

    objects = pl.read_csv(csv_path, schema_overrides={"r:diaObjectId": pl.String}, columns=["r:diaObjectId"])["r:diaObjectId"].to_list()
    light_curve_data = pl.read_parquet(light_curve_data_path)
    light_curve_data_filtered = light_curve_data.filter(pl.col("r:diaObjectId").cast(pl.Utf8).is_in(objects))

    groups = {key[0]: group_df for key, group_df in light_curve_data_filtered.group_by("r:diaObjectId", maintain_order=True)}
    
    for object_id in objects:
        group_df = groups.get(int(object_id))
        if group_df is None:
            continue
        band_df = group_df.filter((pl.col("r:band") == band) & (pl.col("r:psfFlux") > 0)).sort("r:midpointMjdTai")

        # As the bazin fit needs to fit 5 params we should have at least a decent number of points to get an okay fit with the minimum for the function to be 6.
        if len(band_df) < 6:
                print (f"Band {band} has {len(band_df)} which doesn't pass the 6 point cut-off")
                continue
        
        flux = band_df["r:psfFlux"].to_numpy().astype(np.float64)
        flux_err = band_df["r:psfFluxErr"].to_numpy().astype(np.float64)
        time = band_df["r:midpointMjdTai"].to_numpy().astype(np.float64)
        peak_flux = band_df["r:psfFlux"].max()
        peak_time = band_df["r:midpointMjdTai"].max()

        lightcurves.append((time, flux, flux_err))
        object_ids_in_order.append(str(object_id))

    results = bazin.many(lightcurves, fill_value=fill_value, n_jobs= -1)

    for object_id, row in zip(object_ids_in_order, results):
        bazin_results["r:diaObjectId"].append(object_id)
        bazin_results["amplitude"].append(row[0])
        bazin_results["baseline"].append(row[1])
        bazin_results["reference_time"].append(row[2])
        bazin_results["rise_time"].append(row[3])
        bazin_results["fall_time"].append(row[4])
        bazin_results["reduced_chi2"].append(row[5])

    bazin_results_df = pl.DataFrame(bazin_results)

    return bazin_results_df.write_csv(output_path) 

def bazin_curve(t: np.ndarray, amplitude: float, t0: float, tau_rise: float, tau_fall: float, baseline: float) -> np.ndarray:
    """The Bazin et al. (2009) model, used here just to draw the fitted curve
    for visual inspection (feets returns the fitted parameters, not the curve
    itself)."""
    return amplitude * np.exp(-(t - t0) / tau_fall) / (1 + np.exp(-(t - t0) / tau_rise)) + baseline


def plot_bazin_fit_single_object(group_df: pd.DataFrame, band: str, fit_result: dict, diaObjectId: str, output_path: Path) -> None:
    """
    Quick visual check: plots the band's actual flux data points with the fitted
    Bazin curve overlaid, so the fit can be sanity-checked by eye.
    """

    band_df = group_df.filter((pl.col("r:band") == band) & (pl.col("r:psfFlux") > 0)).sort("r:midpointMjdTai")

    t = band_df["r:midpointMjdTai"].to_numpy().astype(np.float64)
    flux = band_df["r:psfFlux"].to_numpy().astype(np.float64)
    flux_err = band_df["r:psfFluxErr"].to_numpy().astype(np.float64)

    t_fine = np.linspace(t.min(), t.max(), 300)
    fit_flux = bazin_curve(
        t_fine,
        fit_result["amplitude"],
        fit_result["reference_time"],
        fit_result["rise_time"],
        fit_result["fall_time"],
        fit_result["baseline"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(t, flux, yerr=flux_err, fmt="o", color="darkred", label=f"{band}-band data", capsize=3, markersize=5)
    ax.plot(t_fine, fit_flux, "--", color="purple", label=f"Bazin fit (chi2={fit_result['reduced_chi2']:.2f})")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux (nanojansky)")
    ax.set_title(f"Bazin fit - {diaObjectId} ({band}-band)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f"{diaObjectId}_{i}_band.png")
    print (f"saved the plot of object {diaObjectId}")
    plt.close(fig)


def plot_bazin_fit_batch(dict_path: Path, csv_path: Path, light_curve_data_path: Path, band: str, output_path: Path) -> None:
    """
    Plots the light curve of the objects data in the band requested and overlays the fitted Bazin 
    line on the plot. 

    This is to check that the Bazin fit is both doing what we want it to do and gives up data for visual 
    inspection which is both valuable and neccessary. 
    """

    objects = pl.read_csv(csv_path, schema_overrides={"r:diaObjectId": pl.String}, columns=["r:diaObjectId"])["r:diaObjectId"].to_list()
    light_curve_data = pl.read_parquet(light_curve_data_path)
    light_curve_data_filtered = light_curve_data.filter(pl.col("r:diaObjectId").cast(pl.Utf8).is_in(objects))

    bazin_df = pl.read_csv(dict_path, schema_overrides={"r:diaObjectId": pl.String})
    bazin_fit_dict = bazin_df.to_pandas().set_index("r:diaObjectId").to_dict(orient="index")

    groups = {key[0]: group_df for key, group_df in light_curve_data_filtered.group_by("r:diaObjectId", maintain_order=True)}

    for object_id in objects:
        group_df = groups.get(int(object_id))
        if group_df is None:
            continue
        fit_result = bazin_fit_dict.get(object_id)
        if fit_result is None:
            print(f"Object {object_id} did not have any data in that band and couldn't be fitted to and thus won't be plotted.")
            continue
        plot_bazin_fit_single_object(group_df, band, fit_result, object_id, output_path)

if __name__ == "__main__":

    # object = ["170235219334922248"]
    # print (fit_bazin_single_object(object, "r", 10000, "mcmc"))
    bands = ["u", "g", "r", "i", "z", "y"]
    for i in bands:
        csv_path = Path.home() / "Documents"/"University"/"Honours"/"Honours_Project"/"data"/"test_data"/"20_off_variation_test_objects.csv"
        result = bazin_batch_fits(csv_path, "/home/riley/Documents/University/Honours/Honours_Project/data/light_curve_data/combined_light_curve_data.parquet", 
                            i, 
                            "mcmc", 
                            2000, 
                            Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "bazin_fits_results" / "20_object_test" / f"bazin_fits_{i}_band_results.csv")
        plot_bazin_fit_batch(Path.home() /"Documents"/"University"/"Honours"/"Honours_Project"/"data"/"bazin_fits_results"/"20_object_test"/ f"bazin_fits_{i}_band_results.csv", 
                        Path.home() / "Documents"/"University"/"Honours"/"Honours_Project"/"data"/"test_data"/"20_off_variation_test_objects.csv",
                        Path.home() / "Documents"/"University"/"Honours"/"Honours_Project"/"data"/"light_curve_data"/"combined_light_curve_data.parquet",
                        i,  
                        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "images" / "bazin_fits_plots" / "20_object_test" / f"{i}_band")