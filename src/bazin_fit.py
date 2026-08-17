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

def fit_bazin_single_object(object_ids: list, light_curve_data_path: Path, band: str, algorithm: str, mcmc_niter: int) -> pl.DataFrame:
    """Fits a Bazin fit to the single band of an objects light curve and returns the amplitude, baseline, reference time, 
    rise time, fall time and reduced chi squared values. 
    
    We are using this fit so that we can filter out objects that pass the linear slope fit but do not posses SN-like 
    shapes in their light curve. This should remove objects that have a rise but this function also uses the decline 
    of the object as well."""

    object_ids = [str(object_id) for object_id in object_ids]

    light_curve_data = (pl.read_parquet(light_curve_data_path).with_columns(pl.col("r:diaObjectId").cast(pl.String)))
    light_curve_data_filtered = light_curve_data.filter(pl.col("r:diaObjectId").is_in(object_ids))

    groups = {str(key[0]): group_df for key, group_df in light_curve_data_filtered.group_by("r:diaObjectId", maintain_order=True)}

    bazin_results = {"r:diaObjectId":[],
                    "amplitude":[], 
                    "baseline":[],
                    "reference_time":[],
                    "rise_time":[],
                    "fall_time":[],
                    "reduced_chi2":[]}

    for object_id in object_ids:
        group_df = groups.get(object_id)
        if group_df is None:
            print (f"Object {object_id} has no light curve data for this band")
            continue
        band_df = group_df.filter((pl.col("r:band") == band) & (pl.col("r:psfFlux") > 0)).sort("r:midpointMjdTai")

        # As the bazin fit needs to fit 5 params we should have at least a decent number of points to get an okay fit with the minimum for the function to be 6.
        if len(band_df) < 6:
                print (f"Band {band} has {len(band_df)} which doesn't pass the 6 point cut-off")
                continue

        flux = band_df["r:psfFlux"].to_numpy().astype(np.float64)
        flux_err = band_df["r:psfFluxErr"].to_numpy().astype(np.float64)
        time = band_df["r:midpointMjdTai"].to_numpy().astype(np.float64)

        # intial parameter calculation

        peak_index = np.argmax(flux)
        peak_flux = flux[peak_index]
        peak_time = time[peak_index]

        baseline_init = np.percentile(flux, 10)

        amplitude_init = peak_flux - baseline_init

        rise_time_init = peak_time - time.min()
        fall_time_init = time.max() - peak_time


        initial_params = [
    10000.0,   # amplitude
    5000.0,    # baseline
    61120.0,   # reference time
    35.0,      # rise time
    100.0,     # fall time
]

        print(f"\nObject: {object_id}")
        print(f"Band: {band}")
        print("Initial parameters:")
        print(f"  amplitude      = {amplitude_init}")
        print(f"  baseline       = {baseline_init}")
        print(f"  reference_time = {peak_time}")
        print(f"  rise_time      = {rise_time_init}")
        print(f"  fall_time      = {fall_time_init}")
        
        bazin = licu.BazinFit(
            algorithm=algorithm,
            mcmc_niter=mcmc_niter,
            init=initial_params,
        )

        result = bazin(
            time,
            flux,
            flux_err,
        )

        print("\nFit result:")
        print(dict(zip(bazin.names, result)))

        bazin_results["r:diaObjectId"].append(object_id)
        bazin_results["amplitude"].append(result[0])
        bazin_results["baseline"].append(result[1])
        bazin_results["reference_time"].append(result[2])
        bazin_results["rise_time"].append(result[3])
        bazin_results["fall_time"].append(result[4])
        bazin_results["reduced_chi2"].append(result[5])

    bazin_results_df = pl.DataFrame(bazin_results)

    # if output_path is not None:
    #     bazin_results_df.write_csv(output_path)

    return bazin_results_df

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
    light_curve_data_filtered = light_curve_data.filter(pl.col("r:diaObjectId").cast(pl.String).is_in(objects))

    groups = {key[0]: group_df for key, group_df in light_curve_data_filtered.group_by("r:diaObjectId", maintain_order=True)}
    
    for object_id in objects:
        group_df = groups.get(str(object_id))
        if group_df is None:
            continue
        band_df = group_df.filter((pl.col("r:band") == band) & (pl.col("r:psfFlux") > 0)).sort("r:midpointMjdTai").unique(subset=["r:midpointMjdTai"], keep="first", maintain_order=True)

        time = band_df["r:midpointMjdTai"].to_numpy()

        if not np.all(np.isfinite(time)):
            print("Non-finite:", object_id)

        if np.any(np.diff(time) <= 0):
            print("Not strictly increasing:", object_id)
            print(time)

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
        group_df = groups.get((object_id))
        if group_df is None:
            continue
        fit_result = bazin_fit_dict.get(object_id)
        print (fit_result)
        if fit_result is None:
            print(f"Object {object_id} did not have any data in that band and couldn't be fitted to and thus won't be plotted.")
            continue
        if plot_bazin_fit_single_object(group_df, band, fit_result, object_id, output_path):
            print(f"{object_id} has been plotted.")
        

if __name__ == "__main__":

    # object = ["313699514821640259"]
    # print (fit_bazin_single_object(object, "/home/riley/Documents/University/Honours/Honours_Project/data/light_curve_data/combined_light_curve_data.parquet", "i", "mcmc", 2000))
    bands = ["u", "g", "r", "i", "z", "y"]
    for i in bands:
        csv_path = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "timing_cuts_results" / "timing_cuts_passed.csv"
        result = bazin_batch_fits(csv_path, "/home/riley/Documents/University/Honours/Honours_Project/data/light_curve_data/combined_light_curve_data.parquet", 
                            i, 
                            "mcmc", 
                            2000, 
                            Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "bazin_fits_results" / f"bazin_fits_{i}_band_results.csv")
        plot_bazin_fit_batch(Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "bazin_fits_results" / f"bazin_fits_{i}_band_results.csv", 
                        csv_path,
                        Path.home() / "Documents"/"University"/"Honours"/"Honours_Project"/"data"/"light_curve_data"/"combined_light_curve_data.parquet",
                        i,  
                    Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "images" / "bazin_fits_plots" / f"{i}_band")