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

from pathlib import Path

import light_curve as licu
import numpy as np
import polars as pl


def fit_bazin_single_object(
    object_ids: list,
    light_curve_data_path: Path,
    band: str,
    algorithm: str,
    mcmc_niter: int,
) -> pl.DataFrame:
    """Fits a Bazin fit to the single band of an objects light curve and returns the amplitude, baseline, reference time,
    rise time, fall time and reduced chi squared values.

    We are using this fit so that we can filter out objects that pass the linear slope fit but do not posses SN-like
    shapes in their light curve. This should remove objects that have a rise but this function also uses the decline
    of the object as well."""

    object_ids = [str(object_id) for object_id in object_ids]

    light_curve_data = pl.read_parquet(light_curve_data_path).with_columns(
        pl.col("r:diaObjectId").cast(pl.String)
    )
    light_curve_data_filtered = light_curve_data.filter(
        pl.col("r:diaObjectId").is_in(object_ids)
    )

    groups = {
        str(key[0]): group_df
        for key, group_df in light_curve_data_filtered.group_by(
            "r:diaObjectId", maintain_order=True
        )
    }

    bazin_results = {
        "r:diaObjectId": [],
        "amplitude": [],
        "baseline": [],
        "reference_time": [],
        "rise_time": [],
        "fall_time": [],
        "reduced_chi2": [],
    }

    for object_id in object_ids:
        group_df = groups.get(object_id)
        if group_df is None:
            print(f"Object {object_id} has no light curve data for this band")
            continue
        band_df = group_df.filter(
            (pl.col("r:band") == band) & (pl.col("r:psfFlux") > 0)
        ).sort("r:midpointMjdTai")

        # As the bazin fit needs to fit 5 params we should have at least a decent number of points to get an okay fit with the minimum for the function to be 6.
        if len(band_df) < 6:
            print(
                f"Band {band} has {len(band_df)} which doesn't pass the 6 point cut-off"
            )
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
            10000.0,  # amplitude
            5000.0,  # baseline
            61120.0,  # reference time
            35.0,  # rise time
            100.0,  # fall time
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


def bazin_batch_fits(
    csv_path: Path,
    light_curve_data_path: Path,
    band: str,
    algorithm: str,
    mcmc_niter: int,
    output_path: Path,
) -> pl.DataFrame:
    """Batch Bazin fits for batch testing and processing of the Bazin fitting to many objects.

    This is to be used for batch processing of light curves so that many objects can be fit to at once
    and it just goes really fast."""

    bazin = licu.BazinFit(algorithm=algorithm, mcmc_niter=mcmc_niter)

    lightcurves = []
    object_ids_in_order = []

    bazin_results = {
        "r:diaObjectId": [],
        "amplitude": [],
        "baseline": [],
        "reference_time": [],
        "rise_time": [],
        "fall_time": [],
        "reduced_chi2": [],
    }

    # This is so when an object cannot be fitted by the Bazin fit the exception doesn't cause a crash and instead logs a Nan for the object
    fill_value = np.nan

    objects = pl.read_csv(
        csv_path,
        schema_overrides={"r:diaObjectId": pl.String},
        columns=["r:diaObjectId"],
    )["r:diaObjectId"].to_list()

    light_curve_data = pl.read_parquet(
        light_curve_data_path,
        columns=[
            "r:diaObjectId",
            "r:band",
            "r:psfFlux",
            "r:psfFluxErr",
            "r:midpointMjdTai",
        ],
    ).with_columns(pl.col("r:diaObjectId").cast(pl.String))

    light_curve_data_filtered = light_curve_data.filter(
        pl.col("r:diaObjectId").is_in(objects) & (pl.col("r:band") == band)
    )

    groups = {
        key[0]: group_df
        for key, group_df in light_curve_data_filtered.group_by(
            "r:diaObjectId", maintain_order=True
        )
    }

    for object_id in objects:
        group_df = groups.get(str(object_id))
        peak_mjd = group_df.sort("r:psfFlux", descending=True,)["r:midpointMjdTai"][0]
        if group_df is None:
            continue
        band_df = (
            group_df.filter(
                pl.col("r:psfFlux").is_finite()
                & pl.col("r:psfFluxErr").is_finite()
                & (pl.col("r:psfFluxErr") > 0)
                & pl.col("r:midpointMjdTai").is_between(peak_mjd - 30.0, peak_mjd + 70.0)
            )
            .sort("r:midpointMjdTai")
            .unique(subset=["r:midpointMjdTai"], keep="first", maintain_order=True)
        )

        time = band_df["r:midpointMjdTai"].to_numpy()

        if not np.all(np.isfinite(time)):
            print("Non-finite:", object_id)

        if np.any(np.diff(time) <= 0):
            print("Not strictly increasing:", object_id)
            print(time)

        # As the bazin fit needs to fit 5 params we should have at least a decent number of points to get an okay fit with the minimum for the function to be 6.
        if len(band_df) < 6:
            print(
                f"Band {band} has {len(band_df)} which doesn't pass the 6 point cut-off"
            )
            continue

        flux = band_df["r:psfFlux"].to_numpy().astype(np.float64)
        flux_err = band_df["r:psfFluxErr"].to_numpy().astype(np.float64)
        time = band_df["r:midpointMjdTai"].to_numpy().astype(np.float64)

        lightcurves.append((time, flux, flux_err))
        object_ids_in_order.append(str(object_id))

    results = bazin.many(lightcurves, fill_value=fill_value, n_jobs=-1)

    for object_id, row in zip(object_ids_in_order, results):
        bazin_results["r:diaObjectId"].append(object_id)
        bazin_results["amplitude"].append(row[0])
        bazin_results["baseline"].append(row[1])
        bazin_results["reference_time"].append(row[2])
        bazin_results["rise_time"].append(row[3])
        bazin_results["fall_time"].append(row[4])
        bazin_results["reduced_chi2"].append(row[5])

    bazin_results_df = pl.DataFrame(bazin_results)

    bazin_results_df.write_csv(output_path)

    return bazin_results_df


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data" / "forced_photometry_test"

    IMAGE_DIR = PROJECT_DIR / "images"

    LIGHT_CURVE_DATA_DIR = (
        DATA_DIR / "light_curve_data" / "forced_photometry_light_curve_data.parquet"
    )

    OBJECTS = DATA_DIR / "peak_timing_cut_results" / "pass.csv"

    OUTPUT_PATH = DATA_DIR / "bazin_fit_results"

    bazin_batch_fits(
        csv_path=OBJECTS,
        light_curve_data_path=LIGHT_CURVE_DATA_DIR,
        band="i",
        algorithm="mcmc",
        mcmc_niter=2000,
        output_path=OUTPUT_PATH / "bazin_fits_i_band_results.csv",
    )
