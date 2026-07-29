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
from feets import FeatureSpace
import matplotlib.pyplot as plt
import numpy as np
from rate_of_rise_cut import chunked_light_curve_data

BAZIN_FEATURES = [
    "BazinFit_Amplitude",
    "BazinFit_Baseline",
    "BazinFit_ReferenceTime",
    "BazinFit_RiseTime",
    "BazinFit_FallTime",
    "BazinFit_ReducedChi2",
]


def fit_bazin_single_band(group_df: pd.DataFrame, band: str, algorithm: str = "mcmc-lmsder", mcmc_niter: int = 128, lmsder_niter: int = 10) -> dict | None:
    """
    Fits the Bazin model to a single band of one object's light curve using
    feets.extractors.BazinFit.

    Returns a dict with the fitted amplitude, baseline, reference time, rise time,
    fall time, and reduced chi-squared, or None if there isn't enough data in this
    band to attempt a fit.
    """
    band_df = group_df[group_df["r:band"] == band].sort_values("r:midpointMjdTai")
    flux = band_df["r:psfFlux"].values
    flux_err = band_df["r:psfFluxErr"].values

    # Bazin has 5 free parameters; require a reasonable minimum of points to
    # attempt a meaningful fit at all.
    if len(band_df) < 6:
        return None

    fs = FeatureSpace(only=BAZIN_FEATURES, algorithm=algorithm, mcmc_niter=mcmc_niter, lmsder_niter=lmsder_niter)

    lc = {
        "time": band_df["r:midpointMjdTai"].values,
        "flux": band_df["r:psfFlux"].values,
        "flux_error": band_df["r:psfFluxErr"].values,
        "magnitude": band_df["r:psfFlux"].values,
        "error": band_df["r:psfFluxErr"].values,
    }

    try:
        features = fs.extract(**lc)
    except Exception as e:
        print(f"Bazin fit failed for band {band}: {e}")
        return None

    result = features[0]

    return {
        "band": band,
        "amplitude": result["BazinFit_Amplitude"],
        "baseline": result["BazinFit_Baseline"],
        "t0": result["BazinFit_ReferenceTime"],
        "tau_rise": result["BazinFit_RiseTime"],
        "tau_fall": result["BazinFit_FallTime"],
        "reduced_chi2": result["BazinFit_ReducedChi2"],
        "n_points": len(band_df),
    }

def bazin_curve(t: np.ndarray, amplitude: float, t0: float, tau_rise: float, tau_fall: float, baseline: float) -> np.ndarray:
    """The Bazin et al. (2009) model, used here just to draw the fitted curve
    for visual inspection (feets returns the fitted parameters, not the curve
    itself)."""
    return amplitude * np.exp(-(t - t0) / tau_fall) / (1 + np.exp(-(t - t0) / tau_rise)) + baseline


def plot_bazin_fit(group_df: pd.DataFrame, band: str, fit_result: dict, diaObjectId: str) -> None:
    """
    Quick visual check: plots the band's actual flux data points with the fitted
    Bazin curve overlaid, so the fit can be sanity-checked by eye.
    """
    band_df = group_df[group_df["r:band"] == band].sort_values("r:midpointMjdTai")

    t = band_df["r:midpointMjdTai"].values
    flux = band_df["r:psfFlux"].values
    flux_err = band_df["r:psfFluxErr"].values

    t_fine = np.linspace(t.min(), t.max(), 300)
    fit_flux = bazin_curve(
        t_fine,
        fit_result["amplitude"],
        fit_result["t0"],
        fit_result["tau_rise"],
        fit_result["tau_fall"],
        fit_result["baseline"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(t, flux, yerr=flux_err, fmt="o", color="black", label=f"{band}-band data", capsize=3, markersize=5)
    ax.plot(t_fine, fit_flux, "--", color="red", label=f"Bazin fit (chi2={fit_result['reduced_chi2']:.2f})")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux (nanojansky)")
    ax.set_title(f"Bazin fit - {diaObjectId} ({band}-band)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    test_id = ["170235219334922248"]
    light_curve_info = chunked_light_curve_data(test_id, 500)

    for object_id, group_df in light_curve_info.groupby("r:diaObjectId"):
        result = fit_bazin_single_band(group_df, "r",)
        print (result)
        # plot_bazin_fit(group_df, "r", result, object_id)
        