from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from early_excess_salt_detection import loading_light_curve_data_and_objects


def fifty_percent_flux_calculation(
    objects_path: Path,
    light_curve_data_path: Path,
    band: str,
) -> pl.DataFrame:
    """Calculates the 50 per cent peak flux threshold for a specified band.

    Args:
        objects_path (Path): Path to the object catalog file.
        light_curve_data_path (Path): Path to the light curve data file.
        band (str): Band pass identifier to filter on (e.g., 'g').

    Returns:
        pl.DataFrame: Filtered light curve data with peak and 50% peak flux columns.
    """
    light_curve_data = loading_light_curve_data_and_objects(
        objects_path=objects_path, light_curve_data_path=light_curve_data_path
    )

    # Filter band and compute peak and 50% threshold in a single vectorized pass
    return light_curve_data.filter(pl.col("r:band") == band).with_columns(
        pl.col("r:psfFlux").max().over("r:diaObjectId").alias("peak_band_flux"),
        (pl.col("r:psfFlux").max().over("r:diaObjectId") * 0.5).alias(
            "50_per_cent_peak_flux"
        ),
    )


def find_fifty_per_cent_time(
    band_df: pl.DataFrame,
) -> pl.DataFrame:
    """Finds the midpointMjdTai closest to the 50% peak flux during the rising phase.

    Args:
        band_df (pl.DataFrame): Polars dataframe containing band data with peak_band_flux
            and 50_per_cent_peak_flux columns.

    Returns:
        pl.DataFrame: One row per object representing the epoch closest to 50% peak flux.
    """
    # 1. Get the first MJD where peak flux occurs per object (handles potential duplicate peak values)
    peak_mjd_df = (
        band_df.filter(pl.col("r:psfFlux") == pl.col("peak_band_flux"))
        .group_by("r:diaObjectId")
        .agg(pl.col("r:midpointMjdTai").min().alias("peak_mjd"))
    )

    # 2. Join peak MJD, filter to rising phase (<= peak_mjd), and compute delta_flux
    rise_df = (
        band_df.join(peak_mjd_df, on="r:diaObjectId", how="inner")
        .filter(pl.col("r:midpointMjdTai") <= pl.col("peak_mjd"))
        .with_columns(
            (pl.col("r:psfFlux") - pl.col("50_per_cent_peak_flux"))
            .abs()
            .alias("delta_flux")
        )
    )

    # 3. Find the point closest to 50% flux for each object on the rise phase
    fifty_per_cent_df = rise_df.filter(
        pl.col("delta_flux") == pl.col("delta_flux").min().over("r:diaObjectId")
    )

    # 4. Break ties if two epochs are equidistant from 50% flux
    return fifty_per_cent_df.unique(subset=["r:diaObjectId"], keep="first")


def iterate_over_bands(
    objects_dir: Path,
    light_curve_data_dir: Path,
    bands: list[str],
) -> pl.DataFrame:
    """Calculates 50% peak flux epochs across specified bands with fallback priority.

    Objects are evaluated in the order of the `bands` list. An object present in an
    earlier band will not be duplicated by later bands.

    Args:
        objects_dir (Path): Path to object catalog data.
        light_curve_data_dir (Path): Path to light curve data.
        bands (list[str]): List of bands in fallback priority order (e.g. ['g', 'r', 'i']).

    Returns:
        pl.DataFrame: DataFrame containing reference 50% peak flux timing (t50_mjd).
    """
    # Load light curve data once to prevent repeated disk read overhead
    light_curve_data = loading_light_curve_data_and_objects(
        objects_path=objects_dir, light_curve_data_path=light_curve_data_dir
    )

    processed_dfs: list[pl.DataFrame] = []

    for band in bands:
        # Calculate peak for current band
        band_df = light_curve_data.filter(pl.col("r:band") == band).with_columns(
            pl.col("r:psfFlux").max().over("r:diaObjectId").alias("peak_band_flux"),
            (pl.col("r:psfFlux").max().over("r:diaObjectId") * 0.5).alias(
                "50_per_cent_peak_flux"
            ),
        )

        if band_df.is_empty():
            continue

        # Get 50% time for objects in current band
        filtered_result = find_fifty_per_cent_time(band_df=band_df)

        # Anti-join against all previously collected objects to maintain priority order
        if processed_dfs:
            existing_objects = pl.concat(
                [df.select("r:diaObjectId") for df in processed_dfs]
            ).unique()

            filtered_result = filtered_result.join(
                existing_objects, on="r:diaObjectId", how="anti"
            )

        processed_dfs.append(filtered_result)

    if not processed_dfs:
        return pl.DataFrame()

    # Combine results across all priority levels
    final_reference_points = pl.concat(processed_dfs)

    return final_reference_points.rename({"r:midpointMjdTai": "t50_mjd"}).select(
        [
            "r:diaObjectId",
            "r:diaSourceId",
            "r:band",
            "r:psfFlux",
            "r:psfFluxErr",
            "t50_mjd",
            "peak_band_flux",
            "50_per_cent_peak_flux",
            "peak_mjd",
            "delta_flux",
        ]
    )


def keep_observations_upto_t50(
    objects: Path,
    light_curve_data_path: Path,
    reference_df: pl.DataFrame,
) -> pl.DataFrame:
    """Filters light curve observations to keep only points up to t50_mjd.

    Args:
        objects (Path): Path to the object catalog file.
        light_curve_data_path (Path): Path to the light curve data file.
        reference_df (pl.DataFrame): Reference DataFrame containing `t50_mjd` per object
            from `iterate_over_bands`.

    Returns:
        pl.DataFrame: Light curve observations up to `t50_mjd` for each object's reference band.
    """
    light_curve_data = loading_light_curve_data_and_objects(
        objects_path=objects,
        light_curve_data_path=light_curve_data_path,
    )

    # Join on both object ID and band to retain observations for the chosen reference band
    return (
        light_curve_data.join(
            reference_df.select("r:diaObjectId", "r:band", "t50_mjd"),
            on=["r:diaObjectId", "r:band"],
            how="inner",
        )
        .filter(pl.col("r:midpointMjdTai") <= pl.col("t50_mjd"))
        .select(
            [
                "r:diaObjectId",
                "r:diaSourceId",
                "r:band",
                "r:midpointMjdTai",
                "r:psfFlux",
                "r:psfFluxErr",
                "t50_mjd",
            ]
        )
    )


def power_law_evaluator():
    """Just a function that was used to test the behaviour of the power law to see that the outputted flux increases with time with exponential behaviour."""

    A_g = 2

    alpha_g = 2

    t_fl = 100

    amplitudes = {
        "g": A_g,
        # "r": A_r,
        # "i": A_i,
        # "z": A_z,
    }

    alphas = {
        "g": alpha_g,
        # "r": alpha_r,
        # "i": alpha_i,
        # "z": alpha_z,
    }

    times = np.linspace(95, 110, 50)

    after_first_light = times >= t_fl

    g_band_flux = np.zeros_like(times)

    g_band_flux[after_first_light] = (
        amplitudes["g"] * (times[after_first_light] - t_fl) ** alphas["g"]
    )

    print(g_band_flux)


def power_law_model(
    times: np.ndarray | list[float],
    t_fl: float,
    A_prime: float,
    alpha: float,
) -> np.ndarray:
    """Computes the early power-law flux model for SN Ia light curves (Müller-Bravo et al., 2026).

    Model: f(t) = A_prime * ((t - t_fl) / 10)^alpha  for t >= t_fl
            f(t) = 0                                   for t < t_fl

    Args:
        times (np.ndarray): Rest-frame observation times up to t50.
        t_fl (float): Time of first light in rest-frame days relative to t0.
        A_prime (float): Power-law amplitude coefficient at t - t_fl = 10 days.
        alpha (float): Power-law index.

    Returns:
        np.ndarray: Model flux values evaluated at input times.
    """
    t_arr = np.asarray(times, dtype=float)
    model_flux = np.zeros_like(t_arr)

    # Mask observations occurring after first light
    mask = t_arr > t_fl
    if not np.any(mask):
        return model_flux

    # Numerically stable formulation of A' * 10^(-alpha) * (t - t_fl)^alpha
    model_flux[mask] = A_prime * ((t_arr[mask] - t_fl) / 10.0) ** alpha

    return model_flux


def rest_frame_conversion(
    object_salt_data: Path,
    pre_t50_objects_data: pl.DataFrame,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Converts observer-frame MJD times and first-detection times to rest-frame days relative to t0.

    Formula: t_rest = (t_obs - t0) / (1 + z)

    Args:
        object_salt_data (Path): Path to CSV containing SALT fit parameters (t0, z_used, t_fd).
        pre_t50_objects_data (pl.DataFrame): DataFrame containing light curve observations up to t50.
        output_path (Path | None): Optional path to save the resulting DataFrame as CSV.

    Returns:
        pl.DataFrame: DataFrame containing rest-frame times ('rest_frame_time', 't_fd_rest').
    """
    # 1. Read required SALT parameters using schema overrides
    salt_data = pl.read_csv(
        object_salt_data,
        columns=["r:diaObjectId", "t0", "z_used", "t_fd"],
        schema={
            "r:diaObjectId": pl.String,
            "t0": pl.Float64,
            "z_used": pl.Float64,
            "t_fd": pl.Float64,
        },
    )

    # 2. Join SALT parameters and compute rest-frame transformations in a single pass
    rest_frame_df = pre_t50_objects_data.join(
        salt_data, on="r:diaObjectId", how="inner"
    ).with_columns(
        ((pl.col("r:midpointMjdTai") - pl.col("t0")) / (1.0 + pl.col("z_used"))).alias(
            "rest_frame_time"
        ),
        ((pl.col("t_fd") - pl.col("t0")) / (1.0 + pl.col("z_used"))).alias("t_fd_rest"),
    )

    # 3. Export to CSV if output path is specified
    if output_path is not None:
        rest_frame_df.write_csv(output_path)

    return rest_frame_df


def log_prior_function(
    theta: np.ndarray,
    band_data: dict,
    t_fd_rest: float,
) -> float:
    """Computes log-prior using Equation A2 from Müller-Bravo et al. (2026).

    P(A', alpha) proportional to A'^(-1) * 10^(-alpha)
    => log P(A', alpha) = -ln(A') - alpha * ln(10)

    Args:
        theta (np.ndarray): Parameter array [t_fl, A_prime_1, alpha_1, ...].
        band_data (dict): Dictionary of active fitting bands.
        t_fd_rest (float): Rest-frame time of first detection.

    Returns:
        float: Log-prior probability (-inf if out of bounds).
    """
    t_fl = theta[0]

    # First light must precede first detection
    if not (-100.0 < t_fl < t_fd_rest):
        return -np.inf

    log_prior = 0.0
    num_bands = len(band_data)

    for index in range(num_bands):
        a_prime = theta[1 + 2 * index]
        alpha = theta[2 + 2 * index]

        # Enforce positive amplitude & reasonable physical bounds on exponent
        if a_prime <= 0.0:
            return -np.inf

        # Constrain alpha to avoid floating-point overflow during MCMC steps
        if not (1.0 <= alpha <= 1e8):
            return -np.inf

        # Equation A2 prior term: -ln(A') - alpha * ln(10)
        log_prior += -np.log(a_prime) - (alpha * np.log(10.0))

    return log_prior


def log_likelihood_function(
    theta: np.ndarray,
    band_data: dict,
) -> float:
    """Computes the Gaussian log-likelihood for early light curve power-law fitting across bands.

    Args:
        theta (np.ndarray): Parameter vector [t_fl, A_prime_1, alpha_1, ...].
        band_data (dict): Nested dictionary structured as:
            {
                'g': {'times': np.ndarray, 'flux': np.ndarray, 'flux_error': np.ndarray},
                ...
            }

    Returns:
        float: Total log-likelihood value across all fitted bands.
    """
    t_fl = theta[0]
    log_likelihood = 0.0

    for index, (band, data) in enumerate(band_data.items()):
        a_prime = theta[1 + 2 * index]
        alpha = theta[2 + 2 * index]

        times = data["times"]
        flux = data["flux"]
        flux_err = data["flux_error"]

        # Safeguard against bad/zero error values
        if np.any(flux_err <= 0) or np.any(np.isnan(flux_err)):
            return -np.inf

        model_flux = power_law_model(
            times=times,
            t_fl=t_fl,
            A_prime=a_prime,
            alpha=alpha,
        )

        residual = flux - model_flux
        inv_variance = 1.0 / (flux_err**2)

        # Standard Gaussian log-likelihood
        log_likelihood += -0.5 * np.sum(
            (residual**2) * inv_variance + np.log(2.0 * np.pi * (flux_err**2))
        )

    return float(log_likelihood)


def log_probability_function(
    theta: np.ndarray,
    band_data: dict,
    t_fd_rest: float,
) -> float:
    """Computes the total log-posterior probability (log-prior + log-likelihood).

    Args:
        theta (np.ndarray): Parameter vector [t_fl, A_prime_1, alpha_1, ...].
        band_data (dict): Dictionary containing time, flux, and error arrays per band.
        t_fd_rest (float): Rest-frame time of first detection.

    Returns:
        float: Total log-probability (-inf if prior or likelihood is invalid).
    """
    lp = log_prior_function(
        theta=theta,
        band_data=band_data,
        t_fd_rest=t_fd_rest,
    )

    # Short-circuit likelihood calculation if prior is invalid (-inf)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood_function(
        theta=theta,
        band_data=band_data,
    )

    if not np.isfinite(ll):
        return -np.inf

    return float(lp + ll)


def fitting_power_law(
    rest_frame_objects: pl.DataFrame,
    n_steps: int,
    fit_df_output_path: Path | None = None,
) -> pl.DataFrame:
    """Fits the early excess power-law model using emcee MCMC across rest-frame light curves.

    Args:
        rest_frame_objects (pl.DataFrame): Rest-frame transformed observations up to t50.
        n_steps (int): Total MCMC steps to run per object.
        fit_df_output_path (Path | None): Optional CSV output path for fitted parameters.

    Returns:
        pl.DataFrame: Summary table containing posterior percentiles for fitted parameters.
    """
    results = {
        "r:diaObjectId": [],
        "t_fl": [],
        "t_fl_err_low": [],
        "t_fl_err_high": [],
        "A_prime_g": [],
        "A_prime_g_err_low": [],
        "A_prime_g_err_high": [],
        "alpha_g": [],
        "alpha_g_err_low": [],
        "alpha_g_err_high": [],
        "A_prime_r": [],
        "A_prime_r_err_low": [],
        "A_prime_r_err_high": [],
        "alpha_r": [],
        "alpha_r_err_low": [],
        "alpha_r_err_high": [],
        "A_prime_i": [],
        "A_prime_i_err_low": [],
        "A_prime_i_err_high": [],
        "alpha_i": [],
        "alpha_i_err_low": [],
        "alpha_i_err_high": [],
        "t_fd": [],
        "t_fd_rest": [],
        "z_used": [],
        "t50_mjd": [],
        "sampler_nwalkers": [],
        "sampler_nsteps": [],
    }

    # Dynamically scale burn-in to avoid discarding more steps than were executed
    burn_in = min(500, int(n_steps * 0.3))

    for (object_id,), object_df in rest_frame_objects.group_by(
        "r:diaObjectId", maintain_order=True
    ):
        band_data = {}
        for current_band in ["g", "r", "i"]:
            band_df = object_df.filter(pl.col("r:band") == current_band)
            if band_df.height > 0:
                band_data[current_band] = {
                    "times": band_df["rest_frame_time"].to_numpy(),
                    "flux": band_df["r:psfFlux"].to_numpy(),
                    "flux_error": band_df["r:psfFluxErr"].to_numpy(),
                }

        if not band_data:
            continue

        t_fd_rest = object_df["t_fd_rest"].first()
        fit_bands = list(band_data.keys())

        # Construct parameter vector [t_fl, A_prime_1, alpha_1, ...]
        # Center initial t_fl safely prior to first detection
        initial_t_fl = min(-18.0, t_fd_rest - 2.0)
        theta = [initial_t_fl]

        for _ in fit_bands:
            theta.extend([2.0, 2.0])  # A' = 2.0, alpha = 2.0 (fireball index)

        theta = np.array(theta)
        ndim = len(theta)
        nwalkers = max(32, 2 * ndim)

        # Initialize walkers with tight scatter to guarantee valid initial log-prior
        pos = theta + 1e-3 * np.random.randn(nwalkers, ndim)

        # Clamp initial walker positions into strict valid prior bounds
        pos[:, 0] = np.clip(pos[:, 0], -45.0, t_fd_rest - 0.1)
        for b_idx in range(len(fit_bands)):
            pos[:, 1 + 2 * b_idx] = np.maximum(pos[:, 1 + 2 * b_idx], 0.01)  # A' > 0
            pos[:, 2 + 2 * b_idx] = np.clip(
                pos[:, 2 + 2 * b_idx], 1.0, 4.5
            )  # 1 <= alpha <= 5

        sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=log_probability_function,
            args=(band_data, t_fd_rest),
        )

        sampler.run_mcmc(initial_state=pos, nsteps=n_steps, progress=False)

        flat_samples = sampler.get_chain(discard=burn_in, thin=10, flat=True)

        # Extract percentiles for t_fl
        q16, t_fl_val, q84 = np.percentile(flat_samples[:, 0], [16, 50, 84])
        results["t_fl"].append(float(t_fl_val))
        results["t_fl_err_low"].append(float(t_fl_val - q16))
        results["t_fl_err_high"].append(float(q84 - t_fl_val))

        # Extract band-specific posteriors
        for band in ["g", "r", "i"]:
            if band in fit_bands:
                index = fit_bands.index(band)
                a_idx, alpha_idx = 1 + 2 * index, 2 + 2 * index

                a_q16, a_val, a_q84 = np.percentile(
                    flat_samples[:, a_idx], [16, 50, 84]
                )
                results[f"A_prime_{band}"].append(float(a_val))
                results[f"A_prime_{band}_err_low"].append(float(a_val - a_q16))
                results[f"A_prime_{band}_err_high"].append(float(a_q84 - a_val))

                al_q16, al_val, al_q84 = np.percentile(
                    flat_samples[:, alpha_idx], [16, 50, 84]
                )
                results[f"alpha_{band}"].append(float(al_val))
                results[f"alpha_{band}_err_low"].append(float(al_val - al_q16))
                results[f"alpha_{band}_err_high"].append(float(al_q84 - al_val))
            else:
                results[f"A_prime_{band}"].append(None)
                results[f"A_prime_{band}_err_low"].append(None)
                results[f"A_prime_{band}_err_high"].append(None)

                results[f"alpha_{band}"].append(None)
                results[f"alpha_{band}_err_low"].append(None)
                results[f"alpha_{band}_err_high"].append(None)

        results["r:diaObjectId"].append(object_id)
        results["t_fd"].append(object_df["t_fd"].first())
        results["t_fd_rest"].append(object_df["t_fd_rest"].first())
        results["z_used"].append(object_df["z_used"].first())
        results["t50_mjd"].append(object_df["t50_mjd"].first())
        results["sampler_nsteps"].append(n_steps)
        results["sampler_nwalkers"].append(nwalkers)

    results_df = pl.DataFrame(results)

    if fit_df_output_path is not None:
        results_df.write_csv(fit_df_output_path)

    return results_df


def excluding_excess_window(
    rest_frame_conversion_df: pl.DataFrame,
    band: str | list[str],
    excess_window_days: float = 3.0,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Removes observations falling within the early excess window [t_fd_rest, t_fd_rest + excess_window_days].

    Args:
        rest_frame_conversion_df (pl.DataFrame): Rest-frame transformed light curve data.
        band (str | list[str]): Passband identifier(s) to exclude early excess from (e.g., 'g' or ['g', 'r']).
        excess_window_days (float, optional): Window size in rest-frame days after first detection. Defaults to 3.0.
        output_path (Path | None, optional): Optional CSV output path for filtered data. Defaults to None.

    Returns:
        pl.DataFrame: DataFrame with points inside the specified excess window removed.
    """
    bands_to_filter = [band] if isinstance(band, str) else band

    # Identify points strictly inside the excess window for target bands
    in_excess_window = pl.col("r:band").is_in(bands_to_filter) & pl.col(
        "rest_frame_time"
    ).is_between(
        pl.col("t_fd_rest"),
        pl.col("t_fd_rest") + excess_window_days,
        closed="both",
    )

    # Keep all observations outside the excess window
    filtered_df = rest_frame_conversion_df.filter(in_excess_window.not_())

    if output_path is not None:
        filtered_df.write_csv(output_path)

    return filtered_df


def walkers_chain_plot(
    sampler_results: emcee.EnsembleSampler,
    labels: list[str],
    discard: int = 0,
    output_path: Path | None = None,
) -> None:
    """Plot the walkers behaviour as a function of the chain length to see how they converge and if there is any degeneracies between the parameters.

    Args:
        sampler_results (emcee.EnsembleSampler): The results from the sampling process which is the second index of the result from the fitting power law function.
        labels (list[str]): List of labels that are used for the labels of the y axis for the plot.
        discard (int, optional): The number of steps to discard before doing the plotting. Defaults to 0.
        output_path (Path | None, optional): Output path to where the plot can be saved if desired. Defaults to None.
    """

    samples = sampler_results.get_chain(
        discard=discard,
    )

    (
        nsteps,
        nwalkers,
        ndim,
    ) = samples.shape

    fig, axes = plt.subplots(
        ndim,
        1,
        figsize=(10, 2 * ndim),
        sharex=True,
    )

    if ndim == 1:
        axes = [axes]

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(
            0,
            nsteps,
        )
        if labels:
            ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("Step Number")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(
            output_path,
        )
    plt.show()


def fit_sample_parameters_plot(
    pl_fit_params_df: pl.DataFrame,
    rest_frame_no_eex: pl.DataFrame,
    full_rest_data_df: pl.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Plots the median sampled power-law values against light curve observations and pulls.

    Args:
        pl_fit_params_df (pl.DataFrame): DataFrame output from fitting_power_law.
        rest_frame_no_eex (pl.DataFrame): DataFrame with excess window removed.
        full_rest_data_df (pl.DataFrame): Complete rest-frame light curve DataFrame.
        output_path (Path | None, optional): Output directory to save generated plots.
    """
    # 1. Fixed indexing bug: iterate directly over pl_fit_params_df
    for (diaObjectId,), object_df in pl_fit_params_df[0].group_by(
        "r:diaObjectId", maintain_order=True
    ):
        rest_band_data = {}
        band_data = {}

        for current_band in ["g", "r", "i"]:
            # Complete dataset for plotting
            rest_band_df = full_rest_data_df.filter(
                (pl.col("r:band") == current_band)
                & (pl.col("r:diaObjectId") == diaObjectId)
            )
            if rest_band_df.height > 0:
                rest_band_data[current_band] = {
                    "times": rest_band_df["rest_frame_time"].to_numpy(),
                    "flux": rest_band_df["r:psfFlux"].to_numpy(),
                    "flux_error": rest_band_df["r:psfFluxErr"].to_numpy(),
                }

            # Fitted dataset (without excess window)
            band_df = rest_frame_no_eex.filter(
                (pl.col("r:band") == current_band)
                & (pl.col("r:diaObjectId") == diaObjectId)
            )
            if band_df.height > 0:
                band_data[current_band] = {
                    "times": band_df["rest_frame_time"].to_numpy(),
                    "flux": band_df["r:psfFlux"].to_numpy(),
                    "flux_error": band_df["r:psfFluxErr"].to_numpy(),
                }

        # Filter bands that actually have parameters fit
        fit_bands = [
            band
            for band in band_data
            if object_df[f"A_prime_{band}"].item() is not None
        ]

        if not fit_bands:
            continue

        model_parameters = {
            band: {
                "A_prime": object_df[f"A_prime_{band}"].item(),
                "alpha": object_df[f"alpha_{band}"].item(),
            }
            for band in fit_bands
        }

        t_fl_med = object_df["t_fl"].item()
        t_fd_rest = object_df["t_fd_rest"].item()

        # Build smooth grid across the full rest_band_data time span
        model_grids = {}
        model_flux = {}
        for band in fit_bands:
            times_full = rest_band_data[band]["times"]
            model_grids[band] = np.linspace(np.min(times_full), np.max(times_full), 300)
            model_flux[band] = power_law_model(
                times=model_grids[band],
                t_fl=t_fl_med,
                A_prime=model_parameters[band]["A_prime"],
                alpha=model_parameters[band]["alpha"],
            )

        fig, (axes_fit, axes_resid) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            height_ratios=[3, 1],
            figsize=(8, 5),
        )

        for band in fit_bands:
            times = rest_band_data[band]["times"]
            flux = rest_band_data[band]["flux"]
            flux_error = rest_band_data[band]["flux_error"]

            in_ranges = (times >= t_fd_rest) & (times <= t_fd_rest + 3)

            # Excess points (faded)
            plot_data = axes_fit.errorbar(
                x=times[in_ranges],
                y=flux[in_ranges],
                yerr=flux_error[in_ranges],
                fmt="o",
                capsize=3,
                mec="white",
                mew=1,
                alpha=0.3,
            )

            data_colour = plot_data.lines[0].get_color()

            # Non-excess points (solid)
            axes_fit.errorbar(
                x=times[~in_ranges],
                y=flux[~in_ranges],
                yerr=flux_error[~in_ranges],
                fmt="o",
                capsize=3,
                mec="white",
                color=data_colour,
                mew=1,
                label=f"lsst{band}",
            )

            # Smooth power law model curve
            axes_fit.plot(
                model_grids[band],
                model_flux[band],
                color=data_colour,
            )

            # Model evaluated at observation points for residual calculation
            model_at_data = power_law_model(
                times=times,
                t_fl=t_fl_med,
                A_prime=model_parameters[band]["A_prime"],
                alpha=model_parameters[band]["alpha"],
            )

            residuals = (flux - model_at_data) / flux_error

            # Residuals for excess points (faded - matched to top plot)
            axes_resid.errorbar(
                x=times[in_ranges],
                y=residuals[in_ranges],
                fmt="o",
                capsize=3,
                mec="white",
                color=data_colour,
                mew=1,
                alpha=1,
            )

            # Residuals for non-excess points (solid - matched to top plot)
            axes_resid.errorbar(
                x=times[~in_ranges],
                y=residuals[~in_ranges],
                fmt="o",
                capsize=3,
                mec="white",
                color=data_colour,
                mew=1,
                alpha=0.3,
            )

        # Draw reference lines and regions
        axes_resid.axhline(c="black")
        axes_resid.axhspan(
            -5, 5, color="grey", linestyle=":", alpha=0.12, label=r"$5 \sigma$ region"
        )
        axes_resid.axhspan(
            -3, 3, color="grey", linestyle="--", alpha=0.25, label=r"$3 \sigma$ region"
        )

        for ax in (axes_fit, axes_resid):
            ax.axvline(
                x=t_fd_rest,
                linestyle="--",
                label=rf"$t_{{fd}}={t_fd_rest:.2f}$ d" if ax == axes_fit else None,
                color="#0072B2",
            )
            ax.axvline(
                x=t_fd_rest + 3,
                linestyle=":",
                label=rf"$t_{{fd}} + 3d ={t_fd_rest + 3:.2f}$ d"
                if ax == axes_fit
                else None,
                color="#0072B2",
            )

        axes_fit.axvline(
            t_fl_med,
            linestyle="--",
            label=rf"$t_{{\mathrm{{fl}}}}={t_fl_med:.2f}$ d",
            color="#D55E00",
        )

        axes_fit.set_ylabel("Flux (nJy)")
        axes_fit.set_title(f"{diaObjectId}")
        axes_resid.set_xlabel(r"Rest-frame days since $B$-band peak")
        axes_resid.set_ylabel("Pull")
        axes_fit.legend(loc="best", frameon=False)

        plt.tight_layout()

        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path / f"{diaObjectId}_eex_power_law_fit.jpg",
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()
        plt.close(fig)


def eex_detection(
    pl_fit_params_df: pl.DataFrame,
    full_rest_frame_lc_df: pl.DataFrame,
    pass_output_path: Path | None = None,
    fail_output_path: Path | None = None,
    values_output_path: Path | None = None,
) -> pl.DataFrame:
    """Calculates pulls for rest-frame light curves and flags early excess candidates.

    Objects are flagged as having an early excess if they have pull >= 5 across at least
    2 distinct nights within the excess window [t_fd_rest, t_fd_rest + 3].

    Args:
        pl_fit_params_df (pl.DataFrame): Output DataFrame from fitting_power_law.
        full_rest_frame_lc_df (pl.DataFrame): Full rest-frame light curve observations.
        pass_output_path (Path | None): Path to save passing objects CSV.
        fail_output_path (Path | None): Path to save failing objects CSV.
        values_output_path (Path | None): Path to save detailed pull calculations CSV.

    Returns:
        pl.DataFrame: pl_fit_params_df augmented with an 'early_excess' boolean column.
    """
    pull_records = []

    # Iterate over fitted objects directly
    for (diaObjectId,), object_df in pl_fit_params_df[0].group_by(
        "r:diaObjectId", maintain_order=True
    ):
        rest_band_data = {}

        for current_band in ["g", "r", "i"]:
            rest_band_df = full_rest_frame_lc_df.filter(
                (pl.col("r:band") == current_band)
                & (pl.col("r:diaObjectId") == diaObjectId)
            )

            if rest_band_df.height > 0:
                rest_band_data[current_band] = {
                    "diaSourceId": rest_band_df["r:diaSourceId"].to_numpy(),
                    "times": rest_band_df["rest_frame_time"].to_numpy(),
                    "flux": rest_band_df["r:psfFlux"].to_numpy(),
                    "flux_error": rest_band_df["r:psfFluxErr"].to_numpy(),
                }

        available_bands = list(rest_band_data.keys())
        fit_bands = [
            b for b in available_bands if object_df[f"A_prime_{b}"].item() is not None
        ]

        if not fit_bands:
            continue

        t_fl_med = object_df["t_fl"].item()

        for band in fit_bands:
            times = rest_band_data[band]["times"]
            flux = rest_band_data[band]["flux"]
            flux_err = rest_band_data[band]["flux_error"]

            model_at_data = power_law_model(
                times=times,
                t_fl=t_fl_med,
                A_prime=object_df[f"A_prime_{band}"].item(),
                alpha=object_df[f"alpha_{band}"].item(),
            )

            pulls = (flux - model_at_data) / flux_err

            for source_id, p in zip(rest_band_data[band]["diaSourceId"], pulls):
                pull_records.append(
                    {
                        "r:diaSourceId": source_id,
                        "pull": float(p),
                    }
                )

    if not pull_records:
        pull_df = pl.DataFrame(schema={"r:diaSourceId": pl.String, "pull": pl.Float64})
    else:
        pull_df = pl.DataFrame(pull_records)

    # Join pull values onto main light curve without duplicating metadata columns
    full_lc_with_pulls_df = full_rest_frame_lc_df.join(
        pull_df, on="r:diaSourceId", how="left"
    )

    early_window = pl.col("rest_frame_time").is_between(
        pl.col("t_fd_rest"),
        pl.col("t_fd_rest") + 3.0,
        closed="both",
    )

    # Count distinct excess nights (pull >= 5 within window)
    eex_candidates = full_lc_with_pulls_df.filter(
        early_window & (pl.col("pull") >= 5.0)
    ).with_columns(pl.col("r:midpointMjdTai").floor().cast(pl.Int64).alias("night"))

    eex_nights = eex_candidates.group_by("r:diaObjectId").agg(
        pl.col("night").n_unique().alias("excess_nights")
    )

    full_lc_with_pulls_df = (
        full_lc_with_pulls_df.join(eex_nights, on="r:diaObjectId", how="left")
        .with_columns(pl.col("excess_nights").fill_null(0))
        .with_columns((pl.col("excess_nights") >= 2).alias("early_excess"))
    )

    passing_object_ids = full_lc_with_pulls_df.filter(pl.col("early_excess"))[
        "r:diaObjectId"
    ].unique()

    fit_params_with_eex = pl_fit_params_df.with_columns(
        pl.col("r:diaObjectId").is_in(passing_object_ids).alias("early_excess")
    )

    if pass_output_path:
        fit_params_with_eex.filter(pl.col("early_excess")).write_csv(pass_output_path)

    if fail_output_path:
        fit_params_with_eex.filter(~pl.col("early_excess")).write_csv(fail_output_path)

    if values_output_path:
        full_lc_with_pulls_df.write_csv(values_output_path)

    return fit_params_with_eex


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    SALT_OBJECT_DATA_PATH = (
        DATA_DIR / "eex_results" / "salt_method" / "eex_cut_results" / "values.csv"
    )

    OBJECT_DIR = (
        DATA_DIR / "eex_results" / "salt_method" / "eex_cut_results" / "values.csv"
    )

    OBJECT_PASS_OUTPUT_PATH = (
        DATA_DIR / "eex_results" / "simple_power_law" / "results" / "pass.csv"
    )

    OBJECT_FAIL_OUTPUT_PATH = (
        DATA_DIR / "eex_results" / "simple_power_law" / "results" / "fail.csv"
    )

    OBJECT_VALUES_OUTPUT_PATH = (
        DATA_DIR / "eex_results" / "simple_power_law" / "results" / "values.csv"
    )

    PLOTS_OUTPUT_PATH = IMAGE_DIR / "early_excess_plots" / "simple_power_law"

    TEST_OBJECT_DIR = (
        DATA_DIR
        / "eex_results"
        / "simple_power_law"
        / "testing_results"
        / "test_object.csv"
    )

    TEST_OBJECT_PASS_OUTPUT_PATH = (
        DATA_DIR / "eex_results" / "simple_power_law" / "testing_results" / "pass.csv"
    )

    TEST_OBJECT_FAIL_OUTPUT_PATH = (
        DATA_DIR / "eex_results" / "simple_power_law" / "testing_results" / "fail.csv"
    )

    TEST_OBJECT_VALUES_OUTPUT_PATH = (
        DATA_DIR / "eex_results" / "simple_power_law" / "testing_results" / "values.csv"
    )

    LIGHT_CURVE_DATA_DIR = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    BANDS = [
        "g",
        "r",
        "i",
    ]

    LABELS = [r"$t_{fl}$", r"$A^{\prime}$", r"$\alpha$"]

    test_result = iterate_over_bands(
        objects_dir=OBJECT_DIR,
        light_curve_data_dir=LIGHT_CURVE_DATA_DIR,
        bands=BANDS,
    )

    test_result = keep_observations_upto_t50(
        objects=OBJECT_DIR,
        light_curve_data_path=LIGHT_CURVE_DATA_DIR,
        reference_df=test_result,
    )

    full_rest_data = rest_frame_conversion(
        object_salt_data=OBJECT_DIR,
        pre_t50_objects_data=test_result,
    )

    for band in BANDS:
        rest_data_no_eex = excluding_excess_window(
            rest_frame_conversion_df=full_rest_data,
            band=band,
        )

    test_result = fitting_power_law(
        rest_frame_objects=rest_data_no_eex,
        n_steps=2000,
    )

    # walkers_chain_plot(
    #     sampler_results=test_result[1],
    #     labels=LABELS,
    #     discard=500,
    # )

    eex_detection(
        pl_fit_params_df=test_result,
        full_rest_frame_lc_df=full_rest_data,
        pass_output_path=OBJECT_PASS_OUTPUT_PATH,
        fail_output_path=OBJECT_FAIL_OUTPUT_PATH,
        values_output_path=OBJECT_VALUES_OUTPUT_PATH,
    )

    fit_sample_parameters_plot(
        pl_fit_params_df=test_result,
        rest_frame_no_eex=rest_data_no_eex,
        full_rest_data_df=full_rest_data,
        output_path=PLOTS_OUTPUT_PATH,
    )
