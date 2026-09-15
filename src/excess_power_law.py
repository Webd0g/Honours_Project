from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import Math, display

from early_excess_salt_detection import loading_light_curve_data_and_objects


def fifty_percent_flux_calculation(
    objects_path: Path,
    light_curve_data_path: Path,
    band: str,
) -> pl.DataFrame:
    """This function finds the time when the observed flux across each band was 50 per cent of the g band flux.

    Args:
        object_path (Path): The path to the CSV where all the objects are.
        light_curve_data_path (Path): The path to the light curve data of all objects.

    Returns:
        pl.DataFrame: Returns a dataframe with an new column that is the mjd time where the observed flux is 50 per cent of the peak in the g band.
    """
    # Reading and loading the light curve data and finding the light curve data that matches the objects that we have loaded in.
    light_curve_data = loading_light_curve_data_and_objects(
        objects_path=objects_path, light_curve_data_path=light_curve_data_path
    )

    # Getting only the g band detections and finding the peak g band flux for each object.
    band_df = light_curve_data.filter(pl.col("r:band") == band).with_columns(
        pl.col("r:psfFlux").max().over("r:diaObjectId").alias("peak_band_flux")
    )

    band_df = band_df.with_columns(
        (pl.col("peak_band_flux") * 0.5).alias("50_per_cent_peak_flux")
    )

    return band_df


def find_fifty_per_cent_time(
    band_df: pl.DataFrame,
) -> pl.DataFrame:
    """Here we find the midpointMjdTai that aligns closest with the quantity of the 50 % flux column.

    Args:
        fifty_per_cent_band_df (pl.DataFrame): The polars dataframe that has all the data from the fifty per cent band flux calcualtion.
    Returns:
        pl.DataFrame: Returns the dataframe that was passed to this function with a new column identifying the epoch that is closest to the 50 per cent peak flux calculation.
    """
    # Finding the peak row so we can track the MJD for that peak.
    peak_df = band_df.filter(
        pl.col("r:psfFlux") == pl.col("peak_band_flux")
    ).with_columns(pl.col("r:midpointMjdTai").alias("peak_mjd"))

    # Join the df that contains the peak info and the df with all the objects in it.
    peak_df = band_df.join(
        peak_df.select("r:diaObjectId", "peak_mjd"), on="r:diaObjectId", how="left"
    )

    # Finding the difference between the peak in the g band for the objects and all the fluxes for the other measurements for the objects.
    difference_df = peak_df.with_columns(
        (pl.col("r:psfFlux") - pl.col("50_per_cent_peak_flux"))
        .abs()
        .alias("delta_flux")
    ).sort(["r:diaObjectId", "r:midpointMjdTai"])

    # Adding a column that is the previous flux measurement for each row.
    difference_df = difference_df.with_columns(
        pl.col("r:psfFlux").shift(1).over("r:diaObjectId").alias("previous_flux")
    )

    # Adding a column that is the next flux measurement for each row.
    difference_df = difference_df.with_columns(
        pl.col("r:psfFlux").shift(-1).over("r:diaObjectId").alias("next_flux")
    )

    # Filter to retain only the points for the objects that are before the peak flux in g band.
    rise_df = difference_df.filter(pl.col("r:midpointMjdTai") <= pl.col("peak_mjd"))

    # Filtering to points that are apart of the rise component where the previous measurement was less than the middle measurement and the next measurement is higher flux.
    filtered_df = rise_df.filter(
        (pl.col("previous_flux") < pl.col("r:psfFlux"))
        & (pl.col("next_flux") > pl.col("r:psfFlux"))
    )

    # Adding a column that is the minimum delta_flux measurement for each object to find the point that is the closest to the 50 per cent flux calculation.
    fifty_per_cent_point_df = filtered_df.with_columns(
        pl.col("delta_flux").min().over("r:diaObjectId").alias("min_abs_delta_flux")
    )

    # Here we find where the delta flux is the smallest value over each diaObjectId, however it only finds 14 rows?
    filtered_df = fifty_per_cent_point_df.filter(
        (pl.col("delta_flux").abs()) == pl.col("min_abs_delta_flux")
    )

    return filtered_df


def iterate_over_bands(
    objects_dir: Path,
    light_curve_data_dir: Path,
    bands: list[str],
) -> pl.DataFrame:
    """Iterates over the bands that you feed as a list of strings, this is so that if an object isn't well sampled in one band but you would like to keep it you can try other bands and they weill get appended.

    Args:
        band_df (pl.DataFrame): The dataframe that is the result from the 50 per cent peak flux calcuation.
        bands (list[str]): The list of the bands as string entries

    Returns:
        pl.DataFrame: The dataframe with the row that is the 50 per cent flux calculatio which is how we find the time to fit to for the power law.
    """

    # Iterates through the bands that have been entered in the list
    for band in bands:
        result = fifty_percent_flux_calculation(
            objects_path=objects_dir,
            light_curve_data_path=light_curve_data_dir,
            band=band,
        )

        filtered_result = find_fifty_per_cent_time(
            band_df=result,
        )
        if band == "g":
            g_result = filtered_result
        elif band == "r":
            r_result = filtered_result
        elif band == "i":
            i_result = filtered_result

    r_only = r_result.join(
        g_result.select("r:diaObjectId"),
        on="r:diaObjectId",
        how="anti",
    )

    i_only = i_result.join(
        r_result.select("r:diaObjectId"),
        on="r:diaObjectId",
        how="anti",
    )

    final_reference_points = pl.concat([g_result, r_only, i_only])

    return final_reference_points.rename({"r:midpointMjdTai": "t50_mjd"}).select(
        [
            "r:diaObjectId",
            "r:band",
            "r:psfFlux",
            "r:psfFluxErr",
            "t50_mjd",
            "peak_band_flux",
            "50_per_cent_peak_flux",
            "peak_mjd",
            "delta_flux",
            "previous_flux",
            "next_flux",
        ]
    )


def keep_observations_upto_t50(
    objects: Path,
    light_curve_data_path: Path,
    reference_df: pl.DataFrame,
) -> pl.DataFrame:
    """Filtering step to keep only observations up to the 50 per cent point before fitting to that point.

    Args:
        objects (Path): Path to the objects of SN Ia that will be filtered.
        light_curve_data (Path): Path to the light curve data for all the objects.
        reference_df (pl.DataFrame): This is the dataframe resulting from the fifty_percent_flux_calculation, find_fifty_per_cent_time and iterate_over_bands functions.

    Returns:
        pl.DataFrame: Returns a polars dataframe that contains only the points up to the 50 per cent time in either the g or i bands.
    """

    light_curve_data = loading_light_curve_data_and_objects(
        objects_path=objects,
        light_curve_data_path=light_curve_data_path,
    )

    t50_df = reference_df

    filtered_light_curve_data = (
        light_curve_data.join(t50_df, on="r:diaObjectId", how="inner")
        .filter(
            pl.col("r:midpointMjdTai") <= pl.col("t50_mjd"),
        )
        .select(
            [
                "r:diaObjectId",
                "r:band",
                "r:midpointMjdTai",
                "r:psfFlux",
                "r:psfFluxErr",
                "t50_mjd",
            ]
        )
    )

    return filtered_light_curve_data


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
    times: np.ndarray,
    t_fl: float,
    A_prime: float,
    alpha: float,
) -> np.ndarray:
    """Computes the power law eqution output as an array of fluxes given the input of the parameters.

    Args:
        times (np.ndarray): The array of times leading up to the 50 per cent point that was computed earlier.
        t_fl (float): Time of first light value which is expressed with respect to the time of salt fitted t0. Thus this quantity should be in the rest-frame time.
        A_prime (float): The amplitude coeefficent of the power law.
        alpha (float): The exponent component of the power law.

    Returns:
        np.ndarray: Returns an array of the flux quantities resulting from the MCMC fitting of this equation to the inputted dataframe of flux points.
    """

    model_flux = np.zeros_like(
        times,
        dtype=float,
    )

    A = A_prime * 10 ** (-alpha)

    after_first_light = times >= t_fl

    model_flux[after_first_light] = A * (times[after_first_light] - t_fl) ** alpha

    return model_flux


def rest_frame_conversion(
    object_salt_data: Path,
    pre_t50_objects_data: pl.DataFrame,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Converts the mjd times from observer frame to rest-frame time observations.

    Args:
        object_salt_data (Path): Path to the SALT data for the objects so that we can derive the time of first light properly.

    Returns:
        pl.DataFrame: Returns a Polars dataframe that we can then use for the computation of the 50 per cent flux and so on functions.
    """

    salt_object_data = pl.read_csv(
        object_salt_data,
        columns=[
            "r:diaObjectId",
            "t0",
            "z_used",
            "t_fd",
        ],
        schema_overrides={
            "r:diaObjectId": pl.String,
            "t0": pl.Float64,
            "z_used": pl.Float64,
            "t_fd": pl.Float64,
        },
    )

    salt_object_data = pre_t50_objects_data.join(
        salt_object_data,
        on="r:diaObjectId",
        how="inner",
    )

    rest_frame_time = salt_object_data.with_columns(
        ((pl.col("r:midpointMjdTai") - pl.col("t0")) / (1 + pl.col("z_used"))).alias(
            "rest_frame_time"
        ),
    ).with_columns(
        ((pl.col("t_fd") - pl.col("t0")) / (1 + pl.col("z_used"))).alias("t_fd_rest"),
    )

    if output_path is not None:
        rest_frame_time.write_csv(output_path)

    return rest_frame_time


def log_prior_function(
    theta: np.ndarray,
    t_fd_rest: float,
) -> float:
    """The priors of the fitting in log form to work in log probabilities when doing the MCMC fitting.

    Args:
        theta (np.ndarray): A numpy array for the parameters to check the fitted ranges in.
        t_fd_rest (pl.Float64): The rest frame time of first detection for the object.

    Returns:
        pl.Float64: _description_
    """

    t_fl = theta[0]

    if not (-100 < t_fl < t_fd_rest):
        return -np.inf

    if A_prime_g <= 0 or A_prime_r <= 0 or A_prime_i <= 0:
        return -np.inf

    if (
        not (1 <= alpha_g <= (1 * 10**8))
        or not (1 <= alpha_r <= (1 * 10**8))
        or not (1 <= alpha_i <= (1 * 10**8))
    ):
        return -np.inf

    return (
        -np.log(A_prime_g)
        - alpha_g * np.log(10)
        - np.log(A_prime_r)
        - alpha_r * np.log(10)
        - np.log(A_prime_i)
        - alpha_i * np.log(10)
    )


def log_likelihood_function(
    theta: np.ndarray,
    band_data: dict,
) -> float:
    """The log likelihood function given the values of theta that have been guessed by the MCMC algorithm and the flux computed by the power law model to assess how good of a guess the algorithm has made.

    Args:
        theta (np.ndarray): An array of the guessed parameters from the algorithm of the three power law parameters.
        times (np.ndarray): An array of the times to enter into the power law for the object to estimate the flux.
        flux (np.ndarray): An array of real flux values that were observed.
        flux_error (np.ndarray): The psf error of the flux measurements as an array.

    Returns:
        float: The likelihood of the estimated parameters for the power law flux model.
    """

    fit_bands = list(band_data.keys())

    log_likelihood = 0.0

    t_fl = theta[0]

    for index, band in enumerate(fit_bands):
        A_prime = theta[1 + 2 * index]
        alpha = theta[2+2 * index]

        times = band_data[band]["times"]
        flux = band_data[band]["flux"]
        flux_error = band_data[band]["flux_error"]

        model_flux = power_law_model(
        times=times,
        t_fl=t_fl,
        A_prime=A_prime,
        alpha=alpha,
    )

    g_times = band_data["g"]["times"]
    g_flux = band_data["g"]["flux"]
    g_flux_error = band_data["g"]["flux_error"]

    r_times = band_data["r"]["times"]
    r_flux = band_data["r"]["flux"]
    r_flux_error = band_data["r"]["flux_error"]

    i_times = band_data["i"]["times"]
    i_flux = band_data["i"]["flux"]
    i_flux_error = band_data["i"]["flux_error"]

    (
        t_fl,
        A_prime_g,
        alpha_g,
        A_prime_r,
        alpha_r,
        A_prime_i,
        alpha_i,
    ) = theta

    g_model_flux = power_law_model(
        times=g_times,
        t_fl=t_fl,
        A_prime=A_prime_g,
        alpha=alpha_g,
    )

    r_model_flux = power_law_model(
        times=r_times,
        t_fl=t_fl,
        A_prime=A_prime_r,
        alpha=alpha_r,
    )

    i_model_flux = power_law_model(
        times=i_times,
        t_fl=t_fl,
        A_prime=A_prime_i,
        alpha=alpha_i,
    )

    g_residual = g_flux - g_model_flux

    r_residual = r_flux - r_model_flux

    i_residual = i_flux - i_model_flux

    g_log_likelihood = -0.5 * np.sum(
        (g_residual / g_flux_error) ** 2 + np.log(2 * np.pi * g_flux_error**2)
    )

    r_log_likelihood = -0.5 * np.sum(
        (r_residual / r_flux_error) ** 2 + np.log(2 * np.pi * r_flux_error**2)
    )

    i_log_likelihood = -0.5 * np.sum(
        (i_residual / i_flux_error) ** 2 + np.log(2 * np.pi * i_flux_error**2)
    )

    # Return the output of the log likelihood equation.
    return g_log_likelihood + r_log_likelihood + i_log_likelihood


def log_probability_function(
    theta: np.ndarray,
    band_data: dict,
    t_fd_rest: float,
) -> float:
    """The probability that these parameters of the fit with respect to the real fluxes and flux errors are determined.

    Args:
        theta (np.ndarray): The array that contains that paremeters that are used to perform the power law fit.
        t_fd_rest (float): The time of first detection in the rest frame for the object.
        times (np.ndarray): The array of times for each dectection for the object.
        flux (np.ndarray): The array of the measured fluxes for the object.
        flux_error (np.ndarray): The psfFlux measured error in the flux measurements.

    Returns:
        float: The calculated overall probability of the fitted parameters
    """

    lp = log_prior_function(
        theta=theta,
        t_fd_rest=t_fd_rest,
    )
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood_function(
        theta=theta,
        band_data=band_data,
    )


def fitting_power_law(
    rest_frame_objects: pl.DataFrame,
    n_steps: int,
    fit_df_output_path: Path | None = None,
) -> pl.DataFrame:
    """Fitting the power law and iterating and minimising is done in this function, using the dataframe from the rest-frame conversion and the prior functions.

    Args:
        rest_frame_objects (pl.DataFrame): The dataframe with the lightcurve data up to the fifty per cent point and the times of the observations converted to the rest-frame times.
        fit_df_output_path (Path | None, optional): The output path for the dataframe of the fit parameters that the mimnimisation arrived on a sa CSV. Defaults to None.

    Returns:
        pl.DataFrame: The dataframe of the fit parameters for the power law.
    """

    results = {
        "r:diaObjectId": [],
        "t_fl": [],
        "t_fl_err_low": [],
        "t_fl_err_high": [],
        "A_prime": [],
        "A_prime_err_low": [],
        "A_prime_err_high": [],
        "alpha": [],
        "alpha_err_low": [],
        "alpha_err_high": [],
        "t_fd": [],
        "t_fd_rest": [],
        "z_used": [],
        "n_points": [],
        "t50_mjd": [],
        "sampler_nwalkers": [],
        "sampler_nsteps": [],
        # "acceptance_fraction": [],
        # "tau_t_fl": [],
        # "tau_A_prime": [],
        # "tau_alpha": [],
        # "converged": [],
    }

    for object_id, object_df in rest_frame_objects.group_by("r:diaObjectId"):
        # Filtering the dataframe to just the individual band.
        rest_frame_objects_df = object_df

        band_data = {}

        for current_band in [
            "g",
            "r",
            "i",
        ]:
            band_df = object_df.filter(pl.col("r:band") == current_band)

            if band_df.height > 0:
                band_data[current_band] = {
                    # Making Numpy arrays of the columns for the iteration and fitting.
                    "times": band_df["rest_frame_time"].to_numpy(),
                    "flux": band_df["r:psfFlux"].to_numpy(),
                    "flux_error": band_df["r:psfFluxErr"].to_numpy(),
                }

        # Finding the first detection for the object
        t_fd_rest = rest_frame_objects_df["t_fd_rest"].first()

        # Bands present in the object that will be used for the fit.
        fit_bands = list(band_data.keys())

        # Initial guess for theta for this test object.
        theta = [
            -18,
        ]

        # Adding extra priors if more bands are present.
        for band in fit_bands:
            theta.extend(
                [
                    2.0,
                    4.0,
                ]
            )

        # Converting the list to a numpy array.
        theta = np.array(theta)

        # Rough scatter approximation for the fitting vector theta.
        scatter = [
            0.1,
        ]

        # Adding extra scatter to extra extra prior for the detected bands
        for band in fit_bands:
            scatter.extend(
                [
                    0.5,
                    0.5,
                ]
            )

        # Converting scatter list to numpy array.
        scatter = np.array(scatter)

        # Beginning position for the MCMC fitting process.
        pos = theta + scatter * np.random.randn(100, len(theta))

        # Initialising the ensemble sample from the emcee library
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=log_probability_function,
            args=(
                band_data,
                t_fd_rest,
            ),
        )

        # Running the MCMC fitting algorithm.
        sampler.run_mcmc(
            initial_state=pos,
            nsteps=n_steps,
            progress=True,
        )

        flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)

        print(flat_samples)

        q16, t_fl, q84 = np.percentile(flat_samples[:, 0], [16, 50, 84])

        results["t_fl"].append(float(t_fl))
        results["t_fl_err_low"].append(float(t_fl - q16))
        results["t_fl_err_high"].append(float(q84 - t_fl))

        q16, A_prime, q84 = np.percentile(flat_samples[:, 1], [16, 50, 84])

        results["A_prime"].append(float(A_prime))
        results["A_prime_err_low"].append(float(A_prime - q16))
        results["A_prime_err_high"].append(float(q84 - A_prime))

        q16, alpha, q84 = np.percentile(flat_samples[:, 2], [16, 50, 84])

        results["alpha"].append(float(alpha))
        results["alpha_err_low"].append(float(alpha - q16))
        results["alpha_err_high"].append(float(q84 - alpha))

        results["r:diaObjectId"].append(object_id[0])
        results["t_fd"].append(rest_frame_objects_df["t_fd"].item(0))
        results["t_fd_rest"].append(rest_frame_objects_df["t_fd_rest"].item(0))
        results["z_used"].append(rest_frame_objects_df["z_used"].item(0))
        results["n_points"].append(len(band_data[current_band]["times"]))
        results["sampler_nsteps"].append(n_steps)
        results["sampler_nwalkers"].append(nwalkers)
        results["t50_mjd"].append(rest_frame_objects_df["t50_mjd"].item(0))

    results = pl.DataFrame(results)

    return results, sampler


def excluding_excess_window(
    rest_frame_conversion_df: pl.DataFrame,
    band: str,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Remove any points in the excess window from the dataframe that will be used for the power law fitting process.

    Args:
        rest_frame_conversion_df (pl.DataFrame): dataframe of the points from the rest-frame conversion function.
        output_path (Path | None, optional): Optional CSV output path of the dataframe from this fucntion. Defaults to None.

    Returns:
        pl.DataFrame: The dataframe with the data points that are in the excess window removed before the fitting.
    """

    # Defining the excess window (for use we take the t_fd and add 3 days remove any points that have a time in that range.)
    rest_frame_objects_df = rest_frame_conversion_df.filter(
        ~(
            (pl.col("r:band") == band)
            & pl.col("rest_frame_time").is_between(
                pl.col("t_fd_rest"),
                pl.col("t_fd_rest") + 5,
                closed="both",
            )
        )
    )

    if output_path is not None:
        rest_frame_objects_df.write_csv(
            output_path,
        )

    return rest_frame_objects_df


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
    sampling_results: pl.DataFrame,
    rest_frame_objects_df: pl.DataFrame,
    band: str,
    output_path: Path | None = None,
) -> None:
    """Plots the median sampled values from the sampling of parameters for the power law against the real data of the light curves of the objects.

    Args:
        sampling_results (pl.DataFrame): The results of the chain from the sampling of the power law, this should be the resulting dataframe output from the power law fitting function.
        output_path (Path | None, optional): Desired output path for the plots of the input polars df. Defaults to None.
    """

    for object_id, object_df in sampling_results[0].group_by("r:diaObjectId"):
        diaObjectId = object_df["r:diaObjectId"]

        # Filter for the light curve data of this object in the desired band too.
        object_data_filter_df = rest_frame_objects_df.filter(
            (pl.col("r:diaObjectId") == diaObjectId) & (pl.col("r:band") == band),
        )

        # Grabbing the data needed to produce the real light curve of the object.
        times = object_data_filter_df["rest_frame_time"].to_numpy()
        flux = object_data_filter_df["r:psfFlux"].to_numpy()
        flux_error = object_data_filter_df["r:psfFluxErr"].to_numpy()

        # Grabbing the time of first detection for the object
        t_fd_rest = object_df["t_fd_rest"].to_numpy()

        # Extract median parameters from the sampling results dataframe
        t_fl_med = object_df["t_fl"].to_numpy()
        A_prime_med = object_df["A_prime"].to_numpy()
        alpha_med = object_df["alpha"].to_numpy()

        t_grid = np.linspace(times.min(), times.max(), 300)

        model_flux = power_law_model(
            times=t_grid,
            t_fl=t_fl_med,
            A_prime=A_prime_med,
            alpha=alpha_med,
        )

        plt.figure(figsize=(8, 5))

        plt.errorbar(
            times,
            flux,
            yerr=flux_error,
            fmt="o",
            capsize=3,
            label="Observed data",
        )

        plt.plot(
            t_grid,
            model_flux,
            label="Median power-law fit",
        )

        plt.axvline(
            x=t_fd_rest,
            linestyle="--",
            label=rf"$t_{{fd}}={t_fd_rest.item():.2f}$ d",
        )

        plt.axvline(
            x=t_fd_rest.item() + 3,
            linestyle="--",
            label=rf"$t_{{fd}} + 3d ={t_fd_rest.item() + 3:.2f}$ d",
        )

        plt.axvline(
            t_fl_med,
            linestyle="--",
            label=rf"$t_{{\mathrm{{fl}}}}={t_fl_med.item():.2f}$ d",
            color="#D55E00",
        )

        plt.xlabel("Rest-frame time (days)")
        plt.ylabel("Flux (nJy)")
        plt.title(f"{diaObjectId[0]}")
        plt.legend()
        plt.tight_layout()
        plt.show()


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
        DATA_DIR
        / "salt_fitting"
        / "ia_search"
        / "with_z_results"
        / "colour_stretch_results"
        / "pass.csv"
    )

    TEST_OBJECT_DIR = (
        DATA_DIR
        / "eex_results"
        / "simple_power_law"
        / "testing_results"
        / "test_object.csv"
    )

    TEST_OUTPUT_PATH = (
        DATA_DIR
        / "eex_results"
        / "simple_power_law"
        / "testing_results"
        / "test_object_output_path.csv"
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

    for band in BANDS:
        test_result = iterate_over_bands(
            objects_dir=TEST_OBJECT_DIR,
            light_curve_data_dir=LIGHT_CURVE_DATA_DIR,
            bands=BANDS,
        )

    test_result = keep_observations_upto_t50(
        objects=TEST_OBJECT_DIR,
        light_curve_data_path=LIGHT_CURVE_DATA_DIR,
        reference_df=test_result,
    )

    rest_frame_test_result = rest_frame_conversion(
        object_salt_data=TEST_OBJECT_DIR,
        pre_t50_objects_data=test_result,
    )

    # excluding_excess_window(
    #     rest_frame_conversion_df=test_result,
    #     band="g",
    # )

    test_result = fitting_power_law(
        rest_frame_objects=rest_frame_test_result,
        n_steps=2000,
    )

    # walkers_chain_plot(
    #     sampler_results=test_result[1],
    #     labels=LABELS,
    #     discard=500,
    # )

    # fit_sample_parameters_plot(
    #     sampling_results=test_result,
    #     rest_frame_objects_df=rest_frame_test_result,
    #     band=band,
    # )
