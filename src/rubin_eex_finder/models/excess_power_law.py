from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from rubin_eex_finder.models.early_excess_salt_detection import (
    loading_light_curve_data_and_objects,
)
from rubin_eex_finder.visualisation.style import set_publication_style

set_publication_style()

# from rubin_eex_finder.visualisation.fits import fit_sample_parameters_plot


def fifty_percent_flux_calculation(
    objects_path: Path,
    light_curve_data_path: Path,
    band: str,
    fraction: float = 0.5,
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
        (pl.col("peak_band_flux") * fraction).alias("50_per_cent_peak_flux")
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
    fraction: float = 0.5,
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
            fraction=fraction,
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
            "r:diaSourceId",
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
                "r:diaSourceId",
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


def excluding_excess_window(
    rest_frame_conversion_df: pl.DataFrame,
    window_length: int,
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
            pl.col("rest_frame_time").is_between(
                pl.col("t_fd_rest"),
                pl.col("t_fd_rest") + window_length,
                closed="both",
            )
        )
    )

    if output_path is not None:
        rest_frame_objects_df.write_csv(
            output_path,
        )

    return rest_frame_objects_df


def log_prior_function(
    theta: np.ndarray,
    band_data: dict,
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

    if not (-100.0 < t_fl < t_fd_rest):
        return -np.inf

    log_prior = 0.0

    fit_bands = list(band_data.keys())

    for index, band in enumerate(fit_bands):
        A_prime = theta[1 + 2 * index]
        alpha = theta[2 + 2 * index]

        if A_prime <= 0:
            return -np.inf

        if not 1 <= alpha <= 1e8:
            return -np.inf

        log_prior += -np.log(A_prime) - alpha * np.log(10)

    return log_prior


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
        alpha = theta[2 + 2 * index]

        times = band_data[band]["times"]
        flux = band_data[band]["flux"]
        flux_error = band_data[band]["flux_error"]

        model_flux = power_law_model(
            times=times,
            t_fl=t_fl,
            A_prime=A_prime,
            alpha=alpha,
        )

        residual = flux - model_flux

        log_likelihood += -0.5 * np.sum(
            (residual / flux_error) ** 2 + np.log(2 * np.pi * flux_error**2)
        )

    return log_likelihood


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
        band_data=band_data,
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
        # "acceptance_fraction": [],
        # "tau_t_fl": [],
        # "tau_A_prime": [],
        # "tau_alpha": [],
        # "converged": [],
    }

    for object_id, object_df in rest_frame_objects.group_by("r:diaObjectId"):
        # Filtering the dataframe to just the individual band.
        t_fd_rest = object_df["t_fd_rest"].first()

        # Inject 3 synthetic baseline points per band (0 nJy flux, 300 nJy error)
        # fake_times = [t_fd_rest - 2.0]
        # synthetic_rows = []
        # for current_band in ["g", "r", "i"]:
        #     for f_time in fake_times:
        #         synthetic_rows.append(
        #             {
        #                 "r:diaObjectId": object_id[0],
        #                 "r:band": current_band,
        #                 "rest_frame_time": f_time,
        #                 "r:psfFlux": 0.0,
        #                 "r:psfFluxErr": 300.0,
        #                 "t_fd_rest": t_fd_rest,
        #                 "t_fd": object_df["t_fd"].first(),
        #                 "z_used": object_df["z_used"].first(),
        #                 "t50_mjd": object_df["t50_mjd"].first(),
        #             }
        #         )

        # synthetic_df = pl.DataFrame(synthetic_rows)
        # augmented_object_df = pl.concat([object_df, synthetic_df], how="diagonal").sort(
        #     "rest_frame_time"
        # )

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
        t_fd_rest = object_df["t_fd_rest"].first()

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

        q16, t_fl, q84 = np.percentile(flat_samples[:, 0], [16, 50, 84])

        results["t_fl"].append(float(t_fl))
        results["t_fl_err_low"].append(float(t_fl - q16))
        results["t_fl_err_high"].append(float(q84 - t_fl))

        for band in ["g", "r", "i"]:
            if band in fit_bands:
                index = fit_bands.index(band)
                A_prime_index = 1 + 2 * index
                alpha_index = 2 + 2 * index

                q16, A_prime, q84 = np.percentile(
                    flat_samples[:, A_prime_index], [16, 50, 84]
                )

                results[f"A_prime_{band}"].append(float(A_prime))
                results[f"A_prime_{band}_err_low"].append(float(A_prime - q16))
                results[f"A_prime_{band}_err_high"].append(float(q84 - A_prime))

                q16, alpha, q84 = np.percentile(
                    flat_samples[:, alpha_index], [16, 50, 84]
                )

                results[f"alpha_{band}"].append(float(alpha))
                results[f"alpha_{band}_err_low"].append(float(alpha - q16))
                results[f"alpha_{band}_err_high"].append(float(q84 - alpha))

            else:
                # Append None for missing bands to preserve exact column heights
                results[f"A_prime_{band}"].append(None)
                results[f"A_prime_{band}_err_low"].append(None)
                results[f"A_prime_{band}_err_high"].append(None)

                results[f"alpha_{band}"].append(None)
                results[f"alpha_{band}_err_low"].append(None)
                results[f"alpha_{band}_err_high"].append(None)

        results["r:diaObjectId"].append(object_id[0])
        results["t_fd"].append(object_df["t_fd"].item(0))
        results["t_fd_rest"].append(object_df["t_fd_rest"].item(0))
        results["z_used"].append(object_df["z_used"].item(0))
        results["sampler_nsteps"].append(n_steps)
        results["sampler_nwalkers"].append(nwalkers)
        results["t50_mjd"].append(object_df["t50_mjd"].item(0))

    results = pl.DataFrame(results)

    return results, sampler


def eex_detection(
    pl_fit_params_df: pl.DataFrame,
    full_rest_frame_lc_df: pl.DataFrame,
    window_length: int,
    pass_output_path: Path | None = None,
    fail_output_path: Path | None = None,
    values_output_path: Path | None = None,
) -> pl.DataFrame:
    """Reconstruct the fluxes at the datapoint times from the power law fit parameters, calculate pulls and check for objects with pulls greater than 5 and save them as "passed".

    Args:
        pl_fit_params_df (pl.DataFrame): Dataframe that has the objects pl fit parameters resulting from the MCMC sampling.
        full_rest_frame_lc_df (pl.DataFrame): Dataframe containing the light curve data in rest frame that will be fit to.
        pass_output_path (Path | None, optional): Output path for all the passing objects from the excess check. Defaults to None.
        fail_output_path (Path | None, optional): Output path for all the failing objects from the excess check. Defaults to None.
        values_output_path (Path | None, optional): Output path for all the values resulting from the excess check. Defaults to None.

    Returns:
        pl.DataFrame: The values dataframe that is the same as the values CSV.
    """

    pull_records = []

    for object_id, object_df in pl_fit_params_df.group_by(
        "r:diaObjectId",
        maintain_order=True,
    ):
        diaObjectId = object_df["r:diaObjectId"].item()

        rest_band_data = {}

        for current_band in [
            "g",
            "r",
            "i",
        ]:
            # Filter for the light curve data of this object in the desired band too.
            rest_band_df = full_rest_frame_lc_df.filter(
                (pl.col("r:band") == current_band)
                & (pl.col("r:diaObjectId") == diaObjectId),
            )

            if rest_band_df.height > 0:
                rest_band_data[current_band] = {
                    "diaSourceId": rest_band_df["r:diaSourceId"].to_numpy(),
                    "times": rest_band_df["rest_frame_time"].to_numpy(),
                    "flux": rest_band_df["r:psfFlux"].to_numpy(),
                    "flux_error": rest_band_df["r:psfFluxErr"].to_numpy(),
                }

        avaliable_bands = list(rest_band_data.keys())

        model_parameters = {}
        fit_bands = []

        for band in avaliable_bands:
            a_prime = object_df[f"A_prime_{band}"].item()
            alpha = object_df[f"alpha_{band}"].item()

            if a_prime is not None and alpha is not None:
                model_parameters[band] = {
                    "A_prime": a_prime,
                    "alpha": alpha,
                }
                fit_bands.append(band)

        if not fit_bands:
            continue

        t_fl_med = object_df["t_fl"].item()

        for band in fit_bands:
            times = rest_band_data[band]["times"]
            flux = rest_band_data[band]["flux"]
            flux_error = rest_band_data[band]["flux_error"]

            model_at_data = power_law_model(
                times=times,
                t_fl=t_fl_med,
                A_prime=model_parameters[band]["A_prime"],
                alpha=model_parameters[band]["alpha"],
            )

            pull = (flux - model_at_data) / (flux_error)

            for source_id, t, p in zip(
                (rest_band_data[band]["diaSourceId"]), times, pull
            ):
                pull_records.append(
                    {
                        "r:diaObjectId": diaObjectId,
                        "r:diaSourceId": source_id,
                        "r:band": band,
                        "rest_frame_time": t,
                        "pull": p,
                    }
                )

    pull_df = pl.DataFrame(pull_records)
    full_lc_with_pulls_df = full_rest_frame_lc_df.join(
        pull_df,
        on=[
            "r:diaSourceId",
        ],
        how="left",
    )

    early_window = pl.col("rest_frame_time").is_between(
        pl.col("t_fd_rest"),
        pl.col("t_fd_rest") + window_length,
    )

    eex_candidates = full_lc_with_pulls_df.filter(
        early_window & (pl.col("pull") >= 5)
    ).with_columns(pl.col("r:midpointMjdTai").floor().cast(pl.Int64).alias("night"))

    eex_nights = eex_candidates.group_by("r:diaObjectId").agg(
        pl.col("night").n_unique().alias("excess_nights")
    )

    full_lc_with_pulls_df = (
        full_lc_with_pulls_df.join(
            other=eex_nights,
            on="r:diaObjectId",
            how="left",
        )
        .with_columns(pl.col("excess_nights").fill_null(0))
        .with_columns((pl.col("excess_nights") >= 2).alias("early_excess"))
    )

    passing_object_ids = (
        full_lc_with_pulls_df.filter(pl.col("early_excess"))
        .select("r:diaObjectId")
        .unique()["r:diaObjectId"]
    )

    fit_params_with_eex = pl_fit_params_df.with_columns(
        pl.col("r:diaObjectId")
        .is_in(passing_object_ids.implode())
        .alias("early_excess")
    )

    if pass_output_path:
        fit_params_with_eex.filter(pl.col("early_excess")).write_csv(pass_output_path)

    if fail_output_path:
        fit_params_with_eex.filter(~pl.col("early_excess")).write_csv(fail_output_path)

    if values_output_path:
        full_lc_with_pulls_df.write_csv(values_output_path)

    return fit_params_with_eex


def fit_sample_parameters_plot(
    pl_fit_params_df: pl.DataFrame,
    rest_frame_no_eex: pl.DataFrame,
    window_length: int,
    full_rest_data_df: pl.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Plots the median sampled values from the sampling of parameters for the power law against the real data of the light curves of the objects.

    Args:
        pl_fit_params_df (pl.DataFrame): The results of the chain from the sampling of the power law, this should be the resulting dataframe output from the power law fitting function.
        output_path (Path | None, optional): Desired output path for the plots of the input polars df. Defaults to None.
    """

    band_colours = {
        "u": "#00008B",
        "g": "#0080FF",  # Deep Sky Blue
        "r": "#00E575",  # Bluish Green / Emerald
        "i": "#FFD700",  # Yellow
        "z": "#FF4500",  # Vermilion / Red-Orange
        "y": "#A52A2A",
    }

    for object_id, object_df in pl_fit_params_df.group_by("r:diaObjectId"):
        diaObjectId = object_df["r:diaObjectId"].item()
        t_fd_rest = object_df["t_fd_rest"].item()

        # fake_times = [t_fd_rest - 5]
        # synthetic_rows = []
        # for current_band in ["g", "r", "i"]:
        #     for f_time in fake_times:
        #         synthetic_rows.append(
        #             {
        #                 "r:diaObjectId": object_id[0],
        #                 "r:band": current_band,
        #                 "rest_frame_time": f_time,
        #                 "r:psfFlux": 0.0,
        #                 "r:psfFluxErr": 300.0,
        #                 "t_fd_rest": t_fd_rest,
        #                 "t_fd": object_df["t_fd"].first(),
        #                 "z_used": object_df["z_used"].first(),
        #                 "t50_mjd": object_df["t50_mjd"].first(),
        #             }
        #         )

        # synthetic_df = pl.DataFrame(synthetic_rows)
        # augmented_object_df = pl.concat(
        #     [full_rest_data_df, synthetic_df], how="diagonal"
        # ).sort("rest_frame_time")
        # diaObjectId = object_df["r:diaObjectId"].item()

        rest_band_data = {}

        band_data = {}

        for current_band in [
            "g",
            "r",
            "i",
        ]:
            # Filter for the light curve data of this object in the desired band too.
            rest_band_df = full_rest_data_df.filter(
                (pl.col("r:band") == current_band)
                & (pl.col("r:diaObjectId") == diaObjectId),
            )

            if rest_band_df.height > 0:
                rest_band_data[current_band] = {
                    "times": rest_band_df["rest_frame_time"].to_numpy(),
                    "flux": rest_band_df["r:psfFlux"].to_numpy(),
                    "flux_error": rest_band_df["r:psfFluxErr"].to_numpy(),
                }

            band_df = rest_frame_no_eex.filter(
                (pl.col("r:band") == current_band)
                & (pl.col("r:diaObjectId") == diaObjectId),
            )

            if band_df.height > 0:
                band_data[current_band] = {
                    # Making Numpy arrays of the columns for the iteration and fitting.
                    "times": band_df["rest_frame_time"].to_numpy(),
                    "flux": band_df["r:psfFlux"].to_numpy(),
                    "flux_error": band_df["r:psfFluxErr"].to_numpy(),
                }

        fit_bands = list(band_data.keys())

        # Grabbing the time of first detection for the object
        model_parameters = {}

        for band in fit_bands:
            model_parameters[band] = {
                "A_prime": object_df[f"A_prime_{band}"].item(),
                "alpha": object_df[f"alpha_{band}"].item(),
            }

        t_fl_med = object_df["t_fl"].item()
        t_fd_rest = object_df["t_fd_rest"].item()

        model_grids = {}

        for band in fit_bands:
            times = rest_band_data[band]["times"]

            model_grids[band] = np.linspace(
                np.min(times),
                np.max(times),
                300,
            )

        model_flux = {}

        for band in fit_bands:
            model_flux[band] = power_law_model(
                times=model_grids[band],
                t_fl=t_fl_med,
                A_prime=model_parameters[band]["A_prime"],
                alpha=model_parameters[band]["alpha"],
            )

        (
            _fig,
            (axes_fit, axes_resid),
        ) = plt.subplots(
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

            in_ranges = (times >= t_fd_rest) & (times <= t_fd_rest + window_length)

            data_colour = band_colours.get(
                band,
                "black",
            )

            axes_fit.errorbar(
                x=times[in_ranges],
                y=flux[in_ranges],
                yerr=flux_error[in_ranges],
                fmt="o",
                capsize=3,
                mec="white",
                mew=1,
                alpha=0.3,
            )

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

            axes_fit.plot(
                model_grids[band],
                model_flux[band],
                color=data_colour,
            )

            # Model evaluated at the actual observation times
            model_at_data = power_law_model(
                times=times,
                t_fl=t_fl_med,
                A_prime=model_parameters[band]["A_prime"],
                alpha=model_parameters[band]["alpha"],
            )

            residuals = (flux - model_at_data) / (flux_error)

            axes_resid.errorbar(
                x=times[in_ranges],
                y=residuals[in_ranges],
                fmt="o",
                capsize=3,
                mec="white",
                color=data_colour,
                mew=1,
            )

            axes_resid.errorbar(
                x=times[~in_ranges],
                y=residuals[~in_ranges],
                fmt="o",
                capsize=3,
                mec="white",
                alpha=0.3,
                color=data_colour,
                mew=1,
            )

        axes_resid.axhline(
            c="black",
        )

        axes_resid.axhspan(
            -5,
            5,
            color="grey",
            linestyle=":",
            alpha=0.12,
            label=r"$5 \sigma$ region",
        )

        axes_resid.axhspan(
            -3,
            3,
            color="grey",
            linestyle="--",
            alpha=0.25,
            label=r"$3 \sigma$ region",
        )

        axes_fit.axvline(
            x=t_fd_rest,
            linestyle="--",
            label=rf"$t_{{fd}}={t_fd_rest:.2f}$ d",
            color="#0072B2",
        )

        axes_fit.axvline(
            x=t_fd_rest + window_length,
            linestyle=":",
            label=rf"$t_{{fd}} + {window_length} ={t_fd_rest + window_length:.2f}$ d",
            color="#0072B2",
        )

        axes_fit.axvspan(
            xmin=t_fd_rest,
            xmax=t_fd_rest + window_length,
            alpha=0.15,
            color="#E94E77",
            label="Points excluded from fit",
        )

        axes_resid.axvline(
            x=t_fd_rest,
            linestyle="--",
            label=rf"$t_{{fd}}={t_fd_rest:.2f}$ d",
            color="#0072B2",
        )

        axes_resid.axvline(
            x=t_fd_rest + window_length,
            linestyle=":",
            label=rf"$t_{{fd}} + {window_length} ={t_fd_rest + window_length:.2f}$ d",
            color="#0072B2",
        )

        axes_fit.axvline(
            t_fl_med,
            linestyle="--",
            label=rf"$t_{{\mathrm{{fl}}}}={t_fl_med:.2f}$ d",
            color="#D55E00",
        )

        axes_fit.set_ylabel("Flux (nJy)")
        axes_resid.set_xlabel(r"Rest-frame days before $B$-band peak")
        axes_resid.set_ylabel("Pull")
        axes_fit.legend(
            loc="best",
            frameon=False,
        )
        axes_fit.set_title(label=f"{object_id[0]}")
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path / f"{diaObjectId}_eex_power_law_fit.jpg")
            print(f"Plotted and saved Object {object_id[0]}")
            plt.close()
        else:
            print(f"Plotted Object {object_id[0]}")
            plt.show()
            plt.close()


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    OBJECT_DIR = (
        DATA_DIR
        / "eex_results"
        / "salt_method"
        / "eex_cut_results"
        / "4_day_window"
        / "values.csv"
    )

    OBJECT_PASS_OUTPUT_PATH = (
        DATA_DIR
        / "eex_results"
        / "simple_power_law"
        / "results"
        / "3_day_window"
        / "pass.csv"
    )

    OBJECT_FAIL_OUTPUT_PATH = (
        DATA_DIR
        / "eex_results"
        / "simple_power_law"
        / "results"
        / "3_day_window"
        / "fail.csv"
    )

    OBJECT_VALUES_OUTPUT_PATH = (
        DATA_DIR
        / "eex_results"
        / "simple_power_law"
        / "results"
        / "3_day_window"
        / "values.csv"
    )

    PLOTS_OUTPUT_PATH = (
        IMAGE_DIR
        / "early_excess_plots"
        / "simple_power_law"
        / "3_day_window"
        / "values"
    )

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

    POWER_LAW_PARAMS = (
        DATA_DIR
        / "eex_results"
        / "simple_power_law"
        / "results"
        / "3_day_window"
        / "pass.csv"
    )

    BANDS = [
        "g",
        "r",
        "i",
    ]

    LABELS = [r"$t_{fl}$", r"$A^{\prime}$", r"$\alpha$"]

    # 1. Compute reference rest-frame data
    ref_points = iterate_over_bands(
        objects_dir=OBJECT_DIR,
        light_curve_data_dir=LIGHT_CURVE_DATA_DIR,
        bands=BANDS,
        fraction=0.5,
    )
    pre_t50_data = keep_observations_upto_t50(
        objects=OBJECT_DIR,
        light_curve_data_path=LIGHT_CURVE_DATA_DIR,
        reference_df=ref_points,
    )
    full_rest_data = rest_frame_conversion(
        object_salt_data=OBJECT_DIR,
        pre_t50_objects_data=pre_t50_data,
    )

    # ----------------------------------------------------
    # RUN 1: Simple Power Law (3-Day Window)
    # ----------------------------------------------------
    rest_data_no_eex_3d = excluding_excess_window(
        rest_frame_conversion_df=full_rest_data,
        window_length=3,
    )
    # pl_results_3d, sampler_3d = fitting_power_law(
    #     rest_frame_objects=rest_data_no_eex_3d,
    #     n_steps=2000,
    # )
    # eex_detection(
    #     pl_fit_params_df=pl_results_3d,
    #     full_rest_frame_lc_df=full_rest_data,
    #     window_length=3,
    #     pass_output_path=OBJECT_PASS_OUTPUT_PATH,
    #     fail_output_path=OBJECT_FAIL_OUTPUT_PATH,
    #     values_output_path=OBJECT_VALUES_OUTPUT_PATH,
    # )

    pl_results_3d = pl.read_csv(source=DATA_DIR
        / "eex_results"
        / "simple_power_law"
        / "results"
        / "3_day_window"
        / "pass.csv", schema_overrides={"r:diaObjectId": pl.String})

    fit_sample_parameters_plot(
        pl_fit_params_df=pl_results_3d,
        rest_frame_no_eex=rest_data_no_eex_3d,
        window_length=3,
        full_rest_data_df=full_rest_data,
        output_path=IMAGE_DIR
        / "early_excess_plots"
        / "simple_power_law"
        / "raw_data_results"
        / "3_day_window"
        / "values",
    )

    # ----------------------------------------------------
    # RUN 2: Simple Power Law (4-Day Window)
    # ----------------------------------------------------
    rest_data_no_eex_4d = excluding_excess_window(
        rest_frame_conversion_df=full_rest_data,
        window_length=4,
    )
    # pl_results_4d, sampler_4d = fitting_power_law(
    #     rest_frame_objects=rest_data_no_eex_4d,
    #     n_steps=2000,
    # )

    # eex_detection(
    #     pl_fit_params_df=pl_results_4d,
    #     full_rest_frame_lc_df=full_rest_data,
    #     window_length=4,
    #     pass_output_path=DATA_DIR
    #     / "eex_results"
    #     / "simple_power_law"
    #     / "results"
    #     / "4_day_window"
    #     / "pass.csv",
    #     fail_output_path=DATA_DIR
    #     / "eex_results"
    #     / "simple_power_law"
    #     / "results"
    #     / "4_day_window"
    #     / "fail.csv",
    #     values_output_path=DATA_DIR
    #     / "eex_results"
    #     / "simple_power_law"
    #     / "results"
    #     / "4_day_window"
    #     / "values.csv",
    # )

    pl_results_4d = pl.read_csv(source=DATA_DIR
        / "eex_results"
        / "simple_power_law"
        / "results"
        / "4_day_window"
        / "pass.csv", schema_overrides={"r:diaObjectId": pl.String})

    fit_sample_parameters_plot(
        pl_fit_params_df=pl_results_4d,
        rest_frame_no_eex=rest_data_no_eex_4d,
        window_length=4,
        full_rest_data_df=full_rest_data,
        output_path=IMAGE_DIR
        / "early_excess_plots"
        / "simple_power_law"
        / "raw_data_results"
        / "4_day_window"
        / "values",
    )
