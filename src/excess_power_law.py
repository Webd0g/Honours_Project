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
        elif band == "i":
            i_result = filtered_result

    i_only = i_result.join(
        g_result.select("r:diaObjectId"),
        on="r:diaObjectId",
        how="anti",
    )

    final_reference_points = pl.concat([g_result, i_only])

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

    t_fl, A_prime, alpha = theta

    if not (-100 < t_fl < t_fd_rest):
        return -np.inf

    if A_prime <= 0:
        return -np.inf

    if not (1 <= alpha <= (1 * 10**8)):
        return -np.inf

    return -np.log(A_prime) - alpha * np.log(10)


def log_likelihood_function(
    theta: np.ndarray,
    times: np.ndarray,
    flux: np.ndarray,
    flux_error: np.ndarray,
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

    t_fl, A_prime, alpha = theta

    model_flux = power_law_model(
        times=times,
        t_fl=t_fl,
        A_prime=A_prime,
        alpha=alpha,
    )

    residual = flux - model_flux

    # Return the output of the log likelihood equation.
    return -0.5 * np.sum(
        (residual / flux_error) ** 2 + np.log(2 * np.pi * flux_error**2)
    )


def log_probability_function(
    theta: np.ndarray,
    times: np.ndarray,
    flux: np.ndarray,
    flux_error: np.ndarray,
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
        times=times,
        flux=flux,
        flux_error=flux_error,
    )

def fitting_power_law(rest_frame_objects: pl.DataFrame, fit_df_output_path: Path | None = None,) -> pl.DataFrame:
    """_summary_

    Args:
        rest_frame_objects (pl.DataFrame): _description_
        fit_df_output_path (Path | None, optional): _description_. Defaults to None.

    Returns:
        pl.DataFrame: _description_
    """

# def mcmc_fitting(n_walkers: int, n_dimensions: int, log_prob: float, arguments: tuple, ):

#     sampler = emcee.EnsembleSampler(nwalkers=)

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

    LIGHT_CURVE_DATA_DIR = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    BANDS = [
        "g",
        "i",
    ]

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

    test_result = rest_frame_conversion(
        object_salt_data=TEST_OBJECT_DIR,
        pre_t50_objects_data=test_result,
    )

    test_result = test_result.filter(pl.col("r:band") == "g")

    times = test_result["rest_frame_time"].to_numpy()
    flux = test_result["r:psfFlux"].to_numpy()
    flux_error = test_result["r:psfFluxErr"].to_numpy()

    t_fd_rest = test_result["t_fd_rest"].first()

    # Initial guess for theta for this test object.
    theta = np.array([-18, 2.0, 4.0])

    scatter = np.array([0.1, 0.05, 0.05])

    pos = theta + scatter * np.random.randn(100, 3)

    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=ndim,
        log_prob_fn=log_probability_function,
        args=(
            times,
            flux,
            flux_error,
            t_fd_rest,
        ),
    )
    sampler.run_mcmc(
        initial_state=pos,
        nsteps=2000    ,
        progress=True,
    )

    # fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    # samples = sampler.get_chain()
    # labels = [r"$t_{fl}$", r"$A^{\prime}$", r"$\alpha$"]
    # for i in range(ndim):
    #     ax = axes[i]
    #     ax.plot(samples[:, :, i], "k", alpha=0.3)
    #     ax.set_xlim(0, len(samples))
    #     ax.set_ylabel(labels[i])
    #     ax.yaxis.set_label_coords(-0.1, 0.5)

    # axes[-1].set_xlabel("step number")

    # tau = sampler.get_autocorr_time()
    # print(tau)

    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)

    t_fl_med = np.median(flat_samples[:, 0])
    A_prime_med = np.median(flat_samples[:, 1])
    alpha_med = np.median(flat_samples[:, 2])

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
        t_fl_med,
        linestyle="--",
        label=rf"$t_{{\mathrm{{fl}}}}={t_fl_med:.2f}$",
    )

    plt.xlabel("Rest-frame time (days)")
    plt.ylabel("Flux")
    plt.title("Single-object power-law fit")
    plt.legend()
    plt.tight_layout()
    plt.show()
