from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import sncosmo
from astropy.table import Table


def spectro_delta_chi_against_chi_SN_plot(
    path_to_ii_delta_chi_results: Path,
    path_to_ia_delta_chi_results: Path,
    path_to_photometric_delta_chi_results: Path,
    band: str,
    output_path: Path | None = None,
) -> None:
    """Produces the plots for the constant minus the chi squared for the Bazin fit results to visualise the removal of the objects using this cut. This function also adds in the spectroscopic objects as different shape and colour points.

    Args:
        path_to_ii_delta_chi_results (Path): Path to the CSV file containing the delta chi squared results for the ii SN.
        path_to_ia_delta_chi_results (Path): Path to the CSV file containing the delta chi squared results for the ia SN.
        path_to_photometric_delta_chi_results (Path): Path to the CSV file containing the delta chi squared results for the photometric objects.
        band (str): The band that the bazin chi sqaured and constant fit was done in.
        output_path (Path | None, optional): The output path of the plot. Defaults to None.
    """

    ii_delta_chi_df = (
        pl.read_csv(path_to_ii_delta_chi_results)
        .filter((pl.col("bazin_reduced_chi2") > 0) & (pl.col("delta_reduced_chi2") > 0))
        .select(
            "r:diaObjectId",
            "bazin_reduced_chi2",
            "delta_reduced_chi2",
        )
        .with_columns(pl.lit("Core-Collapse").alias("class"))
    )

    ia_delta_chi_df = (
        pl.read_csv(path_to_ia_delta_chi_results)
        .filter((pl.col("bazin_reduced_chi2") > 0) & (pl.col("delta_reduced_chi2") > 0))
        .select(
            "r:diaObjectId",
            "bazin_reduced_chi2",
            "delta_reduced_chi2",
        )
        .with_columns(pl.lit("SN Ia").alias("class"))
    )

    photometric_delta_chi_df = (
        pl.read_csv(path_to_photometric_delta_chi_results)
        .filter((pl.col("bazin_reduced_chi2") > 0) & (pl.col("delta_reduced_chi2") > 0))
        .select(
            "r:diaObjectId",
            "bazin_reduced_chi2",
            "delta_reduced_chi2",
        )
        .with_columns(pl.lit("Photometric").alias("class"))
    )

    delta_chi_df = pl.concat(
        [
            photometric_delta_chi_df,
            ii_delta_chi_df,
            ia_delta_chi_df,
        ]
    ).unique(
        subset="r:diaObjectId",
        keep="last",
        maintain_order=True,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    photometric = delta_chi_df.filter(pl.col("class") == "Photometric")

    ia = delta_chi_df.filter(pl.col("class") == "SN Ia")

    cc = delta_chi_df.filter(pl.col("class") == "Core-Collapse")

    ax.scatter(
        photometric["bazin_reduced_chi2"],
        photometric["delta_reduced_chi2"],
        color="#999999",
        alpha=0.3,
        label="Photometric",
        s=10,
    )

    ax.scatter(
        ia["bazin_reduced_chi2"],
        ia["delta_reduced_chi2"],
        color="#0072B2",
        alpha=0.9,
        label="Spectroscopic SN Ia",
        zorder=3,
        s=15,
        marker="^",
    )

    ax.scatter(
        cc["bazin_reduced_chi2"],
        cc["delta_reduced_chi2"],
        color="#D55E00",
        alpha=0.9,
        label="Spectroscopic CC",
        zorder=3,
        s=15,
        marker="s",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    lower = min(xlim[0], ylim[0])
    upper = max(xlim[1], ylim[1])

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="black",
        label="Selection boundary",
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    ax.set_xlabel(rf"$\chi^2_{{shape}}$ ({band} band)")
    ax.set_ylabel(rf"$\chi^2_{{constant}} - \chi^2_{{shape}}$ ({band} band)")

    ax.legend()

    if output_path is not None:
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path / f"{band}_band_constant_minus_bazin.png",
            bbox_inches="tight",
        )

    plt.show()


def photometric_delta_chi_against_chi_SN_plot(
    objects: Path,
    band: str,
    output_path: Path | None = None,
) -> None:
    """Makes a plot of the delta chi squared score as a function of the reduced chi squared score for the bazin fit in a band to visualise what objects have been removed using this cut.

    Args:
        objects (Path): Path to the CSV that hosts the objects and the chi squard scores for the plot.
        band (str): The band that the plot uses the data from.
        output_path (Path | None, optional): The path that the plot will be outputted to. Defaults to None.
    """

    delta_chi_df = (
        pl.read_csv(objects)
        .filter((pl.col("bazin_reduced_chi2") > 0) & (pl.col("delta_reduced_chi2") > 0))
        .select("r:diaObjectId", "bazin_reduced_chi2", "delta_reduced_chi2", "category")
    )

    sn_like_df = delta_chi_df.filter(pl.col("category") == "SN-like")

    non_sn_like_df = delta_chi_df.filter(pl.col("category") == "Non-SN-like")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        sn_like_df["bazin_reduced_chi2"],
        sn_like_df["delta_reduced_chi2"],
        color="#0072B2",
        alpha=0.9,
        label="SN-like",
        zorder=3,
        s=30,
    )

    ax.scatter(
        non_sn_like_df["bazin_reduced_chi2"],
        non_sn_like_df["delta_reduced_chi2"],
        color="#D55E00",
        alpha=0.9,
        label="Non-SN-like",
        zorder=3,
        s=30,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    lower = min(xlim[0], ylim[0])
    upper = max(xlim[1], ylim[1])

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="black",
        label="Selection boundary",
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    ax.set_xlabel(rf"$\chi^2_{{shape}}$ ({band} band)")
    ax.set_ylabel(rf"$\chi^2_{{constant}} - \chi^2_{{shape}}$ ({band} band)")

    ax.legend()

    if output_path is not None:
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path / f"{band}_band_constant_minus_bazin_100_objects.png",
            bbox_inches="tight",
        )

    plt.show()


def spectro_strip_plots(
    sn_ii_spectro_fitted_peak_timing_results: Path,
    sn_ia_spectro_fitted_peak_timing_results: Path,
    output_path: Path | None = None,
) -> None:
    """Makes a strip plot of the results for the spectroscopic classes of SN that we have and what their delta fitted peak times were in the i and r bands.

    Args:
        spectro_fitted_peak_timing_results (Path): Path object to the results for the spectro objects after they have passed through the fitted peak timing function.
        output_path (Path): The Path object where the plot will be outputted to.
    """

    # Set the params of the strip plot
    sns.set_theme(
        style="whitegrid",
    )

    # Load in the results from the fitted peak timing csv file
    sn_ia_spectro_results = pl.read_csv(
        source=sn_ia_spectro_fitted_peak_timing_results,
        columns=[
            "r:diaObjectId",
            "peak_time_difference",
        ],
        schema_overrides={
            "r:diaObjectId": pl.String,
            "peak_time_difference": pl.Float64,
        },
    ).with_columns(
        pl.lit("SN Ia").alias("spec_class"),
    )
    sn_ii_spectro_results = pl.read_csv(
        source=sn_ii_spectro_fitted_peak_timing_results,
        columns=[
            "r:diaObjectId",
            "peak_time_difference",
        ],
        schema_overrides={
            "r:diaObjectId": pl.String,
            "peak_time_difference": pl.Float64,
        },
    ).with_columns(
        pl.lit("SN CC").alias("spec_class"),
    )

    spectro_results = pl.concat(
        [
            sn_ia_spectro_results,
            sn_ii_spectro_results,
        ],
    )

    colours = {
        "SN Ia": "#0072B2",
        "SN CC": "#D55E00",
    }

    sns.stripplot(
        data=spectro_results,
        x="peak_time_difference",
        y="spec_class",
        hue="spec_class",
        order=["SN Ia", "SN CC"],
        hue_order=["SN Ia", "SN CC"],
        jitter=0.2,
        dodge=False,
        palette=colours,
    )
    plt.axvline(
        x=50,
        ls="--",
        label=r"$|t^i_{max} - t^r_{max}|$ = 50 d",
    )

    plt.legend()
    plt.ylabel("")
    plt.xlabel(r"$|t^i_{max} - t^r_{max}|$ (days)")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path / "spectro_fitted_peak_delta_times.png")
    plt.show()


def off_var_chi_vs_i_flux_plot(
    bazin_fits_data_path: Path,
    off_var_chi_squared_result: pl.DataFrame,
    light_curve_data_path: Path,
    sn_like_objects: list[str],
) -> None:
    """Plots the off variation chi squared results as a function of the peak flux of the object in the i band to derive a meaningful empriical cut for the objects.

    Args:
        bazin_fits_data_path (Path): Path to the CSV that has the results from the Bazin fits for the objects in the i band.
        off_var_chi_squared_result (pl.DataFrame): The dataframe that has the results from the off variation chi squared calculation.
        light_curve_data_path (Path): The path to the light curve data for all the objects in the i band.
        sn_like_objects (list[str]): The list of strings that are confirmed SN-like objects so the plot shows colour coding to find the empirical cutting region.
    """

    # Read CSV with overrides for schema
    objects = (
        pl.read_csv(bazin_fits_data_path, columns=["r:diaObjectId"])
        .with_columns(pl.col("r:diaObjectId").cast(pl.Utf8))["r:diaObjectId"]
        .to_list()
    )

    # Read Parquet file
    light_curve_data = pl.read_parquet(light_curve_data_path)

    # Filter DataFrame to include only the objects of interest
    filter_conditions = [
        pl.col("r:band") == "i",
        pl.col("r:diaObjectId").cast(pl.String).is_in(objects),
    ]
    light_curve_data_filtered = light_curve_data.filter(filter_conditions)

    # Group by 'r:diaObjectId' and calculate peak flux for each group
    peak_i_band_flux_data = light_curve_data_filtered.group_by("r:diaObjectId").agg(
        [
            pl.col("r:psfFlux").max().alias("peak_i_band_flux"),
        ]
    )

    peak_i_band_flux_data = peak_i_band_flux_data.with_columns(
        pl.col("r:diaObjectId").cast(pl.String)
    )

    chi2_object_peak_i_band_flux = peak_i_band_flux_data.join(
        off_var_chi_squared_result, "r:diaObjectId", "inner"
    )

    x_values = chi2_object_peak_i_band_flux.select(
        pl.col("peak_i_band_flux")
    ).to_series()
    y_values = chi2_object_peak_i_band_flux.select(pl.col("chi2_off")).to_series()
    labels = chi2_object_peak_i_band_flux.select(pl.col("r:diaObjectId")).to_series()

    colors = [
        color
        for color, obj_id in zip(
            ["red" if obj_id in sn_like_objects else "blue" for obj_id in labels],
            labels,
        )
    ]

    plt.scatter((x_values), y_values, c=colors, alpha=0.7)

    plt.legend(
        title="r:diaObjectId",
        handles=[
            plt.Line2D(
                [0], [0], marker="o", color="red", markersize=10, linestyle="None"
            ),
            plt.Line2D(
                [0], [0], marker="o", color="blue", markersize=10, linestyle="None"
            ),
        ],
        labels=["SN-like", "Non-SN-like"],
    )

    # plt.scatter(chi2_object_peak_i_band_flux.select(pl.col("peak_i_band_flux")), chi2_object_peak_i_band_flux.select(pl.col("chi2_off")))
    plt.xscale("log")
    plt.yscale("log")
    plt.title("The off-var chi squared as a function of peak i band flux")
    plt.xlabel("Peak i band flux (nanoJanskys)")
    plt.ylabel("Off Variation chi squared score.")
    plt.show()


def bar_chart_pass_fractions(
    passing_ia_objects: Path,
    failing_ia_objects: Path,
    passing_ii_objects: Path,
    failing_ii_objects: Path,
    output_path: Path | None = None,
) -> None:
    """Creates a bar chart that represents the passing fractions for whatever objects you would like.

    Args:
        objects (Path): Path to the CSV file of the objects with the pass/fail info with it
        output_path (Path | None, optional): The path where the plot will be outputted. Defaults to None.
    """

    passing_ias = pl.read_csv(
        passing_ia_objects,
        schema_overrides={
            "r:diaObjectId": pl.String,
        },
    ).with_columns(
        pl.lit("SN Ia").alias("spec_class"),
        pl.lit(True).alias("rise_slope_pass"),
    )

    failing_ias = pl.read_csv(
        failing_ia_objects,
        schema_overrides={
            "r:diaObjectId": pl.String,
        },
    ).with_columns(
        pl.lit("SN Ia").alias("spec_class"),
        pl.lit(False).alias("rise_slope_pass"),
    )

    passing_iis = pl.read_csv(
        passing_ii_objects,
        schema_overrides={
            "r:diaObjectId": pl.String,
        },
    ).with_columns(
        pl.lit("SN CC").alias("spec_class"),
        pl.lit(True).alias("rise_slope_pass"),
    )

    failing_iis = pl.read_csv(
        failing_ii_objects,
        schema_overrides={
            "r:diaObjectId": pl.String,
        },
    ).with_columns(
        pl.lit("SN CC").alias("spec_class"),
        pl.lit(False).alias("rise_slope_pass"),
    )

    combined = pl.concat(
        [
            passing_ias,
            failing_ias,
            passing_iis,
            failing_iis,
        ]
    )

    summary = (
        combined.group_by(pl.col("spec_class"))
        .agg(
            pl.len().alias("n_total"),
            pl.col("rise_slope_pass").sum().alias("n_pass"),
        )
        .with_columns(
            (pl.col("n_pass") / pl.col("n_total")).alias("pass_fraction"),
        )
    )

    plt.subplots(figsize=(6, 4))

    colours = {
        "SN Ia": "#0072B2",
        "SN CC": "#D55E00",
    }

    hatches = {
        "SN Ia": "//",
        "SN CC": "xx",
    }

    edgecolours = {
        "SN Ia": "#05507B",
        "SN CC": "#743504",
    }

    bars = plt.bar(
        (
            summary["spec_class"][0],
            summary["spec_class"][1],
        ),
        summary["pass_fraction"],
        color=[colours[spec_class] for spec_class in summary["spec_class"]],
        alpha=0.8,
        edgecolor=[edgecolours[spec_class] for spec_class in summary["spec_class"]],
        hatch=[hatches[spec_class] for spec_class in summary["spec_class"]],
    )

    plt.bar_label(
        bars,
        labels=[
            f"{n_pass}/{n_total} ({((n_pass / n_total) * 100):.2f} %)"
            for n_pass, n_total in zip(
                summary["n_pass"],
                summary["n_total"],
            )
        ],
        padding=3,
    )

    plt.grid(
        axis="y",
        linestyle="-",
        color="grey",
        alpha=1,
    )

    plt.ylim(0, 1)
    plt.ylabel("Pass fraction")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    if output_path is not None:
        plt.savefig(output_path)


def plot_salt_phase_diagnostic(
    phase_results_path: Path,
    light_curve_data_path: Path,
    band_map: dict,
    output_dir: Path | None = None,
) -> None:
    """This function makes plots of the SALT fitted objects but plots the tau parameters as vertical lines to understand where objects are failing and if the phase cuts are doing what they are supposed to be doing, a diagnostic.

    Args:
        phase_results_path (Path): Path to the results from the phase cuts values.
        light_curve_data_path (Path): Path to the light curve data for the objects.
        output_dir (Path): Path to the directory where the user wants the plots saved.
    """
    phase_results = pl.read_csv(
        phase_results_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
            "z_used": pl.Float64,
            "t0": pl.Float64,
        },
    )

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

    tau_edges = [-10, 5, 20, 35]

    for row in phase_results.iter_rows(named=True):
        object_id = row["r:diaObjectId"]
        redshift = row["z_used"]
        if not row["fit_success"]:
            continue

        object_data = light_curve_data.filter(pl.col("r:diaObjectId") == object_id)

        if redshift is None:
            continue

        data = Table(
            {
                "time": object_data["r:midpointMjdTai"].to_numpy(),
                "band": [band_map[b] for b in object_data["r:band"].to_list()],
                "flux": object_data["r:psfFlux"].to_numpy(),
                "fluxerr": object_data["r:psfFluxErr"].to_numpy(),
                "zp": np.full(object_data.height, 31.4),
                "zpsys": np.full(object_data.height, "ab"),
            }
        )
        # Rebuilding the fitted model from the SALT fitted parameters of the object
        model = sncosmo.Model(source="salt3")

        model.set(
            z=redshift,
            t0=row["t0"],
            x0=row["x0"],
            x1=row["x1"],
            c=row["c"],
        )

        fig = sncosmo.plot_lc(
            data,
            model=model,
        )

        time_edges = [tau * (1 + redshift) for tau in tau_edges]

        phase_regions = [
            (time_edges[0], time_edges[1], "Peak", "#56B4E9"),
            (time_edges[1], time_edges[2], "Decline", "#009E73"),
            (time_edges[2], time_edges[3], "Tail", "#D55E00"),
        ]

        n_panels = len(fig.axes) // 2  # Total number of passband panels
        flux_axes = fig.axes[:n_panels]  # The first half are ALWAYS the flux panels
        pull_axes = fig.axes[n_panels:]  # The second half are ALWAYS the pull panels

        for ax in flux_axes:
            for tau, t_edge in zip(tau_edges, time_edges):
                ax.axvline(
                    t_edge,
                    linestyle="--",
                    linewidth=0.8,
                    color="#595959",
                    alpha=0.7,
                    zorder=1,
                )

                for t_start, t_end, name, colour in phase_regions:
                    ax.axvspan(
                        t_start,
                        t_end,
                        color=colour,
                        alpha=0.08,
                        zorder=0,
                    )

                for tau, t_edge in zip(tau_edges, time_edges):
                    sign = "+" if tau > 0 else ""
                    ax.text(
                        x=t_edge,
                        y=0.98,
                        s=rf"$\tau = {sign}{tau}$ d",
                        transform=ax.get_xaxis_transform(),
                        rotation=90,
                        verticalalignment="top",
                        horizontalalignment="right",
                        fontsize=8,
                        color="#333333",
                        alpha=0.85,
                    )

        if output_dir is not None:
            fig.savefig(output_dir / f"{object_id}_phase_diagnostic.png")
        plt.close(fig)


def strip_plot(
    data_path: Path,
    output_path: Path | None = None,
) -> None:
    """Produces a strip plot that shows the distribution of the differences in times between the two peaks for the objects.

    Args:
        data_path (Path): Path to the data CSV.
        output_path (Path | None, optional): Path to where the plot will be outputted. Defaults to None.
    """

    data = pl.read_csv(
        data_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
            "category": pl.String,
        },
        columns=["r:diaObjectId", "delta_peak_mjd", "category"],
    )

    data = data.with_columns(
        pl.col("delta_peak_mjd").cast(
            pl.Float64,
            strict=False,
        ),
    )

    sns.set_theme(
        style="whitegrid",
    )

    delta_peak_mjd = data["delta_peak_mjd"]

    objects = data["category"]
    print(objects)

    colours = {
        " SN-like": "#0072B2",
        " Non-SN-like": "#D55E00",
    }

    sns.stripplot(
        y=objects,
        x=delta_peak_mjd,
        data=data,
        hue="category",
        palette=colours,
        order=[
            " SN-like",
            " Non-SN-like",
        ],
        jitter=0.15,
    )

    plt.axvline(
        50,
        linestyle="--",
        label=r"$\Delta t_{\rm peak}=50$ d",
    )
    plt.legend()
    plt.ylabel("")
    plt.xlabel(r"$\Delta t_{\mathrm{peak}}$ (days)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def spectro_strip_plot(
    passing_ia_objects: Path,
    failing_ia_objects: Path,
    passing_ii_objects: Path,
    failing_ii_objects: Path,
    output_path: Path | None = None,
) -> None:
    """Produces a strip plot for the spectrospoic sample of objects, with the delta mjd on the x axis and the categories on the y axis.

    Args:
        passing_ia_objects (Path): Path to the passing ia objects.
        failing_ia_objects (Path): Path to the failing ia objects.
        passing_ii_objects (Path): Path to the passing II objects.
        failing_ii_objects (Path): Path to the failing II objects.
        output_path (Path | None, optional): The path to where the plot should be outputted to. Defaults to None.
    """

    passing_ias = (
        pl.read_csv(
            passing_ia_objects,
            schema_overrides={
                "r:diaObjectId": pl.String,
            },
            columns=[
                "r:diaObjectId",
                "delta_peak_mjd",
            ],
        )
        .with_columns(
            pl.lit("SN Ia").alias("spec_class"),
        )
        .with_columns(
            pl.col("delta_peak_mjd").cast(
                pl.Float64,
                strict=False,
            ),
        )
    )

    failing_ias = (
        pl.read_csv(
            failing_ia_objects,
            schema_overrides={
                "r:diaObjectId": pl.String,
            },
            columns=[
                "r:diaObjectId",
                "delta_peak_mjd",
            ],
        )
        .with_columns(
            pl.lit("SN Ia").alias("spec_class"),
        )
        .with_columns(
            pl.col("delta_peak_mjd").cast(
                pl.Float64,
                strict=False,
            ),
        )
    )

    passing_iis = (
        pl.read_csv(
            passing_ii_objects,
            schema_overrides={
                "r:diaObjectId": pl.String,
            },
            columns=[
                "r:diaObjectId",
                "delta_peak_mjd",
            ],
        )
        .with_columns(
            pl.lit("SN CC").alias("spec_class"),
        )
        .with_columns(
            pl.col("delta_peak_mjd").cast(
                pl.Float64,
                strict=False,
            ),
        )
    )

    failing_iis = (
        pl.read_csv(
            failing_ii_objects,
            schema_overrides={
                "r:diaObjectId": pl.String,
            },
            columns=[
                "r:diaObjectId",
                "delta_peak_mjd",
            ],
        )
        .with_columns(
            pl.lit("SN CC").alias("spec_class"),
        )
        .with_columns(
            pl.col("delta_peak_mjd").cast(
                pl.Float64,
                strict=False,
            ),
        )
    )

    spectro_df = pl.concat(
        [
            passing_ias,
            failing_ias,
            passing_iis,
            failing_iis,
        ]
    )

    sns.set_theme(
        style="whitegrid",
    )

    delta_peak_mjd = spectro_df["delta_peak_mjd"]

    objects = spectro_df["spec_class"]

    colours = {
        "SN Ia": "#0072B2",
        "SN CC": "#D55E00",
    }

    sns.stripplot(
        y=objects,
        x=delta_peak_mjd,
        data=spectro_df,
        hue="spec_class",
        jitter=0.2,
        palette=colours,
        order=["SN Ia", "SN CC"],
    )

    plt.axvline(
        50,
        linestyle="--",
        label=r"$\Delta t_{\rm peak}=50$ d",
    )
    plt.legend()
    plt.ylabel("")
    plt.xlabel(r"$\Delta t_{\mathrm{peak}}$ (days)")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    PROJECT_DIR = (
        Path.home() / "Documents" / "University" / "Honours" / "Honours_Project"
    )

    DATA_DIR = PROJECT_DIR / "data"

    IMAGE_DIR = PROJECT_DIR / "images"

    OBJECTS = (
        DATA_DIR
        / "salt_fitting"
        / "ia_search"
        / "with_z_results"
        / "phase_cuts_results"
        / "fail.csv"
    )

    LIGHT_CURVE_DATA = (
        DATA_DIR / "light_curve_data" / "combined_light_curve_data.parquet"
    )

    OUTPUT_PATH = IMAGE_DIR / "salt_plots" / "ia_search" / "phase_cuts" / "fail"

    BAND_MAP = {
        "u": "lsstu",
        "g": "lsstg",
        "r": "lsstr",
        "i": "lssti",
        "z": "lsstz",
        "y": "lssty",
    }

    plot_salt_phase_diagnostic(
        phase_results_path=OBJECTS,
        light_curve_data_path=LIGHT_CURVE_DATA,
        band_map=BAND_MAP,
        output_dir=OUTPUT_PATH,
    )
