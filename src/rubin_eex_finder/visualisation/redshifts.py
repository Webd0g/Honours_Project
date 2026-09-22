from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def host_photo_z_distro(
    object_path: Path, output_path: Path | None = None
) -> pl.DataFrame:
    """This function finds the distribution of the host photo redshifts of the sample from the Supernova Selection cuts i.e. up until the fitted peak timing cut.

    Args:
        object_path (Path): The path object to the CSV of the objects.
        output_path (Path): The path of the destination for the outputted CSV file.

    Returns:
        pl.DataFrame: The polars dataframe that contains the inforamtion collected by this function added to the input CSV file.
    """

    with_photoz = pl.read_csv(
        object_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
            "host_photoz": pl.Float64,
            "host_photoz_err": pl.Float64,
        },
    ).filter(
        pl.col("host_photoz").is_not_null()
        & pl.col("host_photoz").is_finite()
        & (pl.col("host_photoz") > 0)
    )

    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)

    print(
        with_photoz.select("host_photoz", "host_photoz_err").describe(
            [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]
        )
    )

    redshifts = with_photoz["host_photoz"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(redshifts, bins=30)

    plt.xlabel("Host photometric redshifts")
    plt.ylabel("Number of objects")
    plt.title("Host Photometric Redshift Distribution")

    plt.tight_layout()
    plt.show()

    return with_photoz


def host_photoz_vs_salt3_z(
    object_path: Path, plot_output_path: Path | None = None
) -> None:
    """This function plots the host photo z of the objects against the blindly fitted SALT3 redshifts.

    Args:
        object_path (Path): Path to the CSV with the objects to plot
        plot_output_path (Path | None, optional): The output path that the plot saves to. Defaults to None.
    """

    salt_fitted_objects = pl.read_csv(
        object_path,
        schema_overrides={"host_photoz": pl.Float64, "fitted_z": pl.Float64},
    )

    host_photoz = salt_fitted_objects["host_photoz"].to_numpy()
    salt_fitted_z = salt_fitted_objects["fitted_z"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.scatter(salt_fitted_z, host_photoz)

    ax = plt.gca()

    lower = min(ax.get_xlim()[0], ax.get_ylim()[0])

    upper = max(ax.get_xlim()[1], ax.get_ylim()[1])

    ax.plot([lower, upper], [lower, upper], "r--", label="unity")

    plt.xlabel("SALT3 fitted redshift")
    plt.ylabel("Host photo redshift")
    plt.title("Host photometric redshifts as a function of SALT3 fitted redshifts.")

    plt.tight_layout()
    plt.savefig(plot_output_path / "host_photoz_vs_salt3z.png")
    plt.show()


def normalised_z_residuals_plot(
    object_path: Path, plot_output_path: Path | None = None
) -> None:
    """This function plots the normalised SALT3 fitted redshift residuals as a function of the host redshift with a horizontal line at 0.

    Args:
        object_path (Path): Path to the CSV with the objects to plot
        plot_output_path (Path | None, optional): The output path that the plot saves to. Defaults to None.
    """

    salt_fitted_objects = pl.read_csv(
        object_path,
        schema_overrides={"host_photoz": pl.Float64, "delta_z_normalised": pl.Float64},
    )

    host_photoz = salt_fitted_objects["host_photoz"].to_numpy()
    delta_z_normalised = salt_fitted_objects["delta_z_normalised"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.scatter(host_photoz, delta_z_normalised)

    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Host photo redshift")
    plt.ylabel("Residual in normalised SALT3 fitted redshift")
    plt.title(
        "Residuals of normalised SALT3 fitted redshift as a function of host photo Z"
    )

    plt.tight_layout()
    plt.savefig(plot_output_path / "residuals_salt3_fits_vs_hostz.png")
    plt.show()

