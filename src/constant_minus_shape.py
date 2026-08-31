"""Here we will fit the objects with a constant and determine that object's reduced chi square with the constant fit to the i band data.
Then the bazin fit reduced chi squared in the i band will be subtracted from that constant fit. The values obtained for each object from that subtraction
will then be scatter plotted as a function of the reduced chi squared score of the Bazin fit in the i band.

This is the most significant cut that can be done for photmetric selection and shoudl hopefully result in SN-like obejcts being greatly separated from
objects that are more flat than SN-like."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

SN_LIKE_OBJECTS = [
    "170028511067636654",
    "170019716323082348",
    "170032900274651160",
    "170032917163016282",
    "170046089928376424",
    "170235219334922248",
    "170261592629837872",
    "313699514821640259",
    "313835836426682441",
    "313945770546430083",
    "313963405286309936",
    "170257199649521669",
    "170411112000913487",
    "170028511473959008",
    "170094456357257301",
    "170094456400773135",
    "170226393328648251",
    "170230797176406143",
    "170296772058939493",
    "313765480704245975",
    "313831458352922664",
    "313871013530042570",
    "313928194288779300",
    "313932564866269223",
    "313932564880424967",
    "313932595758366765",
    "313936964384981047",
    "313945769043820590",
    "313945770525982796",
    "313950161909317636",
    "313958953395945530",
    "313963406691402058",
    "313967766285058088",
    "313980947917701210",
    "313985346353234012",
    "314011741926195228",
    "170032882395906068",
    "170274770629689377",
    "170274770656952330",
    "170274770671632438",
    "170389126953566274",
    "170433140735082813",
    "170433141132493221",
    "313708289017249906",
    "313708289051328625",
    "313761043604045880",
    "313783031112400976",
    "313831458447818885",
    "313831458523316231",
    "313853496686280764",
    "313853497291309068",
]

NOT_SN_LIKE_OBJECTS = [
    "170019696299475118",
    "170019696327786549",
    "170019696328835126",
    "170019696414818363",
    "170019696423731323",
    "170019696438935593",
    "170019696467247193",
    "170019696472490057",
    "170019696475111620",
    "170019696484548684",
    "170019716273274936",
    "170019716279566437",
    "170019716313645184",
    "170019716320460966",
    "170019716324130833",
    "170019716333043819",
    "170019716335140929",
    "170019716412735616",
    "170019716433707059",
    "170019716456251517",
    "170019717277810782",
    "170028485637046567",
    "170028485637570618",
    "170028485641240585",
    "170028485648056414",
    "170028485652775010",
    "170028485657493519",
    "170028485660639330",
    "170028485663260764",
    "170028485666930763",
    "170028485674270849",
    "170028485674795036",
    "170028485675319420",
    "170028485677940837",
    "170028485678465209",
    "170028485681086597",
    "170028485686329432",
    "170028485686853721",
    "170028485695242343",
    "170028485697339398",
    "170028485698912382",
    "170028485702058098",
    "170028485707300870",
    "170028485710971008",
    "170028485716738057",
    "170028485719359556",
    "170028485725126711",
    "170028485725651088",
    "170028485726699537",
    "170028485727748115",
    "170028485731418130",
]


def constant_fit(flux: np.ndarray, flux_err: np.ndarray) -> tuple:
    """
    Fit a constant flux model to a light curve.
    """

    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)

    # Inverse-variance weights
    weights = 1.0 / flux_err**2

    # Best-fitting constant
    constant = np.sum(weights * flux) / np.sum(weights)

    # Chi-squared
    chi2 = np.sum(((flux - constant) / flux_err) ** 2)

    # One fitted parameter: constant
    dof = len(flux) - 1

    reduced_chi2 = chi2 / dof

    return constant, reduced_chi2


def constant_batch_fits(
    object_path: Path, light_curve_data_path: Path, band: str
) -> pl.DataFrame:

    objects = pl.read_csv(
        object_path,
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
            "r:diaSourceId",
        ],
    ).with_columns(pl.col("r:diaObjectId").cast(pl.String))

    light_curve_data = light_curve_data.filter(
        pl.col("r:diaObjectId").is_in(objects)
        & (pl.col("r:band") == band)
        & (pl.col("r:psfFlux") > 0)
        & pl.col("r:psfFlux").is_finite()
        & pl.col("r:psfFluxErr").is_finite()
        & (pl.col("r:psfFluxErr") > 0)
    )

    results = {"r:diaObjectId": [], "constant": [], "reduced_chi2": []}

    groups = {
        key[0]: group_df
        for key, group_df in light_curve_data.group_by(
            "r:diaObjectId", maintain_order=True
        )
    }

    for object_id in objects:
        group_df = groups.get(object_id)

        if group_df is None:
            continue

        band_df = group_df.sort("r:midpointMjdTai").unique(
            subset=["r:midpointMjdTai"], keep="first", maintain_order=True
        )

        if len(band_df) < 2:
            continue

        flux = band_df["r:psfFlux"].to_numpy().astype(np.float64)

        flux_err = band_df["r:psfFluxErr"].to_numpy().astype(np.float64)

        # Remove invalid uncertainties
        valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)

        flux = flux[valid]
        flux_err = flux_err[valid]

        if len(flux) < 2:
            continue

        constant, reduced_chi2 = constant_fit(
            flux,
            flux_err,
        )

        results["r:diaObjectId"].append(object_id)
        results["constant"].append(constant)
        results["reduced_chi2"].append(reduced_chi2)

    result_df = pl.DataFrame(results)

    return result_df


def compute_delta_chi(
    object_path: Path,
    light_curve_data_path: Path,
    band: str,
    bazin_fit_data_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    """Here we compute the difference between the constant fit and the Bazin
    fit and append it as a column to the df."""

    # Fit with a constant for the objects
    constant_fit_result = constant_batch_fits(object_path, light_curve_data_path, band)

    # Renaming the columns for the constant chi squared
    constant_fit_result = constant_fit_result.rename(
        {"reduced_chi2": "constant_reduced_chi2"}
    )

    bazin_fit_chi_squared = pl.read_csv(bazin_fit_data_path).with_columns(
        pl.col("r:diaObjectId").cast(pl.String)
    )

    # Renaming the column for the Bazin chi squared
    bazin_fit_chi_squared = bazin_fit_chi_squared.rename(
        {"reduced_chi2": "bazin_reduced_chi2"}
    )

    # Joining the two data sources into one df making it easy to compare the columns of the data and add a new column for the delta
    delta_chi_df = constant_fit_result.join(
        bazin_fit_chi_squared, on="r:diaObjectId", how="inner", suffix="bazin"
    ).with_columns(
        (pl.col("constant_reduced_chi2") - pl.col("bazin_reduced_chi2")).alias(
            "delta_reduced_chi2"
        )
    )

    delta_chi_df.write_csv(output_path)

    return delta_chi_df


def delta_chi_against_chi_SN_plot(
    path_to_ii_delta_chi_results: Path,
    path_to_ia_delta_chi_results: Path,
    path_to_photometric_delta_chi_results: Path,
    output_path: Path,
    band: str,
) -> None:

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
            ia_delta_chi_df,
            ii_delta_chi_df,
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    photometric = delta_chi_df.filter(pl.col("class") == "Photometric")

    ia = delta_chi_df.filter(pl.col("class") == "SN Ia")

    cc = delta_chi_df.filter(pl.col("class") == "Core-Collapse")

    ax.scatter(
        photometric["bazin_reduced_chi2"],
        photometric["delta_reduced_chi2"],
        color="black",
        alpha=0.7,
        label="Photometric",
    )

    ax.scatter(
        ia["bazin_reduced_chi2"],
        ia["delta_reduced_chi2"],
        color="green",
        alpha=0.9,
        label="Spectroscopic SN Ia",
        zorder=3,
    )

    ax.scatter(
        cc["bazin_reduced_chi2"],
        cc["delta_reduced_chi2"],
        color="red",
        alpha=0.9,
        label="Spectroscopic CC",
        zorder=3,
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

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(rf"Bazin reduced $\chi^2$ ({band} band)")
    ax.set_ylabel(rf"$\Delta\chi^2_\nu$ ({band} band)")
    ax.set_title(f"Constant vs Bazin fit comparison in {band} band")

    ax.legend()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        output_path / f"{band}_band_constant_minus_bazin.png",
        bbox_inches="tight",
    )

    plt.show()


def delta_chi_check(path_to_delta_chi_results: Path):
    """Here we separate the objects that are above the unity line from those that are below the unity line."""

    delta_chi_df = pl.read_csv(path_to_delta_chi_results)

    delta_chi_df = delta_chi_df.with_columns(
        (pl.col("delta_reduced_chi2") > pl.col("bazin_reduced_chi2")).alias(
            "passes_delta_chi_cut"
        )
    )

    delta_chi_df = delta_chi_df.with_columns(
        (pl.col("delta_reduced_chi2") <= pl.col("bazin_reduced_chi2")).alias(
            "failed_delta_chi_cut"
        )
    )

    return delta_chi_df


def save_pass_fail_objects(
    path_to_delta_chi_results: Path, output_path_pass: Path, output_path_fail: Path
):
    """Here we write the results of which objects passed and which objects failed the to their respective CSV files."""

    delta_chi_df = delta_chi_check(path_to_delta_chi_results)

    delta_chi_df_pass = delta_chi_df.filter(pl.col("passes_delta_chi_cut"))

    delta_chi_df_fail = delta_chi_df.filter(pl.col("failed_delta_chi_cut"))

    delta_chi_df_pass.write_csv(output_path_pass)

    delta_chi_df_fail.write_csv(output_path_fail)


if __name__ == "__main__":
    TEST_DATA = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "test_data"
        / "100_constant_minus_shape_test_objects.csv"
    )

    LIGHT_CURVE_DATA = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "light_curve_data"
        / "combined_light_curve_data.parquet"
    )

    TEST_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "100_object_test.csv"
    )

    BAZIN_FITS_DATA_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "bazin_fits_results"
        / "bazin_fits_i_band_results.csv"
    )

    CANDIDATES = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "timing_cuts_results"
        / "timing_cuts_passed.csv"
    )

    G_BAND_VALUES_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "g_band"
        / "constant_minus_shape_values.csv"
    )

    G_BAND_PASS_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "g_band"
        / "pass.csv"
    )

    G_BAND_FAIL_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "g_band"
        / "fail.csv"
    )

    G_BAND_PLOT_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "images"
        / "constant_minus_shape_plots"
        / "g_band"
        / "g_band_constant_minus_shape.png"
    )

    R_BAND_VALUES_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "r_band"
        / "constant_minus_shape_values.csv"
    )

    R_BAND_PASS_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "r_band"
        / "pass.csv"
    )

    R_BAND_FAIL_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "r_band"
        / "fail.csv"
    )

    R_BAND_PLOT_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "images"
        / "constant_minus_shape_plots"
        / "r_band"
        / "r_band_constant_minus_shape.png"
    )

    I_BAND_VALUES_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "i_band"
        / "constant_minus_shape_values.csv"
    )

    I_BAND_PASS_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "i_band"
        / "pass.csv"
    )

    I_BAND_FAIL_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "i_band"
        / "fail.csv"
    )

    I_BAND_PLOT_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "images"
        / "constant_minus_shape_plots"
        / "i_band"
        / "i_band_constant_minus_shape.png"
    )

    Z_BAND_VALUES_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "z_band"
        / "constant_minus_shape_values.csv"
    )

    Z_BAND_PASS_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "z_band"
        / "pass.csv"
    )

    Z_BAND_FAIL_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "constant_minus_shape_results"
        / "z_band"
        / "fail.csv"
    )

    Z_BAND_PLOT_OUTPUT_PATH = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "images"
        / "constant_minus_shape_plots"
        / "z_band"
        / "z_band_constant_minus_shape.png"
    )

    PATH_TO_IA_SPECTRO_RESULTS = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "bazin_fits_results"
        / "spectroscopic_tests"
        / "ia_tests"
        / "bazin_fits_i_band_results.csv"
    )

    PATH_TO_II_SPECTRO_RESULTS = (
        Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "bazin_fits_results"
        / "spectroscopic_tests"
        / "ii_tests"
        / "bazin_fits_i_band_results.csv"
    )

    compute_delta_chi(
        object_path=PATH_TO_II_SPECTRO_RESULTS,
        light_curve_data_path=LIGHT_CURVE_DATA,
        band="i",
        bazin_fit_data_path=PATH_TO_II_SPECTRO_RESULTS,
        output_path=Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "sn_ii_test_results"
        / "constant_minus_shape_results"
        / "values.csv",
    )

    delta_chi_against_chi_SN_plot(
        path_to_ii_delta_chi_results=Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "data"
        / "sn_ii_test_results"
        / "constant_minus_shape_results"
        / "values.csv",
        path_to_ia_delta_chi_results=(
            Path.home()
            / "Documents"
            / "University"
            / "Honours"
            / "Honours_Project"
            / "data"
            / "sn_ia_test_results"
            / "constant_minus_shape_results"
            / "values.csv"
        ),
        path_to_photometric_delta_chi_results=(
            Path.home()
            / "Documents"
            / "University"
            / "Honours"
            / "Honours_Project"
            / "data"
            / "constant_minus_shape_results"
            / "constant_minus_shape_values.csv"
        ),
        output_path=Path.home()
        / "Documents"
        / "University"
        / "Honours"
        / "Honours_Project"
        / "images"
        / "results_plots",
        band="i",
    )

    save_pass_fail_objects(
        path_to_delta_chi_results="/home/riley/Documents/University/Honours/Honours_Project/data/sn_ia_test_results/constant_minus_shape_results/values.csv",
        output_path_pass="/home/riley/Documents/University/Honours/Honours_Project/data/sn_ia_test_results/constant_minus_shape_results/pass.csv",
        output_path_fail="/home/riley/Documents/University/Honours/Honours_Project/data/sn_ia_test_results/constant_minus_shape_results/fail.csv",
    )
