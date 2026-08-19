"""This code computes the off main-variation chi squared result which will help remove those objects that have variable fluxes that might pass the Bazin fit.
Bazin 2009 state this was a complemantary cut to the cut from the Bazin chi square and constant chi square fit as this can remove objects that will pass that
cut."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from light_curve_plotter import lightcurve_plotting

# Path to the directory where all the Bazin fits CSV are
BAZIN_FITS_DATA_DIR = (
    Path.home()
    / "Documents"
    / "University"
    / "Honours"
    / "Honours_Project"
    / "data"
    / "bazin_fits_results"
)

LIGHT_CURVE_DATA_DIR = (
    Path.home()
    / "Documents"
    / "University"
    / "Honours"
    / "Honours_Project"
    / "data"
    / "light_curve_data"
)

VARIABILITY_WINDOW_PLOT_OUTPUT_DIR = (
    Path.home()
    / "Documents"
    / "University"
    / "Honours"
    / "Honours_Project"
    / "images"
    / "off_variation_plots"
)

OFF_VAR_CHI2_CSV_OUTPUT_DIR = (
    Path.home()
    / "Documents"
    / "University"
    / "Honours"
    / "Honours_Project"
    / "data"
    / "off_variation_results"
)

SN_LIKE_OBJECTS = (
    "170019716323082348",
    "313928194151940140",
    "313853517466435621",
    "313853501446815761",
    "313761043604045880",
    "313761042284412983",
    "313690717384867850",
    "170270398108139547",
    "170094456400773135",
    "170094456357257301",
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
    "313677516803932212",
    "313695087643066440",
    "313765480559017992",
    "313871014435488133",
)

NOT_SN_LIKE_OBJECTS = (
    "170019717135204495",
    "170019696305242310",
    "170019696442081285",
    "170019716324130933",
    "170019716327800983",
    "170019716479320072",
    "170028485639143520",
    "170028485639143548",
    "170028485642813476",
    "170028485648056327",
    "170028529317052437",
    "170028500628013064",
    "170028510469947486",
    "170028510471521062",
    "170028510473617524",
    "170028510474141813",
    "170028510489870374",
    "170028510495637583",
    "170028510507696173",
    "170028510508744828",
    "170028510522900564",
    "170028510526046240",
    "170028510528143378",
    "170028510531813398",
    "170028510542299163",
    "170028510605738072",
    "170028510610980932",
    "170028510612029446",
    "170028510629330965",
    "170028510636146781",
    "170028510639816802",
    "170028510639816858",
    "170028510660788996",
    "170028510666555431",
    "170028510668128342",
    "170028510675992678",
    "170028510687526916",
    "170028510737334281",
    "170028510737858665",
    "170028510745722944",
    "170028510748869541",
    "170028510756208687",
    "170028510807064685",
    "170028510824890492",
    "170028510891999237",
    "170028510916640778",
    "170028510922407946",
    "170028511008391720",
    "170028511061868555",
    "170028511067636654",
)


def variability_window(
    i_band_bazin_fits_path: Path, light_curve_data_path: Path
) -> pl.DataFrame:
    """Here is where we find the main variability window for each objects Bazin fit.

    We do this so that we can seperate the points of main variation from the points that
    aren't in the main varaition of the Bazin fit."""

    bazin_i_data = pl.read_csv(
        i_band_bazin_fits_path,
        schema_overrides={
            "r:diaObjectId": pl.String,
            "reference_time": pl.Float64,
            "rise_time": pl.Float64,
            "fall_time": pl.Float64,
            "baseline": pl.Float64,
        },
        columns=[
            "r:diaObjectId",
            "reference_time",
            "rise_time",
            "fall_time",
            "baseline",
        ],
    )

    light_curve_data = pl.read_parquet(light_curve_data_path)
    light_curve_data = light_curve_data.with_columns(
        pl.col("r:diaObjectId").cast(pl.String)
    )

    bazin_i_data = bazin_i_data.with_columns(
        (
            pl.col("reference_time")
            + pl.col("rise_time")
            * (pl.col("fall_time") / pl.col("rise_time") - 1).log()
        ).alias("peak_time")
    )

    bazin_i_data = bazin_i_data.with_columns(
        [
            (pl.col("peak_time") - (3 * pl.col("rise_time"))).alias("var_start"),
            (pl.col("peak_time") + (3 * pl.col("fall_time"))).alias("var_end"),
        ]
    )

    bazin_light_curve_data = light_curve_data.join(
        bazin_i_data.select(
            [
                "r:diaObjectId",
                "var_start",
                "var_end",
            ]
        ),
        on="r:diaObjectId",
        how="inner",
    )
    off_var_data = bazin_light_curve_data.filter(
        (pl.col("r:midpointMjdTai") < pl.col("var_start"))
        | (pl.col("r:midpointMjdTai") > pl.col("var_end"))
    )
    return off_var_data


def plotting_off_var_data(
    i_band_bazin_fits_path: Path, light_curve_data_path: Path, output_path: Path
) -> None:
    """Here we plot the data points
    that are off the main variation.
    This is done iteratively to ensure that we aren't cutting
    off any points that are in the main variation."""

    off_var_data = variability_window(i_band_bazin_fits_path, light_curve_data_path)
    for (diaObjectId,), off_variation_data in off_var_data.group_by("r:diaObjectId"):
        diaObjectId = str(diaObjectId)
        lightcurve_plotting(off_variation_data, diaObjectId, output_path)


def load_band_baselines(bazin_fits_paths: dict[str, Path]):
    """Here we want to load the baselines from
    the Bazin fits for the objects accross all bands
    so that when we do compute the off variation
    chi squared we can use the baselines from each band."""

    baseline_tables = []

    for band, path in bazin_fits_paths.items():
        baseline_df = pl.read_csv(
            path,
            schema_overrides={"r:diaObjectId": pl.String, "baseline": pl.Float64},
            columns=["r:diaObjectId", "baseline"],
        )

        baseline_df = baseline_df.with_columns(pl.lit(band).alias("r:band"))

        baseline_tables.append(baseline_df)

    baselines = pl.concat(baseline_tables)

    return baselines


def off_var_chi_squared(
    i_band_bazin_fits_path: Path,
    bazin_fits_paths: dict[str, Path],
    light_curve_data_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    """Here we compute the chi squared for the data points that are considered
    off the main variation of the SN

    We do this so we can make a cut based on this parameter empirically using the SN-like
    objects as our ground truth."""

    off_var_window_df = variability_window(
        i_band_bazin_fits_path, light_curve_data_path
    )

    band_baselines = load_band_baselines(bazin_fits_paths)

    off_var_window_df = off_var_window_df.with_columns(pl.col("r:band").cast(pl.String))

    off_var_window_df = off_var_window_df.join(
        band_baselines, on=["r:diaObjectId", "r:band"], how="inner"
    )

    off_var_window_df = off_var_window_df.with_columns(
        ((pl.col("r:psfFlux") - pl.col("baseline")) / pl.col("r:psfFluxErr"))
        .pow(2)
        .alias("chi2_term")
    )

    result = (
        off_var_window_df.group_by("r:diaObjectId")
        .agg(
            [
                pl.col("chi2_term").sum().alias("chi2_sum"),
                pl.col("chi2_term").count().alias("n_points"),
            ]
        )
        .with_columns((pl.col("chi2_sum") / pl.col("n_points")).alias("chi2_off"))
    )
    result.write_csv(f"{output_path}.csv")
    return result


def off_var_chi_vs_i_flux_plot(
    bazin_fits_data_path: Path,
    off_var_chi_squared_result: pl.DataFrame,
    light_curve_data_path: Path,
) -> None:
    """Here we plot the off-variation chi squared score against
    the value of the peak flux in the i band for the object. This will
    help us make the empirical cut for objects that are SN-like and
    not SN-like."""

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
            ["red" if obj_id in SN_LIKE_OBJECTS else "blue" for obj_id in labels],
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


if __name__ == "__main__":
    # print(variability_window(BAZIN_FITS_DATA_DIR / "20_object_test" / "bazin_fits_i_band_results.csv", LIGHT_CURVE_DATA_DIR / "combined_light_curve_data.parquet"))
    # print(off_var_chi_squared(BAZIN_FITS_DATA_DIR / "20_object_test" / "bazin_fits_i_band_results.csv", LIGHT_CURVE_DATA_DIR / "combined_light_curve_data.parquet"))
    BAZIN_TEST_DIR = BAZIN_FITS_DATA_DIR / "20_object_test"

    bazin_fits_paths = {
        "u": BAZIN_TEST_DIR / "bazin_fits_u_band_results.csv",
        "g": BAZIN_TEST_DIR / "bazin_fits_g_band_results.csv",
        "r": BAZIN_TEST_DIR / "bazin_fits_r_band_results.csv",
        "i": BAZIN_TEST_DIR / "bazin_fits_i_band_results.csv",
        "z": BAZIN_TEST_DIR / "bazin_fits_z_band_results.csv",
        "y": BAZIN_TEST_DIR / "bazin_fits_y_band_results.csv",
    }

    result = off_var_chi_squared(
        i_band_bazin_fits_path=bazin_fits_paths["i"],
        bazin_fits_paths=bazin_fits_paths,
        light_curve_data_path=(
            LIGHT_CURVE_DATA_DIR / "combined_light_curve_data.parquet"
        ),
        output_path=(OFF_VAR_CHI2_CSV_OUTPUT_DIR / "20_object_test"),
    )

    off_var_chi_vs_i_flux_plot(
        bazin_fits_paths["i"],
        result,
        LIGHT_CURVE_DATA_DIR / "combined_light_curve_data.parquet",
    )
    # plotting_off_var_data(BAZIN_FITS_DATA_DIR / "100_object_test" / "bazin_fits_i_band_results.csv", LIGHT_CURVE_DATA_DIR / "combined_light_curve_data.parquet", VARIABILITY_WINDOW_PLOT_OUTPUT_DIR / "100_object_test")
