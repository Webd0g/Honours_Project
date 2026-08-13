"""Here we will fit the objects with a constant and determine that object's reduced chi square with the constant fit to the i band data. 
Then the bazin fit reduced chi squared in the i band will be subtracted from that constant fit. The values obtained for each object from that subtraction
will then be scatter plotted as a function of the reduced chi squared score of the Bazin fit in the i band.

This is the most significant cut that can be done for photmetric selection and shoudl hopefully result in SN-like obejcts being greatly separated from 
objects that are more flat than SN-like."""

import polars as pl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


SN_LIKE_OBJECTS = ["170019716323082348",
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
"313871014435488133"]

NOT_SN_LIKE_OBJECTS = ["170019717135204495",
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
"170028511067636654"]

def constant_fit(flux: np.ndarray, flux_err: np.ndarray)-> tuple:
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
    chi2 = np.sum(
        ((flux - constant) / flux_err) ** 2
    )

    # One fitted parameter: constant
    dof = len(flux) - 1

    reduced_chi2 = chi2 / dof

    return constant, reduced_chi2

def constant_batch_fits(object_path: Path, light_curve_data_path: Path, band: str) -> pl.DataFrame:

    objects = (pl.read_csv(object_path, schema_overrides={"r:diaObjectId": pl.String}, columns=["r:diaObjectId"],)["r:diaObjectId"].to_list())

    light_curve_data = (pl.read_parquet(light_curve_data_path).with_columns(pl.col("r:diaObjectId").cast(pl.String)))

    light_curve_data = light_curve_data.filter(pl.col("r:diaObjectId").is_in(objects))

    results = {"r:diaObjectId": [],"constant": [],"reduced_chi2": []}

    groups = {key[0]: group_df for key, group_df in light_curve_data.group_by("r:diaObjectId",maintain_order=True)}

    for object_id in objects:

        group_df = groups.get(object_id)

        if group_df is None:
            continue

        band_df = (group_df.filter((pl.col("r:band") == band) & (pl.col("r:psfFlux") > 0)).sort("r:midpointMjdTai"))

        if len(band_df) < 2:
            continue

        flux = (band_df["r:psfFlux"].to_numpy().astype(np.float64))

        flux_err = (band_df["r:psfFluxErr"].to_numpy().astype(np.float64))

        # Remove invalid uncertainties
        valid = (np.isfinite(flux)& np.isfinite(flux_err)& (flux_err > 0))

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

def compute_delta_chi(object_path: Path, light_curve_data_path: Path, band: str, bazin_fit_i_band_data_path: Path, output_path: Path) -> pl.DataFrame:
    """Here we compute the difference between the constant fit and the Bazin
    fit and append it as a column to the df. """

    # Fit with a constant for the objects
    constant_fit_result = constant_batch_fits(object_path, light_curve_data_path, band)

    # Renaming the columns for the constant chi squared
    constant_fit_result = constant_fit_result.rename({"reduced_chi2": "constant_reduced_chi2"})

    bazin_fit_chi_squared = pl.read_csv(bazin_fit_i_band_data_path).with_columns(pl.col("r:diaObjectId").cast(pl.String))

    # Renaming the column for the Bazin chi squared 
    bazin_fit_chi_squared = bazin_fit_chi_squared.rename({"reduced_chi2": "bazin_reduced_chi2"})

    # Joining the two data sources into one df making it easy to compare the columns of the data and add a new column for the delta
    delta_chi_df = (constant_fit_result.join(bazin_fit_chi_squared, on="r:diaObjectId", how="inner", suffix="bazin").with_columns((pl.col("constant_reduced_chi2") - pl.col("bazin_reduced_chi2")).alias("delta_reduced_chi2")))

    delta_chi_df.write_csv(output_path)

    return delta_chi_df

def delta_chi_agaisnt_chi_SN_plot(path_to_delta_chi_results: Path, output_path: Path) -> None:
    """Here we plot the difference between the fit by a constant and the shape fit as a function of the shape fit in the i band."""

    delta_chi_df = pl.read_csv(path_to_delta_chi_results)

    labels = delta_chi_df.select("r:diaObjectId").to_series()

    colors = ["red" if str(obj_id) in SN_LIKE_OBJECTS else "blue" for obj_id in labels]
    plt.scatter(delta_chi_df["bazin_reduced_chi2"], delta_chi_df["delta_reduced_chi2"], c=colors, alpha=0.7)

    plt.legend(title="r:diaObjectId", handles=[plt.Line2D([0], [0], marker='o', color='red', markersize=10, linestyle='None'),
                                            plt.Line2D([0], [0], marker='o', color='blue', markersize=10, linestyle='None')],
            labels=['SN-like', 'Non-SN-like'])

    plt.xscale("log")
    plt.yscale("log")

    ax = plt.gca()

    lower = min(ax.get_xlim()[0], ax.get_ylim()[0])

    upper = max(ax.get_xlim()[1], ax.get_ylim()[1])

    ax.plot([lower, upper], [lower, upper], "r--", label="unity")
    plt.title("Delta chi squared as a function of bazin chi squared in the i band")
    plt.xlabel("Chi squared of the bazin fit in the i band")
    plt.ylabel("delta chi squared in the i band")
    plt.savefig(output_path)
    plt.show()

def delta_chi_check(path_to_delta_chi_results: Path):
    """Here we separate the objects that are above the unity line from those that are below the unity line."""

    delta_chi_df = pl.read_csv(path_to_delta_chi_results)

    delta_chi_df = delta_chi_df.with_columns((pl.col("delta_reduced_chi2") > pl.col("bazin_reduced_chi2")).alias("passes_delta_chi_cut"))

    delta_chi_df = delta_chi_df.with_columns((pl.col("delta_reduced_chi2") <= pl.col("bazin_reduced_chi2")).alias("failed_delta_chi_cut"))

    return delta_chi_df

def save_pass_fail_objects(path_to_delta_chi_results: Path, output_path_pass: Path, output_path_fail: Path):
    """Here we write the results of which objects passed and which objects failed the to their respective CSV files."""

    delta_chi_df = delta_chi_check(path_to_delta_chi_results)

    delta_chi_df_pass = delta_chi_df.filter(pl.col("passes_delta_chi_cut") == "true")

    delta_chi_df_fail = delta_chi_df.filter(pl.col("failed_delta_chi_cut") == "true")

    delta_chi_df_pass.write_csv(output_path_pass)

    delta_chi_df_fail.write_csv(output_path_fail)

if __name__ == "__main__":

    TEST_DATA = Path.home() / "Documents"/"University"/"Honours"/"Honours_Project"/"data"/"test_data"/"20_test_objects.csv"

    LIGHT_CURVE_DATA = Path.home() / "Documents"/"University"/"Honours"/"Honours_Project"/"data"/ "light_curve_data" / "combined_light_curve_data.parquet"

    TEST_OUTPUT_PATH = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "constant_minus_shape_results" / "20_object_test.csv"

    BAZIN_FITS_I_BAND_PATH = Path.home() / "Documents" / "University" / "Honours" / "Honours_Project" / "data" / "bazin_fits_results" / "20_object_test" / "bazin_fits_i_band_results.csv"

    delta_chi_agaisnt_chi_SN_plot("/home/riley/Documents/University/Honours/Honours_Project/data/constant_minus_shape_results/20_object_test/20_object_test.csv",
                                "/home/riley/Documents/University/Honours/Honours_Project/Figure.png")

    compute_delta_chi(TEST_DATA, LIGHT_CURVE_DATA, "i", BAZIN_FITS_I_BAND_PATH, "/home/riley/Documents/University/Honours/Honours_Project/data/constant_minus_shape_results/20_object_test/20_object_test.csv")

    save_pass_fail_objects("/home/riley/Documents/University/Honours/Honours_Project/data/constant_minus_shape_results/20_object_test/20_object_test.csv", 
                        "/home/riley/Documents/University/Honours/Honours_Project/data/constant_minus_shape_results/20_object_test/20_object_test_pass.csv",
                        "/home/riley/Documents/University/Honours/Honours_Project/data/constant_minus_shape_results/20_object_test/20_object_test_fail.csv")