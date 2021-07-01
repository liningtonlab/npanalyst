import os
import tempfile
from pathlib import Path
from zipfile import ZipFile
import shutil
import pandas as pd

from npanalyst import configuration, core, cli
import pprint

from pandas._testing import assert_frame_equal


# # Define relative path to input files
HERE = Path(__file__).parent

# mzML files
INPUT_FILE_MZMLS = HERE / "data/BioMAP_mzml_input.zip"

# Bioactivity readout table
INPUT_FILE_ACTIVITY_FILE = HERE / "data/Activity.csv"

# Replicate compared basketed CSVs
OUTPUT_FILE_REPLICATED = HERE / "data/replicated_mzml_result.zip"

# Basketed CSV file output
OUTPUT_FILE_BASKETED = HERE / "data/basketed_mzml.csv"

# Activity association network graphml
OUTPUT_GRAPHML = HERE / "data/network.graphml"

# Community-related outputs: graphml, table output and assay table for heatmap generation
OUTPUT_COMMUNITY = HERE / "data/community"


# INPUT_FILE_MZMLS = HERE / "data/small_test_input.zip"
# OUTPUT_FILE_REPLICATED = HERE / "data/small_test_result.zip"


# # Test config settings (ms parameter and AS and CS threshold)
def config_parameter():
    """This test shall guarantee that the loaded settings are identical to those, used to
    obtain the reference/ground truth results."""
    configd = configuration.load_config(config_path=None)

    assert configd["ACTIVITYTHRESHOLD"] == 2
    assert configd["CLUSTERTHRESHOLD"] == 0.3
    assert configd["MINREPS"] == 2
    assert configd["ERRORINFO"]["PrecMz"] == ('ppm', 30.0)
    assert configd["ERRORINFO"]["RetTime"] == ('window', 0.03)


def mzml_replicate_comparison():
    """Test for the replicate comparison step. The BioMAP mzML dataset is used to generate the
    replicate-compared csv files. A full dataframe by dataframe comparison is performed to ensure
    identical csv files."""
    # # Create temporary folder for result and test files
    tmpdir = tempfile.mkdtemp()

    # # Open and extract zip file that contains the 2775 mzML files
    with ZipFile(Path(INPUT_FILE_MZMLS), 'r') as zip:
        zip.extractall(Path(tmpdir, "mzml_files"))

    # # Perform replicate comparison
    # # Workers has to be set to 1, it would not run for more workers,
    # # returns error that some workers try to attempt calculations before all of them done.
    cli.run_replicate(input_path=Path(tmpdir, "mzml_files"),
                      output_path=Path(tmpdir),
                      workers=1,
                      verbose=True,
                      config=None)


    # # Test length of generated replicated files (=925)
    length = len(os.listdir(Path(tmpdir, "replicated")))
    assert length == 925

    # # Get replicated zip output file with expected output files and extract them
    with ZipFile(Path(OUTPUT_FILE_REPLICATED), 'r') as zip:
        zip.extractall(Path(tmpdir, "expected_replicated_results"))

    # # Catch all csv files from the expected results
    replicate_file_names = os.listdir(Path(tmpdir, "expected_replicated_results"))

    # # Define the columns that shall be compared. For now the UniqueFiles entry is randomly ordered
    # TODO: Either fix function that produces the replicated basketed files (most likely a set is used)
    #  or reorder before testing. But requires to stringsplit the entry.
    COLS = ['Unnamed: 0', 'PrecMz', 'RetTime', 'PrecIntensity', 'UniqueFiles']
    COLS_TO_MISS = {'UniqueFiles'}

    # # Compare the expected replicated files with the produced files
    for rep in replicate_file_names:
        result_file = pd.read_csv(Path(tmpdir,"expected_replicated_results", rep), usecols=list(set(COLS) - COLS_TO_MISS))
        test_file = pd.read_csv(Path(tmpdir, "replicated", rep), usecols=list(set(COLS) - COLS_TO_MISS))

        assert_frame_equal(result_file, test_file)

    # # Remove temporary folder. Windows would not delete all files.
    # # Python 3.11 seems to enable the ignore_errors function also for tempfile.TemporaryDirectory() which
    # # is the nicer context manager option.
    shutil.rmtree(tmpdir, ignore_errors=True)


def mzml_basket_building():
    # # Create temporary folder for result and test files
    tmpdir = tempfile.mkdtemp()

    # # Get replicated zip output file with expected output files and extract them
    with ZipFile(Path(OUTPUT_FILE_REPLICATED), 'r') as zip:
        zip.extractall(Path(tmpdir))

    cli.run_basketing(input_path=Path(tmpdir),
                      output_path=Path(tmpdir),
                      verbose=True,
                      config=None)

    COLS = ['PrecMz', 'RetTime', 'PrecIntensity', 'UniqueFiles']
    COLS_TO_MISS = {''}

    # # Compare the expected replicated files with the produced files
    result_file = pd.read_csv(OUTPUT_FILE_BASKETED, usecols=list(set(COLS) - COLS_TO_MISS))
    test_file = pd.read_csv(Path(tmpdir, "basketed.csv"), usecols=list(set(COLS) - COLS_TO_MISS))

    assert_frame_equal(result_file, test_file)


def mzml_bioactivity_mapping():
    # # Create temporary folder for result and test files
    tmpdir = tempfile.mkdtemp()

    cli.run_activity(input_path=OUTPUT_FILE_BASKETED,
                     output_path=tmpdir,
                     activity_path=INPUT_FILE_ACTIVITY_FILE,
                     verbose=True,
                     include_web_output=False,
                     config=None)

    # # Compare table output of the 9834 features

    # # Compare graphml file (connectivity,
    # Compare graphml x y coordinates, activity / cluster score, name, type_

    # # Compare communities output folder
    # Count communities

    # # Go through all files and compare

    # tables

    # assay data

    # graphml files



if __name__ == '__main__':

    mzml_replicate_comparison()








