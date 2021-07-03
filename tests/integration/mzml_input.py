import os
import tempfile
from pathlib import Path
from zipfile import ZipFile
import shutil
import pandas as pd
import networkx as nx
import time

from npanalyst import configuration, cli

from pandas._testing import assert_frame_equal


def graphml_assertion(reference_path, test_path):
    """This function converts the graphml files into a dataframe
    with all the attributes. These dataframes are compared."""
    G = nx.read_graphml(Path(reference_path))
    attr_list = []
    for node in G.nodes(data=True):
        attr_list.append(node[1])

    reference_graphml_df = pd.DataFrame(attr_list)
    # Sort index by "x" position and also sort columns
    # This is only true for the community graphml files
    # TODO: There seems to be random effect present, when the community graphs are produced.
    #  The graphml files are per se identical, but the order changes.
    reference_graphml_df.sort_values(by=["x"], inplace=True, ignore_index=True)
    reference_graphml_df = reference_graphml_df.reindex(sorted(reference_graphml_df.columns), axis=1)

    G = nx.read_graphml(Path(test_path))
    attr_list = []
    for node in G.nodes(data=True):
        attr_list.append(node[1])

    test_graphml_df = pd.DataFrame(attr_list)
    test_graphml_df.sort_values(by=["x"], inplace=True, ignore_index=True)
    test_graphml_df = test_graphml_df.reindex(sorted(test_graphml_df.columns), axis=1)

    assert_frame_equal(reference_graphml_df, test_graphml_df)


def dataframe_assertion(reference_path, test_path):
    """This function reads the respective dataframe and compares
    the two files."""
    result_table = pd.read_csv(reference_path)
    test_table = pd.read_csv(Path(test_path))

    assert_frame_equal(result_table, test_table)


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
OUTPUT_GRAPHML = HERE / "data/network_mzml.graphml"

# Feature table with assigned activity and cluster score
OUTPUT_TABLE = HERE / "data/table_mzml.csv"

# Community-related outputs: graphml, table output and assay table for heatmap generation
OUTPUT_COMMUNITY = HERE / "data/communities"


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
                      verbose=False,
                      config=None)

    # # Test length of generated replicated files (=925)
    length = len(os.listdir(Path(tmpdir, "replicated")))
    assert length == 925

    # # Get replicated zip output file with expected output files and extract them
    with ZipFile(Path(OUTPUT_FILE_REPLICATED), 'r') as zip:
        zip.extractall(Path(tmpdir, "expected_replicated_results"))

    # # Catch all csv files from the expected results
    replicate_file_names = os.listdir(Path(tmpdir, "expected_replicated_results"))

    # # Compare the expected replicated files with the produced files
    for rep in replicate_file_names:
        dataframe_assertion(reference_path=Path(tmpdir, "expected_replicated_results", rep),
                            test_path=Path(tmpdir, "replicated", rep))



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
                      verbose=False,
                      config=None)

    # # Compare the expected basketed file with the produced file
    dataframe_assertion(reference_path=Path(OUTPUT_FILE_BASKETED),
                        test_path=Path(tmpdir, "basketed.csv"))

    shutil.rmtree(tmpdir, ignore_errors=True)


def mzml_bioactivity_mapping():
    # # Create temporary folder for result and test files
    tmpdir = tempfile.mkdtemp()

    cli.run_activity(input_path=OUTPUT_FILE_BASKETED,
                     output_path=Path(tmpdir),
                     activity_path=INPUT_FILE_ACTIVITY_FILE,
                     verbose=False,
                     include_web_output=False,
                     config=None)

    # # Compare table output of the 9834 features
    dataframe_assertion(reference_path=Path(OUTPUT_TABLE),
                        test_path=Path(tmpdir, "table.csv"))

    # Validate graphml x y coordinates, activity / cluster score, name, type_, etc.
    graphml_assertion(reference_path=Path(OUTPUT_GRAPHML),
                      test_path=Path(tmpdir, "network.graphml"))

    # # Compare communities output folder
    # Count communities
    nr_communities = len(os.listdir(Path(tmpdir, "communities")))
    assert nr_communities == 17

    # # Go through all community folders and compare

    for community in [str(i) for i in range(17)]:
        print("Validate community nr:" + community)

        # tables
        dataframe_assertion(reference_path=Path(OUTPUT_COMMUNITY, community, "table.csv"),
                            test_path=Path(tmpdir, "communities", community, "table.csv"))

        # assay data
        dataframe_assertion(reference_path=Path(OUTPUT_COMMUNITY, community, "assay.csv"),
                            test_path=Path(tmpdir, "communities", community, "assay.csv"))

        # graphml files

        graphml_assertion(reference_path=Path(OUTPUT_COMMUNITY, community, "network.graphml"),
                          test_path=Path(tmpdir, "communities", community, "network.graphml"))

    # Remove temp folder
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':

    start = time.time()

    config_parameter()

    mzml_replicate_comparison()

    mzml_basket_building()

    mzml_bioactivity_mapping()

    print("This testing took: " + str(round((time.time() - start) / 60, 2)) + " minutes.")









