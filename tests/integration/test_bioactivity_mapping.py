import os
import tempfile
from pathlib import Path
import shutil
import pandas as pd
import networkx as nx
import time

from npanalyst import configuration, cli

from pandas._testing import assert_frame_equal


# # Helper functions
def graphml_assertion(reference_path, test_path):
    """This function converts the graphml files into a dataframe
    with all the attributes. These dataframes are compared."""
    G = nx.read_graphml(Path(reference_path))
    attr_list = []
    for node in G.nodes(data=True):
        attr_list.append(node[1])

    reference_graphml_df = pd.DataFrame(attr_list)
    # Remove x and y position before comparing dataframes
    reference_graphml_df = reference_graphml_df.drop(["x", "y"], axis=1)
    reference_graphml_df.sort_values(by=["MaxPrecIntensity"], inplace=True, ignore_index=True)
    reference_graphml_df = reference_graphml_df.reindex(sorted(reference_graphml_df.columns), axis=1)

    G = nx.read_graphml(Path(test_path))
    attr_list = []
    for node in G.nodes(data=True):
        attr_list.append(node[1])

    test_graphml_df = pd.DataFrame(attr_list)
    # Remove x and y position before comparing dataframes
    test_graphml_df = test_graphml_df.drop(["x", "y"], axis=1)
    test_graphml_df.sort_values(by=["MaxPrecIntensity"], inplace=True, ignore_index=True)
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

# Bioactivity readout table
INPUT_FILE_ACTIVITY_FILE = HERE / "data/Activity.csv"

# Basketed CSV file output
OUTPUT_FILE_BASKETED = HERE / "data/basketed_mzml.csv"

# Activity association network graphml
OUTPUT_GRAPHML = HERE / "data/network_mzml.graphml"

# Feature table with assigned activity and cluster score
OUTPUT_TABLE = HERE / "data/table_mzml.csv"

# Community-related outputs: graphml, table output and assay table for heatmap generation
OUTPUT_COMMUNITY = HERE / "data/communities"


def test_config_parameter():
    """This test shall guarantee that the loaded settings are identical to those, used to
    obtain the reference/ground truth results."""
    configd = configuration.load_config(config_path=None)

    assert configd["ACTIVITYTHRESHOLD"] == 2
    assert configd["CLUSTERTHRESHOLD"] == 0.3


def test_bioactivity_mapping():
    """Test for bioactivity mapping function. A pre-built basketed.csv file is used as the input file.
    The network.graphml, the table.csv (features table) and the individual community-related files are
    compared through a dataframe-by-dataframe analysis."""

    # # Create temporary folder for result and test files
    tmpdir = tempfile.mkdtemp()

    # # Run activity readout mapper function
    cli.run_activity(input_path=Path(OUTPUT_FILE_BASKETED),
                     output_path=Path(tmpdir),
                     activity_path=INPUT_FILE_ACTIVITY_FILE,
                     verbose=False,
                     include_web_output=False,
                     config=None)

    # # Compare table output of the 9834 features
    dataframe_assertion(reference_path=Path(OUTPUT_TABLE),
                        test_path=Path(tmpdir, "table.csv"))

    # # Validate graphml x y coordinates, activity / cluster score, name, type_, etc.
    graphml_assertion(reference_path=Path(OUTPUT_GRAPHML),
                      test_path=Path(tmpdir, "network.graphml"))

    # # Compare communities output folder
    # # Count communities
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

    # # Remove temp folder
    shutil.rmtree(tmpdir, ignore_errors=True)


# if __name__ == '__main__':
#
#     start = time.time()
#
#     test_config_parameter()
#
#     test_bioactivity_mapping()
#
#     print("This testing took: " + str(round((time.time() - start) / 60, 2)) + " minutes.")