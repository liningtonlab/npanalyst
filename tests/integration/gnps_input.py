import tempfile
from pathlib import Path
import shutil
import pandas as pd
import time
from npanalyst import cli

from pandas._testing import assert_frame_equal


# Helper function
def dataframe_assertion(reference_path, test_path):
    """This function reads the respective dataframe and compares
    the two files."""
    result_table = pd.read_csv(reference_path)
    # # This resorting is just a safe-guard to assure that rows are ordered properly and error messages are
    # # due to wrong values, not due to interchanged rows
    result_table.sort_values(by=["UniqueFiles", "PrecMz", "RetTime"], ignore_index=True, inplace=True)

    test_table = pd.read_csv(Path(test_path))
    test_table.sort_values(by=["UniqueFiles", "PrecMz", "RetTime"], ignore_index=True, inplace=True)

    assert_frame_equal(result_table, test_table)


# # Define relative path to input files
HERE = Path(__file__).parent

# GNPS graphml file that is converted into the basketed format used by NPAnalyst.
INPUT_GNPS_FILE ="data/GNPS_input.graphml"

# Basketed CSV file output
OUTPUT_FILE_BASKETED = HERE / "data/GNPS_expected_basketed.csv"


def gnps_import():
    """Test the import function for GNPS graphml files. An exhaustive dataframe-by-dataframe comparison
    is performed."""

    # # Create a temporary folder
    tmpdir = tempfile.mkdtemp()

    # # Import GNPS file and convert it into the basketed.csv file format,
    # # saving it in the temporary folder
    cli.run_import(
        input_path=Path(INPUT_GNPS_FILE),
        output_path=Path(tmpdir),
        mstype="gnps",
        verbose=False)

    # # Compare the expected basketed file with the produced file
    dataframe_assertion(reference_path=Path(OUTPUT_FILE_BASKETED),
                        test_path=Path(tmpdir, "basketed.csv"))

    # # Remove the temp folder
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':

    start = time.time()

    gnps_import()

    print("This testing took: " + str(round((time.time() - start) / 60, 2)) + " minutes.")