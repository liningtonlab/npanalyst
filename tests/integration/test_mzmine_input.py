import tempfile
from pathlib import Path
import shutil
import pandas as pd
import time
import re

from npanalyst import cli

from pandas._testing import assert_frame_equal


# Helper function
def dataframe_assertion(reference_path, test_path):
    """This function reads the respective dataframe and compares
    the two files."""
    result_table = pd.read_csv(reference_path)
    result_table.sort_values(
        by=["UniqueFiles", "PrecMz", "RetTime"], ignore_index=True, inplace=True
    )

    test_table = pd.read_csv(Path(test_path))
    test_table.sort_values(
        by=["UniqueFiles", "PrecMz", "RetTime"], ignore_index=True, inplace=True
    )

    assert_frame_equal(result_table, test_table)


# # Define relative path to input files
HERE = Path(__file__).parent

# MZmine csv export dataframe that is converted into the basketed format used by NPAnalyst.
INPUT_MZMINE_FILE = HERE / "data/mzmine_input.csv"

# Basketed CSV file output
OUTPUT_FILE_BASKETED = HERE / "data/basketed_mzml.csv"


def test_mzmine_import():
    """Test the import function for MZmine csv files. An exhaustive dataframe-by-dataframe comparison
    is performed."""

    # # Create a temporary folder
    tmpdir = tempfile.mkdtemp()

    # # Import MZmine csv file and convert it into the basketed.csv file format,
    # # saving it in the temporary folder
    cli.run_import(
        input_path=Path(INPUT_MZMINE_FILE),
        output_path=Path(tmpdir),
        mstype="mzmine",
        verbose=False,
    )

    # # Prepare basketed_mzml.csv for comparison with basketed.csv

    b_df = pd.read_csv(Path(OUTPUT_FILE_BASKETED))

    # Rename lengthy file names from mzML output.
    b_df["UniqueFiles"] = b_df["UniqueFiles"].apply(
        lambda x: re.sub(r"_iDTs_cppis_..mzML", ".raw", x)
    )
    b_df["UniqueFiles"] = b_df["UniqueFiles"].apply(
        lambda x: "|".join(sorted(list(set(x.split("|")))))
    )

    # Exchange intensities for plain 1s. This is only important for the mock MZmine table
    # that was generated from a basketed file.
    b_df["PrecIntensity"] = [1.0 for n in range(b_df.shape[0])]
    b_df["MinPrecIntensity"] = [1.0 for n in range(b_df.shape[0])]
    b_df["MaxPrecIntensity"] = [1.0 for n in range(b_df.shape[0])]

    b_df.to_csv(Path(tmpdir, "exp_basketed.csv"), index=False)

    # # Compare the expected basketed file with the produced file
    dataframe_assertion(
        reference_path=Path(tmpdir, "exp_basketed.csv"),
        test_path=Path(tmpdir, "basketed.csv"),
    )

    # # Remove the temp folder
    shutil.rmtree(tmpdir, ignore_errors=True)


# if __name__ == '__main__':
#
#     start = time.time()
#
#     test_mzmine_import()
#
#     print("This testing took: " + str(round((time.time() - start) / 60, 2)) + " minutes.")
