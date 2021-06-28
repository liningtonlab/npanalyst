import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd

from npanalyst import exceptions
from npanalyst.logging import get_logger

logger = get_logger()

# TODO: Cleanup this note
"""
To keep the data structure the same, we will try to mimic our data format when handling mzMl data: 
"BasketID","Frequency","PrecMz","PrecIntensity","RetTime","SampleList","ACTIVITY_SCORE","CLUSTER_SCORE"

MZmine data sets contain peak area, which is ignored. We do not have BasketID, Frequency, or 
PrecIntensity, Activity and Cluster scores either.

This converter takes an mzmine file and converts it to into the following table format:
"PrecMz", "RetTime", "SampleList".

Note: SampleList is a list

REQUIRED: an mzmine file in CSV format
OUTPUT: pandas dataframe with "PrecMz", "RetTime", "SampleList" column names
FUTURE: will allow mzTab, XML, SQL and MetaboAnalyst file formats as well
"""


def fix_long_dtype(fpath: Path) -> tempfile.TemporaryFile:
    """Patch to resolve networkX versions which do not support `long` as an integer datatype (networkx 2.5 I think)"""
    temp_f = tempfile.TemporaryFile()
    with open(fpath, encoding="utf-8") as f:
        for l in f.readlines():
            temp_f.write(l.replace('attr.type="long"', 'attr.type="int"').encode())
    temp_f.seek(0)
    return temp_f


def gnps(input_file: Path) -> pd.DataFrame:
    """Convert the GNPS molecular network to a list of basketed features
    with the same columns as the `basketed.csv` output from the mzML pipeline.
    """
    # check for proper extension
    if not input_file.suffix == ".graphml":
        raise exceptions.InvalidFormatError(
            "Only graphml files supported for gnps conversions."
        )

    baskets = []
    # samples should be a string of files separated by |
    create_row = lambda precmz, rt, inten, samples: {
        "PrecMz": float(precmz),
        "RetTime": float(rt),
        "PrecIntensity": float(inten),
        "MaxPrecIntensity": float(inten),
        "MinPrecIntensity": float(inten),
        "UniqueFiles": samples,
    }
    try:
        G = nx.read_graphml(input_file)
    except KeyError:
        temp_f = fix_long_dtype(input_file)
        G = nx.read_graphml(temp_f)
        temp_f.close()

    for _, ndata in G.nodes(data=True):
        logger.debug(ndata)
        # TODO: determine if this is consistent for all GNPS networks
        # may need a try except or more robust solution
        # NOTE: I have manually tested one classic MN and one FBMN input
        inten = ndata.get("sum(precursor intensity)")
        baskets.append(
            create_row(
                ndata.get("precursor mass"),
                ndata.get("RTMean"),
                # Mean, Max, Min Itensity all the same from GNPS -> 0.0
                inten,
                ndata.get("UniqueFileSources"),
            )
        )
    return pd.DataFrame(baskets)


def mzmine(input_file: Path) -> pd.DataFrame:
    pass
    # # check for proper extension
    # try:
    #     if not (data_path.suffix == ".csv"):
    #         raise exceptions.InvalidFormatError(
    #             "Only CSV supported for mzmine conversions, for now."
    #         )
    #     else:
    #         data_file = pd.read_csv(data_path)
    # except:
    #     print("Using Recchia format")
    #     data_file = data_path

    # # load activity file
    # activity_file = pd.read_csv(act_path)

    # # determine the sample column and save samples
    # try:
    #     if configd["FILENAMECOL"] in activity_file.keys():
    #         actSamples = set(activity_file[configd["FILENAMECOL"]])
    #     else:
    #         actSamples = set(activity_file.iloc[:, 0])
    # except:
    #     actSamples = set(activity_file.iloc[:, 0])

    # logging.debug(f"Found {len(actSamples)} unique samples")

    # basketSamples = data_file.columns.tolist()
    # basketList = []
    # found = False

    # # map basket sample names to activity samples - in order
    # for basketSample in basketSamples:
    #     for actSample in actSamples:
    #         if actSample in basketSample:
    #             found = True
    #             basketList.append(actSample)
    #             break
    #     if not found:
    #         if re.search("row", basketSample):
    #             basketList.append("ADD")
    #         else:  # if it contains the word "row" probably not a sample
    #             basketList.append("NA")
    #             print("Could not find", basketSample, "in activity file!")

    # if not found:
    #     print("This is not proper mzml format.")
    #     print("Require two columns with the names: 'row m/z' and 'row retention time'")
    #     sys.exit()

    # newTable = pd.DataFrame()
    # print("Prepared the new table")

    # for row in data_file.itertuples(index=False):  # pass through each row
    #     currSamples = set()
    #     currRow = []
    #     values = []
    #     for col in range(0, data_file.shape[1]):  # pass through each column
    #         try:
    #             if float(row[col]) > 0:
    #                 if basketList[col] == "ADD":
    #                     currRow.append(row[col])  # PrecMz and RetTime
    #                 elif basketList[col] != "NA":
    #                     currSamples.add(basketList[col])
    #                     values.append(row[col])
    #         except:
    #             pass

    #     if len(currSamples) > 1:
    #         avg_value = sum(values) / len(values)
    #         currRow.append(avg_value)  # PrecIntensity is average
    #         currRow.append("|".join(currSamples))  # SampleList
    #         currRow.append(len(currSamples))  # frequency
    #         # change from row-wise to column-wise
    #         dfRow = pd.DataFrame(currRow).transpose()
    #         newTable = newTable.append(dfRow)

    # newTable.columns = ["PrecMz", "RetTime", "PrecIntensity", "Sample", "freq"]

    # newTable.to_csv(
    #     configd["OUTPUTDIR"].joinpath("basketed.csv"), index=False, quoting=None
    # )

    # print("Saving basketed file, as basketed.csv")
