import re
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
    """Convert the MZmine feature list to a list of basketed features
    with the same columns as the `basketed.csv` output from the mzML pipeline.
    """
    # check for proper extension
    if not input_file.suffix == ".csv":
        raise exceptions.InvalidFormatError(
            "Only CSV supported for mzmine conversions."
        )
    baskets = []
    # samples should be a string of files separated by |
    create_row = lambda precmz, rt, mean_inten, max_inten, min_inten, samples: {
        "PrecMz": float(precmz),
        "RetTime": float(rt),
        "PrecIntensity": float(mean_inten),
        "MaxPrecIntensity": float(max_inten),
        "MinPrecIntensity": float(min_inten),
        "UniqueFiles": samples,
    }
    df = pd.read_csv(input_file)
    # TODO: determine if this is consistent for MZmine inputs
    data_cols = ["row m/z", "row retention time", "row identity (main ID)"]
    sample_cols = [x for x in df.columns if x not in data_cols]
    # Unpivot MZmine input file, keeping index for groupby
    df1 = df.melt(id_vars=data_cols, value_vars=sample_cols, ignore_index=False)
    for _, group in df1.groupby(df1.index):
        group_presence = group[group["value"] > 0]
        mz = group_presence.iloc[0]["row m/z"]
        rt = group_presence.iloc[0]["row retention time"]
        mean_inten = group_presence["value"].mean()
        max_inten = group_presence["value"].max()
        min_inten = group_presence["value"].min()
        pa_pattern = re.compile(" peak area", re.IGNORECASE)
        samples = "|".join(
            sorted(pa_pattern.sub("", x) for x in group_presence["variable"].unique())
        )
        baskets.append(create_row(mz, rt, mean_inten, max_inten, min_inten, samples))
    return pd.DataFrame(baskets)
