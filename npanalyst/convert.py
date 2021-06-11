import pandas as pd
import numpy as np
import regex as re

from typing import Union, List, Dict
from pathlib import Path
import sys
import argparse
import logging

from regex.regex import split

from . import exceptions

pd.set_option("precision", 15)

PATH = Union[Path, str]

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


def mzmine(act_path, data_path, configd):
    # check for proper extension
    try:
        if not (data_path.suffix == ".csv"):
            raise exceptions.InvalidFormatError(
                "Only CSV supported for mzmine conversions, for now."
            )
        else:
            data_file = pd.read_csv(data_path)
    except:
        print("Using Recchia format")
        data_file = data_path

    # load activity file
    activity_file = pd.read_csv(act_path)

    # determine the sample column and save samples
    try:
        if configd["FILENAMECOL"] in activity_file.keys():
            actSamples = set(activity_file[configd["FILENAMECOL"]])
        else:
            actSamples = set(activity_file.iloc[:, 0])
    except:
        actSamples = set(activity_file.iloc[:, 0])

    print(f"Found {len(actSamples)} unique samples")

    basketSamples = data_file.columns.tolist()
    basketList = []
    found = False

    # map basket sample names to activity samples - in order
    for basketSample in basketSamples:
        for actSample in actSamples:
            if actSample in basketSample:
                found = True
                basketList.append(actSample)
                break
        if not found:
            if re.search("row", basketSample):
                basketList.append("ADD")
            else:  # if it contains the word "row" probably not a sample
                basketList.append("NA")
                print("Could not find", basketSample, "in activity file!")

    if not found:
        print("This is not proper mzml format.")
        print("Require two columns with the names: 'row m/z' and 'row retention time'")
        sys.exit()

    newTable = pd.DataFrame()
    print("Prepared the new table")

    for row in data_file.itertuples(index=False):  # pass through each row
        currSamples = set()
        currRow = []
        values = []
        for col in range(0, data_file.shape[1]):  # pass through each column
            try:
                if float(row[col]) > 0:
                    if basketList[col] == "ADD":
                        currRow.append(row[col])  # PrecMz and RetTime
                    elif basketList[col] != "NA":
                        currSamples.add(basketList[col])
                        values.append(row[col])
            except:
                pass

        if len(currSamples) > 1:
            avg_value = sum(values) / len(values)
            currRow.append(avg_value)  # PrecIntensity is average
            currRow.append("|".join(currSamples))  # SampleList
            currRow.append(len(currSamples))  # frequency
            # change from row-wise to column-wise
            dfRow = pd.DataFrame(currRow).transpose()
            newTable = newTable.append(dfRow)

    newTable.columns = ["PrecMz", "RetTime", "PrecIntensity", "Sample", "freq"]

    newTable.to_csv(
        configd["OUTPUTDIR"].joinpath("basketed.csv"), index=False, quoting=None
    )

    print("Saving basketed file, as basketed.csv")


def gnps(act_path, data_path, configd):
    # check for proper extension
    if not (data_path.suffix == ".graphml"):
        print("Only graphml files supported for gnps conversions.")
        sys.exit()

    # load activity file
    activity_file = pd.read_csv(act_path)

    # determine the sample column and save samples
    try:
        if configd["FILENAMECOL"] in activity_file.keys():
            actSamples = set(activity_file[configd["FILENAMECOL"]])
    except:
        actSamples = set(activity_file.iloc[:, 0])

    print(f"Found {len(actSamples)} unique samples")

    newTable = pd.DataFrame()

    # output = {"PrecMz": None,"RetTime": None,"PrecIntensity": None,"Sample": list(),"freq": 0}
    keys = dict()
    currRow = []
    currValue = ""
    currSamples = set()

    with open(data_path, "r") as f:
        for line in f:
            # print("curr line", line)
            if re.search("<node", line):  # we started a new sample
                if currRow:
                    currRow.append(str("|".join(currSamples)))  # add samples
                    currRow.append(len(currSamples))  # add frequency
                    if len(currSamples) > 1:
                        dfRow = pd.DataFrame(currRow).transpose()
                        newTable = newTable.append(dfRow)
                currRow = []
                currSamples = set()
                next

            if re.search("<key", line):  # figure out the sample header structure
                name = re.sub("['/<>?\"]", "", line)
                id = name.split("id=")[-1].strip()

                if re.search("precursor mass", name):
                    keys[id] = "PrecMz"
                elif re.search("RTMean", name):
                    keys[id] = "RTMean"
                elif re.search("precursor intensity", name):
                    keys[id] = "PrecIntensity"
                elif re.search("UniqueFileSources", name):
                    keys[id] = "Sample"

            elif re.search("<data", line):  # gather data values
                currValue = re.findall(">(.*)<", line)
                currKey = re.findall('<data key="(.*?)">', line)

                for key in keys:
                    if currKey[0] == key:
                        if (
                            keys[key] == configd["FILENAMECOL"]
                        ):  # we found the sample key
                            for i in currValue[0].split(
                                "|"
                            ):  # check each sample against activity file
                                found = False
                                for actSample in actSamples:
                                    if actSample in i:
                                        found = True
                                        currSamples.add(actSample)
                                if not found:
                                    print("No match in Basketed file for", i)
                            # currRow.append(str("|".join(currSamples)))      # add samples
                            # currRow.append(len(currSamples))        # add frequency
                            # if(len(currSamples) > 1):
                            #     dfRow = pd.DataFrame(currRow).transpose()
                            #     newTable = newTable.append(dfRow)
                        else:
                            currRow.append(currValue[0])

    newTable.columns = ["PrecIntensity", "PrecMz", "RetTime", "Sample", "freq"]
    newTable = newTable[["PrecMz", "RetTime", "PrecIntensity", "Sample", "freq"]]
    newTable.to_csv(
        configd["OUTPUTDIR"].joinpath("basketed.csv"), index=False, quoting=None
    )
    print("Saving file to basketed.csv")


def recchiaFormat(data_file):

    splitColumns = []

    for row in data_file.index:  # pass through each row
        m = re.findall("([0-9]+\.[0-9]+)_([0-9]+\.[0-9]+)", row)
        if m:
            rettime = m[0][0]
            precmz = m[0][1]
            new_row = {"row m/z": precmz, "row retention time": rettime}
            splitColumns.append(new_row)
        else:
            pass

    splitTable = pd.DataFrame(splitColumns, columns=["row m/z", "row retention time"])

    col_names = list(splitTable.columns) + list(data_file.columns)
    df_merged = pd.concat(
        [splitTable.reset_index(drop=True), data_file.reset_index(drop=True)],
        axis=1,
        ignore_index=True,
    )
    df_merged.columns = col_names

    return df_merged


def default(act_path, data_path, configd):

    transpose = False
    progene = False

    # check for proper extension
    try:
        if not (data_path.suffix == ".csv" and act_path.suffix == ".csv"):
            print("Only CSV supported for general conversions, for now.")
            sys.exit()
    except:
        pass

    # load data and activity file
    try:
        data_file = pd.read_csv(data_path)
        # print(data_file.head())
    except:
        print("Error: could not load data file ", str(data_path))
        sys.exit()

    counter = 0
    # check to see if the header row contains the (#.#)_(#.#) pattern where /1 is the retention time and /2 is the m/z value
    for col in data_file.columns:
        try:
            counter += 1
            m = re.findall("([0-9]+\.[0-9]+)_([0-9]+\.[0-9]+)", col)
            if m:
                data_file = pd.DataFrame(data_file).set_index("Compound").transpose()
                transpose = True
                break
            elif col == "Raw abundance":
                # print("Discovered Progensis - only Raw abundance column calculated")
                progene = counter
                break
        except:
            print("Could not run check on", col)

    if transpose:
        df_merged = recchiaFormat(data_file)
        mzmine(act_path, df_merged, configd)

    elif progene:
        print("Found Progenesis data format.")
        data_file = pd.read_csv(data_path, skiprows=2).set_index("Compound")
        data_file = data_file.iloc[:, progene:]

        endCol = 0
        samples = []
        # skip columns until "Raw abundance"
        for col in data_file.columns:
            splitColumn = col.split("_")
            if len(splitColumn) == 2:
                endCol += 1
                samples.append(col.split("_")[1])
            else:
                # omit all columns after this
                break

        data_file = data_file.iloc[:, :endCol]
        data_file.columns = samples
        df_merged = recchiaFormat(data_file)
        mzmine(act_path, df_merged, configd)

    else:
        print("Data file not in Recchia format")
        mzmine(act_path, data_path, configd)
