import pandas as pd
import numpy as np
import regex as re

from typing import Union, List, Dict
from pathlib import Path
import sys
import argparse
import logging

pd.set_option('precision', 15)

PATH = Union[Path, str]

"""
    To keep the data structure the same, we will try to mimic our data format when handling mzMl data: 
    "BasketID","Frequency","PrecMz","PrecIntensity","RetTime","SampleList","ACTIVITY_SCORE","CLUSTER_SCORE"

    Mzmine data sets contain peak area, which is ignored. We do not have BasketID, Frequency, or 
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
    if not (data_path.suffix == ".csv"):
        print("Only CSV supported for mzmine conversions, for now.")
        sys.exit()

    # load data and activity file
    data_file = pd.read_csv(data_path)
    activity_file = pd.read_csv(act_path)

    # determine the sample column and save samples
    try:
        if configd["FILENAMECOL"] in activity_file.keys():
            actSamples = set(activity_file[configd["FILENAMECOL"]])
    finally:
        actSamples = set(activity_file.iloc[:,0])

    print (f"Found {len(actSamples)} unique samples")

    basketSamples = data_file.columns.tolist()
    basketList = []
    # map basket sample names to activity samples - in order
    for basketSample in basketSamples:
        found = False
        for actSample in actSamples:
            if actSample in basketSample:
                found = True
                basketList.append(actSample)
                break
        if not found:
            basketList.append("NA")
            if not re.search("row", basketSample):      # if it contains the word "row" probably not a sample
                print ("Could not find", basketSample, "in activity file!")

    newTable = pd.DataFrame()

    for row in data_file.itertuples(index=False):   # pass through each row
        currSamples = set()
        currRow = []
        for col in range(0, data_file.shape[1]):    # pass through each column
            if row[col] > 0:
                if basketList[col] != "NA":
                    currSamples.add(basketList[col])
                else:
                    currRow.append(row[col])    # PrecMz and RetTime
        currRow.append(None)            # PrecIntensity is empty
        currRow.append("|".join(currSamples))  # SampleList
        currRow.append(len(currSamples))  # frequency
        if(len(currSamples) > 1):
            dfRow = pd.DataFrame(currRow).transpose()   # change from row-wise to column-wise
            newTable = newTable.append(dfRow)

    newTable.columns = ["PrecMz", "RetTime", "PrecIntensity", "Sample", "freq"]

    newTable.to_csv("basketed.csv", index=False, quoting=None)

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
    finally:
        actSamples = set(activity_file.iloc[:,0])

    print (f"Found {len(actSamples)} unique samples")

    newTable = pd.DataFrame()

    #output = {"PrecMz": None,"RetTime": None,"PrecIntensity": None,"Sample": list(),"freq": 0}
    keys = dict()
    currRow = []
    currValue = ""
    currSamples = set()

    with open(data_path, 'r') as f:
        for line in f:
            if re.search("<node", line):        # we started a new sample
                currRow = []
                currSamples = set()
                next

            if re.search("<key", line):         # figure out the sample header structure
                name = re.sub('[\'/<>?"]', '', line)
                id = name.split('id=')[-1].strip()

                if re.search("precursor mass", name):
                    keys[id] = 'PrecMz'
                elif re.search("RTMean", name):
                    keys[id] = "RTMean"
                #elif re.search("precursor intensity", name):
                #    keys[id] = "PrecIntensity"
                elif re.search("UniqueFileSources", name):
                    keys[id] = "Sample"
        
            elif re.search("<data", line):          # gather data values
                currValue = re.findall('>(.*)<', line)
                currKey = re.findall('<data key="(.*?)">', line)
                
                for key in keys:
                    if (currKey[0] == key):
                        if keys[key] == configd["FILENAMECOL"]:     # we found the sample key
                            for i in currValue[0].split("|"):       # check each sample against activity file
                                found = False
                                for actSample in actSamples:
                                    if (actSample in i):
                                        found = True 
                                        currSamples.add(actSample)
                                if not found:
                                    print ("No match in Basketed file for", i)                              
                            currRow.append("")          # no intensity column
                            currRow.append(str("|".join(currSamples)))      # add samples
                            currRow.append(len(currSamples))        # add frequency
                            if(len(currSamples) > 1):
                                dfRow = pd.DataFrame(currRow).transpose()
                                newTable = newTable.append(dfRow)
                        else:
                            currRow.append(currValue[0])

                #print ("VALUE: " + str(value) + " KEY: " + str(key))

    newTable.columns = ["PrecMz", "RetTime", "PrecIntensity", "Sample", "freq"]
    newTable.to_csv("basketed.csv", index=False, quoting=None)
    print ("Saving file to basketed.csv")
