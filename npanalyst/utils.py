import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import pymzml
from rtree import index

import re 

import sys


PATH = Union[Path, str]


def gen_rep_df_paths(datadir: PATH, extension: str, samples: Dict) -> Tuple[str, List[Path]]:
    """
    generator func which yields concatenated replica file dataframes

    Args:
        datadir (str or Path): data directory path
        ext: extension we are considering
    """
    #extensions = [
    #    ".mzml",
    #    ".csv",
    #]  # allow mzml or csv, if files are mixed this will cause issues
    sd = Path(datadir)
    csvs = [f for f in sd.iterdir() if f.suffix.lower() == extension]
    logging.info("Collected replicate files:")
    logging.debug(f"{csvs}")
    repd = defaultdict(list)

    for sample in samples:
        found = False
        for fname in csvs:
            if re.search(f'{sample}', str(fname.stem), re.IGNORECASE):
                repd[sample].append(fname)
                found = True

        if not found:
            print ("Could not find an mzml file for", sample,"check activity and/or the mzml file.")
            print (sample, "will be ignored.")

    check_replicates(repd)

    #for fname in csvs:
    #    sample = fname.stem.split("_")[0]
    #    repd[sample].append(
    #        fname
    #    ) # parse rule.. probably needs to be more flexible

    for sample, files in repd.items():
        if len(files) >= 1:
            yield (sample, files)


def gen_error_cols(df: pd.DataFrame, qcols: List, ERRORINFO: Dict) -> Dict:
    """
    Uses the global ERRORINFO dict to generate
    error windows for each of the qcols.
    Mutates dataframe inplace for some memory conservation.
    possible error types are:
    * ppm - parts per million
    * perc - percentage
    * factor - a multiplier (ie 10 = 10x)
    * window - a standard fixed error window

    Args:
        df (pandas.DataFrame): input dataframe to calc error windows (modified in place)
    """

    for dcol in qcols:
        einfo = ERRORINFO[dcol]
        col = df[dcol]
        etype, evalue = einfo
        if etype == "ppm":
            efunc = lambda x: x * (evalue * 1e-6)
        if etype == "perc":
            efunc = lambda x: x * (evalue / 100)
        if etype == "factor":
            efunc = lambda x: x * evalue
        if etype == "window":
            efunc = lambda x: evalue
        if etype is None:
            efunc = lambda x: 0
        errors = col.apply(efunc)
        df[f"{dcol}_low"] = df[dcol] - errors
        df[f"{dcol}_high"] = df[dcol] + errors


def get_rects(df: pd.DataFrame, errorcols: Iterable) -> np.ndarray:
    """
    get the hyperrectangles defined by the error funcs. assumes error cols are present in df.

    Args:
        errorcols (iterable): the error cols to make rectangles from
        df (pd.DataFrame): datafame with error cols in format <datacol>_low and <datacol>_high

    Returns:
        np.ndarray: array of hyperrectangles in format [[x_low,y_low...x_high,y_high]]
    """
    order = errorcols
    return df[order].values


def build_rtree(df: pd.DataFrame, errorcols: Iterable) -> index.Index:
    """
    build an rtree index from a dataframe for fast range queries.
    dataframe needs to have error cols pre-calced

    Args:
        df (pd.DataFrame): dataframe with error cols

    Returns:
        rtree.Index: a rtree index built from df rectangles
    """
    dims = len(errorcols) // 2
    p = index.Property()
    p.dimension = dims
    p.interleaved = False
    rgen = ((i, r, None) for i, r in enumerate(get_rects(df, errorcols)))
    idx = index.Index(rgen, properties=p)
    return idx


def gen_con_comps(rtree: index.Index, rects: Iterable, pbar: bool = False):
    """
    Generate connected components subgraphs for a graph where nodes are hyperrectangles
    and edges are overlapping hyperrectangles. This is done using the rtree index and
    a depth first search.

    Args:
        rtree (rtree.Index): rtree index to use
        rects (iterable): array like object of rectangles used to build the rtree
        pbar (bool, optional): Defaults to False. Whether or not to display a progress bar
    """

    seen = set()

    for i, _ in enumerate(rects):
        if i in seen:
            continue
        else:
            search_idxs = [i]
            c = {i}
            while search_idxs:
                search = search_idxs.pop()
                try:
                    neighbors = set(rtree.intersection(rects[search]))
                except Exception as e:
                    print(e)
                    print(rects[search])
                    raise e

                for n in neighbors - seen:  # set math
                    c.add(n)
                    search_idxs.append(n)
                    seen.add(n)
            yield c


def reduce_to_ms1(df: pd.DataFrame, configd: dict) -> pd.DataFrame:
    """takes a dataframe w/ ms2 data in "tidy dataformat" and
    reduces it down to a ms1 df with a ms2df object stored in MS2Info

    Args:
        df (pd.DataFrame): ms2 dataframe in tidy (rectangluar) format
        FILENAMECOL (str): filename column (needed for de-replication)

    Returns:
        df: a ms1df which has the ms2 data in MS2Info column
    """
    MS1COLS = configd["MS1COLS"]
    FILENAMECOL = configd["FILENAMECOL"]
    gb = df.groupby(MS1COLS + [FILENAMECOL])
    ms1_data = []
    for gbi, ms2df in gb:
        fname = gbi[-1]
        ms2df = ms2df.copy()
        ms2df[FILENAMECOL] = [fname] * len(ms2df)
        # ms1_data.append(list(gbi)+[ms2df.to_json(orient='split',index=False)])
        # ms1_data.append(list(gbi)+[ms2df.to_json()])
        ms1_data.append(list(gbi) + [ms2df])
    cols = MS1COLS + [FILENAMECOL, "MS2Info"]
    ms1df = pd.DataFrame(ms1_data, columns=cols)

    return ms1df


def _average_data_rows(
    cc_df: pd.DataFrame, datacols: Iterable, calc_basket_info: bool = False
) -> List:
    """average (mean) the datacols values. optionally calculates the bounds
    of each resulting basket. basket_info will be serialized as json.

    Args:
        cc_df (pd.DataFrame): connected component dataframe
        datacols (iterable): the columns whose data should be averaged
        calc_basket_info (bool, optional): Defaults to False. flag to compute bounds of basket parameters

    Returns:
        list: row of averaged values (+ optional basket_info json)
    """

    avgd = list(cc_df[list(datacols)].mean())
    if calc_basket_info:
        basket_info = {
            cn: [float(cc_df[cn].min()), float(cc_df[cn].max())] for cn in datacols
        }
        basket_info["n"] = len(cc_df)
        avgd.append(json.dumps(basket_info))
    return avgd


def combine_ms2(cc_df: pd.DataFrame, configd: Dict) -> pd.DataFrame:
    """combine the ms2 data for a given connected component graph of ms1 ions.
    this is done the same way as the ms1 matching (rtree index) but uses the columns
    defined in global MS2COLSTOMATCH

    Args:
        cc_df (pd.DataFrame): conncected component dataframe
        min_reps (int, optional): Defaults to 2. minimum number of replicates required for ms2 ion to be included

    Returns:
        pd.DataFrame: combined ms2 dataframe
    """

    # ms2dfs = [pd.read_json(ms2df,orient='split') for ms2df in cc_df['MS2Info']]
    # ms2dfs = [pd.read_json(ms2df) for ms2df in cc_df['MS2Info']]
    MS2COLSTOMATCH = configd["MS2COLSTOMATCH"]
    MS2ERRORCOLS = configd["MS2ERRORCOLS"]
    FILENAMECOL = configd["FILENAMECOL"]
    MS2COLS = configd["MS2COLS"]
    ERRORINFO = configd["ERRORINFO"]
    MINREPS = configd["MINREPS"]
    ms2dfs = cc_df["MS2Info"].values.tolist()
    ms2df = pd.concat(ms2dfs, sort=True)
    # print(ms2df.columns)
    if ms2df.shape[0] > 1:
        gen_error_cols(ms2df, MS2COLSTOMATCH, ERRORINFO)
        rects = get_rects(ms2df, MS2ERRORCOLS)
        rtree = build_rtree(ms2df, MS2ERRORCOLS)
        ccs = gen_con_comps(rtree, rects)
        data = []
        file_col = []
        for cc in ccs:
            if len(cc) > 1:
                cc_df = ms2df.iloc[list(cc)]
                uni_files = set(cc_df[FILENAMECOL].values)
                if len(uni_files) >= MINREPS:
                    data.append(_average_data_rows(cc_df, MS2COLS))
                    file_col.append("|".join(uni_files))
            #     else:
            #         data.append([None]*len(MS2COLS))
            # else:
            #     data.append([None]*len(MS2COLS))

        avg_ms2 = pd.DataFrame(data, columns=MS2COLS)
        avg_ms2[FILENAMECOL] = file_col
    else:
        avg_ms2 = ms2df
    # return avg_ms2.to_json(orient='split',index=False) #note that to read back to df orient='split' must be set in pd.read_json()
    return avg_ms2


def _combine_rows(cc_df: pd.DataFrame, configd: Dict, ms2: bool = False) -> List:
    """combine ms1 rows and optionally ms2 information in conncected component
    dataframe

    Args:
        cc_df (pd.DataFrame): conncected component dataframe
        min_reps (int): minumum number of replicates (number of uniqe files) that must be in connected component
        ms2 (bool): whether or not to combine ms2 data

    Returns:
        list: averaged values from rows of cc_df and optionally ms2 info and/or basket info
    """
    calc_basket_info = configd["CALCBASKETINFO"]
    ms1vals = _average_data_rows(
        cc_df, configd["MS1COLS"], calc_basket_info=calc_basket_info
    )

    if ms2 and configd["MSLEVEL"] == 2:
        ms2vals = combine_ms2(cc_df, configd)
        return ms1vals + [ms2vals]
    else:
        return ms1vals


def proc_con_comps(
    ccs: Set, df: pd.DataFrame, configd: Dict, min_reps: int, ms2: bool = False
) -> pd.DataFrame:
    """
    Takes the connected components from the overlapping hyperrectangles and averages (mean)
    the data values from which the error was generated. Unique filenames are concatenated with a
    '|' delimiter. Only subgraphs with multiple nodes are used and further filtered for only those
    which come from at least `min_reps` unique files.


    Args:
        ccs (set): connected component subgraph dataframe indices
        df (pd.DataFrame): the dataframe which the connected component subgraphs were calculated from
        min_reps (int): Minimum number of files needed in subgraph to be used
        ms2(bool, optional): Defaults to False. Whether or not to average MS2 data

    Returns:
        pd.DataFrame: newdata frame with data cols and file name col.
    """

    data = []
    file_col = []
    FILENAMECOL = configd["FILENAMECOL"]
    datacols = configd["MS1COLS"]
    calc_basket_info = configd["CALCBASKETINFO"]
    for cc in ccs:
        # if len(cc) > 1:
        cc_df = df.iloc[list(cc)]
        uni_files = set(cc_df[FILENAMECOL].values)
        if len(uni_files) >= min_reps:
            file_col.append("|".join(uni_files))
            avgd = _combine_rows(cc_df, configd, ms2=ms2)
            data.append(avgd)
        else:
            continue
    cols = datacols[:]
    if calc_basket_info:
        cols += ["BasketInfo"]
    if ms2 & configd["MSLEVEL"] == 2:
        cols += ["MS2Info"]

    ndf = pd.DataFrame(data, columns=cols)
    ndf[FILENAMECOL] = file_col

    return ndf


def make_repdf(datadir: PATH) -> pd.DataFrame:
    """
    Make a concatonated dataframe from all the replicated data files.
    Assumes filenames end in 'Replicated.csv'
    Args:
        datadir (str or Path): the directory with the replicated data files.

    Returns:
        pd.DataFrame: a dataframe with all replicate files concatenated
    """

    sd = Path(datadir)
    csvs = [f for f in sd.iterdir() if f.name.lower().endswith("replicated.csv")]
    dfs = [pd.read_csv(f) for f in csvs]
    return pd.concat(dfs, sort=False)


def _read_json(ms2json, i):
    """helper func that can be serialized for multiproc json de-serialization"""
    return i, pd.read_json(ms2json)


def _make_error_col_names(qcols: Iterable) -> List:
    """helper func to make error column names of the form
    <col_name>_low ... <col_name>_high

    Args:
        qcols (iterable): an iterable of column names used for matching

    Returns:
        list: list of error col names in non-interleaved order
    """
    error_cols = [f"{dcol}_low" for dcol in qcols]
    error_cols = error_cols + [f"{dcol}_high" for dcol in qcols]
    return error_cols


def _run2df(mzrun: pymzml.run.Reader) -> pd.DataFrame:
    data = []
    specl = [s for s in mzrun if s]
    for spec in specl:

        scantime = spec.scan_time[0]
        mslevel = spec.ms_level
        if mslevel == 1:  # MS
            lower_scan_limit = spec["MS:1000501"]
            upper_scan_limit = spec["MS:1000500"]

            try:
                mzi = spec.peaks("centroided")
                # Filter out bad peaks
                if mzi.shape == 0:
                    continue
                # Apply simple filtering
                mzi = mzi[
                    (mzi[:, 0] > lower_scan_limit) & (mzi[:, 0] < upper_scan_limit)
                ]
                specdata = mzi.tolist()

            except (AttributeError, IndexError) as e:
                logging.warning(e)
                specdata = []

            if spec["MS:1000130"] is not None:
                mode = "+"
            elif spec["MS:1000129"] is not None:
                mode = "-"
        # note the following won't work with current builds of pymzml
        # either need to include hacked version as sub module of this or fix the package
        # elif mslevel == 0: # UV
        #     continue
        # not dealing w/ UV for now
        # specdataL = spec.peaks("raw").tolist()
        # mode = None

        else:
            continue
            # raise NotImplementedError("Only MS1 mzML data supported for now...")

        for mz, inte in specdata:
            data.append(
                [mz, inte, scantime, mslevel, mode,]
            )
    df = pd.DataFrame(
        data, columns=["PrecMz", "PrecIntensity", "RetTime", "mslevel", "mode"]
    )
    return df


def mzml_to_df(mzml_path: PATH, configd: Dict) -> pd.DataFrame:
    # mzml_path = Path(mzml_path)
    run = pymzml.run.Reader(str(mzml_path))
    df = _run2df(run)
    fname = Path(mzml_path).stem
    logging.debug(f"{configd['FILENAMECOL']} -> {fname}")
    df[configd["FILENAMECOL"]] = fname
    logging.debug("Loaded file: %s", fname)
    return df


def _update(pbar, future):
    """callback func for future object to update progress bar"""
    pbar.update()

def sameFileFormat(path: PATH, unknown: List) -> bool:
    ext = set()
    file_names = list(path) + unknown

    allowed_ext = ['.csv', '.mzml', '.graphml']
    
    for files in file_names:
        if (Path(files).is_file()):
            ext.add(Path(files).suffix.lower())
            if (len(ext) > 1):
                print(ext.pop(),"is a different format to",ext.pop())
                return False
        if (Path(files).is_dir()):
            pass
    print ("Checked all file extensions same format", ext.pop())
    return True

def check_sample_names(actdf, basket, configd):
    # save all activity file sample names into a set
    actSamples = set(actdf.index)

    # save all basket file sample names into a set
    basketSamples = set()
    for record in basket:
        matched_samples = set(record[configd["FILENAMECOL"]].split("|"))
        basketSamples = basketSamples.union(matched_samples)

    matches =  set()
    mismatches = set()

    for actSample in actSamples:
        found_match = False
        for basketSample in basketSamples:
            if (actSample in basketSample):
                found_match = True
                matches.add(actSample)

        if not found_match:
            print ("No match in Basketed file for", actSample)
            mismatches.add(actSample)
    
    print ("Activity/Basket matches:", str(len(matches)))
    print ("Activity/Basket mismatches:", str(len(mismatches)))

    return mismatches, matches

def get_samples (actdf, samplecol) -> Dict:
    df = pd.read_csv(actdf, index_col = None)
    try: 
        if df[samplecol]:
            return set((df[samplecol]))
    except:
        return (set(df[df.columns[0]]))

def check_replicates(df):
    num_replicates = set()
    replicateNumError = False
    for sample in df:
        num_replicates.add(len(df[sample]))
        if (len(num_replicates) == 2 and not replicateNumError):
            print ("Sample", sample, "has different number of replicates:", len(df[sample]), "compared to", num_replicates)
            replicateNumError = True
    
    if replicateNumError:
        print ("Different number of replicates:", num_replicates)
    elif (len(num_replicates) > 0):
        print ("Autodetected:", num_replicates.pop(), "replicates per sample")