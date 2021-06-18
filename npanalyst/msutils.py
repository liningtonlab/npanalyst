import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np
import pandas as pd
import pymzml
from rtree import index

from npanalyst import exceptions
from npanalyst.logging import get_logger

logger = get_logger()


def generate_rep_df_paths(datadir: Path) -> Iterator[Tuple[str, List[Path]]]:
    """
    generator func which yields concatenated replica file dataframes

    Args:
        datadir (str or Path): data directory path
        ext: extension we are considering
    """
    # allow mzml or csv, if files are mixed this will cause issues
    # extensions = ("mzml", "csv")
    datadir = Path(datadir)
    files = [f for f in datadir.iterdir() if f.suffix.lower().endswith("mzml")]
    logger.debug(files)
    repd = defaultdict(list)
    for fpath in files:
        repd[fpath.stem.split("_")[0]].append(
            fpath
        )  # parse rule.. probably needs to be more flexible

    for sample, files in repd.items():
        if len(files) >= 1:
            paths = [f for f in files]
            yield (sample, paths)


def make_error_col_names(qcols: Iterable) -> List:
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


def add_error_cols(df: pd.DataFrame, qcols: List, ERRORINFO: Dict) -> None:
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
        elif etype == "perc":
            efunc = lambda x: x * (evalue / 100)
        elif etype == "factor":
            efunc = lambda x: x * evalue
        elif etype == "window":
            efunc = lambda x: evalue
        elif etype is None:
            efunc = lambda x: 0
        else:
            raise exceptions.InvalidErrorType
        errors = col.apply(efunc)
        df[f"{dcol}_low"] = df[dcol] - errors
        df[f"{dcol}_high"] = df[dcol] + errors


def get_hyperrectangles(df: pd.DataFrame, errorcols: Iterable) -> np.ndarray:
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
    rgen = ((i, r, None) for i, r in enumerate(get_hyperrectangles(df, errorcols)))
    idx = index.Index(rgen, properties=p)
    return idx


def generate_connected_components(
    rtree: index.Index, rects: Iterable
) -> Iterator[Set[int]]:
    """
    Generate connected components subgraphs for a graph where nodes are hyperrectangles
    and edges are overlapping hyperrectangles. This is done using the rtree index and
    a depth first search.

    Args:
        rtree (rtree.Index): rtree index to use
        rects (iterable): array like object of hyperrectangles used to build the rtree

    Returns:
        Generator of sets of indices for each connect component
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
                    logger.error(e)
                    logger.error(rects[search])
                    raise e

                for n in neighbors - seen:  # set math
                    c.add(n)
                    search_idxs.append(n)
                    seen.add(n)
            yield c


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


def _combine_rows(cc_df: pd.DataFrame, configd: Dict) -> List:
    """combine ms1 rows in conncected component dataframe

    Args:
        cc_df (pd.DataFrame): conncected component dataframe
        min_reps (int): minumum number of replicates (number of uniqe files) that must be in connected component

    Returns:
        list: averaged values from rows of cc_df and/or basket info
    """
    calc_basket_info = configd["CALCBASKETINFO"]
    ms1vals = _average_data_rows(
        cc_df, configd["MS1COLS"], calc_basket_info=calc_basket_info
    )
    return ms1vals


def collapse_connected_components(
    ccs: Set, df: pd.DataFrame, configd: Dict, min_reps: int
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

    Returns:
        pd.DataFrame: newdata frame with data cols and file name col.
    """

    data = []
    file_col = []
    FILENAMECOL = configd["FILENAMECOL"]
    datacols = configd["MS1COLS"]
    calc_basket_info = configd["CALCBASKETINFO"]
    for cc in ccs:
        cc_df = df.iloc[list(cc)]
        uni_files = set(cc_df[FILENAMECOL].values)
        if len(uni_files) >= min_reps:
            file_col.append("|".join(uni_files))
            avgd = _combine_rows(cc_df, configd)
            data.append(avgd)
        else:
            continue
    cols = datacols[:]  # copy list
    if calc_basket_info:
        cols += ["BasketInfo"]

    ndf = pd.DataFrame(data, columns=cols)
    ndf[FILENAMECOL] = file_col

    return ndf


def make_repdf(datadir: Path) -> pd.DataFrame:
    """
    Make a concatonated dataframe from all the replicated data files.
    Assumes filenames end in 'replicated.csv' (as done in the replicate step from this toolchain)
    Args:
        datadir (str or Path): the directory with the replicated data files.

    Returns:
        pd.DataFrame: a dataframe with all replicate files concatenated
    """

    sd = Path(datadir)
    csvs = [f for f in sd.iterdir() if f.name.lower().endswith("replicated.csv")]
    dfs = [pd.read_csv(f) for f in csvs]
    return pd.concat(dfs, sort=False)


def _run2df(mzrun: pymzml.run.Reader) -> pd.DataFrame:
    """Convert pymyzml.run into DataFrame"""
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
                logger.warning(e)
                specdata = []

            if spec["MS:1000130"] is not None:
                mode = "+"
            elif spec["MS:1000129"] is not None:
                mode = "-"

        else:
            continue

        for mz, inte in specdata:
            data.append([mz, inte, scantime, mslevel, mode])
    df = pd.DataFrame(
        data, columns=["PrecMz", "PrecIntensity", "RetTime", "mslevel", "mode"]
    )
    return df


def mzml_to_df(mzml_path: Path, configd: Dict) -> pd.DataFrame:
    """Read MS data from mzML file and return a DataFrame."""
    run = pymzml.run.Reader(str(mzml_path))
    df = _run2df(run)
    fname = Path(mzml_path).stem
    logger.debug(f"{configd['FILENAMECOL']} -> {fname}")
    df[configd["FILENAMECOL"]] = fname
    logger.debug("Loaded file: %s", mzml_path)
    return df
