import gc
import json
import logging
import os
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import networkx as nx

import numpy as np
import pandas as pd
import pymzml
from rtree import index
from tqdm import tqdm


def setup_logging(verbose=False):
    """setup logging

    Args:
        verbose (bool): If True logging level=DEBUG, else WARNING
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level)


def _make_error_col_names(qcols):
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


def load_config(config_path=None):
    """loads the config_path config file and stores a bunch of values as globals
        config_path (str, optional): Defaults to 'default.cfg'.
            path to the config file, default can be overridden.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.joinpath("default.json")

    try:
        with open(config_path) as f:
            config = json.load(f)
    except OSError as e:
        logging.error(e)
        raise e

    configd = {}
    MS1COLS = config["MSFileInfo"]["MS1Cols"].split(",")
    configd["MS1COLS"] = MS1COLS
    MS1COLSTOMATCH = config["MSFileInfo"]["MS1ColsToMatch"].split(",")
    configd["MS1COLSTOMATCH"] = MS1COLSTOMATCH

    try:
        MS2COLS = config["MSFileInfo"]["MS2Cols"].split(",")
        configd["MS2COLS"] = MS2COLS
        MS2COLSTOMATCH = config["MSFileInfo"]["MS2ColsToMatch"].split(",")
        configd["MS2COLSTOMATCH"] = MS2COLSTOMATCH
        MS2ERRORCOLS = _make_error_col_names(MS2COLSTOMATCH)
        configd["MS2ERRORCOLS"] = MS2ERRORCOLS
    except KeyError:
        pass

    ERRORINFO = {}
    for name, tup in config["Tolerances"].items():
        etype, ev = tup.split(",")
        if etype == "None":
            etype = None
        if ev == "None":
            ev = None
        else:
            ev = float(ev)
        ERRORINFO[name] = (etype, ev)
    configd["ERRORINFO"] = ERRORINFO
    FILENAMECOL = config["MSFileInfo"]["FileNameCol"]
    configd["FILENAMECOL"] = FILENAMECOL
    MS1ERRORCOLS = _make_error_col_names(MS1COLSTOMATCH)
    configd["MS1ERRORCOLS"] = MS1ERRORCOLS

    configd["CalcBasketInfo"] = config["BasketInfo"]["CalcBasketInfo"]
    configd["BasketMSLevel"] = int(config["BasketInfo"]["BasketMSLevel"])
    configd["MINREPS"] = int(config["ReplicateInfo"]["RequiredReplicates"])
    configd["MSLEVEL"] = int(config["MSFileInfo"]["MSLevel"])

    # Network information
    configd["ActivityThreshold"] = float(config["NetworkInfo"]["ActivityThreshold"])
    configd["ClusterThreshold"] = float(config["NetworkInfo"]["ClusterThreshold"])

    logging.debug("Config loaded: \n%s", configd)
    return configd


def _run2df(mzrun):
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


def mzml_to_df(mzml_path):
    run = pymzml.run.Reader(mzml_path)
    df = _run2df(run)
    fname = Path(mzml_path).stem
    df["Sample"] = fname
    logging.debug("Loaded Sample: %s", fname)
    return df


def gen_rep_df_paths(datadir):
    """
    generator func which yields concatenated replica file dataframes

    Args:
        datadir (str): data directory path
    """
    extensions = [
        "mzml",
        "csv",
    ]  # allow mzml or csv, if files are mixed this will cause issues
    sd = os.scandir(datadir)
    csvs = [f.name for f in sd if f.name.lower().split(".")[-1] in extensions]
    logging.debug(csvs)
    repd = defaultdict(list)
    for fname in csvs:
        repd[fname.split("_")[0]].append(
            fname
        )  # parse rule.. probably needs to be more flexible

    for sample, files in repd.items():
        if len(files) >= 1:
            paths = [os.path.join(datadir, f) for f in files]
            yield (sample, paths)


def gen_error_cols(df, qcols, ERRORINFO):
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


def get_rects(df, errorcols):
    """
    get the hyperrectangles defined by the error funcs. assumes error cols are present in df.

    Args:
        errorcols (iterable): the error cols to make rectangles from
        df (pd.DataFrame): datafame with error cols in format <datacol>_low and <datacol>_high

    Returns:
        np.array: array of hyperrectangles in format [[x_low,y_low...x_high,y_high]]
    """
    order = errorcols
    return df[order].values


def build_rtree(df, errorcols):
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


def gen_con_comps(rtree, rects, pbar=False):
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
    if pbar:
        to_it = tqdm(range(len(rects)))
    else:
        to_it = range(len(rects))

    for i in to_it:
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


def reduce_to_ms1(df, configd):
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


def _average_data_rows(cc_df, datacols, calc_basket_info=False):
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


def combine_ms2(
    cc_df, configd,
):
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


def _combine_rows(cc_df, configd, ms2):
    """combine ms1 rows and optionally ms2 information in conncected component
    dataframe

    Args:
        cc_df (pd.DataFrame): conncected component dataframe
        min_reps (int): minumum number of replicates (number of uniqe files) that must be in connected component
        ms2 (bool): whether or not to combine ms2 data
        calc_basket_info (bool, optional): Defaults to False. whether or not to calculate basket info (spans)

    Returns:
        list: averaged values from rows of cc_df and optionally ms2 info and/or basket info
    """
    calc_basket_info = configd["CalcBasketInfo"]
    ms1vals = _average_data_rows(
        cc_df, configd["MS1COLS"], calc_basket_info=calc_basket_info
    )

    if ms2 and configd["MSLEVEL"] == 2:
        ms2vals = combine_ms2(cc_df, configd)
        return ms1vals + [ms2vals]
    else:
        return ms1vals


def proc_con_comps(ccs, df, configd, min_reps, ms2=False):
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
    calc_basket_info = configd["CalcBasketInfo"]
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


def _proc_one(sample, df_paths, configd, datadir):
    """
    Process one replica sample. The replicated file is saved as ./Replicated/<sample>_Replicated.csv

    Args:
        sample (str): sample name
        df_paths (list): list of paths to replica files to be loaded

    Returns:
        str: "DONE" when completed
    """

    FILENAMECOL = configd["FILENAMECOL"]
    MS1COLSTOMATCH = configd["MS1COLSTOMATCH"]
    MS1ERRORCOLS = configd["MS1ERRORCOLS"]
    ERRORINFO = configd["ERRORINFO"]
    # calc_basket_info = configd['CalcBasketInfo']

    logging.debug(df_paths)
    logging.debug(";".join(map(str, [MS1ERRORCOLS, ERRORINFO])))

    if df_paths[0].lower().endswith("csv"):
        dfs = [reduce_to_ms1(pd.read_csv(p), FILENAMECOL) for p in df_paths]
    else:  # mzML data
        dfs = [mzml_to_df(p) for p in df_paths]  # assumes only MS1 data is present

    df = pd.concat(dfs, sort=True)
    df = df.reset_index()
    gen_error_cols(df, MS1COLSTOMATCH, ERRORINFO)
    rtree = build_rtree(df, MS1ERRORCOLS)
    con_comps = gen_con_comps(rtree, get_rects(df, MS1ERRORCOLS))
    ndf = proc_con_comps(con_comps, df, configd, configd["MINREPS"])
    if configd["MSLEVEL"] == 2:
        ndf["MS2Info"] = [ms2df.to_json() for ms2df in ndf["MS2Info"]]

    ndf.to_csv(datadir.joinpath("Replicated").joinpath(f"{sample}_Replicated.csv"))
    gc.collect()  # attempt to fix rtree index memory leak...
    return "DONE"


def proc_folder(datadir, configd):
    """process a folder of sample data replicates. output files will be saved in ./Replacted

    Args:
        datadir (str): data directory of sample replicates
    """

    try:
        os.mkdir(datadir.joinpath("Replicated"))
    except OSError:
        pass
    paths = list(gen_rep_df_paths(datadir))
    for sample, df in tqdm(paths, desc="proc_folder"):
        _proc_one(sample, df, configd, datadir)


def _update(pbar, future):
    """callback func for future object to update progress bar"""
    pbar.update()


def mp_proc_folder(datadir, configd, max_workers=0):
    """
    multi proccesor version of proc_folder. by default will use cpu_count - 2 workers.

    process a folder of sample data replicates. output files will be saved in ./Replacted

    Args:
        datadir (str): data direcory of sample replicates
        calc_basket_info (bool,optional): Defaults to False. Bool on whether or
            not to save bin info as json strings.
        max_workers (int, optional): Defaults to None. If provided will use that
            many workers for processing. If there is limited system memory this might be good to set low.
    """

    try:
        os.mkdir(datadir.joinpath("Replicated"))
    except OSError:
        pass

    if max_workers == 0:
        max_workers = os.cpu_count()

    paths = list(gen_rep_df_paths(datadir))
    pbar = tqdm(desc="proc_folder", total=len(paths))
    samples_left = len(paths)
    paths_iter = iter(paths)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        while samples_left:

            for sample, paths in paths_iter:
                # fut = ex.submit(_proc_one,sample,paths,FILENAMECOL,datadir,calc_basket_info)
                fut = ex.submit(_proc_one, sample, paths, configd, datadir)
                fut.add_done_callback(partial(_update, pbar))
                futs[fut] = sample
                if len(futs) > max_workers:
                    break

            for fut in as_completed(futs):
                res = fut.result()
                logging.debug(res)
                del futs[fut]
                samples_left -= 1
                break


def make_repdf(datadir):
    """
    Make a concatonated dataframe from all the replicated data files.
    Assumes filenames end in 'Replicated.csv'
    Args:
        datadir (str): the directory with the replicated data files.

    Returns:
        pd.DataFrame: a dataframe with all replicate files concatenated
    """

    sd = os.scandir(datadir)
    csvs = [f.name for f in sd if f.name.lower().endswith("replicated.csv")]
    dfs = [pd.read_csv(os.path.join(datadir, f)) for f in csvs]
    return pd.concat(dfs, sort=False)


def _read_json(ms2json, i):
    """helper func that can be serialized for multiproc json de-serialization"""
    return i, pd.read_json(ms2json)


def basket(datadir, configd):
    """
    Basket all the replicates in a directory in to a single file called Baskted.csv in datadir
    Unique file names are kept and deliminated with a '|'

    Args:
        datadir (str): the directory of replicated files.
    """
    FILENAMECOL = configd["FILENAMECOL"]
    MS1COLS = configd["MS1COLS"]
    MS1ERRORCOLS = configd["MS1ERRORCOLS"]
    ERRORINFO = configd["ERRORINFO"]
    ms2 = configd["BasketMSLevel"] == 2
    print("Loading Rep Files")
    df = make_repdf(datadir)
    orig_len = df.shape[0]
    if ms2:  # de-serialize the json df's w/ multiproc
        with ProcessPoolExecutor() as ex:
            futs = [
                ex.submit(_read_json, ms2json, i)
                for i, ms2json in enumerate(df["MS2Info"])
            ]
        ms2dfs = []
        for f in tqdm(as_completed(futs), total=orig_len):
            ms2dfs.append(f.result())
        ms2dfs.sort(key=lambda x: x[0])
        df["MS2Info"] = [x[1] for x in ms2dfs]

    # need to handle multiple file name cols from legacy/mixed input files
    df[FILENAMECOL] = np.where(df[FILENAMECOL].isnull(), df["Sample"], df[FILENAMECOL])
    df.dropna(subset=[FILENAMECOL], inplace=True)
    print(f"Dropped {orig_len-df.shape[0]} rows missing values in {FILENAMECOL}")
    gen_error_cols(df, configd["MS1COLSTOMATCH"], ERRORINFO)
    print("Making Rtree Index")
    rtree = build_rtree(df, MS1ERRORCOLS)
    print("Generating Baskets")
    con_comps = gen_con_comps(rtree, get_rects(df, MS1ERRORCOLS), pbar=True)
    ndf = proc_con_comps(con_comps, df, configd, min_reps=1, ms2=ms2)
    #     ndf['MS2Info'] = [ms2df.to_json(orient='split',index=False) for ms2df in ndf['MS2Info']]
    #     ndf['freq'] = [len(s.split('|')) for s in ndf[FILENAMECOL]]
    ndf["freq"] = ndf[FILENAMECOL].apply(lambda x: len(x.split("|")))
    ndf.to_csv(os.path.join(datadir, "Basketed.csv"), index=False)


######################
#  Actiivity Mapping #
######################


def filename2sample(filename, fn_delim="_", sampleidx=1):
    sample = filename.split(fn_delim)[sampleidx]
    return sample


def filenames2samples(filenames, delim="|", fn_delim="_", sampleidx=0):
    samples = {
        filename.split(fn_delim)[sampleidx] for filename in filenames.split(delim)
    }
    return samples


def synth_fp(act_df, samples):
    to_cat = get_fps(act_df, samples)
    return np.vstack(to_cat).mean(axis=0)


def get_fps(fpd, samples):
    to_cat = []
    for samp in samples:
        try:
            to_cat.append(fpd.loc[samp].values)
        except KeyError as e:
            logging.warning(e)
    if not to_cat:
        raise KeyError("No Fingerprints found...")
    return np.asarray(to_cat)


def cluster_score(fpd, samples):
    """
    Cluster score is the average of the off diagonal elements of the Pearson
    correlation matrix of all the fingerprints for the extracts a feature
    appears in.
    """
    fps = get_fps(fpd, samples)  # Get matrix of fingerprints
    j = fps.shape[0]
    if j == 1:
        return 0.0
    # Easy pairwise correlation in pandas
    corr = pd.DataFrame(np.transpose(fps)).corr("pearson").values
    score = np.sum(corr[np.triu_indices_from(corr, k=1)]) / ((j ** 2 - j) / 2)
    return score


def load_basket_data(bpath: Path, configd) -> list:
    if not isinstance(bpath, Path):
        bpath = Path(bpath)
    df = pd.read_csv(bpath.resolve())
    MS1COLS = configd["MS1COLS"]
    FILENAMECOL = configd["FILENAMECOL"]
    cols_to_keep = MS1COLS + [FILENAMECOL]
    ms1df = pd.DataFrame(list(set(df[cols_to_keep].itertuples(index=False))))
    baskets = []
    for bd in ms1df.to_dict("records"):
        bd["samples"] = filenames2samples(bd[FILENAMECOL])
        baskets.append(bd)
    return baskets


def load_activity_data(apath: str, samplecol: int = 0) -> pd.DataFrame:
    """
    Take activity file path and make dataframe with loaded data
    Add filename as column for future grouping
    """
    p = apath if isinstance(apath, Path) else Path(apath)
    dfs = []
    if p.is_dir():
        filenames = [sd.name for sd in os.scandir(p) if sd.name.lower().endswith("csv")]
        for fname in filenames:
            df = pd.read_csv(p.joinpath(fname)).fillna(value=0)
            name = os.path.splitext(fname)[0]
            df["filename"] = name
            dfs.append(df)
    if p.is_file():
        name = os.path.splitext(p.name)[0]
        df = pd.read_csv(apath).fillna(value=0)
        df["filename"] = name
        dfs.append(df)

    big_df = pd.concat(dfs)
    big_df.set_index(big_df.columns[samplecol], inplace=True)
    return big_df


SCORET = namedtuple("Score", "activity cluster")


def score_baskets(baskets, act_df):
    scores = defaultdict(dict)
    grouped = act_df.groupby("filename")
    # for i, bask in tqdm(enumerate(baskets),desc='Scoring Baskets'):
    for i, bask in enumerate(baskets):
        samples = bask["samples"]
        for actname, fpd in grouped:
            num_fpd = fpd[[c for c in fpd.columns if c != "filename"]]
            try:
                sfp = synth_fp(num_fpd, samples)
                act_score = np.sum(sfp ** 2)
                clust_score = cluster_score(num_fpd, samples)
                scores[actname][i] = SCORET(act_score, clust_score)
            except KeyError as e:
                logging.warning(e)

    scores = dict(scores)
    return scores


def make_bokeh_input(baskets, scored, output):
    """produce output CSV consistent with bokeh server input

    Args:
        baskets (list): List of basketed data loaded with load_baskets
        scored (dict): Dict of scores from score_baskets
    """
    logging.debug("Writing Bokeh output...")
    scores = scored.get("Activity")
    data = []
    for i, bask in enumerate(baskets):
        bid = f"Basket_{i}"
        freq = len(bask["samples"])
        samplelist = "['{0}']".format("', '".join(bask["samples"]))
        try:
            act = scores[i].activity
            clust = scores[i].cluster
        except KeyError:
            act, clust = None, None

        row = (
            bid,
            freq,
            bask["PrecMz"],
            bask["PrecIntensity"],
            bask["RetTime"],
            samplelist,
            act,
            clust,
        )
        data.append(row)
    columns = (
        "BasketID",
        "Frequency",
        "PrecMz",
        "PrecIntensity",
        "RetTime",
        "SampleList",
        "ACTIVITY_SCORE",
        "CLUSTER_SCORE",
    )
    df = pd.DataFrame(data, columns=columns)
    # df.to_excel("HIFAN.xlsx")
    outfile = output.joinpath("HIFAN.csv").as_posix()
    df.to_csv(outfile, index=False, quoting=1, doublequote=False, escapechar=" ")


_BASKET_KEYS = ["PrecMz", "RetTime", "PrecIntensity"]
BINFO = namedtuple(
    "Basket",
    [
        "id",
        "freq",
        "samples",
        *[k for k in _BASKET_KEYS],
        "activity_score",
        "cluster_score",
    ],
)


def make_cytoscape_input(baskets, scored, output, act_thresh=5000, clust_thresh=0.25):
    logging.debug("Writing Cytoscape output...")
    scores = scored.get("Activity")
    edges = []
    basket_info = []
    samples = set()
    for i, bask in enumerate(baskets):
        bid = f"Basket_{i}"
        try:
            score = scores[i]
        except KeyError as e:
            logging.warning(e)
            score = SCORET(0, 0)
        if score.activity > act_thresh and abs(score.cluster) > clust_thresh:
            samples.update(bask["samples"])
            for samp in bask["samples"]:
                edges.append((bid, samp))
            basket_info.append(
                BINFO(
                    bid,
                    len(bask["samples"]),
                    ";".join(list(bask["samples"])),
                    *[round(bask[k], 4) for k in _BASKET_KEYS],
                    round(score.activity, 2),
                    round(score.cluster, 2),
                )
            )

    # Construct graph and write outputs
    G = nx.Graph()
    for samp in samples:
        G.add_node(samp, type_="sample")
    for b in basket_info:
        G.add_node(b.id, **b._asdict(), type_="basket")
    for e in edges:
        G.add_edge(*e)

    logging.debug(nx.info(G))
    outfile_gml = output.joinpath("HIFAN.graphml").resolve()
    outfile_cyjs = output.joinpath("HIFAN.cyjs").resolve()
    nx.write_graphml(G, outfile_gml, prettyprint=True)

    data = nx.cytoscape_data(G)
    # Pre-calculate and add layout
    pos = nx.spring_layout(G)
    pos_dict = dict()
    scale = len(pos) * 10
    for node, (x, y) in pos.items():
        x = x * scale
        y = y * scale
        pos_dict[node] = {"x": x, "y": y}

    for d in data["elements"]["nodes"]:
        posi = pos_dict.get(d.get("data").get("id"))
        d["position"] = posi

    with open(outfile_cyjs, "w") as fout:
        fout.write(json.dumps(data, indent=2))


def load_and_generate_act_outputs(basket_path, act_path, configd):
    baskets = load_basket_data(basket_path, configd)
    activity_df = load_activity_data(act_path)
    scores = score_baskets(baskets, activity_df)
    outputdir = configd["outputdir"]
    make_bokeh_input(baskets, scores, outputdir)
    make_cytoscape_input(
        baskets,
        scores,
        outputdir,
        act_thresh=configd["ActivityThreshold"],
        clust_thresh=configd["ClusterThreshold"],
    )
