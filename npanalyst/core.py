import gc
import json
import logging
from joblib import Parallel, delayed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from npanalyst import activity, utils, create_community, exceptions
from npanalyst.utils import PATH

HERE = Path(__file__).resolve().parent


def load_config(config_path: Optional[PATH] = None) -> Dict:
    """loads the config_path config file and stores a bunch of values in a flatten dict
    config_path (str, optional): Defaults to 'default.json'.
        path to the config file, defaults can be overridden.
    """
    if config_path is None:
        config_path = HERE / "default.json"

    try:
        with open(config_path) as f:
            config = json.load(f)
    except OSError as e:
        logging.error(e)
        raise e

    MS1COLSTOMATCH = config["MSFileInfo"]["MS1ColsToMatch"].split(",")
    configd = {
        "FILENAMECOL": config["MSFileInfo"]["FileNameCol"],
        "MS1COLS": config["MSFileInfo"]["MS1Cols"].split(","),
        "MS1COLSTOMATCH": MS1COLSTOMATCH,
        "MS1ERRORCOLS": utils.make_error_col_names(MS1COLSTOMATCH),
        "CALCBASKETINFO": config["BasketInfo"]["CalcBasketInfo"],
        "BASKETMSLEVEL": int(config["BasketInfo"]["BasketMSLevel"]),
        "MINREPS": int(config["ReplicateInfo"]["RequiredReplicates"]),
        "MSLEVEL": int(config["MSFileInfo"]["MSLevel"]),
        "ACTIVITYTHRESHOLD": float(config["NetworkInfo"]["ActivityThreshold"]),
        "CLUSTERTHRESHOLD": float(config["NetworkInfo"]["ClusterThreshold"]),
    }

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

    logging.debug(f"Config loaded: \n{json.dumps(configd, indent=2)}")
    return configd


def replicate_compare_sample(
    sample: str, data_paths: List[Path], configd: Dict, outputdir: Path
) -> None:
    """
    Process one replica sample. The replicated file is saved as ./replicated/<sample>_Replicated.csv

    Args:
        sample (str): sample name
        data_paths (list): list of paths to replica files to be loaded
    """
    MS1COLSTOMATCH = configd["MS1COLSTOMATCH"]
    MS1ERRORCOLS = configd["MS1ERRORCOLS"]
    ERRORINFO = configd["ERRORINFO"]

    logging.info(f"Loading {len(data_paths)} MS data files for {sample}")
    logging.debug(data_paths)
    logging.debug(";".join(map(str, [MS1ERRORCOLS, ERRORINFO])))
    dfs = [utils.mzml_to_df(p, configd) for p in data_paths]
    df = pd.concat(dfs, sort=True).reset_index(drop=True)

    utils.add_error_cols(df, MS1COLSTOMATCH, ERRORINFO)
    rtree = utils.build_rtree(df, MS1ERRORCOLS)
    con_comps = utils.generate_connected_components(
        rtree, utils.get_hyperrectangles(df, MS1ERRORCOLS)
    )
    ndf = utils.collapse_connected_components(
        con_comps, df, configd, configd["MINREPS"]
    )
    ndf.to_csv(outputdir.joinpath("replicated").joinpath(f"{sample}_replicated.csv"))
    gc.collect()  # attempt to fix rtree index memory leak...
    logging.debug(f"{sample} done processing!")


def process_replicates(
    datadir: Path,
    outputdir: Path,
    configd: Dict,
    max_workers: int = -1,
) -> None:
    """
    multi proccesor version of replicate_compare_sample. by default will use cpu_count workers.

    process a folder of sample data replicates. output files will be saved in ./Replicated

    Args:
        datadir (str): data directory of sample replicates
        outputdir (str): output directory
        max_workers (int, optional): Defaults to None. If provided will use that
            many workers for processing. If there is limited system memory this might be good to set low.
    """
    outputdir.joinpath("replicated").mkdir(exist_ok=True)
    paths_iter = utils.generate_rep_df_paths(datadir)
    Parallel(n_jobs=max_workers)(
        delayed(replicate_compare_sample)(sample, paths, configd, datadir, outputdir)
        for sample, paths in paths_iter
    )


def basket(datadir: Path, configd: Dict) -> None:
    """
    Basket all the replicates in a directory in to a single file called Basketed.csv in datadir
    Unique file names are kept and deliminated with a '|'

    Args:
        datadir (str or Path): the directory of replicated files.
    """
    datadir = Path(datadir)
    FILENAMECOL = configd["FILENAMECOL"]
    # MS1COLS = configd["MS1COLS"]
    MS1ERRORCOLS = configd["MS1ERRORCOLS"]
    ERRORINFO = configd["ERRORINFO"]
    logging.info("Loading Rep Files")
    df = utils.make_repdf(datadir)
    orig_len = df.shape[0]

    # need to handle multiple file name cols from legacy/mixed input files
    df[FILENAMECOL] = np.where(df[FILENAMECOL].isnull(), df["Sample"], df[FILENAMECOL])
    df.dropna(subset=[FILENAMECOL], inplace=True)
    logging.info(f"Dropped {orig_len-df.shape[0]} rows missing values in {FILENAMECOL}")
    utils.add_error_cols(df, configd["MS1COLSTOMATCH"], ERRORINFO)
    logging.info("Making Rtree Index")
    rtree = utils.build_rtree(df, MS1ERRORCOLS)
    logging.info("Generating Baskets")
    con_comps = utils.generate_connected_components(
        rtree, utils.get_hyperrectangles(df, MS1ERRORCOLS)
    )
    ndf = utils.collapse_connected_components(con_comps, df, configd, min_reps=1)
    ndf["freq"] = ndf[FILENAMECOL].apply(lambda x: len(x.split("|")))
    # create the Basketed.csv file
    ndf.to_csv(datadir.joinpath("basketed.csv"), index=False)


def load_and_generate_act_outputs(basket_path, act_path, configd):
    baskets = activity.load_basket_data(basket_path, configd)
    activity_df = activity.load_activity_data(act_path)
    # Scores comes back as dict for if multiple activity files
    # TODO: eliminate dict

    # need to check and make sure that the samples in the baskets and activity file match
    mismatches, matches = utils.check_sample_names(activity_df, baskets, configd)
    if mismatches:
        logging.debug("Sample names in basket and activity file differ!")
        logging.debug(
            "The following samples were removed from the analysis:", mismatches
        )

    logging.debug("The following samples are kept:", matches)
    # only keep the matches
    activity_df = activity_df.loc[matches]

    if len(activity_df) < 3:
        logging.error(
            "There are fewer than 3 matches between the activity and msdata files ... exiting"
        )
        raise exceptions.MismatchedDataError

    scores = activity.score_baskets(baskets, activity_df, configd)

    logging.debug("SCORES", scores)
    activity.make_heatmap_input(activity_df, configd["OUTPUTDIR"])
    activity.make_bokeh_input(baskets, scores, configd["OUTPUTDIR"])
    activity.make_cytoscape_input(baskets, scores, configd["OUTPUTDIR"])


def create_communitites(act_path, outdir):
    logging.debug("Building clusters ... ")

    # create cluster folder and structure with json files for heatmaps
    return create_community.run(act_path, outdir)


def setup_logging(verbose: bool = False):
    """setup logging

    Args:
        verbose (bool): If True logging level=DEBUG, else WARNING
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level)
