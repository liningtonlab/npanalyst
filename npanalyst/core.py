import gc
import json
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from npanalyst import activity, community_detection, msutils, convert
from npanalyst.logging import get_logger

logger = get_logger()


HERE = Path(__file__).resolve().parent


def load_config(config_path: Optional[Path] = None) -> Dict:
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
        logger.error("Could not find config file")
        raise e
    except JSONDecodeError as e:
        logger.error("Invalid JSON config file")
        raise e

    MS1COLSTOMATCH = config["MSFileInfo"]["MS1ColsToMatch"].split(",")
    configd = {
        "FILENAMECOL": config["MSFileInfo"]["FileNameCol"],
        "MS1COLS": config["MSFileInfo"]["MS1Cols"].split(","),
        "MS1COLSTOMATCH": MS1COLSTOMATCH,
        "MS1ERRORCOLS": msutils.make_error_col_names(MS1COLSTOMATCH),
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

    logger.debug(f"Config loaded: \n{json.dumps(configd, indent=2)}")
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

    logger.info(f"Loading {len(data_paths)} MS data files for {sample}")
    logger.debug(data_paths)
    dfs = [msutils.mzml_to_df(p, configd) for p in data_paths]
    df = pd.concat(dfs, sort=True).reset_index(drop=True)

    msutils.add_error_cols(df, MS1COLSTOMATCH, ERRORINFO)
    rtree = msutils.build_rtree(df, MS1ERRORCOLS)
    con_comps = msutils.generate_connected_components(
        rtree, msutils.get_hyperrectangles(df, MS1ERRORCOLS)
    )
    ndf = msutils.collapse_connected_components(
        con_comps, df, configd, configd["MINREPS"]
    )
    ndf.to_csv(outputdir.joinpath("replicated").joinpath(f"{sample}_replicated.csv"))
    logger.debug(f"{sample} done processing - Found {len(ndf)} features.")
    gc.collect()  # attempt to fix rtree index memory leak...


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
    outputdir.joinpath("replicated").mkdir(exist_ok=True, parents=True)
    paths_iter = msutils.generate_rep_df_paths(datadir)
    Parallel(n_jobs=max_workers, backend="multiprocessing")(
        delayed(replicate_compare_sample)(sample, paths, configd, outputdir)
        for sample, paths in paths_iter
    )


def basket_replicated(datadir: Path, output_dir: Path, configd: Dict) -> None:
    """
    Basket all the replicates in a directory in to a single file called Basketed.csv in datadir
    Unique file names are kept and deliminated with a '|'
    """
    FILENAMECOL = configd["FILENAMECOL"]
    # MS1COLS = configd["MS1COLS"]
    MS1ERRORCOLS = configd["MS1ERRORCOLS"]
    ERRORINFO = configd["ERRORINFO"]
    logger.info("Loading Rep Files")
    df = msutils.make_repdf(datadir)
    orig_len = df.shape[0]

    # need to handle multiple file name cols from legacy/mixed input files
    df[FILENAMECOL] = np.where(df[FILENAMECOL].isnull(), df["Sample"], df[FILENAMECOL])
    df.dropna(subset=[FILENAMECOL], inplace=True)
    logger.info(f"Dropped {orig_len-df.shape[0]} rows missing values in {FILENAMECOL}")
    msutils.add_error_cols(df, configd["MS1COLSTOMATCH"], ERRORINFO)
    logger.info("Making Rtree Index")
    rtree = msutils.build_rtree(df, MS1ERRORCOLS)
    logger.info("Computing connected components")
    con_comps = msutils.generate_connected_components(
        rtree, msutils.get_hyperrectangles(df, MS1ERRORCOLS)
    )
    logger.info("Generating Baskets")
    ndf = msutils.collapse_connected_components(con_comps, df, configd, min_reps=1)
    ndf["freq"] = ndf[FILENAMECOL].apply(lambda x: len(x.split("|")))
    # Sort baskets by RT then MZ
    ndf.sort_values(["RetTime", "PrecMz"], inplace=True)
    logger.info(f"Found a total of {len(ndf)} basketed features")
    logger.info("Saving output file")
    # create the basketed.csv file
    ndf.to_csv(output_dir.joinpath("basketed.csv"), index=False)


def import_data(input_file: Path, output_dir: Path, mstype: str) -> None:
    """Convert the GNPS molecular network or  MZmine feature list to a list of
    basketed features with the same columns as the `basketed.csv` output from
    the mzML pipeline.

    Saves the CSV files as `output_dir/basketed.csv`.
    """
    # Extra guard - probably unnecessary
    assert mstype in ("gnps", "mzmine")
    if mstype == "gnps":
        logger.info(f"Importing molecular network from {input_file}")
        basket_df = convert.gnps(input_file)
    elif mstype == "mzmine":
        logger.info(f"Importing MZmine features from {input_file}")
        basket_df = convert.mzmine(input_file)
    # create the basketed.csv file
    logger.info(f"Found a total of {len(basket_df)} basketed features")
    logger.info("Saving output file")
    output_dir.mkdir(exist_ok=True, parents=True)
    # Sort baskets by RT then MZ
    basket_df.sort_values(["RetTime", "PrecMz"], inplace=True)
    basket_df.to_csv(output_dir.joinpath("basketed.csv"), index=False)


def bioactivity_mapping(
    basket_path: Path,
    output_dir: Path,
    activity_path: Path,
    configd: Dict,
    include_web_output: bool,
) -> None:
    """Performs compound activity mapping on basketed features."""
    logger.debug("Loading baskets and activity data")
    baskets = activity.load_basket_data(basket_path, configd)
    activity_df = activity.load_activity_data(activity_path)
    logger.info("Computing activity and cluster scores")
    scores = activity.score_baskets(baskets, activity_df)
    basket_df = activity.create_output_table(baskets, scores)
    G = activity.create_association_network(baskets, scores)
    logger.info("Computing network communities")
    G, communities = create_communitites(G, activity_df, basket_df)
    logger.info("Saving output files")
    output_dir.mkdir(exist_ok=True, parents=True)
    activity.save_table_output(basket_df, output_dir)
    activity.save_association_network(
        G, output_dir, include_web_output=include_web_output
    )
    activity.save_communities(
        communities, output_dir, include_web_output=include_web_output
    )


def create_communitites(
    G: nx.Graph,
    activity_df: pd.DataFrame,
    basket_df: pd.DataFrame,
) -> Tuple[nx.Graph, List[community_detection.Community]]:
    """Detect communities in compound activity mapping association network and returns
    newly annotated network and a list of Community named tuples for export
    """
    logger.info("Building communities ...")
    communities = community_detection.louvain(G)
    # Add the community number as a new attribute ('community') to each sample and basket node
    community_detection.add_community_as_node_attribute(G, communities)
    community_df = community_detection.community_assignment_df(G)
    communities = community_detection.detect_communities(
        activity_df, community_df, basket_df, G
    )
    return G, communities
