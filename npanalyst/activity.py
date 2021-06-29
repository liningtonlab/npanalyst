import json
from collections import namedtuple
from npanalyst import community_detection
from pathlib import Path
from typing import List, Dict
from joblib import Parallel, delayed
import networkx as nx
import numpy as np
import pandas as pd

import re

# from networkx.readwrite import json_graph

from npanalyst.logging import get_logger

logger = get_logger()

Score = namedtuple("Score", "activity cluster")


def filenames2samples(filenames: str, all_samples: List) -> List:
    """Aligns activity sames to filenames string for each basketed feature
    #NOTE: This function matches case sensitive for samples and filenames.
    """
    delims = "[_.-]"
    found_samples = set()
    match = re.findall(f"{delims}?({'|'.join(all_samples)}){delims}", filenames)
    if match:
        found_samples.update(match)
    return sorted(list(found_samples))


def feature_synthetic_fp(act_df: pd.DataFrame, samples: List) -> np.ndarray:
    to_cat = get_samples_fps(act_df, samples)
    return np.vstack(to_cat).mean(axis=0)


def get_samples_fps(fpd: pd.DataFrame, samples: List) -> np.ndarray:
    to_cat = []
    for samp in samples:
        try:
            to_cat.append(fpd.loc[samp].values)
        except KeyError as e:
            logger.warning(e)
    if not to_cat:
        raise KeyError("No Fingerprints found...")
    return np.asarray(to_cat)


def cluster_score(fpd: pd.DataFrame, samples: List) -> float:
    """
    Cluster score is the average of the off diagonal elements of the Pearson
    correlation matrix of all the fingerprints for the extracts a feature
    appears in.
    """
    fps = get_samples_fps(fpd, samples)  # Get matrix of fingerprints
    j = fps.shape[0]
    if j == 1:
        return 0.0
    # Easy pairwise correlation in pandas
    corr = pd.DataFrame(np.transpose(fps)).corr("pearson").values
    score = np.nansum(corr[np.triu_indices_from(corr, k=1)]) / ((j ** 2 - j) / 2.0)
    return score


def load_basket_data(
    bpath: Path, activity_df: pd.DataFrame, configd: Dict
) -> List[Dict]:
    """Loads basketed data and aligns samples from Activity file to filenames"""
    activity_samples = activity_df.index.to_list()
    bpath = Path(bpath)
    df = pd.read_csv(bpath.resolve())
    mmcols = []
    for col in configd["BASKETMINMAXCOLS"]:
        mmcols += [f"Min{col}", f"Max{col}"]
    MS1COLS = configd["BASKETFEATURES"] + mmcols
    FILENAMECOL = configd["FILENAMECOL"]
    cols_to_keep = list(set(MS1COLS + [FILENAMECOL]))
    ms1df = df[cols_to_keep].copy(deep=True)
    baskets = []

    for bd in ms1df.to_dict("records"):
        bd["samples"] = filenames2samples(bd[FILENAMECOL], activity_samples)
        baskets.append(bd)

    return baskets


def load_activity_data(path: Path, samplecol: int = 0) -> pd.DataFrame:
    """
    Take activity file path and make dataframe with loaded data
    Add filename as column for future grouping.

    Sets the samplecol as the index
    """
    # df = pd.read_csv(path).fillna(value=0)  # na is not the same as 0!
    df = pd.read_csv(path)
    df.set_index(df.columns[samplecol], inplace=True)
    return df


def score_basket(basket: Dict, activity_df: pd.DataFrame) -> Score:
    """Compute the activity and cluster score for a single basket"""
    samples = basket["samples"]
    sfp = feature_synthetic_fp(activity_df, samples)
    act_score = np.sum(sfp ** 2)
    clust_score = cluster_score(activity_df, samples)
    return Score(act_score, clust_score)


def score_baskets(
    baskets: List[Dict], activity_df: pd.DataFrame, max_workers=-1
) -> List[Score]:
    """Compute the activity and cluster score for all baskets in a parallelized fashion."""
    scores = Parallel(n_jobs=max_workers, backend="multiprocessing")(
        delayed(score_basket)(bask, activity_df) for bask in baskets
    )

    return scores


def create_feature_table(baskets: List[Dict], scores: List[Score]) -> pd.DataFrame:
    """produce output CSV consistent with bokeh server input

    Args:
        baskets (list): List of basketed data loaded with load_baskets
        scores (Score): Score namedtuple from score_baskets
    """
    logger.debug("Generating tabular output...")
    data = []
    for i, bask in enumerate(baskets):
        # bid = f"Basket_{i}"
        bid = i
        # samplelist = json.dumps(sorted(bask["samples"]))
        samplelist = "|".join(sorted(bask["samples"]))
        try:
            act = scores[i].activity
            clust = scores[i].cluster

            row = (
                bid,
                bask["PrecMz"],
                bask["PrecIntensity"],
                bask["MinPrecIntensity"],
                bask["MaxPrecIntensity"],
                bask["RetTime"],
                samplelist,
                act,
                clust,
            )
            data.append(row)
        except KeyError:
            # act, clust = None, None
            pass

    columns = (
        "BasketID",
        "PrecMz",
        "PrecIntensity",
        "MinPrecIntensity",
        "MaxPrecIntensity",
        "RetTime",
        "SampleList",
        "ACTIVITY_SCORE",
        "CLUSTER_SCORE",
    )
    df = pd.DataFrame(data, columns=columns)
    return df


_BASKET_KEYS = [
    "PrecMz",
    "RetTime",
    "PrecIntensity",
    "MinPrecIntensity",
    "MaxPrecIntensity",
]
Basket = namedtuple(
    "Basket",
    [
        "id",
        "freq",
        "samples",
        # BASKET KEYS - MS data to carry forward
        # These values are rounded in Network format
        "PrecMz",
        "RetTime",
        "PrecIntensity",
        "MinPrecIntensity",
        "MaxPrecIntensity",
        # Actvity Data to carry forward
        "activity_score",
        "cluster_score",
    ],
)


def create_association_network(
    baskets: List[Dict], scores: List[Score], configd: Dict
) -> nx.Graph:
    """Generate an assocation network of baskets and sampples
    filtering based on the ACTIVITYTHRESHOLD and CLUSTERTHRESHOLD.
    """
    logger.info("Generating association network...")
    edges = []
    # List of named tuples for convenient node creation
    # using built in **as_dict()
    basket_info = []
    samples = set()
    activity_scores = []
    act_thresh = configd["ACTIVITYTHRESHOLD"]
    clust_thresh = configd["CLUSTERTHRESHOLD"]

    # Need to remove basket ids that were removed during the automatic cutoff threshold
    for i, bask in enumerate(baskets):
        bid = i
        act = scores[i].activity
        activity_scores.append(act)
        clust = scores[i].cluster
        # TODO: verify this is the correct intention of filtering
        if act >= act_thresh and clust >= clust_thresh:
            samples.update(bask["samples"])
            for samp in bask["samples"]:
                edges.append((bid, samp))
            # Format this baskets for use in the networks
            basket_info.append(
                Basket(
                    bid,
                    len(bask["samples"]),
                    ";".join(list(bask["samples"])),
                    *[round(bask[k], 4) for k in _BASKET_KEYS],
                    round(act, 2),
                    round(clust, 2),
                )
            )

    # Construct graph
    G = nx.Graph()
    for samp in sorted(samples):
        G.add_node(samp, type_="sample")

    for b in sorted(basket_info):
        G.add_node(b.id, **b._asdict(), type_="basket")

    for e in sorted(edges):
        G.add_edge(*e)

    logger.debug("Precomputing network layout")
    # Pre-calculate and add layout
    # use random_state to always produce same positions
    pos = nx.spring_layout(G, seed=np.random.RandomState(42))
    for node, (x, y) in pos.items():
        G.nodes[node]["x"] = float(x)
        G.nodes[node]["y"] = float(y)

    return G


# def add_network_styling(G: nx.Graph) -> None:
#     """Add's Nima's styling to graph inplace"""
#         G.nodes[samp]["radius"] = 6
#         G.nodes[samp]["depth"] = 0
#         G.nodes[samp]["color"] = "rgb(51,51,51)"

#         # set node size based on activity score value - should range between 3 to 10 like scatterplot
#         # output_start + ((output_end - output_start) * (input - input_start)) / (input_end - input_start)
#         node_size = round(
#             3
#             + ((10 - 3) * (G.nodes[b.id]["activity_score"] - min(activity_scores)))
#             / (max(activity_scores) - min(activity_scores))
#         )

#         # G.nodes[b.id]['radius'] = 4
#         G.nodes[b.id]["radius"] = node_size
#         G.nodes[b.id]["depth"] = 1
#         # G.nodes[b.id]['color'] = "rgb(97, 205, 187)"

#         # colors are hard-coded - change this for future versions
#         if G.nodes[b.id]["cluster_score"] > 0.75:
#             color = "rgb(165,0,38)"  # red color
#         elif G.nodes[b.id]["cluster_score"] > 0.5:
#             color = "rgb(215,48,39)"
#         elif G.nodes[b.id]["cluster_score"] > 0.25:
#             color = "rgb(244,109,67)"
#         elif G.nodes[b.id]["cluster_score"] > 0:
#             color = "rgb(253,174,97)"
#         elif G.nodes[b.id]["cluster_score"] > -0.25:
#             color = "rgb(171,217,233)"
#         elif G.nodes[b.id]["cluster_score"] > -0.5:
#             color = "rbg(116,173,209)"
#         elif G.nodes[b.id]["cluster_score"] > -0.75:
#             color = "rgb(69,117,180)"
#         else:
#             color = "rgb(49,54,149)"  # blue color

#         # set color for the basket node
#         G.nodes[b.id]["color"] = color


def save_association_network(
    G: nx.Graph,
    output_dir: Path,
    include_web_output: bool,
) -> None:
    """Save network output(s) to specified output directory"""
    outfile_gml = output_dir.joinpath("network.graphml").resolve()
    logger.debug(f"Saving {outfile_gml}")
    logger.debug(nx.info(G))
    nx.write_graphml(G, outfile_gml, prettyprint=True)

    if include_web_output:
        outfile_cyjs = output_dir.joinpath("network.cyjs").resolve()
        data = nx.cytoscape_data(G)
        # Old code for cytoscape JS positioning information
        #     scale = len(G.nodes()) * 10
        #     for node, ndata in G.nodes(data=True):

        #         x = ndata["x"] * scale
        #         y = ndata["x"] * scale
        #         node_pos = {"x": x, "y": y}

        # # for d in data["elements"]["nodes"]:
        # #     posi = pos_dict.get(d.get("data").get("id"))
        # #     d["position"] = posi
        logger.debug(f"Saving {outfile_cyjs}")
        with open(outfile_cyjs, "w") as fout:
            fout.write(json.dumps(data, indent=2))


def save_table_output(
    df: pd.DataFrame,
    output_dir: Path,
    fstem: str = "table",
    index: bool = False,
    # include_web_output: bool,
) -> None:
    """Pandas To CSV is quite slow here..."""
    fpath = output_dir.joinpath(f"{fstem}.csv").resolve()
    logger.debug(f"Saving {fpath}")
    df.to_csv(fpath, index=index)


def save_communities(
    communities: List[community_detection.Community],
    output_dir: Path,
    include_web_output: bool,
) -> None:
    root_com_dir = output_dir / "communities"
    root_com_dir.mkdir(exist_ok=True)
    for idc, comm in enumerate(communities):
        com_dir = root_com_dir / str(idc)
        com_dir.mkdir(
            exist_ok=True,
        )
        save_association_network(
            comm.graph, com_dir, include_web_output=include_web_output
        )
        save_table_output(comm.table, com_dir, fstem="table")
        save_table_output(comm.assay, com_dir, fstem="assay", index=True)
