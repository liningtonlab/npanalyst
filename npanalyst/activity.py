import json
import logging
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Dict

import networkx as nx
import numpy as np
import pandas as pd

import re

from networkx.readwrite import json_graph

Score = namedtuple("Score", "activity cluster")


# def filename2sample(filename: str, fn_delim: str = "_", sampleidx: int = 1) -> str:
#     sample = filename.split(fn_delim)[sampleidx]
#     return sample


def filenames2samples(
    filenames: List, delim: str = "|", fn_delim: str = "_", sampleidx: int = 0
) -> Dict:

    samples = set()
    for filename in filenames.split(delim):
        if re.search("_[0-9]$", filename):
            samples.add(filename.split(fn_delim)[sampleidx])
        else:
            samples.add(filename)
    # samples = {
    #     #filename.split(fn_delim)[sampleidx] for filename in filenames.split(delim)
    #     filename for filename in filenames.split(delim)
    # }
    return samples


def feature_synthetic_fp(act_df: pd.DataFrame, samples: List) -> np.ndarray:
    to_cat = get_samples_fps(act_df, samples)
    return np.vstack(to_cat).mean(axis=0)


def get_samples_fps(fpd: pd.DataFrame, samples: List) -> np.ndarray:
    to_cat = []
    for samp in samples:
        try:
            to_cat.append(fpd.loc[samp].values)
        except KeyError as e:
            logging.warning(e)
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


def load_basket_data(bpath: Path, configd: Dict) -> List:
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


def load_activity_data(path: Path, samplecol: int = 0) -> pd.DataFrame:
    """
    Take activity file path and make dataframe with loaded data
    Add filename as column for future grouping.

    Sets the samplecol as the index
    """
    path = Path(path)
    dfs = []
    if path.is_dir():
        filenames = [sd for sd in path.iterdir() if sd.suffix.lower().endswith("csv")]
        for fname in filenames:
            # df = pd.read_csv(fname).fillna(value=0)  # na is not the same as 0!
            df = pd.read_csv(fname)
            name = fname.stem
            df["filename"] = name
            dfs.append(df)
    if path.is_file():
        name = path.stem
        # df = pd.read_csv(path).fillna(value=0)  # na is not the same as 0!
        df = pd.read_csv(path)
        df["filename"] = name
        dfs.append(df)

    big_df = pd.concat(dfs)
    big_df.set_index(big_df.columns[samplecol], inplace=True)

    return big_df


def score_baskets(baskets, act_df, configd):
    """Compute the activity and cluster score for all baskets.

    TODO: Double check the filename groupby and numeric feature detection
    """
    scores = defaultdict(dict)
    grouped = act_df.groupby("filename")

    for i, bask in enumerate(baskets):
        samples = bask["samples"]
        for actname, fpd in grouped:
            numeric_feature_df = fpd[[c for c in fpd.columns if c != "filename"]]
            try:
                sfp = feature_synthetic_fp(numeric_feature_df, samples)
                act_score = np.sum(sfp ** 2)
                clust_score = cluster_score(numeric_feature_df, samples)
                scores[actname][i] = Score(act_score, clust_score)
            except KeyError as e:
                logging.warning(e)

    return dict(scores)


def create_output_table(baskets, scored, output):
    """produce output CSV consistent with bokeh server input

    Args:
        baskets (list): List of basketed data loaded with load_baskets
        scored (dict): Dict of scores from score_baskets
    """
    logging.debug("Writing Bokeh output...")
    scores = scored.get("Activity")
    data = []
    for i, bask in enumerate(baskets):
        if scores is not None:
            # bid = f"Basket_{i}"
            bid = i
            freq = len(bask["samples"])
            samplelist = json.dumps(sorted(bask["samples"]))
            try:
                act = scores[i].activity
                clust = scores[i].cluster

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
            except KeyError:
                # act, clust = None, None
                pass

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
    outfile = output.joinpath("table.csv").resolve()
    df.to_csv(outfile, index=False, quoting=1, doublequote=False, escapechar=" ")


_BASKET_KEYS = ["PrecMz", "RetTime", "PrecIntensity"]
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
        # Actvity Data to carry forward
        "activity_score",
        "cluster_score",
    ],
)


def create_association_network(baskets, scored, output):
    logging.debug("Writing Cytoscape output...")
    scores = scored.get("Activity")
    edges = []
    basket_info = []
    samples = set()
    activity_scores = []

    # Need to remove basket ids that were removed during the automatic cutoff threshold
    for i, bask in enumerate(baskets):
        if scores is not None:
            bid = i
            try:
                act = scores[i].activity
                activity_scores.append(act)
                clust = scores[i].cluster
                samples.update(bask["samples"])

                for samp in bask["samples"]:
                    edges.append((bid, samp))

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
                # logging.debug(basket_info)

            except KeyError as e:
                logging.warning(e)

    # Construct graph and write outputs
    G = nx.Graph()
    for samp in samples:
        G.add_node(samp, type_="sample")
        G.nodes[samp]["radius"] = 6
        G.nodes[samp]["depth"] = 0
        G.nodes[samp]["color"] = "rgb(51,51,51)"
    for b in basket_info:
        G.add_node(b.id, **b._asdict(), type_="basket")
        # set node size based on activity score value - should range between 3 to 10 like scatterplot
        # output_start + ((output_end - output_start) * (input - input_start)) / (input_end - input_start)
        nodeSize = round(
            3
            + ((10 - 3) * (G.nodes[b.id]["activity_score"] - min(activity_scores)))
            / (max(activity_scores) - min(activity_scores))
        )

        # G.nodes[b.id]['radius'] = 4
        G.nodes[b.id]["radius"] = nodeSize
        G.nodes[b.id]["depth"] = 1
        # G.nodes[b.id]['color'] = "rgb(97, 205, 187)"

        # colors are hard-coded - change this for future versions
        if G.nodes[b.id]["cluster_score"] > 0.75:
            color = "rgb(165,0,38)"  # red color
        elif G.nodes[b.id]["cluster_score"] > 0.5:
            color = "rgb(215,48,39)"
        elif G.nodes[b.id]["cluster_score"] > 0.25:
            color = "rgb(244,109,67)"
        elif G.nodes[b.id]["cluster_score"] > 0:
            color = "rgb(253,174,97)"
        elif G.nodes[b.id]["cluster_score"] > -0.25:
            color = "rgb(171,217,233)"
        elif G.nodes[b.id]["cluster_score"] > -0.5:
            color = "rbg(116,173,209)"
        elif G.nodes[b.id]["cluster_score"] > -0.75:
            color = "rgb(69,117,180)"
        else:
            color = "rgb(49,54,149)"  # blue color

        # set color for the basket node
        G.nodes[b.id]["color"] = color

    for e in edges:
        G.add_edge(*e)

    logging.debug(nx.info(G))
    outfile_gml = output.joinpath("network.graphml").resolve()
    outfile_cyjs = output.joinpath("network.cyjs").resolve()
    outfile_json = output.joinpath("network.json").resolve()
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

    with open(outfile_json, "w") as fout:
        jsonData = json_graph.node_link_data(G)
        fout.write(json.dumps(jsonData, indent=2))


def create_activity_heatmap(activity_df, output):
    """produce json file in record format to be used as a heatmap

    Args:
        activity_df a pandas dataframe with activity values
    """
    logging.debug("Writing Heatmap output...")

    # save the big dataframe as json file for heatmap - remove filename column first though
    heatmap_df = activity_df.drop(columns=["filename"])  # remove filename column
    heatmap_df = heatmap_df.rename_axis(
        "Sample"
    ).reset_index()  # add index back as a column
    result = heatmap_df.to_json(orient="records", index=True)
    parsed = json.loads(result)

    outfile = output.joinpath("activity.json").resolve()

    with open(outfile, "w") as fout:
        fout.write(json.dumps(parsed, indent=2))
