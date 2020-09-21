import json
import logging
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Union, List, Dict

import networkx as nx
import numpy as np
import pandas as pd

PATH = Union[Path, str]


def filename2sample(filename: str, fn_delim: str = "_", sampleidx: int = 1) -> str:
    sample = filename.split(fn_delim)[sampleidx]
    return sample


def filenames2samples(
    filenames: List, delim: str = "|", fn_delim: str = "_", sampleidx: int = 0
) -> Dict:
    samples = {
        filename.split(fn_delim)[sampleidx] for filename in filenames.split(delim)
    }
    return samples


def synth_fp(act_df: pd.DataFrame, samples: List) -> np.ndarray:
    to_cat = get_fps(act_df, samples)
    return np.vstack(to_cat).mean(axis=0)


def get_fps(fpd: pd.DataFrame, samples: List) -> np.ndarray:
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
    fps = get_fps(fpd, samples)  # Get matrix of fingerprints
    j = fps.shape[0]
    if j == 1:
        return 0.0
    # Easy pairwise correlation in pandas
    corr = pd.DataFrame(np.transpose(fps)).corr("pearson").values
    score = np.nansum(corr[np.triu_indices_from(corr, k=1)]) / ((j ** 2 - j) / 2.0)
    return score


def load_basket_data(bpath: PATH, configd: Dict) -> List:
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


def load_activity_data(apath: PATH, samplecol: int = 0) -> pd.DataFrame:
    """
    Take activity file path and make dataframe with loaded data
    Add filename as column for future grouping
    """
    p = Path(apath)
    dfs = []
    if p.is_dir():
        filenames = [sd for sd in p.iterdir() if sd.suffix.lower().endswith("csv")]
        for fname in filenames:
            df = pd.read_csv(fname).fillna(value=0)
            name = fname.stem
            df["filename"] = name
            dfs.append(df)
    if p.is_file():
        name = p.stem
        df = pd.read_csv(p).fillna(value=0)
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
    outfile = output.joinpath("NPAnalyst.csv").as_posix()
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
        if score.activity >= act_thresh and abs(score.cluster) >= clust_thresh:
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
    outfile_gml = output.joinpath("NPAnalyst.graphml").as_posix()
    outfile_cyjs = output.joinpath("NPAnalyst.cyjs").as_posix()
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


def auto_detect_threshold(scores):
    return None
