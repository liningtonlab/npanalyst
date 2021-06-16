import json
import logging
import os
from collections import defaultdict

import community as louvain_modularity
import networkx as nx
import pandas as pd
from networkx.readwrite import json_graph
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler


def add_community_as_node_attribute(graph, community_list, community_key="community"):
    """

    :param graph:
    :param community_list:
    :param community_key:
    :return:
    """

    community_dict = {}
    # TODO: Refactor this function for clarity
    for n, i in enumerate(community_list):
        for name in i:
            community_dict[name] = n
    nx.set_node_attributes(graph, community_dict, community_key)
    print("Community assignment has been added to the graph meta data.")


def community_assignment_df(graph, community_key="community", type_key="type_"):
    """

    :param graph:
    :param community_key:
    :param type_key:
    :return:
    """

    community_df = pd.DataFrame(
        {
            "node": [i for i in graph.nodes()],
            "community": [*nx.get_node_attributes(graph, community_key).values()],
            "type": [*nx.get_node_attributes(graph, type_key).values()],
        }
    )

    return community_df


def prune_assay_df_by_community(assay_df, community_df, graph, output=None):
    """

    :param assay_df:
    :param community_df:
    :param graph:
    :param output:
    :return:
    """

    community_count = max(community_df["community"])

    for community in range(community_count + 1):

        # # Create the community folders and save the data
        # # TODO: This line creates the new folder. For the UNIX system, this has to be changed.
        # os.mkdir(os.path.join(output, "community" + str(community)))
        os.makedirs(os.path.join(output, "clusters", str(community)), exist_ok=True)
        # # TODO: These 2 lines create the path to place the figures in. For the UNIX system, this has to be changed.

        # # Save the assay_df subset
        # outfile_assay_df_subset = os.path.join(output, "community/" + str(community) + "/assay_df_community_"
        #                                       + str(community) + ".csv")

        samples = (
            community_df["node"]
            .loc[
                (community_df["community"] == community)
                & (community_df["type"] == "sample")
            ]
            .tolist()
        )

        assay_df_subset = assay_df.loc[samples, :]

        # # Create a correlation matrix to sort the dataframe by bioassay relatedness
        if assay_df_subset.shape[0] >= 3:
            # # Calculate correlation distance between all normalized samples, create a single linkage matrix and
            # # order the samples optimal
            X = assay_df_subset.to_numpy(dtype="float64")
            sc = StandardScaler(with_std=True)
            X_norm = sc.fit_transform(X)
            distance_matrix = pdist(X_norm, "correlation")
            linkage_matrix = linkage(
                distance_matrix, method="complete", optimal_ordering=True
            )

            optimized_order = assay_df_subset.index.values[leaves_list(linkage_matrix)]
            assay_df_subset = assay_df_subset.reindex(optimized_order)

        # assay_df_subset.to_csv(outfile_assay_df_subset)

        # write assay data as json file
        comm = assay_df_subset.rename_axis("Sample").reset_index()
        result = comm.to_json(orient="records", index=True)
        parsed = json.loads(result)
        f = open(output + "/clusters/" + str(community) + "/activity.json", "w")
        f.write(json.dumps(parsed, indent=2))
        f.close()

        # # Save the subgraph that only contains nodes from the respective community
        outfile_graphml = os.path.join(
            output, "clusters/" + str(community) + "/network.graphml"
        )

        nodes = community_df["node"][community_df["community"] == community].tolist()

        subgraph = nx.subgraph(graph, nodes)
        nx.write_graphml(subgraph, outfile_graphml)

        # output network.json file
        jsonData = json_graph.node_link_data(subgraph)
        outfile_json = open(
            output + "/clusters/" + str(community) + "/network.json", "w"
        )
        outfile_json.write(json.dumps(jsonData, indent=2))
        outfile_json.close()

        # create a table for scatterplot
        table = pd.read_csv("table.csv")  # full table
        basketids = []

        for node in subgraph.nodes():
            try:
                if int(node):
                    basketids.append(node)
            except:
                pass

        output_table = table[table["BasketID"].isin(basketids)]
        outfile_csv = open(output + "/clusters/" + str(community) + "/table.csv", "w")
        output_table.to_csv(outfile_csv)

    return community_count


def louvain(g, weight="weight", resolution=1.0, randomize=False):
    """
    Implementation copied from `cdlib` - https://github.com/GiulioRossetti/cdlib
    to ease installation and remove unnecessary (problematic) dependencies.

    Louvain  maximizes a modularity score for each community.
    The algorithm optimises the modularity in two elementary phases:
    (1) local moving of nodes;
    (2) aggregation of the network.
    In the local moving phase, individual nodes are moved to the community that yields the largest increase in the quality function.
    In the aggregation phase, an aggregate network is created based on the partition obtained in the local moving phase.
    Each community in this partition becomes a node in the aggregate network. The two phases are repeated until the quality function cannot be increased further.

    :param g: a networkx object
    :param weight: str, optional the key in graph to use as weight. Default to 'weight'
    :param resolution: double, optional  Will change the size of the communities, default to 1.
    :param randomize:  boolean, optional  Will randomize the node evaluation order and the community evaluation  order to get different partitions at each call, default False
    :return: list of communities, sorted by size  largest to smalled

    >>> coms = algorithms.louvain(G, weight='weight', resolution=1., randomize=False)

    :References:

    Blondel, Vincent D., et al. `Fast unfolding of communities in large networks. <https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008/meta/>`_ Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.

    .. note:: Reference implementation: https://github.com/taynaud/python-louvain
    """

    coms = louvain_modularity.best_partition(
        g, weight=weight, resolution=resolution, randomize=randomize
    )

    # Reshaping the results
    coms_to_node = defaultdict(list)
    for n, c in coms.items():
        coms_to_node[c].append(n)

    coms_louvain = sorted(
        (list(c) for c in coms_to_node.values()), key=len, reverse=True
    )
    return list(coms_louvain)


def run(activityFile, outdir):
    # Load the bioassay dataframe
    assay_df = pd.read_csv(activityFile, index_col=0)

    os.chdir(outdir)
    # Load the NPAnalyst graph. Only features are retained that are above the defined activity score
    # and cluster score
    G = nx.read_graphml(path="network.graphml")
    # Detect the communities, using the Louvain's algorithm
    communities = louvain(G, randomize=False)
    # Add the community number as a new attribute ('community') to each sample and basket node
    add_community_as_node_attribute(
        G, communities
    )  # only applies this to a subset of the nodes!!!!
    # Create full graph with community annotations
    nx.write_graphml(G, "networkCommunities.graphml")
    # Translate the community assignments into a pandas dataframe that contains: node name, community id, and type_
    # as columns
    community_df = community_assignment_df(G)
    # Prune the graph and the bioassay table using the community id information and save the csv and graphml files.
    maxCount = prune_assay_df_by_community(assay_df, community_df, G, os.getcwd())
    return maxCount
