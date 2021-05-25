import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import gower

import os
import json
from pathlib import Path
# import logging
import networkx as nx
from networkx.readwrite import json_graph
import re
import sys


def initialize_clustering_method(X, cat_vector, method='standardize'):
    """
    This function intends to determine, if hierarchical clustering with euclidean distance (continuous data) and Ward's
    method or Gower distance with complete_linkage method shall be used. This decision depends on the
    cat_vector that indicates if categorical (binary) values are present in the X matrix.
    If only continuous variables are detected, variables will be standardized or min_max scaled.

    :param X: X assay matrix with samples in rows and variables in columns
    :param cat_vector: list / numpy 1-D array with booleans indicating if a variable (column position) is categorical
    :return: X_prepared for further analysis (might be unchanged, if Gower distance was used), linkage method
    """

    if len(cat_vector) < X.shape[1]:
        raise ValueError(
            "Length of cat_vector needs to be identical with variable count (columns) of X")

    if not any(cat_vector):
        if method == 'standardize':
            scaler = StandardScaler(with_mean=True, with_std=True)
        elif method == 'min_max':
            scaler = MinMaxScaler()
        else:
            raise ValueError(
                "Parameter 'method' can only be 'standardize' or 'min_max'.")

        X_prepared = scaler.fit_transform(X)
        X_prepared = pd.DataFrame(X_prepared, index=X.index, columns=X.columns)
        cluster_method = 'HC'
    else:
        X_prepared = X
        cluster_method = 'HC'

    return X_prepared, cluster_method


def create_linkage_matrix(X, cat_vector):
    """
    This function creates the linkage matrix, necessary for the dendrogram plot.
    In case that the X matrix contains any categorical value (any(cat_vector)),
    the Gower distance will be calculated first, and complete linkage will be used
    to create the distance matrix. For continuous data, the Euclidean distance
    with Ward's method will be used to create the distance matrix.
    :param X: X assay matrix with samples in rows and variables in columns
    :param cat_vector: list / numpy 1-D array with booleans indicating if a variable (column position) is categorical
    :return: Linkage matrix, ready for the scipy.dendrogram function.
    """
    if len(cat_vector) < X.shape[1]:
        raise ValueError(
            "Length of cat_vector needs to be identical with variable count (columns) of X")

    if any(cat_vector):
        print('Gower distance is used')
        dist_matrix = squareform(
            gower.gower_matrix(X, cat_features=cat_vector))
        linkage_matrix = linkage(dist_matrix, method='complete')
    else:
        linkage_matrix = linkage(y=X, method='ward', metric='euclidean')

    return linkage_matrix


def find_clusters(X, linkage_matrix=None, method='HC', random_state=42, max_cluster=100):
    """
    This function iteratively tests an increasing cluster counts and calculates the samples
    silhouette scores. It uses the linkage matrix, in case hierarchical clustering was used
    beforehand. Otherwise (only for continuous data), KMeans is used to assign cluster IDs
    to each sample.
    :param X: X assay matrix with samples in rows and variables in columns
    :param linkage_matrix: Linkage matrix, obtained from the create_linkage_matrix function.
    :param method: 'HC' for hierarchical clustering and 'KMeans' for KMeans clustering.
    :param random_state: Seed used by the KMeans function. Enables reproducible clustering results.
    :param max_cluster: Integer value that indicates the maximum cluster count that shall be tested.
    :return: top_n_df dataframe that contains: n_cluster, sample name, predictions, silhouette coefficient,
             average silhouette coefficient.
    """
    if max_cluster > X.shape[0]:
        max_cluster = X.shape[0]

    if method == 'HC':
        if linkage_matrix is None:
            raise NameError(
                "The parameter linkage_matrix needs to be assigned.")

    # # Use the fcluster function to select 2 to 15 clusters (groups) and calculate the silhouette coefficient

    for n_cluster in range(2, max_cluster + 1):
        if method == 'HC':
            y_pred = fcluster(linkage_matrix, t=n_cluster,
                              criterion='maxclust') - 1

        elif method == 'KMeans':
            y_pred = KMeans(n_clusters=n_cluster,
                            random_state=random_state).fit_predict(X)
            print(y_pred)
        else:
            raise ValueError(
                "Parameter 'method' can only be 'HC' or 'KMeans'.")

        # Test if no clusters have been found and all samples have been assigned to one big cluster
        if np.count_nonzero(y_pred) == 0:
            raise ValueError("No clusters detected. Most likely, an inappropriate linkage method or distance measure"
                             "has been applied to the dataset.")

        sample_silhouette_values = silhouette_samples(X, y_pred)
        # Since maxclust can allow to have less clusters than indicated, identify the correct count of clusters

        temp_df = {'n_cluster': np.repeat(n_cluster, X.shape[0]), 'sample': X.index, 'prediction': y_pred,
                   'silhouette_coeff': sample_silhouette_values,
                   'average_silhouette_coeff': np.mean(sample_silhouette_values)}

        if n_cluster == 2:
            top_n_df = pd.DataFrame(temp_df)
        else:
            top_n_df = top_n_df.append(pd.DataFrame(temp_df))

    top_n_df = top_n_df.sort_values(
        by=['average_silhouette_coeff', 'n_cluster'], ascending=[False, True])

    # Add placing to dataframe
    top_n = np.repeat(np.arange(1, max_cluster, 1), X.shape[0])

    top_n_df['top_n'] = top_n

    return top_n_df.reset_index(drop=True)


def dendrogram_plot(linkage_matrix, y_pred, labels=None):
    """
    Function that creates the dendrogram plot. Sample names on X axis are colored by predictions
    (cluster ID).
    :param linkage_matrix: Linkage matrix, obtained from the create_linkage_matrix function.
    :param y_pred: Prediction vector that contains the cluster assignments per sample.
    :param labels: X labels, retrieved from a subset (by cluster count) of the top_n_df.sample column.
    :return: matplotlib.pyplot figure object that can be plotted or saved as graphic. Needs to be closed afterwards.
    """
    if labels is None:
        raise NameError("Parameter 'labels' is missing.")

    # The fcluster function allows differing values for the cluster count, if n_cluster (tested number
    # of clusters) is not obtainable
    cluster_count = max(y_pred + 1)
    dendrogram(linkage_matrix, color_threshold=0,
               above_threshold_color='black')

    # Create dictionary with colors for each cluster.
    label_colors = {label: cm.rainbow(
        float(y_pred[label]) / cluster_count, alpha=0.5) for label in range(len(y_pred))}

    # Plot parameter
    ax = plt.gca()
    ax.set_title('Hierarchical clustering - ' +
                 str(cluster_count) + ' groups', weight='bold')
    ax.set_ylabel('Distance', weight='bold')
    ax.set_xlabel('Sample', loc="center", weight='bold')

    # modify labels - color by cluster(grouping)
    for n, tl in enumerate(ax.get_xticklabels()):
        txt = tl.get_text()
        tl.set_bbox({'mutation_aspect': 0.15, 'ec': 'none'})
        tl.set_backgroundcolor(label_colors[int(n)])
        tl.set_text(labels[n])
    ax.set_xticklabels(labels, rotation=90)

    return


def silhouette_plot(silhouette_df):
    """
    Function that creates the silhouette plot.
    :param silhouette_df: The pd.DataFrame silhouette_df contains the
    :return: matplotlib.pyplot figure object that can be plotted or saved as graphic. Needs to be closed afterwards.
    """

    cluster_count = max(silhouette_df['cluster_id'] + 1)

    # # Silhouette coefficient plot
    fig, ax = plt.subplots()

    # Get individual colors for each cluster
    colors = [cm.rainbow(float(x) / cluster_count)
              for x in silhouette_df['cluster_id']]

    # Draw silhouette barplot
    ax.bar(range(
        silhouette_df.shape[0]), silhouette_df['silhouette_vals'], color=colors, width=1)

    # Get new xtick position that are centered underneath each cluster
    x_ticks = [np.mean(silhouette_df[silhouette_df['cluster_id'] == i]
                       ['position']) for i in range(cluster_count)]

    # print(silhouette_df)
    # X tick labels with adjusted position
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x + 1) for x in range(cluster_count)])

    # Dashed line that indicates average silhouette coefficient
    average_silhouette_coeff = np.mean(silhouette_df['silhouette_vals'])
    ax.axhline(y=average_silhouette_coeff, color='black', linestyle='--')

    # Legend that explains average silhouette coefficient
    lines = [Line2D([0], [0], color='black', linewidth=1.5, linestyle='--')]
    labels = ['Average silhouette coefficient: ' +
              str(round(average_silhouette_coeff, 3))]
    ax.legend(lines, labels, loc='upper right')

    # Force y-axis to always stop at 1 (max silhouette coefficient)
    ax.set_ylim(top=1)

    # Labels and plot title
    ax.set_title('Silhouette plot - ' + str(cluster_count) +
                 ' groups', weight='bold')
    ax.set_xlabel('Group number', weight='bold')
    ax.set_ylabel('Silhouette coefficient', weight='bold')

    return


def top_n_plot_wrapper(top_n_df, linkage_matrix=None, X=None, top_n_results=1, output=None):
    """
    Wrapper function that creates the top 3 dendrogram and silhouette plots.
    If no linkage matrix is present, no dendrogram plot is produced. It uses
    the top_n_df to determine the top 3 cluster counts, using the average silhouette coefficient.
    Since the cluster_count might deviate from the tested n_cluster value due to the fcluster function,
    the top_n cluster can be repetitive (ie. 3 clusters found for n_cluster = 3, 4, 5 and 6.
    But at n_cluster = 7, 7 clusters were found. Therefore ignore the repetitive solutions and only return
    the top_n unique solutions. This list might be shorter than top_n_results.

    :param top_n_results: Return top 'n' results.
    :param top_n_df: Pandas dataframe that contains: n_cluster, sample name, predictions, silhouette coefficient,
                     average silhouette coefficient.
    :param linkage_matrix: Linkage matrix, obtained from the create_linkage_matrix function.
    :param output: Relative path, where the plots shall be saved in individual subfolders.
    :return: New top n subfolders that contain the dendrogram and silhouette plots.
    """

    top_n_results = abs(int(top_n_results))

    # Find top_n unique average silhouette coefficients, sorted from high to low value
    unique_avg_silh = pd.unique(top_n_df['average_silhouette_coeff'])

    unique_top_n = [top_n_df[top_n_df['average_silhouette_coeff']
                             == x]['top_n'].values[0] for x in unique_avg_silh]

    if len(unique_top_n) < top_n_results:
        top_n_results = len(unique_top_n)
        print('Only the top ' + str(len(unique_top_n)) + ' subclusters are shown.')

    output = "subclusters"
    os.makedirs(os.path.join(output), exist_ok=True)

    for n, i in enumerate(unique_top_n[:top_n_results]):

        temp_df = top_n_df[top_n_df['top_n'] == i]
        temp_df = temp_df.sort_values(by='prediction', ascending=True)

        y_pred = temp_df['prediction'].values
        labels = temp_df['sample'].values

        # # TODO: This line creates the new folder. For the UNIX system, this has to be changed.
        # os.mkdir(os.path.join(output, "nr" + str(n + 1) + "_silhouette_coeff"))

        # # # In case a linkage_matrix exists, plot the dendrogram
        # if linkage_matrix is not None:
        #     dendrogram_plot(linkage_matrix, y_pred, labels)
        #     # # TODO: This line creates the path to place the figures in. For the UNIX system, this has to be changed.
        #     # outfile_silhouette = os.path.join(output, "nr" + str(n + 1) + "_silhouette_coeff\\dendrogram_plot_nr"
        #     #   + str(n + 1) + ".svg")
        #     outfile_silhouette = os.path.join(output, "dendrogram_plot.svg")
        #     plt.savefig(outfile_silhouette, bbox_inches='tight')
        #     # plt.show()
        #     plt.close()

        # # # Prepare silhouette plot
        # silhouette_coeff = temp_df['silhouette_coeff']
        # silhouette_df = pd.DataFrame(
        #     {'silhouette_vals': silhouette_coeff, 'cluster_id': y_pred})
        # silhouette_df = silhouette_df.sort_values(
        #     by=['cluster_id', 'silhouette_vals'], ascending=[True, False])
        # silhouette_df['position'] = [x for x in range(silhouette_df.shape[0])]

        # silhouette_plot(silhouette_df)
        # # # TODO: This line creates the path to place the figures in. For the UNIX system, this has to be changed.
        # # outfile_silhouette = os.path.join(output, "nr" + str(n + 1) + "_silhouette_coeff\\silhouette_plot_nr"
        # #   + str(n + 1) + ".svg")
        # outfile_silhouette = os.path.join(output, "silhouette_plot.svg")

        # plt.savefig(outfile_silhouette, bbox_inches='tight')
        # # plt.show()
        # plt.close()

        # # Create subset X (assay data) dataframes as CSVs. The X dataframe has been split by top_n cluster count
        # # (for folder creation)
        if X is not None:
            for m in range(np.max(temp_df['prediction']) + 1):
                # Retrieve sample names per cluster
                samples = temp_df[temp_df['prediction']
                                  == m]['sample'].tolist()
                X_subset = X.loc[samples, :]
                # TODO: These 2 lines create the path to place the figures in. For the UNIX system,
                #  this has to be changed.
                os.makedirs(os.path.join(output, str(m + 1)), exist_ok=True)
                # outfile_X_subset = os.path.join(output, str(m+1), "Activity.csv")
                X_subset = X_subset.rename_axis('Sample').reset_index()
                result = X_subset.to_json(orient="records", index=True)
                parsed = json.loads(result)

                # write file as output
                f = open(output + "/" + str(m+1) + "/activity.json", "w")
                f.write(json.dumps(parsed, indent=2))
                f.close()
                # X_subset.to_csv(outfile_X_subset)

    return


def run(activityFile, outdir):

    X = pd.read_csv(activityFile, index_col=0)
    cat_vector = np.repeat(False, X.shape[1])
    X_prepared, cluster_method = initialize_clustering_method(X, cat_vector, method='standardize')

    # # If Hierarchical clustering shall be performed, produce a linkage matrix
    lm = create_linkage_matrix(X_prepared, cat_vector)

    # # # Find the top 3 clustering solutions, using the silhouette coefficient as a cut-off criterion
    top_n_df = find_clusters(X=X_prepared, linkage_matrix=lm, method=cluster_method)

    
    # change directory to the jobid to output files
    os.chdir(outdir)
    
    # only include the top df
    top_cluster_df = top_n_df[(top_n_df['top_n'] == 1)]
    top_cluster_df.to_csv("subcluster_info.csv")

    # # Retrieve dendrogram and silhouette plots - only pick the best one for now
    top_n_plot_wrapper(top_n_df, lm, X, top_n_results=1)

    cluster_size = top_cluster_df['n_cluster'][0]
    cluster_array = [[] for i in range(cluster_size)]

    # create a dictionary with samples to cluster number
    for index, row in top_cluster_df.iterrows():
        cluster_num = row['prediction']
        sample = row['sample']
        cluster_array[cluster_num].append(sample)

    G = nx.read_graphml('network.graphml')      # full graph 
    table = pd.read_csv('table.csv')            # full table
    output = "subclusters"

    # traverse through each cluster and write the network and table files
    for i in range(cluster_size):
        # outfile_gml = open(output + "/" + str(i+1) + "/network.graphml", "w")  # not working - double check later
        basketids = []
        H = G.edge_subgraph(G.edges(cluster_array[i])).copy()

        # store all the basketids which are integers - probably not the best way to do this
        for node in H.nodes():
            try:
                if int(node): 
                    basketids.append(node)
            except:
                pass

        # nx.write_graphml(H, outfile_gml, prettyprint=True)        # not working double-check later

        # output network.json file
        jsonData = json_graph.node_link_data(H)
        outfile_json = open(output + "/" + str(i+1) + "/network.json", "w")
        outfile_json.write(json.dumps(jsonData, indent=2))
        outfile_json.close()

        # filter the table dataframe for these basketids to create a subtable
        output_table = table[table['BasketID'].isin(basketids)]
        outfile_csv = open(output + "/" + str(i+1) + "/table.csv", "w")
        output_table.to_csv(outfile_csv)

    return int(cluster_size)

if __name__ == '__main__':

    print ("[createCluster.py] Main")

    # filename = sys.argv[1]
    # print("Building clusters ...")
    # run("Activity.csv")

    # table_df = pd.read_csv("table.csv")
    # G = nx.read_graphml("network.graphml")

    # create subsets of tables, based on cluster number and place them in the correct folder as "table.csv"

    # X = pd.read_csv(filename, index_col=0)
    # cat_vector = np.repeat(False, X.shape[1])

    # # Make_blobs environment -- For simulation of clusters
    # n_samples = 15
    # n_features = 60
    # random_state = 42

    # X, y = make_blobs(centers=5, n_samples=n_samples, n_features=n_features, random_state=random_state)

    # X = pd.DataFrame(X, index=['Sample_' + str(i + 1) for i in range(X.shape[0])])

    # # # Test dataset for Gower's distance
    # Xd = {'age': [21, 21, 19, 30, 21, 21, 19, 30, 21, 19, 30, 21, 21, 19, 30],
    #       'gender': [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #       'civil_status': [0, 1, 1, 1, 0, 1, 2, 3, 1, 1, 1, 0, 1, 2, 3],
    #       'salary': [3000.0, 1200.0, 32000.0, 1800.0, 2900.0, 1100.0, 10000.0, 1500.0,
    #                  1200.0, 32000.0, 1800.0, 2900.0, 1100.0, 10000.0, 1500.0],
    #       'has_children': [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
    #       'available_credit': [2200, 100, 22000, 1100, 2000, 100, 6000, 2200, 100, 22000, 1100, 2000, 100, 6000, 2200]
    #     ,  'sample_names': ['sample1', 'sample2', 'sample3', 'sample4',
    #                        'sample5', 'sample6', 'sample7', 'sample8',
    #                        'sample9', 'sample10', 'sample11',
    #                        'sample12', 'sample13', 'sample14', 'sample15']
    #       }
    #
    # # # Use Pandas dataframe as input
    # X = pd.DataFrame(Xd)
    #
    # X = X.set_index('sample_names')
    # cat_vector = [False, True, True, False, True, False]

    # # # BioMAP dataset (Sanghoon), 925 samples, trinary (0 - inactive, 0.5 - mildly active, 1 - active)
    # X = pd.read_csv('BioMAP_activity.csv', index_col=0)
    # cat_vector = np.repeat(True, X.shape[1])
    #
    # # # Michael HIFAN2 RLUS Legacy Screening data
    #
    # X = pd.read_csv('HIFAN2_RLUS_ScreenData_Worksheet_MR.csv', index_col=0)
    #
    # X = X.drop(columns=['Trypanosoma cruzi', 'Trypanosoma brucei', 'Leishmania donovani'])
    # cat_vector = np.repeat(False, X.shape[1])
    # cat_vector[np.array([19, 20, 21])] = True

    # # This vector contains booleans that indicate if a variable (column) in X is of categorical type
    # #  For BioMAP data, all variables are categorical
    # cat_vector = np.repeat(True, X.shape[1])

    # # # Initialize clustering by determining, if data needs to be scaled (continuous data only) and what clustering
    # # # is used
    # X_prepared, cluster_method = initialize_clustering_method(X, cat_vector, method='standardize')

    # # # If Hierarchical clustering shall be performed, produce a linkage matrix
    # lm = create_linkage_matrix(X_prepared, cat_vector)
    # #
    # # # # Find the top 3 clustering solutions, using the silhouette coefficient as a cut-off criterion
    # top_n_df = find_clusters(X=X_prepared, linkage_matrix=lm, method=cluster_method)

    # # only include the top df
    # top_cluster_df = top_n_df[(top_n_df['top_n'] == 1)]
    # top_cluster_df.to_csv("top_cluster.csv")

    # # top_n_df.to_csv("top_n_dataframe.csv")

    # # # Retrieve dendrogram and silhouette plots - only pick the best one for now
    # top_n_plot_wrapper(top_n_df, lm, X, top_n_results=1, output=os.getcwd())
