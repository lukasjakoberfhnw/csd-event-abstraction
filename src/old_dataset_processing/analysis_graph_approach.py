import os
import pm4py
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')
tale_XES_file_location = os.path.join(data_path, 'tale-camerino', 'MRS.xes')

def main():
    log = pm4py.read_xes(tale_XES_file_location)
    log_df = pm4py.convert_to_dataframe(log)

    # 'time:timestamp', 'concept:name'
    processing_df = log_df[['x', 'y', 'z', 'robot', 'payload']].iloc[0:500].copy()

    attributes = processing_df[['x', 'y', 'z']].values
    similarity_matrix = -euclidean_distances(attributes)  # Negative distances for similarity
    print(similarity_matrix)
    threshold = -1
    edge_indices = np.argwhere(similarity_matrix > threshold)
    # show edge indices
    print(edge_indices[-20:])
    print(edge_indices.shape)

    G = nx.DiGraph()

    # add nodes
    for index, row in processing_df.iterrows():
        G.add_node(index, **row.to_dict())

    # # add robot types
    robot_types = processing_df['robot'].unique()
    for robot_type in robot_types:
        G.add_node(robot_type, type='robot')

    # add edges
    for i in range(0, len(processing_df) - 1):
        G.add_edge(i, i+1, edge_type="sequential", weight=1)

    # add edges based on who performs the action/sends the message
    for index, row in processing_df.iterrows():
        G.add_edge(index, row['robot'], edge_type='robot_connection', weight=5)  # Connect event to robot

    # # currently doesn't consider the actual values... just the connections of the nodes
    for edge in edge_indices:
        G.add_edge(edge[0], edge[1], edge_type='similarity', weight=similarity_matrix[edge[0], edge[1]])

    # print number of edges in network
    print(G.number_of_edges())

    # show graph
    # nx.draw(G, with_labels=True)
    # plt.show()

    # Create adjacency matrix
    adj_matrix = nx.to_numpy_array(G, nodelist=G.nodes(), weight='weight')
    print(adj_matrix)

    # check if there are NaN values in the matrix
    print(np.isnan(adj_matrix).any())
    print(np.isinf(adj_matrix).any())
    print(adj_matrix.shape)

    # Ensure no isolated nodes
    row_sums = np.sum(adj_matrix, axis=1)
    isolated_nodes = np.where(row_sums == 0)[0]
    if len(isolated_nodes) > 0:
        print(f"Warning: Found {len(isolated_nodes)} isolated nodes.")
        adj_matrix += np.eye(adj_matrix.shape[0]) * 1e-5  # Add small values to diagonal

    # postprocessing matrix
    np.fill_diagonal(adj_matrix, 0)  # No self-loops
    # Row-normalization for directed graphs
    row_sums = adj_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_matrix = adj_matrix / row_sums[:, np.newaxis]

    normalized_matrix = (normalized_matrix + normalized_matrix.T) / 2  # Symmetrize matrix

    if np.isnan(normalized_matrix).any() or np.isinf(normalized_matrix).any():
        print("Matrix still contains invalid values!")
    else:
        print("Matrix is clean!")




    # # Apply spectral clustering
    clustering = SpectralClustering(n_clusters=6, affinity='precomputed', random_state=0)
    labels = clustering.fit_predict(normalized_matrix)

    # # node_to_cluster = {node: label for node, label in zip(G.nodes(), labels)}

    # # Assign cluster labels to nodes
    # for node, label in zip(G.nodes(), labels):
    #     G.nodes[node]['cluster'] = label

    # # Visualize clusters
    # pos = nx.spring_layout(G)
    # colors = [G.nodes[node]['cluster'] for node in G.nodes()]
    # nx.draw(G, pos, node_color=colors, with_labels=True, node_size=3000, cmap=plt.cm.Set1)
    # plt.show()

    # processing_df['cluster'] = labels[:500]
    # processing_df['concept:name'] = log_df['concept:name'].iloc[0:500]
    # print(processing_df.iloc[0:50])

    # processing_df.to_csv(os.path.join(output_path, 'similarity_cluster_spectral_graphs.csv'))

if __name__ == '__main__':
    main()