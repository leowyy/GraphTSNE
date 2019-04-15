import numpy as np
import time
from scipy.sparse.csgraph import shortest_path

MAX_DISTANCE = 1e6


def get_shortest_path_matrix(adj, verbose=0):
    """
    Computes the all-pairs shortest path matrix for an undirected and unweighted graph
    If a pair of nodes are not connected, places an arbitrarily large value
    """
    if verbose:
        print("Computing all pairs shortest path lengths for {} nodes...".format(adj.shape[0]))
    t_start = time.time()

    path_lengths_matrix = shortest_path(adj, directed=False, unweighted=True)
    path_lengths_matrix[path_lengths_matrix == np.inf] = MAX_DISTANCE

    t_elapsed = time.time() - t_start
    if verbose:
        print("Time to compute shortest paths (s) = {:.4f}".format(t_elapsed))
    return path_lengths_matrix


def neighbor_sampling(adj, minibatch_indices, D_layers):
    """
    Performs neighbor sampling scheme proposed by Hamilton et al (2017)
    Args:
        adj (scipy csr matrix): adjacency matrix of the COMPLETE graph
        minibatch_indices (list or np.array): indices of initial sample
        D_layers (list): maximum number of neighbors sampled per layer,
                    if D_layers[l] = -1, sample all neighbors at layer l
    Returns:
        expanded indices (np.array): indices of expanded sample
    """
    selected_indices = list(minibatch_indices)
    for i in minibatch_indices:
        one_hop_neighbors = adj[i].nonzero()[1]
        if D_layers[0] != -1 and len(one_hop_neighbors) > D_layers[0]:
            one_hop_neighbors = np.random.choice(one_hop_neighbors, size=D_layers[0], replace=False)

        selected_indices += list(one_hop_neighbors)

        for j in one_hop_neighbors:
            two_hop_neighbors = adj[j].nonzero()[1]
            if  D_layers[1] != -1 and len(two_hop_neighbors) > D_layers[1]:
                two_hop_neighbors = np.random.choice(two_hop_neighbors, size=D_layers[1], replace=False)

            selected_indices += list(two_hop_neighbors)
    return np.unique(np.array(selected_indices))
