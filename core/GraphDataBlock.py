import numpy as np
import scipy.sparse as sp
from sklearn import manifold
import torch


class GraphDataBlock(object):
    """
    Attributes:
        edge_to_starting_vertex (scipy coo matrix): matrix of size E x V mapping edges to start vertices
        edge_to_ending_vertex (scipy coo matrix): matrix of size E x V mapping edges to end vertices
        inputs (np.arrray): data feature matrix of size n x f
        labels (np,array): data class label matrix of size n x 1
        adj_matrix (scipy coo matrix): adjacency matrix of size n x n
        original indices (np.array): indices of data points within the original dataset
        precomputed_path_matrix (scipy csr matrix)
    """
    def __init__(self, X, labels, W=None):
        # Convert to torch if numpy array
        if sp.issparse(X):
            X = X.toarray()
        if type(X) is np.ndarray:
            X = torch.from_numpy(X).type(torch.FloatTensor)

        # Get affinity matrix
        if W is None:
            embedder = manifold.SpectralEmbedding(n_components=2)
            W = embedder._get_affinity_matrix(X)

        W = sp.coo_matrix(W)  # sparse matrix

        # Get edge information
        nb_edges = W.nnz
        nb_vertices = W.shape[0]
        self.edge_to_starting_vertex = sp.coo_matrix((np.ones(nb_edges), (np.arange(nb_edges), W.row)),
                                                     shape=(nb_edges, nb_vertices))
        self.edge_to_ending_vertex = sp.coo_matrix((np.ones(nb_edges), (np.arange(nb_edges), W.col)),
                                                   shape=(nb_edges, nb_vertices))

        # Save as attributes
        self.inputs = X             # data matrix
        self.labels = labels        # labels
        self.adj_matrix = W         # affinity matrix

        # Placeholders
        self.original_indices = None
        self.precomputed_path_matrix = None

