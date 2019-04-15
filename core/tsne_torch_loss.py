import numpy as np
import torch
from time import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold.t_sne import _joint_probabilities, _joint_probabilities_nn
from scipy.spatial.distance import squareform

from util.graph_utils import get_shortest_path_matrix
from util.training_utils import get_torch_dtype


dtypeFloat, dtypeLong = get_torch_dtype()


def compute_joint_probabilities(X, perplexity=30, metric='euclidean', method='exact', adj=None, verbose=0):
    """
    Computes the joint probability matrix P from a feature matrix X of size n x f
    Adapted from sklearn.manifold.t_sne
    """

    # Compute pairwise distances
    if verbose > 0: print('Computing pairwise distances...')

    if method == 'exact':
        if metric == 'precomputed':
            D = X
        elif metric == 'euclidean':
            D = pairwise_distances(X, metric=metric, squared=True)
        elif metric == 'cosine':
            D = pairwise_distances(X, metric=metric)
        elif metric == 'shortest_path':
            assert adj is not None
            D = get_shortest_path_matrix(adj, verbose=verbose)

        P = _joint_probabilities(D, desired_perplexity=perplexity, verbose=verbose)
        assert np.all(np.isfinite(P)), "All probabilities should be finite"
        assert np.all(P >= 0), "All probabilities should be non-negative"
        assert np.all(P <= 1), ("All probabilities should be less "
                                "or then equal to one")

        P = squareform(P)

    else:
        # Cpmpute the number of nearest neighbors to find.
        # LvdM uses 3 * perplexity as the number of neighbors.
        # In the event that we have very small # of points
        # set the neighbors to n - 1.
        n_samples = X.shape[0]
        k = min(n_samples - 1, int(3. * perplexity + 1))

        # Find the nearest neighbors for every point
        knn = NearestNeighbors(algorithm='auto', n_neighbors=k,
                               metric=metric)
        t0 = time()
        knn.fit(X)
        duration = time() - t0
        if verbose:
            print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
                n_samples, duration))

        t0 = time()
        distances_nn, neighbors_nn = knn.kneighbors(
            None, n_neighbors=k)
        duration = time() - t0
        if verbose:
            print("[t-SNE] Computed neighbors for {} samples in {:.3f}s..."
                  .format(n_samples, duration))

        # Free the memory used by the ball_tree
        del knn

        if metric == "euclidean":
            # knn return the euclidean distance but we need it squared
            # to be consistent with the 'exact' method. Note that the
            # the method was derived using the euclidean method as in the
            # input space. Not sure of the implication of using a different
            # metric.
            distances_nn **= 2

        # compute the joint probability distribution for the input space
        P = _joint_probabilities_nn(distances_nn, neighbors_nn,
                                    perplexity, verbose)
        P = P.toarray()

    # Convert to torch tensor
    P = torch.from_numpy(P).type(dtypeFloat)

    return P


def tsne_torch_loss(P, y_emb):
    """
    Computes the t-SNE loss, i.e. KL(P||Q). Torch implementation allows from auto-grad.
    Args:
        P (np.array): joint probabilities matrix of size n x n
        y_emb (np.array): low dimensional map of data points, matrix of size n x 2
    Returns:
        C (float): t-SNE loss
    """

    d = 2
    n = P.shape[1]
    v = d - 1.  # degrees of freedom
    eps = 10e-15  # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)

    # Euclidean pairwise distances in the low-dimensional map
    sum_act = torch.sum(y_emb.pow(2), dim=1)
    Q = sum_act + torch.reshape(sum_act, [-1, 1]) + -2 * torch.mm(y_emb, torch.t(y_emb))

    Q = Q / v
    Q = torch.pow(1 + Q, -(v + 1) / 2)
    Q *= 1 - torch.eye(n).type(dtypeFloat)
    Q /= torch.sum(Q)
    Q = torch.clamp(Q, min=eps)
    C = torch.log((P + eps) / (Q + eps))
    C = torch.sum(P * C)
    return C
