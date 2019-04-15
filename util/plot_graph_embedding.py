import numpy as np
import scipy.sparse as sp
from bokeh.palettes import Category20_20, Category20b_20, Accent8
from matplotlib import collections  as mc
import matplotlib.pyplot as plt


def plot_graph_embedding(y_emb, labels, adj, line_alpha=0.2, s=7, title=""):
    """
    Plots the visualization of graph-structured data
    Args:
        y_emb (np.array): low dimensional map of data points, matrix of size n x 2
        labels (np.array): underlying class labels, matrix of size n x 1
        adj (scipy csr matrix): adjacency matrix
    """
    labels = np.array([int(l) for l in labels])
    adj = sp.coo_matrix(adj)

    colormap = np.array(Category20_20 + Category20b_20 + Accent8)

    f, ax = plt.subplots(1, sharex='col', figsize=(6, 4), dpi=800)

    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)

    # Plot edges
    p0 = y_emb[adj.row, :]
    p1 = y_emb[adj.col, :]
    p_0 = [tuple(row) for row in p0]
    p_1 = [tuple(row) for row in p1]

    classA = labels[adj.row]
    classB = labels[adj.col]
    mask = classA == classB
    edge_colormask = mask * (classA + 1) - 1

    lines = list(zip(p_0, p_1))
    lc = mc.LineCollection(lines, linewidths=0.5, colors=colormap[edge_colormask])
    lc.set_alpha(line_alpha)
    ax.add_collection(lc)

    ax.scatter(y_emb[:, 0], y_emb[:, 1], s=s, c=colormap[labels])

    ax.margins(0.1, 0.1)
    plt.autoscale(tight=True)
    plt.tight_layout()