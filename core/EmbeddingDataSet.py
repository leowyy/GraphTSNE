import os
import pickle
import numpy as np
import scipy.sparse as sp
import time
from core.GraphDataBlock import GraphDataBlock
from util.graph_utils import neighbor_sampling


class EmbeddingDataSet():
    """
    Attributes:
        name (str): name of dataset
        data_dir (str): path to dataset folder
        train_dir (str): path to training data file
        test_dir (str): path to test data file
        input_dim (int): number of data features per node
        is_labelled (Boolean): whether underlying class labels are present
        all_data (list[GraphDataBlock]): data inputs packaged into blocks
        all_indices (np.array): input sequence when packaging data into blocks

        inputs (scipy csr matrix): data feature matrix of size n x f
        labels (np.array): data class label matrix of size n x 1
        adj_matrix (scipy csr matrix): adjacency matrix of size n x n
    """

    train_dir = {'cora': 'cora_full.pkl'}

    test_dir = train_dir

    def __init__(self, name, data_dir, train=True):
        self.name = name
        self.data_dir = data_dir
        self.train_dir = EmbeddingDataSet.train_dir[name]
        self.test_dir = EmbeddingDataSet.test_dir[name]
        self.is_labelled = False

        self.all_data = []

        # Extract data from file contents
        data_root = os.path.join(self.data_dir, self.name)
        if train:
            fname = os.path.join(data_root, self.train_dir)
        else:
            assert self.test_dir is not None
            fname = os.path.join(data_root, self.test_dir)
        with open(fname, 'rb') as f:
            file_contents = pickle.load(f)

        self.inputs = file_contents[0]
        self.labels = file_contents[1]
        self.adj_matrix = file_contents[2]

        self.is_labelled = len(self.labels) != 0
        self.input_dim = self.inputs.shape[1]

        self.all_indices = np.arange(0, self.inputs.shape[0])

        # Convert adj to csr matrix
        self.inputs = sp.csr_matrix(self.inputs)
        self.adj_matrix = sp.csr_matrix(self.adj_matrix)

    def create_all_data(self, n_batches=1, shuffle=False, sampling=False, full_path_matrix=None):
        """
        Initialises all_data as a list of GraphDataBlock
        Args:
            n_batches (int): number of blocks to return
            shuffle (Boolean): whether to shuffle input sequence
            sampling (Boolean): whether to expand data blocks with neighbor sampling
        """
        i = 0
        labels_subset = []
        self.all_data = []

        if shuffle:
            np.random.shuffle(self.all_indices)
        else:
            self.all_indices = np.arange(0, self.inputs.shape[0])

        # Split equally
        # TODO: Another option to split randomly
        chunk_sizes = self.get_k_equal_chunks(self.inputs.shape[0], k=n_batches)

        t_start = time.time()

        for num_samples in chunk_sizes:
            mask = sorted(self.all_indices[i: i + num_samples])

            # Perform sampling to obtain local neighborhood of mini-batch
            if sampling:
                D_layers = [9, 14]  # max samples per layer
                mask = neighbor_sampling(self.adj_matrix, mask, D_layers)

            inputs_subset = self.inputs[mask]
            adj_subset = self.adj_matrix[mask, :][:, mask]

            if self.is_labelled:
                labels_subset = self.labels[mask]

            # Package data into graph block
            G = GraphDataBlock(inputs_subset, labels=labels_subset, W=adj_subset)

            # Add original indices from the complete dataset
            G.original_indices = mask

            # Add shortest path matrix from precomputed data if needed
            if full_path_matrix is not None:
                G.precomputed_path_matrix = full_path_matrix[mask, :][:, mask]

            self.all_data.append(G)
            i += num_samples

        t_elapsed = time.time() - t_start
        print('Data blocks of length: ', [len(G.labels) for G in self.all_data])
        print("Time to create all data (s) = {:.4f}".format(t_elapsed))

    def summarise(self):
        print("Name of dataset = {}".format(self.name))
        print("Input dimension = {}".format(self.input_dim))
        print("Number of training samples = {}".format(self.inputs.shape[0]))
        print("Training labels = {}".format(self.is_labelled))

    def get_k_equal_chunks(self, n, k):
        # returns n % k sub-arrays of size n//k + 1 and the rest of size n//k
        p, r = divmod(n, k)
        return [p + 1 for _ in range(r)] + [p for _ in range(k - r)]

    def get_current_inputs(self):
        inputs = self.inputs[self.all_indices]
        labels = self.labels[self.all_indices]
        adj = self.adj_matrix[self.all_indices, :][:, self.all_indices]
        return inputs, labels, adj

    def get_sample_block(self, n_initial, sample_neighbors, verbose=0):
        """
        Returns a subset of data as a GraphDataBlock
        Args:
            n_initial (int): number of samples at the start
            sample_neighbors (Boolean): whether to expand the sample block with neighbor sampling
        Returns:
            G (GraphDataBlock): data subset
        """

        mask = sorted(np.random.choice(self.all_indices, size=n_initial, replace=False))
        if sample_neighbors:
            mask = neighbor_sampling(self.adj_matrix, mask, D_layers=[9, 14])
        inputs = self.inputs[mask]
        labels = self.labels[mask]
        W = self.adj_matrix[mask, :][:, mask]
        G = GraphDataBlock(inputs, labels, W)
        G.original_indices = mask
        if verbose:
            print("Initial set of {} points was expanded to {} points".format(n_initial, len(mask)))
        return G
