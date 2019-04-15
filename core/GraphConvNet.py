import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

from core.GraphConvNetCell import GraphConvNetCell
from util.training_utils import get_torch_dtype


dtypeFloat, dtypeLong = get_torch_dtype()


class GraphConvNet(nn.Module):
    """
    PyTorch implementation of Residual Gated Graph ConvNets
    Adapted from An Experimental Study of Neural Networks for Variable Graphs (ICLR'18)
    Xavier Bresson and Thomas Laurent
    See: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, net_parameters):

        super(GraphConvNet, self).__init__()

        self.name = 'graph_net'

        # parameters
        D = net_parameters['D']
        n_components = net_parameters['n_components']
        H = net_parameters['H']
        L = net_parameters['L']

        # vector of hidden dimensions
        net_layers = []
        for layer in range(L):
            net_layers.append(H)

        # CL cells
        # NOTE: Each graph convnet cell uses *TWO* convolutional operations
        net_layers_extended = [D] + net_layers  # include embedding dim
        L = len(net_layers)
        list_of_gnn_cells = []  # list of NN cells
        for layer in range(L // 2):
            Hin, Hout = net_layers_extended[2 * layer], net_layers_extended[2 * layer + 2]
            list_of_gnn_cells.append(GraphConvNetCell(Hin, Hout))

        # register the cells for pytorch
        self.gnn_cells = nn.ModuleList(list_of_gnn_cells)

        # fc
        Hfinal = net_layers_extended[-1]
        self.fc = nn.Linear(Hfinal, n_components)

        # init
        self.init_weights_Graph_OurConvNet(Hfinal, n_components, 1)

        # print('\nnb of hidden layers=', L)
        # print('dim of layers (w/ embed dim)=', net_layers_extended)
        # print('\n')

        # class variables
        self.L = L
        self.net_layers_extended = net_layers_extended

    def init_weights_Graph_OurConvNet(self, Fin_fc, Fout_fc, gain):

        scale = gain * np.sqrt(2.0 / Fin_fc)
        self.fc.weight.data.uniform_(-scale, scale)
        self.fc.bias.data.fill_(0)

    def forward(self, G):
        # Data matrix
        x = G.inputs

        # Unroll into single vector
        x = x.view(x.shape[0], -1)

        # Pass raw data matrix X directly as input
        x = Variable(torch.FloatTensor(x).type(dtypeFloat), requires_grad=False)

        # graph operators
        # Edge = start vertex to end vertex
        # E_start = E x V mapping matrix from edge index to corresponding start vertex
        # E_end = E x V mapping matrix from edge index to corresponding end vertex
        E_start = G.edge_to_starting_vertex
        E_end = G.edge_to_ending_vertex
        E_start = torch.from_numpy(E_start.toarray()).type(dtypeFloat)
        E_end = torch.from_numpy(E_end.toarray()).type(dtypeFloat)
        E_start = Variable(E_start, requires_grad=False)
        E_end = Variable(E_end, requires_grad=False)

        for layer in range(self.L // 2):
            gnn_layer = self.gnn_cells[layer]
            x = gnn_layer(x, E_start, E_end)  # V x Hfinal

        # FC
        x = self.fc(x)

        return x

    def loss(self, y, y_target):
        loss = nn.MSELoss()(y, y_target) # L2 loss
        return loss

    def pairwise_loss(self, y, y_target, W):
        distances_1 = y_target[W.row, :] - y_target[W.col, :]
        distances_2 = y[W.row, :] - y[W.col, :]
        loss = torch.mean(torch.pow(distances_1.norm(dim=1) - distances_2.norm(dim=1), 2))

        return loss

    def update(self, lr):
        update = torch.optim.Adam(self.parameters(), lr=lr)
        return update

    def update_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    def nb_param(self):
        return self.nb_param
