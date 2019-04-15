import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class GraphConvNetCell(nn.Module):
    """
    PyTorch implementation of Residual Gated Graph ConvNets
    Adapted from An Experimental Study of Neural Networks for Variable Graphs (ICLR'18)
    Xavier Bresson and Thomas Laurent
    See: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, dim_in, dim_out):
        super(GraphConvNetCell, self).__init__()

        # conv1
        self.Ui1 = nn.Linear(dim_in, dim_out, bias=False)
        self.Uj1 = nn.Linear(dim_in, dim_out, bias=False)
        self.Vi1 = nn.Linear(dim_in, dim_out, bias=False)
        self.Vj1 = nn.Linear(dim_in, dim_out, bias=False)
        self.bu1 = torch.nn.Parameter(torch.FloatTensor(dim_out), requires_grad=True)
        self.bv1 = torch.nn.Parameter(torch.FloatTensor(dim_out), requires_grad=True)

        # conv2
        self.Ui2 = nn.Linear(dim_out, dim_out, bias=False)
        self.Uj2 = nn.Linear(dim_out, dim_out, bias=False)
        self.Vi2 = nn.Linear(dim_out, dim_out, bias=False)
        self.Vj2 = nn.Linear(dim_out, dim_out, bias=False)
        self.bu2 = torch.nn.Parameter(torch.FloatTensor(dim_out), requires_grad=True)
        self.bv2 = torch.nn.Parameter(torch.FloatTensor(dim_out), requires_grad=True)

        # bn1, bn2
        self.bn1 = torch.nn.BatchNorm1d(dim_out)
        self.bn2 = torch.nn.BatchNorm1d(dim_out)

        # resnet
        self.R = nn.Linear(dim_in, dim_out, bias=False)

        # init
        self.init_weights_OurConvNetCell(dim_in, dim_out, 1)

    def init_weights_OurConvNetCell(self, dim_in, dim_out, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.Ui1.weight.data.uniform_(-scale, scale)
        self.Uj1.weight.data.uniform_(-scale, scale)
        self.Vi1.weight.data.uniform_(-scale, scale)
        self.Vj1.weight.data.uniform_(-scale, scale)
        scale = gain * np.sqrt(2.0 / dim_out)
        self.bu1.data.fill_(0)
        self.bv1.data.fill_(0)

        # conv2
        scale = gain * np.sqrt(2.0 / dim_out)
        self.Ui2.weight.data.uniform_(-scale, scale)
        self.Uj2.weight.data.uniform_(-scale, scale)
        self.Vi2.weight.data.uniform_(-scale, scale)
        self.Vj2.weight.data.uniform_(-scale, scale)
        scale = gain * np.sqrt(2.0 / dim_out)
        self.bu2.data.fill_(0)
        self.bv2.data.fill_(0)

        # RN
        scale = gain * np.sqrt(2.0 / dim_in)
        self.R.weight.data.uniform_(-scale, scale)

    def forward(self, x, E_start, E_end):
        # E_start, E_end : E x V

        xin = x
        # conv1
        Vix = self.Vi1(x)  # V x H_out
        Vjx = self.Vj1(x)  # V x H_out
        x1 = torch.mm(E_end, Vix) + torch.mm(E_start, Vjx) + self.bv1  # E x H_out, edge gates
        x1 = torch.sigmoid(x1)
        Ujx = self.Uj1(x)  # V x H_out
        x2 = torch.mm(E_start, Ujx)  # V x H_out   
        Uix = self.Ui1(x)  # V x H_out
        # x = Uix + torch.mm(E_end.t(), x1 * x2) + self.bu1  # V x H_out
        indegree = torch.sum(E_end, dim=0) # V
        indegree[indegree==0] = 1
        sum_xj = torch.div(torch.mm(E_end.t(), x1 * x2).t(), indegree).t()
        x = Uix + sum_xj + self.bu1

        # bn1
        x = self.bn1(x)
        # relu1
        x = F.relu(x)
        # conv2
        Vix = self.Vi2(x)  # V x H_out
        Vjx = self.Vj2(x)  # V x H_out
        x1 = torch.mm(E_end, Vix) + torch.mm(E_start, Vjx) + self.bv2  # E x H_out, edge gates
        x1 = torch.sigmoid(x1)
        Ujx = self.Uj2(x)  # V x H_out
        x2 = torch.mm(E_start, Ujx)  # V x H_out
        Uix = self.Ui2(x)  # V x H_out
        sum_xj = torch.div(torch.mm(E_end.t(), x1 * x2).t(), indegree).t()
        x = Uix + sum_xj + self.bu2  # V x H_out
        # bn2
        x = self.bn2(x)
        # addition
        x = x + self.R(xin)
        # relu2
        x = F.relu(x)

        return x
