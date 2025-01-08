import os
from .Graph_construction import *
from .Feature_extractor import *
from functools import wraps
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np

class GCNLayer(nn.Module):

    def __init__(self, in_ft, out_ft, act='prelu', bias=True):

        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else nn.ReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        adj = adj.unsqueeze(0).expand(seq_fts.size(0), -1, -1)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)

class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

def laplacian(W):
    N, N = W.shape
    W = W+torch.eye(N).to(W.device)
    D = W.sum(axis=1)
    D = torch.diag(D**(-0.5))
    out = D@W@D

    return out

def get_A(adj):
    # diag = torch.diag(adj)
    # a_diag = torch.diag_embed(diag)
    # adj = adj - a_diag

    return adj
def get_A2(adj):
    # one = torch.ones_like(adj)
    bs, N, dimen = adj.size()
    A2 = torch.matmul(adj, adj)
    # A2 = A2 + adj
    # diag = torch.diag(A2)
    # a_diag = torch.diag_embed(diag)
    # A2 = A2 - a_diag
    # eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()
    # eyes_like_inf = eyes_like*1e8
    # A2 = F.leaky_relu(A2-eyes_like_inf)
    # A2 = F.softmax(A2, dim = -1)
    # A2 = A2+eyes_like

    return A2

def z_score_normalize(matrix):
    min_val = matrix.min()
    max_val = matrix.max()
    return (matrix - min_val) / (max_val - min_val)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dim):
        super(GraphConvolution, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.att_A, self.att_A2, self.att_mlp = 0, 0, 0
        self.hidden = dim
        self.W_k0, self.W_k1, self.W_k2 = Parameter(
            torch.FloatTensor(out_features, self.hidden)), Parameter(
            torch.FloatTensor(out_features, self.hidden)), Parameter(torch.FloatTensor(out_features, self.hidden))
        self.weight_A, self.weight_A2, self.weight_mlp = Parameter(
            torch.FloatTensor(in_features, out_features)), Parameter(
            torch.FloatTensor(in_features, out_features)), Parameter(torch.FloatTensor(in_features, out_features))
        self.att_vec_A, self.att_vec_A2, self.att_vec_mlp = Parameter(
            torch.FloatTensor(out_features, self.hidden)), Parameter(
            torch.FloatTensor(out_features, self.hidden)), Parameter(
            torch.FloatTensor(out_features, self.hidden))
        self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        std_att = 1. / math.sqrt(self.att_vec_mlp.size(1))
        std_k = 1./math.sqrt(self.W_k0.size(1))
        std_att_vec = 1. / math.sqrt(self.att_vec.size(1))

        self.W_k0.data.uniform_(-std_k, std_k)
        self.W_k1.data.uniform_(-std_k, std_k)
        self.W_k2.data.uniform_(-std_k, std_k)

        self.weight_A.data.uniform_(-stdv, stdv)
        self.weight_A2.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)

        self.att_vec_A.data.uniform_(-std_att, std_att)
        self.att_vec_A2.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_A, output_A2, output_mlp):
        T = 3

        k0 = torch.mean(torch.matmul(output_A, self.W_k0), dim=1, keepdim=True)
        k1 = torch.mean(torch.matmul(output_A2, self.W_k1), dim=1, keepdim=True)
        k2 = torch.mean(torch.matmul(output_mlp, self.W_k2), dim=1, keepdim=True)

        k0 = k0.permute(0, 2, 1)
        k1 = k1.permute(0, 2, 1)
        k2 = k2.permute(0, 2, 1)

        att = torch.softmax(torch.matmul(torch.sigmoid(torch.cat(
            [torch.matmul(torch.matmul((output_A), self.att_vec_A), k2),
             torch.matmul(torch.matmul((output_A2), self.att_vec_A2), k0),
             torch.matmul(torch.matmul((output_mlp), self.att_vec_mlp), k1)], 2)), self.att_vec) / T, 2)

        return (att[:, :, 0].unsqueeze(-1), att[:, :, 1].unsqueeze(-1), att[:, :, 2].unsqueeze(-1))

    def forward(self, inputx , adj_A, adj_A2):
        output_A = F.relu(torch.matmul(adj_A, (torch.matmul(inputx, self.weight_A))))
        output_A2 = F.relu(torch.matmul(adj_A2, (torch.matmul(inputx, self.weight_A2))))
        output_mlp = F.relu(torch.matmul(inputx, self.weight_mlp))

        self.att_A, self.att_A2, self.att_mlp = self.attention((output_A), (output_A2), (output_mlp))

        return 3 * (self.att_A * output_A + self.att_A2 * output_A2 + self.att_mlp * output_mlp)

class Base_model(nn.Module):
    def __init__(self, configs, args):
        super(Base_model, self).__init__()

        indim_fea = configs.window_size#6
        hidden_fea = configs.hidden_channels
        outdim_fea = configs.final_out_channels
        num_node = configs.num_nodes
        num_classes = configs.num_classes
        self.nonlin_map = nn.Linear(indim_fea, indim_fea)
        self.t_l = configs.time_denpen_len#24

        self.hidden_dim = hidden_fea
        self.output_dim = outdim_fea
        self.time_length = configs.convo_time_length
        self.Time_Preprocessing = Feature_extractor_1DCNN_tiny(indim_fea,32,hidden_fea, configs.kernel_size, configs.stride, configs.dropout)
        self.Time_Preprocessing2 = Modern(num_node, hidden_fea,configs.kernel_size, configs.stride, configs.dropout)

        self.MPNN = GCNLayer(hidden_fea, outdim_fea)
        self.logits = nn.Linear(self.time_length * outdim_fea * num_node, num_classes)
        self.SHGCN1 = GraphConvolution(64, 128, 64)

        self.temperature = 1.0
        self.training = True
        self.dropout = 0.7
        self.droprate = 0.3

        self.heatmap_buffer = []
        self.heatmap_counter = 0

    def forward(self, X, self_supervised = False, num_remain = None):
        ## Size of X is (bs, time_length, num_nodes, dimension)
        bs, tlen, num_node, dimension = X.size()
        X = torch.transpose(X, 2, 1) # (bs, num_nodes, time_length, dimension)
        X = torch.reshape(X, [bs*num_node, tlen, dimension])
        X = self.nonlin_map(X)
        TD_input = tr.transpose(X, 1, 2)
        TD_output = self.Time_Preprocessing(TD_input)
        TD_output = tr.transpose(TD_output, 1, 2)
        GC_input = tr.reshape(TD_output, [bs, -1, num_node, self.hidden_dim])
        GC_input = self.Time_Preprocessing2(GC_input)
        GC_input = tr.reshape(GC_input, [-1, num_node, self.hidden_dim])
        Adj_input = Dot_Graph_Construction(GC_input)
        adj_2 = get_A2(Adj_input)
        GC_output = self.SHGCN1(GC_input, Adj_input, adj_2)
        GC_output = tr.reshape(GC_output, [bs, -1, num_node, self.output_dim])
        logits_input = tr.reshape(GC_output, [bs, -1])
        logits = self.logits(logits_input)
        return logits, GC_output

class GCN_layer(nn.Module):
    def __init__(self, input_dimension, out_dimension):
        super(GCN_layer, self).__init__()

        self.nonlinear_FC = nn.Sequential(
            nn.Linear(input_dimension, out_dimension),
            nn.ReLU()
        )

    def forward(self, adj, X):
        adj_X = torch.bmm(adj, X)
        adj_X = self.nonlinear_FC(adj_X)

        return adj, adj_X
#
class prediction(nn.Module):
    def __init__(self, indim_fea, hidden_size):
        super(prediction, self).__init__()
        self.fc1 = nn.Linear(indim_fea, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        features = F.leaky_relu(self.fc1(x))

        features = F.leaky_relu(self.fc2(features))

        return features

class classify(nn.Module):
    def __init__(self, num_class):
        super(classify, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(128, 8)),
            ('relu2', nn.ReLU(inplace=True))
        ]))

        self.cls = nn.Linear(8, num_class)

    def forward(self, features):
        features = self.fc(features)
        cls = self.cls(features)

        return cls
