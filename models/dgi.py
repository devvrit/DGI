import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class GCN_new(nn.Module):
    def __init__(self, inp_dim, hidden_dims, act, dropout=0.2):
        """
        :param inp_dim = dimension of X matrix representing node features
        :param hidden_dims = hidden layer dimensions
        """
        super(GCN_new, self).__init__()

        assert len(hidden_dims)>0
        self.Wr=[]
        self.g = []
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.inp_dim=inp_dim
        self.Wr.append(nn.Linear(inp_dim, hidden_dims[0], bias=True))
        self.Wr[-1].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.Wr[0].weight)
        self.g.append(nn.PReLU() if act == 'prelu' else act)
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_dims[0]))
        for i in range(1, self.num_layers):
            self.Wr.append(nn.Linear(hidden_dims[i-1], hidden_dims[i], bias=True))
            nn.init.xavier_uniform_(self.Wr[-1].weight)
            self.Wr[-1].bias.data.fill_(0.0)
            self.g.append(nn.PReLU() if act == 'prelu' else act)
            self.bns.append(torch.nn.BatchNorm1d(hidden_dims[i]))
        self.Wr = nn.ModuleList(self.Wr)
        self.g = nn.ModuleList(self.g)
        self.dropout=nn.Dropout(dropout)
        #print("self.Wr.weight.size() is: " + str(self.Wr.weight.size()))
        #print("self.Wr.weight is: " + str(self.Wr.weight))
        #print("self.Wr.bias is: " + str(self.Wr.bias))

    def forward(self, A, AX):
        #temp1 = self.g[0](self.Wr[0](AX))
        temp1 = self.g[0](self.bns[i](self.Wr[0](AX)))
        temp = self.dropout(temp1)
        for i in range(1, self.num_layers):
            temp1 = self.Wr[i](temp)
            temp1 = torch.sparse.mm(A, temp1)
            temp1 = self.bns[i](temp1)
            temp1 = self.g[i](temp1)
            temp1 = self.dropout(temp1)
            temp=temp1
        return torch.unsqueeze(temp, 0)
        # temp = torch.sparse.mm(A, self.g(self.Wr(AX)))
        # return torch.unsqueeze(self.act(self.W(temp)), 0)
        #return torch.unsqueeze(self.g(self.Wr(AX)), 0)
        #return torch.unsqueeze(self.act(self.W(self.g(self.Wr(AX)))),0)


class SAGE(torch.nn.Module):
    def __init__(self, inp_dim, hidden_dims, act, dropout=0.0):
        super(SAGE, self).__init__()

        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.g = torch.nn.ModuleList()
        self.convs1.append(nn.Linear(inp_dim, hidden_dims[0]))
        self.convs2.append(nn.Linear(inp_dim, hidden_dims[0]))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dims[0]))
        self.dropout=nn.Dropout(dropout)
        self.g.append(nn.PReLU() if act == 'prelu' else act)

        for i in range(1, self.num_layers):
            self.convs1.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.convs2.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dims[i]))
            self.g.append(nn.PReLU() if act == 'prelu' else act)

    def forward(self, A, X):
        temp=X
        for i in range(self.num_layers):
            temp = self.convs1[i](temp) + self.convs2[i](torch.sparse.mm(A,temp))
            if i<self.num_layers-1:
                temp = self.bns[i](temp)
            temp = self.g[i](temp)
            temp = self.dropout(temp)
        return torch.unsqueeze(temp, 0)
            



# class SAGE(torch.nn.Module):
#     def __init__(self, inp_dim, hidden_dims, act, dropout=0.0):
#         super(SAGE, self).__init__()

#         self.hidden_dims = hidden_dims
#         self.num_layers = len(hidden_dims)
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(SAGEConv(inp_dim, hidden_dims[0]))
#         self.bns = torch.nn.ModuleList()
#         self.bns.append(torch.nn.BatchNorm1d(hidden_dims[0]))
#         for i in range(1, self.num_layers):
#             self.convs.append(SAGEConv(hidden_dims[i-1], hidden_dims[i]))
#             self.bns.append(torch.nn.BatchNorm1d(hidden_dims[i]))
#         # self.convs.append(SAGEConv(hidden_dims[-1], out_channels))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, adj_t, x):
#         for i, conv in enumerate(self.convs):
#             x = conv(x, adj_t)
#             x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         # x = self.convs[-1](x, adj_t)
#         return torch.unsqueeze(x, 0)
#         # return x.log_softmax(dim=-1)

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        # self.gcn = GCN_new(n_in, n_h, activation)
        self.gcn = SAGE(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h[-1])

    def forward(self, A, AX1, AX2, nodes, sparse, msk, samp_bias1, samp_bias2):
        # h_1 = self.gcn(seq1, adj, sparse)
        h_1 = self.gcn(A, AX1)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        # h_2 = self.gcn(seq2, adj, sparse)
        h_2 = self.gcn(A, AX2)

        ret = self.disc(c, h_1[:,nodes,:], h_2[:,nodes,:], samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, A, AX1, sparse, msk):
        # h_1 = self.gcn(seq, adj, sparse)
        h_1 = self.gcn(A, AX1).squeeze(0)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

