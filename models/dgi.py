import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator

class GCN_new(nn.Module):
    def __init__(self, inp_dim, hidden_dims, act):
        """
        :param inp_dim = dimension of X matrix representing node features
        :param hidden_dims = hidden layer dimensions
        """
        super(GCN_new, self).__init__()
        self.hidden_dims = hidden_dims
        self.inp_dim=inp_dim
        self.Wr = nn.Linear(inp_dim, hidden_dims, bias=True)
        self.W = nn.Linear(hidden_dims, hidden_dims, bias=True)
        #torch.manual_seed(786)
        nn.init.xavier_uniform_(self.Wr.weight.data)
        nn.init.xavier_uniform_(self.W.weight.data)
        self.Wr.bias.data.fill_(0.0)
        self.W.bias.data.fill_(0.0)
        self.g = nn.PReLU() if act == 'prelu' else nn.ReLU()
        self.act = nn.PReLU() if act == 'prelu' else nn.ReLU()
        #print("self.Wr.weight.size() is: " + str(self.Wr.weight.size()))
        #print("self.Wr.weight is: " + str(self.Wr.weight))
        #print("self.Wr.bias is: " + str(self.Wr.bias))

    def forward(self, A, AX):
        temp = torch.sparse.mm(A, self.g(self.Wr(AX)))
        return torch.unsqueeze(self.act(self.W(temp)), 0)
        #return torch.unsqueeze(self.g(self.Wr(AX)), 0)
        #return torch.unsqueeze(self.act(self.W(self.g(self.Wr(AX)))),0)

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN_new(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

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

