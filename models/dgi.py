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
        assert len(hidden_dims)>0
        self.hidden_dims = hidden_dims
        self.inp_dim=inp_dim
        self.Wr = nn.Linear(inp_dim, hidden_dims, bias=True)
        nn.init.xavier_uniform_(self.Wr.weight)
        self.g = nn.PReLU() if act == 'prelu' else nn.ReLU()

    def forward(self, AX):
        return torch.unsqueeze(self.g(self.Wr(AX)), 0)

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN_new(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, AX1, AX2, sparse, msk, samp_bias1, samp_bias2):
        # h_1 = self.gcn(seq1, adj, sparse)
        h_1 = self.gcn(AX1)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        # h_2 = self.gcn(seq2, adj, sparse)
        h_2 = self.gcn(AX2)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        # h_1 = self.gcn(seq, adj, sparse)
        h_1 = self.gcn(AX1)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

