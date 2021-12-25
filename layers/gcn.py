import torch
import torch.nn as nn

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

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
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

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

