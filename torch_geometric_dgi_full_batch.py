import os.path as osp

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected, add_remaining_self_loops

# +
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset = 'Pubmed'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid(path, dataset)

dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root="/home/devvrit_03/GraphNN/clean_codes/dataset")
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

data = dataset[0].to(device)
data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])


# -

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.con = GCNConv(hidden_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)
        self.prel = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        x = self.con(x, edge_index)
        x = self.prel(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


model = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(data.x, data.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(e):
    model.eval()
    z, _, _ = model(data.x, data.edge_index)
    torch.save(z, "embedding.pt_epoch_"+str(e))
    #torch.save(z, "embedding.pt")
    return
    acc = model.test(z[train_idx], data.y[train_idx],
                     z[test_idx], data.y[test_idx], max_iter=150)
    return acc


for epoch in range(1, 10):
    model.train()
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch<11 or epoch%5==0:
        test(epoch)
#acc = test()
print(f'Accuracy: {acc:.4f}')


