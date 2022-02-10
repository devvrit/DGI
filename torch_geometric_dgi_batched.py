import os.path as osp

import torch
import torch.nn as nn

from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from ogb.nodeproppred import PygNodePropPredDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
# dataset = Reddit(path)
# data = dataset[0]

dataset = PygNodePropPredDataset(name = "ogbn-products")
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
data = dataset[0].to(device)
data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])


train_loader = NeighborSampler(data.edge_index, node_idx=None,
                               sizes=[25, 20, 10], batch_size=2048,
                               shuffle=True, num_workers=12)

test_loader = NeighborSampler(data.edge_index, node_idx=None,
                              sizes=[25, 20, 10], batch_size=2048,
                              shuffle=False, num_workers=12)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels)
        ])

        self.activations = torch.nn.ModuleList()
        self.activations.extend([
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels)
        ])

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


model = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

x, y = data.x.to(device), data.y.to(device)


def train(epoch):
    model.train()

    total_loss = total_examples = 0
    for batch_size, n_id, adjs in tqdm(train_loader,
                                       desc=f'Epoch {epoch:02d}'):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        pos_z, neg_z, summary = model(x[n_id], adjs)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pos_z.size(0)
        total_examples += pos_z.size(0)

    return total_loss / total_examples


@torch.no_grad()
def test(e):
    model.eval()

    zs = []
    for i, (batch_size, n_id, adjs) in enumerate(test_loader):
        adjs = [adj.to(device) for adj in adjs]
        zs.append(model(x[n_id], adjs)[0])
    z = torch.cat(zs, dim=0)
    torch.save(z, "embedding_products.pt_epoch_" + str(e))
    return
    train_val_mask = data.train_idx | data.valid_idx
    acc = model.test(z[train_val_mask], y[train_val_mask], z[data.test_idx],
                     y[data.test_idx], max_iter=10000)
    return acc


for epoch in range(1, 21):
    model.train()
    loss = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
    test(epoch)

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
