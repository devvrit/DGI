import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics

from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from ogb.nodeproppred import PygNodePropPredDataset

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
# dataset = Reddit(path)
# data = dataset[0]

def Kmeans(x, K=-1, Niter=10, verbose=False):
    #start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space
    x_temp = x.detach()

    temp = set()
    while len(temp)<K:
        temp.add(np.random.randint(0, N))
    c = x_temp[list(temp), :].clone()

    x_i = x_temp.view(N, 1, D) # (N, 1, D) samples
    cutoff = 1
    if K>cutoff:
        c_j = []
        niter=K//cutoff
        rem = K%cutoff
        if rem>0:
            rem=1
        for i in range(niter+rem):
            c_j.append(c[i*cutoff:min(K,(i+1)*cutoff),:].view(1, min(K,(i+1)*cutoff)-(i*cutoff), D))
    else:
        c_j = c.view(1, K, D) # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        #print("iteration: " + str(i))

        # E step: assign points to the closest cluster -------------------------
        if K>cutoff:
            for j in range(len(c_j)):
                if j==0:
                    D_ij = ((x_i - c_j[j]) ** 2).sum(-1)
                else:
                    D_ij = torch.cat((D_ij,((x_i - c_j[j]) ** 2).sum(-1)), dim=-1)
                    # D_ij += ((x_i - c_j[j]) ** 2).sum(-1)
        else:
            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        assert D_ij.size(1)==K
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x_temp)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        # print(Ncl[:10])
        Ncl += 0.00000000001
        c /= Ncl  # in-place division to compute the average
    return cl, c


dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root="/home/devvrit_03/GraphNN/clean_codes/dataset")
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

x, y = data.x.to(device), data.y.view(-1).to(device)


def train(epoch):
    model.train()

    total_loss = total_examples = 0
    it=0
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
        it+=1
#         if it==10:
#             break
    zs = []
    with torch.no_grad():
        model.eval()
        for batch_size, n_id, adjs in tqdm(test_loader, desc=f'Test Epoch {epoch:02d}'):
            adjs = [adj.to(device) for adj in adjs]
            zs.append(model(x[n_id], adjs)[0])
        zs = torch.cat(zs, dim=0)
        zs = torch.nn.functional.normalize(zs)
        y_pred,_ = Kmeans(zs.detach(), y.max()+1)
        nmi = metrics.normalized_mutual_info_score(y.cpu().numpy(), y_pred.cpu().numpy())
        print("nmi: " + str(nmi))
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
#     test(epoch)

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
