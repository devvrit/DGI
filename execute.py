import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random

from models import DGI, LogReg
from utils import process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#dataset = 'cora'
dataset = 'ogbn-products'

# training params
#batch_size = 2708
# batch_size = round(169343//2.7)
nb_epochs = 1000
patience = 25
lr = 0.0001
l2_coef = 0.0
drop_prob = 0.0
hid_units = [256,256]
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

#adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
adj, features, labels, idx_train, idx_val, idx_test = process.load_ogbn(dataset)
features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[-1]
batch_size = round(nb_nodes//256)

labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


#adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
adj = process.normalize_adj(adj)


if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()
features = torch.FloatTensor(features[np.newaxis])
if torch.cuda.is_available():
    print('Using CUDA')
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

print("features_size is: " + str(features.size()))
AX = torch.sparse.mm(sp_adj, features[0])
print("AX calculated, size is: " + str(AX.size()))
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])

model = DGI(ft_size, hid_units, nonlinearity).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)


b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

arr = [i for i in range(nb_nodes)]


for epoch in range(nb_epochs):
    i=0
    tot_loss=0
    random.shuffle(arr)
    #np.random.seed(42)
    idx = np.random.permutation(nb_nodes)
    #print("idx: " + str(idx))
    shuf_fts = features[:, idx, :]
    AX2 = torch.sparse.mm(sp_adj, shuf_fts[0]).to(device)
    #if torch.cuda.is_available():
    #    shuf_fts = shuf_fts.cuda()
    while i<nb_nodes:
        model.train()
        optimiser.zero_grad()
        nodes = arr[i:min(i+batch_size, nb_nodes)]
        lbl_1 = torch.ones(1, len(nodes))
        lbl_2 = torch.zeros(1, len(nodes))
        lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

        #logits = model(AX[nodes], AX2[nodes], sparse, None, None, None)
        logits = model(sp_adj, AX, AX2, nodes, sparse, None, None, None)
        loss = b_xent(logits, lbl)
        #print("epoch " + str(epoch) + " Loss: " + str(loss))

        loss.backward()
        optimiser.step()
        tot_loss+=loss.item()
        #if epoch==0 and i==0:
        #    embeds, _ = model.embed(AX, sparse, None)
        #    torch.save(embeds, "embeddding_dgi_ogbn_arxiv.pt_temp")
        #    del embeds
        i+=batch_size
    model.eval()
    embeds, _ = model.embed(sp_adj, AX, sparse, None)
    torch.save(embeds, "embeddding_dgi_ogbn_arxiv.pt_epoch_" + str(epoch+1))
    print("TOTAL LOSS in epoch " + str(epoch) + " is: " + str(tot_loss))
    del embeds
    if tot_loss < best:
        best = tot_loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break
    if epoch==10:
        #print("weight final is: " + str(model.gcn.Wr.weight.data))
        #print("bias final is: " + str(model.gcn.Wr.bias.data))
        assert 1==1

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_dgi.pkl'))
embeds, _ = model.embed(sp_adj, AX, sparse, None)
print("EMBEDDING CALCULATED")
torch.save(embeds, "embeddding_dgi_ogbn_arxiv.pt")

'''

for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

    loss = b_xent(logits, lbl)

    print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_dgi.pkl'))

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
'''
embeds = embeds.unsqueeze(0)
labels = labels.view(1,nb_nodes,-1)
print("labels size: " + str(labels.size()))
train_embs = embeds[0, idx_train]
val_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]
print("train_embs size: " + str(train_embs.size()))

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)
print("train_lbls size: " + str(train_lbls.size()))

tot = torch.zeros(1)
tot = tot.cuda()

accs = []

for _ in range(50):
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    log.cuda()

    pat_steps = 0
    best_acc = torch.zeros(1)
    best_acc = best_acc.cuda()
    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print(acc)
    tot += acc

print('Average accuracy:', tot / 50)

accs = torch.stack(accs)
print(accs.mean())
print(accs.std())
