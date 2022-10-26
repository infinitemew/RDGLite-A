import json
import numpy as np
import scipy.sparse as sp
import torch as t
import torch.nn as nn
import argparse


# 选取出现较多的属性的个数
attr_range = 2000


def loadfile(fn, num=1):
    """Load a file and return a list of tuple containing $num integers in each line."""
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


def loadattr(fns, e, ent2id):
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    num_features = min(len(fre), attr_range)
    attr2id = {}
    for i in range(num_features):
        attr2id[fre[i][0]] = i
    M = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            M[(ent2id[th[0]], attr2id[th[i]])] = 1.0
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[0])
        col.append(key[1])
        data.append(M[key])

    return sp.coo_matrix((data, (row, col)), shape=(e, num_features))


def get_ae_input(attr):
    return sparse_to_tuple(sp.coo_matrix(attr))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="zh_en")
    args = parser.parse_args()
    return args


args = parse_args()
lang = args.lang
dataset_str = lang
names = [['ent_ids_1', 'ent_ids_2'], ['training_attrs_1', 'training_attrs_2'], ['triples_1', 'triples_2'], ['ref_ent_ids']]
for fns in names:
    for i in range(len(fns)):
        fns[i] = 'data/DBP15K/'+dataset_str+'/'+fns[i]
Es, As, Ts, ill = names

e = len(set(loadfile(Es[0], 1)) | set(loadfile(Es[1], 1)))
ent2id = get_ent2id([Es[0], Es[1]])
attr = loadattr([As[0], As[1]], e, ent2id)



dense = attr.todense()

attr_sparse = []
attrArry = attr.A
for i in range(dense.shape[0]):
    for j in range(dense.shape[1]):
        if attrArry[i][j] != 0:
            attr_sparse.append([i, j, 1.0])

ae_input = get_ae_input(attr)

for i in range(dense.shape[0]):
    for j in range(dense.shape[1]):
        if dense[i, j] != 0.:
            dense[i, j] = j + 1
print('prepare finished')

print('start embedding')
attr_dim = 100
embedding = nn.Embedding(attr_range + 1, attr_dim)
embed = embedding(t.LongTensor(dense))

print('embedding finished')

print('start select tensor')

attr_emb = np.zeros([attr_range, attr_dim])
# 初始化数组
for i in range(attr_range):
    for j in range(attr_dim):
        attr_emb[i][j] = 9999.

for i in range(e):
    for j in range(attr_range):
        if dense[i, j] != 0. and attr_emb[j][0] == 9999.:
            for x in range(attr_dim):
                attr_emb[j][x] = embed.detach().numpy()[i][j][x]

print('start write json')

path = './data/DBP15K/' + dataset_str + '/'

filename = path + dataset_str[0:2] + '/' + 'attr_vector.json'
with open(filename,'w') as file_obj:
    json.dump(attr_emb.tolist(), file_obj)


filename = path + dataset_str[0:2] + '/' + 'ae_adj.json'
with open(filename,'w') as file_obj:
    json.dump(attr_sparse, file_obj)


