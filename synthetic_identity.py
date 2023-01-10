from multiprocessing import Pool
import random

import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import SIGN, AddSelfLoops, ToUndirected
import torch
from tqdm import tqdm

pools = 1
n = 100
max_k = 5
result = []

random.seed(42)


def phi_fj(lf,
           j):
    return np.min(lf[j - 1:] - lf[:1 - j])


F = torch.eye(n)
possible_edges = [(i, j) for j in range(n)
                  for i in range(j)]
# Shuffle edges
random.shuffle(possible_edges)

edge_index_0 = []
edge_index_1 = []

for p in possible_edges:
    edge_index_0.append(p[0])
    edge_index_1.append(p[1])
    G = Data(x=F,
             edge_index=torch.tensor([edge_index_0,
                                      edge_index_1]),
             num_nodes=n)
    G = AddSelfLoops()(G)
    G = ToUndirected()(G)

    G = SIGN(max_k)(G)

    for k in range(max_k + 1):
        X = torch.cat([G["x"]] + [G["x" + str(l)]
                                  for l in range(1, k+1)],
                      axis=1)

        def create_lf(i):
            return np.sort(X[:, i])

        with Pool(pools) as p:
            lfs = [lf for lf in tqdm(p.imap(create_lf, range(X.shape[1])),
                                     total=X.shape[1])]

        def phi_j(j):
            return np.max([phi_fj(lf, j) for lf in lfs])

        with Pool(pools) as p:
            dimension = np.mean([0] + [value for value in tqdm(p.imap(phi_j,
                                                                      range(2,
                                                                            X.shape[0] + 1)),
                                                               total=X.shape[0])])
        dimension = 1 / (dimension**2)
        result.append({"dimension": dimension,
                       "k": k,
                       "n_edges": len(edge_index_0)})
        print(k, "/", max_k + 1)
        print(len(edge_index_0), "/", len(possible_edges))
        print("#########################")

    d = pd.DataFrame(result)
    d.to_csv("data/identity.csv")
