import os

import torch
from sklearn.neighbors import NearestNeighbors as NN
import numpy as np
import pandas as pd
from torch_geometric.transforms import SIGN, AddSelfLoops, ToUndirected
from torch_geometric.datasets import Planetoid, OGB_MAG
from ogb.nodeproppred import PygNodePropPredDataset as PD


def mle(X, k=5, n_jobs=16):
    searcher = NN(n_neighbors=k,
                  n_jobs=n_jobs)
    print("Find neighbors")
    searcher.fit(X)
    distances, indices = searcher.kneighbors()
    distances = np.sort(distances)
    print(distances)
    for d in distances:
        print(d[0])
    nominators = distances[:, -1]
    distances = 1/distances
    print("Compute MLEs")
    sums = np.array([n * dist[:-1] for n, dist
                     in zip(nominators, distances)]).flatten()
    sums = np.array([x for x in sums if x != np.inf])
    print(sums)
    sums = np.log(sums)
    sums = np.mean(sums)
    return sums**(-1)


def main(name="Cora",
         k=5,
         aggregations=5,
         n_jobs=10,
         destination=None):
    result = pd.DataFrame()
    print("Prepare Data")
    if destination is None:
        destination = "data/baseline/" + name + "/"
    if not os.path.isdir(destination):
        os.makedirs(destination)
    if name in {"Cora", "PubMed", "CiteSeer"}:
        data = Planetoid(name=name, root="data")
    elif name == "ogbn-mag":
        data = OGB_MAG(root="data",
                       preprocess="metapath2vec")
    else:
        data = PD(name=name, root="data")
    print("Prepare Graph")
    G = data[0]
    if name == "ogbn-mag":
        G = G.to_homogeneous()
    G = AddSelfLoops()(G)
    G = ToUndirected()(G)
    G = SIGN(k)(G)
    for i in range(aggregations+1):
        X = [G["x"]] + [G["x" + str(j)] for j in range(1, i+1)]
        X = torch.cat(X, axis=1)
        X = np.array(X)
        id = mle(X,
                 k=k,
                 n_jobs=n_jobs)
        result = result.append({"k": i,
                                "ID": id},
                               ignore_index=True)
        print(result)
        result.to_csv(destination + "ids.csv", index=False)
        print(i, "/", aggregations)
    return result


if __name__ == "__main__":
    for name in ["Cora", "PubMed", "CiteSeer",
                 "ogbn-arxiv", "ogbn-mag", "ogbn-products"]:
        print("###################")
        print("Start with ", name)
        main(name)
        print("###################")
