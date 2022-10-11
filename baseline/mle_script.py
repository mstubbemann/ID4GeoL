import os
import argparse
from time import time

from torch_geometric.transforms import SIGN, AddSelfLoops, ToUndirected
from torch_geometric.datasets import Planetoid, OGB_MAG
from ogb.nodeproppred import PygNodePropPredDataset as PD
import numpy as np
import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors as NN


parser = argparse.ArgumentParser()

parser.add_argument("--n_jobs",
                    type=int,
                    default=15)
parser.add_argument("--samples",
                    type=int,
                    default=169343)
parser.add_argument("--name",
                    type=str,
                    default="ogbn-arxiv")
parser.add_argument("--k",
                    type=int,
                    default=5)
parser.add_argument("--n_neighbors",
                    type=int,
                    default=5)
parser.add_argument("--destination",
                    type=str,
                    default="")

args = parser.parse_args()

if not args.destination:
    destination = "data/baseline/" + args.name + ".csv"
    if not os.path.isdir("data/baseline/"):
        os.makedirs("data/baseline")

print("Start with: ", args.name)

print("Load Data")
if args.name in {"Cora", "PubMed", "CiteSeer"}:
    data = Planetoid(name=args.name, root="data")
elif args.name == "ogbn-mag":
    data = OGB_MAG(root="data",
                   preprocess="metapath2vec")
else:
    data = PD(name=args.name, root="data")
print("Prepare Graph")
G = data[0]
if args.name == "ogbn-mag":
    G = G.to_homogeneous()
G = AddSelfLoops()(G)
G = ToUndirected()(G)
G = SIGN(args.k)(G)
rng = np.random.default_rng(13)
result_table = pd.DataFrame()
size = G["x"].shape[0]
if size > args.samples:
    samples = rng.choice(size, args.samples, replace=False)

for i in range(args.k+1):
    print("Start with k: ", i)
    print("Prepare Feature Matrix")
    X = [G["x"]] + [G["x" + str(j)] for j in range(1, i+1)]
    X = torch.cat(X, axis=1)
    X = np.array(X)
    print("Prepare Neighborhood Finder")    
    searcher = NN(n_neighbors=args.n_neighbors+1,
                  n_jobs=args.n_jobs)
    searcher.fit(X)
    print("Shuffle and sample")
    if size > args.samples:
        X = X[samples]
    rng.shuffle(X)
    print("Compute neighbors")
    dists, _ = searcher.kneighbors(X)
    print("Compute ID")
    dists = np.sort(dists)
    dists = dists[:, 1:]
    results = [np.log(A[-1]/A[A > 0][:-1]) for A in dists]
    results = np.concatenate(results)
    result = np.mean(results)**(-1)
    result_table = result_table.append({"k": i,
                                        "ID": result},
                                       ignore_index=True)
    result_table.to_csv(destination, index=False)
    print(result_table)

