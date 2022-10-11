from multiprocessing import Pool
import os
import pandas as pd
import argparse
from torch_geometric.transforms import SIGN, AddSelfLoops, ToUndirected
from tqdm import tqdm
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset as PD

# Command Line Arguments
parser = argparse.ArgumentParser()

parser.add_argument("--pools",
                    type=int,
                    default=15)
parser.add_argument("--num_samples",
                    type=int,
                    default=100000)
parser.add_argument("--name",
                    type=str,
                    default="ogbn-papers100M")
parser.add_argument("--k",
                    type=int,
                    default=5)
parser.add_argument("--destination",
                    type=str,
                    default="")


if not os.path.isdir("data/approximate"):
    os.makedirs("data/approximate/")

args = parser.parse_args()
if not args.destination:
    destination = "data/approximate/" + args.name + ".csv"
else:
    destination = args.destination

# Utility functions


def phi_fj(lf,
           j):
    return np.min(lf[j-1:]-lf[:1-j])


def Obs_diam(X):
    return np.max(np.max(X, axis=0) - np.min(X, axis=0))


print("Start with Computation of Agrregated Vectors")
# Start loading Featrues
data = PD(name=args.name, root="data")
G = data[0]
G = AddSelfLoops()(G)
G = ToUndirected()(G)
G = SIGN(args.k)(G)

print("Compute samples")
n = len(G["x"])
samples = list(n+2 - np.geomspace(n, 2, args.num_samples))
samples = [int(x) for x in samples]
samples = sorted(list(set(samples)))
if samples[0] != 2:
    samples = [2] + samples
if samples[-1] != n:
    samples.append(n)
S = np.array(samples)
gaps = S[1:] - S[:-1]
X = np.array(G["x"], dtype=np.float32)
OD = Obs_diam(X)

# Start computations for Dim
support_values = []
results = pd.DataFrame()

for k in range(args.k+1):
    print("-----------------")
    print("Start with k: ", k)
    if k == 0:
        X = G["x"]
    else:
        X = G["x" + str(k)]
    X = np.array(X, dtype=np.float32)
    X = (1/OD) * X

    print("Start Creating LFS")

    def create_lf(i):
        return np.sort(X[:, i])

    with Pool(args.pools) as p:
        lfs = [lf for lf in tqdm(p.imap(create_lf, range(X.shape[1])),
                                 total=X.shape[1])]

    print("Compute Support Phis")
    current_support_values = []

    def phi_j(j):
        return np.max([phi_fj(lf, j) for lf in lfs])

    with Pool(args.pools) as p:
        for value in tqdm(p.imap_unordered(phi_j,
                                           samples,
                                           chunksize=10),
                          total=len(samples)):
            current_support_values.append(value)
    current_support_values = np.sort(np.array(current_support_values,
                                              dtype=np.float32))
    support_values.append(current_support_values)
    phi_js = np.max(support_values, axis=0)
    min_Delta = (np.sum(phi_js[:-1] * gaps) + phi_js[-1]) * (1/n)
    max_Delta = (np.sum(phi_js[1:] * gaps) + phi_js[0]) * (1/n)
    max_Dim = 1/(min_Delta**2)
    min_Dim = 1/(max_Delta**2)
    error = (max_Dim - min_Dim)/min_Dim
    print("Error:", error)
    results = results.append({"k": k,
                              "min_Dim": min_Dim,
                              "max_Dim": max_Dim,
                              "error": error},
                             ignore_index=True)
    results.to_csv(destination, index=False)
    print(results)
