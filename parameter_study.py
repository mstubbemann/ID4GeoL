from multiprocessing import Pool
import os

from ogb.nodeproppred import PygNodePropPredDataset as PD
import numpy as np
import argparse
from torch_geometric.transforms import SIGN, AddSelfLoops, ToUndirected
import pandas as pd
from tqdm import tqdm
from torch_geometric.datasets import OGB_MAG

# Command Line Arguments
parser = argparse.ArgumentParser()

parser.add_argument("--pools",
                    type=int,
                    default=15)
parser.add_argument("--num_steps",
                    type=int,
                    default=50)
parser.add_argument("--end",
                    type=float,
                    default=0.05)
parser.add_argument("--name",
                    type=str,
                    default="ogbn-arxiv")
parser.add_argument("--destination",
                    type=str,
                    default="")

args = parser.parse_args()
print(args.pools,
      args.num_steps,
      args.end,
      args.name,
      args.destination)

if not os.path.isdir("data/case_study/"):
    os.makedirs("data/case_study/")

if not args.destination:
    destination = "data/case_study/" + args.name + ".csv"

    
# Utility functions

def phi_fj(lf,
           j):
    return np.min(lf[j-1:]-lf[:1-j])


def phi_j(lfs, j):
    return np.array([phi_fj(lf, j) for lf in lfs],
                    dtype=np.float32)


def Obs_diam(X):
    return np.max(np.max(X, axis=0) - np.min(X, axis=0))


def cost(n, j):
    "Return costs for phi_fj"
    return n - (j-1)


def gap_costs(n, x, y):
    """
    Return costs for computing phi_f,x+1,...,phi_f,y-1
    if lfs have length n.
    """
    return sum([cost(n, a) for a in range(x+1, y)])


print("Start with Computation of Agrregated Vectors")
# Start loading Featrues

if args.name == "ogbn-mag":
    data = OGB_MAG(preprocess="metapath2vec", root="data")
else:
    data = PD(name=args.name, root="data")

G = data[0]
if args.name == "ogbn-mag":
    G = G.to_homogeneous()
G = ToUndirected()(G)
G = AddSelfLoops()(G)
G = SIGN(2)(G)

print("Compute samples")
n = len(G["x"])
Xs = [np.array(G["x"], dtype=np.float32)] + [np.array(G["x" + str(i)],
                                                      dtype=np.float32)
                                             for i in range(1, 3)]

print("Compute Sample")
n_supports = list(np.linspace(args.end/args.num_steps,
                              args.end,
                              args.num_steps))

print("Start computations")
d = pd.DataFrame()

for k in [2]:
    print("Start with k: ", k)
    print("Start Creating LFS")
    X = np.hstack(Xs[:k+1])

    def create_lf(i):
        return np.sort(X[:, i])

    with Pool(args.pools) as p:
        lfs = [lf for lf in tqdm(p.imap(create_lf, range(X.shape[1])),
                                 total=X.shape[1])]

    for r_n_samples in n_supports:
        print("##############")
        print("Start with n_samples:", r_n_samples)
        n_samples = int(n*r_n_samples)
        samples = list(n+2 - np.geomspace(n, 2, n_samples))
        samples = [int(x) for x in samples]
        samples = sorted(list(set(samples)))
        if samples[0] != 2:
            samples = [2] + samples
        if samples[-1] != n:
            samples.append(n)
        real_r_n_samples = (len(samples)/n)

        print("Compute Support Phis")

        max_phi = 0  # phi_j for the last support phi_j
        phi_js = []  # store al phi_js
        phi_sets = []  # store for each j which features have to be considered

        def c_phi_j(s):
            return phi_j(lfs, s)

        with Pool(args.pools) as p:
            for phi_lfs in tqdm(p.imap(c_phi_j,
                                       samples),
                                total=len(samples)):
                phi_sets.append(np.where(phi_lfs > max_phi)[0])
                max_phi = np.max(phi_lfs)
                phi_js.append(max_phi)

        phi_sets = phi_sets[1:]

        print("Compute Error")
        S = np.array(samples)
        gaps = S[1:] - S[:-1]
        min_Delta = (np.sum(phi_js[:-1] * gaps) + phi_js[-1]) * (1/n)
        max_Delta = (np.sum(phi_js[1:] * gaps) + phi_js[0]) * (1/n)
        max_Dim = 1/(min_Delta**2)
        min_Dim = 1/(max_Delta**2)
        error = (max_Dim - min_Dim)/min_Dim
        print("Error:", error)

        print("Compute Costs")
        pre_costs = np.sum([len(lfs) * cost(n, s) for s in samples])
        full_gap_costs = np.array([gap_costs(n, x, y)
                                   for x, y in zip(S, S[1:])])
        full_gap_costs = np.array([len(s) for s in phi_sets]) * full_gap_costs
        full_gap_costs = np.sum(full_gap_costs)
        costs_full_computation = len(lfs) * ((n**2/2) - (n/2))

        skipped_costs = 1 - ((pre_costs + full_gap_costs) / costs_full_computation)
        print("Skipped costs:", skipped_costs)
        d = d.append({"r_samples": r_n_samples,
                      "real_r_samples": real_r_n_samples,
                      "k:": k,
                      "min_Dim": min_Dim,
                      "max_Dim": max_Dim,
                      "Error": error,
                      "Skipped Cost": skipped_costs}, ignore_index=True)
        d.to_csv(destination)
        print("##############")
print(d)
