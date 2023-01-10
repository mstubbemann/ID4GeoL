from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
import pandas as pd

pools = 10
n_samples = 100000
rounds = 3


def phi_fj(lf,
           j):
    return np.min(lf[j-1:]-lf[:1-j])


rng = np.random.default_rng(seed=13)


result = []
for n in [1e6, 10e6, 100e6]:
    n = int(n)
    samples = list(n+2 - np.geomspace(n, 2, n_samples))
    samples = [int(x) for x in samples]
    samples = sorted(list(set(samples)))
    if samples[0] != 2:
        samples = [2] + samples
    if samples[-1] != n:
        samples.append(n)
    S = np.array(samples)
    gaps = S[1:] - S[:-1]
    for d in [10, 50, 250]:
        for round in range(rounds):
            X = rng.standard_normal((n, d))

            print("Start Creating LFS")

            def create_lf(i):
                return np.sort(X[:, i])

            with Pool(pools) as p:
                lfs = [lf for lf in tqdm(p.imap(create_lf, range(X.shape[1])),
                                         total=X.shape[1])]

            print("Compute Support Phis")

            def phi_j(j):
                return np.array([phi_fj(lf, j) for lf in lfs],
                                dtype=np.float32)

            max_phi = 0  # phi_j for the last support phi_j
            max_phis = []  # store al phi_js

            with Pool(pools) as p:
                for i, phi_lfs in enumerate(tqdm(p.imap(phi_j,
                                                        samples),
                                                 total=len(samples))):
                    max_phi = np.max(phi_lfs)
                    max_phis.append(max_phi)

            print("Compute Bounds and errors")
            print("Support Sum: ", np.sum(max_phis))
            min_Delta = (np.sum(max_phis[:-1] * gaps) + max_phis[-1]) * (1/n)
            max_Delta = (np.sum(max_phis[1:] * gaps) + max_phis[0]) * (1/n)
            max_Dim = 1/(min_Delta**2)
            min_Dim = 1/(max_Delta**2)
            error = (max_Dim - min_Dim)/min_Dim
            print("Error:", error)
            print("Accuracy:", 1 - error)
            print("##########")
            result.append({"n": n,
                           "d": d,
                           "round": round,
                           "Error": error,
                           "Accuracy": 1 - error})
        D = pd.DataFrame(result)
        D.to_csv("data/random_errors.csv")
