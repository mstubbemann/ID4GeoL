from subprocess import run

names = ["PubMed", "Cora", "CiteSeer",
         "ogbn-arxiv",
         "ogbn-mag",
         "ogbn-products"]

n_neighbors = 5
n_jobs = 15

for name in names:
    print("------")
    print("------")
    print(name)
    run(["python", "baseline/mle_script.py",
         "--name", name,
         "--n_jobs", str(n_jobs),
         "--n_neighbors", str(n_neighbors)])
