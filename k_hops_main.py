from subprocess import run

names = ["PubMed", "Cora", "CiteSeer",
         "ogbn-arxiv", "ogbn-products",
         "ogbn-mag"]

num_samples = 100000
pools = 15

for name in names:
    print("------")
    print("------")
    print(name)
    run(["python", "intrinsic_dimension_k_hops.py",
         "--name", name,
         "--pools", str(pools),
         "--num_samples", str(num_samples)])
    
