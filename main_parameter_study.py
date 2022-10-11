from subprocess import run

names = ["ogbn-products",
         "ogbn-mag",
         "ogbn-arxiv"]

pools = 15
for name in names:
    print("++++++++++++")
    print("Start with: ", name)
    run(["python", "parameter_study.py",
         "--pools", str(pools),
         "--name", name])
    print("++++++++++++")
