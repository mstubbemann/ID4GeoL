from subprocess import run

name = "ogbn-papers100M"
pools = 15

run(["python", "approximate.py",
     "--pools", str(pools)])
