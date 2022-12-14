# Intrinsic Dimension for Large-Scale Geometric Learning

This repository contains code for the paper [Intrinsic Dimension for Large-Scale Geometric Learning](https://openreview.net/forum?id=85BfDdYMBY).

You can cite this work via the following BibTeX.

```BibTeX
@article{
stubbemann2022intrinsic,
title={Intrinsic Dimension for Large-Scale Geometric Learning},
author={Maximilian Stubbemann and Tom Hanika and Friedrich Martin Schneider},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=85BfDdYMBY},
note={}
}
```
## WARNING

Running the scripts will download the needed datasets, which are of different size. 
The largest dataset is **ogbn-papers100M** which needs about 145GB and is only downloaded by the script *main_approximate.py*.The second largest dataset is **ogbn-products**, which needs about 4,5GB.

## Installation
Python 3.10 was used for our experiments.
First create an environment via `[PYTHON3.10COMMAND] -m venv .venv` and then switch into the environment via `source .venv/bin/activate`. Then use `pip install -U pip wheel` and `pip install setuptools==59.5.0`. We made the experience, that other versions of setuptools where not compatible to some of the Pytorch (Lightning) packages. 


### Installing Pytorch and Pytorch Geometric


Install the desired versions of [Pytorch](https://pytorch.org/get-started/locally/) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). For computing the IDs, we used Pytorch 1.11.0 with CUDA=CPU, which can be installed via `pip install torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu` on an Intel Xeon Gold server. 

For the training of the SIGN models, we used Pytorch 1.11.0 with Cuda11.3, which can be installed via `pip install torch==1.11.0+cu113--extra-index-url https://download.pytorch.org/whl/cu113` on a RTX3090.

### Installing the other required packages

This can be done via `pip install -r requirements.txt`.

## Computing IDs

WARNING: All computations were done on a Xeon Gold Server with 16 cores. If you have a lower amount of cores available, the pools variable in the main scripts have to be changed.


### Parameter Study

```python
python main_parameter_study.py
```

### Computing Intrinsic Dimensions

```python
python k_hops_main.py
```

### Baseline MLE Estimator

```python
python baseline/main.py
```

### Approximation of Intrinsic Dimension for ogbn-papers-100M
WARNING: This will download **ogbn-papers100M** which needs 145GB disk space. Additionally, the script will need up to 700GB of RAM.

```python
python main_approximate.py
```


## Classification via SIGN

```python
PYTHONHASHSEED=42 python gnn/sign.py
```

## Experiment on Iteratively Adding Edges to Graph with One-Hot Attributes
```python
python synthetic_identity.py
```

## Errors on Random Data
```python
python error_experiment.py
```
