import os
import random
import gc

from torch_geometric.datasets import Planetoid, OGB_MAG
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch_geometric.transforms import SIGN, AddSelfLoops, ToUndirected
import torch_geometric
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from ogb.nodeproppred import PygNodePropPredDataset as PD

# #########The network##########


class Net(pl.LightningModule):

    def __init__(self,
                 input_features=512,
                 inception_dim=512,
                 classes=47,
                 input_dropout=0.5,
                 k=5,
                 dropout=0.1,
                 weight_decay=0.0001,
                 lr=0.0001):
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.k = k
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_features = input_features
        self.inception_dim = inception_dim
        self.classes = classes
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.linears = nn.ModuleList([nn.Linear(self.input_features,
                                                self.inception_dim)
                                      for _ in range(self.k+1)])
        self.normalizations = nn.ModuleList([nn.BatchNorm1d(self.inception_dim)
                                             for _ in range(self.k+1)])
        self.classification = nn.Linear(self.inception_dim*(k+1), self.classes)
        self.prelu = nn.PReLU()


    def forward(self,
                Xs):
        Xs = [N(L(self.input_dropout(X))) for N, L, X in zip(self.normalizations,
                                                             self.linears,
                                                             Xs)]
        X = torch.cat(Xs, axis=1)
        X = self.prelu(X)
        X = self.dropout(X)
        X = self.classification(X)
        return X

    def training_step(self,
                      batch,
                      batch_idx):
        target = batch[-1]
        output = self(batch[:-1])
        loss = F.cross_entropy(output,
                               target)
        return loss

    def validation_step(self,
                        batch,
                        batch_idx):
        labels = batch[-1]
        Z = self(batch[:-1])
        Z = torch.softmax(Z, dim=-1)
        return {"preds": Z,
                "labels": labels}

    def test_step(self,
                  batch,
                  batch_idx):
        labels = batch[-1]
        Z = self(batch[:-1])
        Z = torch.softmax(Z, dim=-1)
        return {"preds": Z,
                "labels": labels}

    def validation_epoch_end(self, outs):
        scores = [out["preds"] for out in outs]
        scores = torch.cat(scores).detach().cpu().numpy()
        labels = torch.cat([out["labels"] for out in outs]).detach().cpu().numpy()
        preds = np.argmax(scores, axis=1)
        accuracy = accuracy_score(labels, preds)
        self.val_accuracy = accuracy
        self.log("val_accuracy", accuracy,
                 prog_bar=True)
        print("\n")

    def test_epoch_end(self, outs):
        scores = [out["preds"] for out in outs]
        scores = torch.cat(scores).detach().cpu().numpy()
        labels = torch.cat([out["labels"] for out in outs]).detach().cpu().numpy()
        preds = np.argmax(scores, axis=1)
        accuracy = accuracy_score(labels, preds)
        self.test_accuracy = accuracy
        self.log("test_accuracy", accuracy)
        print("\n")

    def configure_optimizers(self):
        return AdamW(params=self.parameters(),
                     lr=self.lr,
                     weight_decay=self.weight_decay)

# ###Data#############


class GraphData(Dataset):

    def __init__(self,
                 Xs,
                 labels):
        self.Xs = Xs
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        Xs = [X[idx] for X in self.Xs]
        return (*Xs, self.labels[idx])


class GraphModule(pl.LightningDataModule):

    def __init__(self,
                 batch_size=4096,
                 name="ogbn-products",
                 k=5):

        super().__init__()
        if name == "ogbn-mag":
            data = OGB_MAG(root="data",
                           preprocess="metapath2vec")
            H = data[0]
        elif name in {"Cora", "PubMed", "CiteSeer"}:
            data = Planetoid(name=name, root="data")
        else:
            data = PD(name=name, root="data")
        G = data[0]
        if name == "ogbn-mag":
            G = G.to_homogeneous()
        G = AddSelfLoops()(G)
        G = ToUndirected()(G)
        G = SIGN(k)(G)
        self.batch_size = batch_size
        self.Xs = [G["x"]] + [G["x" + str(k)]
                              for k in range(1, k+1)]
        if name != "ogbn-mag":
            self.labels = G["y"]

        if name == "ogbn-mag":
            self.labels = H["paper"]["y"]
            self.train_indices = H["paper"]["train_mask"].nonzero(as_tuple=True)[0]
            self.val_indices = H["paper"]["val_mask"].nonzero(as_tuple=True)[0]
            self.test_indices = H["paper"]["test_mask"].nonzero(as_tuple=True)[0]
            self.train_data = GraphData([X[self.train_indices]
                                         for X in self.Xs],
                                        self.labels[self.train_indices])
            self.val_data = GraphData([X[self.val_indices]
                                       for X in self.Xs],
                                      self.labels[self.val_indices])
            self.test_data = GraphData([X[self.test_indices]
                                        for X in self.Xs],
                                       self.labels[self.test_indices])
        elif name not in {"PubMed", "Cora", "CiteSeer"}:
            self.labels = self.labels[:, 0]
            self.split = data.get_idx_split()
            self.train_data = GraphData([X[self.split["train"]] for X in self.Xs],
                                        self.labels[self.split["train"]])
            self.val_data = GraphData([X[self.split["valid"]] for X in self.Xs],
                                      self.labels[self.split["valid"]])
            self.test_data = GraphData([X[self.split["test"]] for X in self.Xs],
                                       self.labels[self.split["test"]])    
        else:
            self.train_data = GraphData([X[G["train_mask"]] for X in self.Xs],
                                        self.labels[G["train_mask"]])
            self.val_data = GraphData([X[G["val_mask"]] for X in self.Xs],
                                      self.labels[G["val_mask"]])
            self.test_data = GraphData([X[G["test_mask"]] for X in self.Xs],
                                       self.labels[G["test_mask"]])

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size)


def main(name="ogbn-products",
         ks=list(range(6)),
         batch_size=50000,
         iterations=10,
         input_features=100,
         input_dropout=0.3,
         inception_dim=512,
         classes=47,
         dropout=0.4,
         weight_decay=0.0001,
         lr=0.001,
         max_epochs=1000,
         gpus=1,
         destination=None):
    # Prepare training
    gpus = gpus if torch.cuda.is_available() else None
    if destination is None:
        destination = "data/gnn_results/" + name + "/"
    if not os.path.isdir(destination):
        os.makedirs(destination)
    result = pd.DataFrame()
    for k in ks:
        print("Start with k: ", k)
        datamodule = GraphModule(batch_size=batch_size,
                                 name=name,
                                 k=k)
        print("Generated Data Module")
        for i in range(1, iterations+1):
            seed = 10 * k + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)
            random.seed(seed)
            torch_geometric.seed.seed_everything(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            net = Net(input_features=input_features,
                      inception_dim=inception_dim,
                      classes=classes,
                      k=k,
                      input_dropout=input_dropout,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr)
            # Some preparation
            logger = TensorBoardLogger(save_dir=destination,
                                       name="logs/")
            stopper = EarlyStopping(monitor="val_accuracy",
                                    patience=15,
                                    check_on_train_epoch_end=False,
                                    mode="max")
            trainer = pl.Trainer(logger=logger,
                                 deterministic=True,
                                 callbacks=[stopper],
                                 gpus=gpus,
                                 max_epochs=max_epochs)
            trainer.fit(model=net,
                        datamodule=datamodule)
            trainer.test(model=net,
                         datamodule=datamodule)
            result = result.append({"val_accuracy": net.val_accuracy,
                                    "test_accuracy": net.test_accuracy,
                                    "k": k,
                                    "i": i},
                                   ignore_index=True)
            result.to_csv(destination + "result.csv")
            # Prevent GPU memory overflow
            del net, trainer, logger, stopper
            gc.collect()
            torch.cuda.empty_cache()
        del datamodule
        gc.collect()
        torch.cuda.empty_cache()
    print(result)


if __name__ == "__main__":
    main()
    main(name="ogbn-mag",
         batch_size=50000,
         input_features=128,
         input_dropout=0,
         inception_dim=512,
         dropout=0.5,
         lr=0.001,
         classes=349)
    main(name="ogbn-arxiv",
         batch_size=50000,
         input_features=128,
         input_dropout=0.1,
         inception_dim=512,
         classes=40,
         dropout=0.5,
         lr=0.001)
    for name, input_features, classes in zip(["PubMed", "Cora", "CiteSeer"],
                                             [500, 1433, 3703],
                                             [3, 7, 6]):
        print("name: ", name)
        main(batch_size=256,
             name=name,
             input_features=input_features,
             inception_dim=64,
             input_dropout=0.5,
             dropout=0.5,
             lr=0.01,
             classes=classes)
