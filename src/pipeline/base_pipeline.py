import pytorch_lightning as pl
import torch as T
from torch.utils.data import DataLoader, Dataset


class BasePipeline(pl.LightningModule):
    min_delta = 1e-6

    def __init__(self, datagen, cg, ncm, batch_size=256):
        super().__init__()
        self.datagen = datagen
        self.cg = cg
        self.ncm = ncm

        self.batch_size = batch_size

    def forward(self, n=1, u=None, do={}):
        return self.ncm(n, u, do)

    def train_dataloader(self):
        return DataLoader(self.datagen, batch_size=self.batch_size, shuffle=True, drop_last=True)
