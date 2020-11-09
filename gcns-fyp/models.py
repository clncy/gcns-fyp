import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder
import pytorch_lightning as pl


class ModelBase(pl.LightningModule):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hyper_params["lr"])

    def training_step(self, train_batch, batch_idx):
        y_hat = self(train_batch.x, train_batch.edge_index, train_batch.batch)  
        loss = F.mse_loss(y_hat, train_batch.y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        y_hat = self(val_batch.x, val_batch.edge_index, val_batch.batch)  
        loss = F.mse_loss(y_hat, val_batch.y)
        self.log("val_loss", loss, on_epoch=True)
        return loss


class GCN(ModelBase):
    def __init__(self, embedding_dimension, hidden_channels, representation_size, output_size, hyper_params):
        super(GCN, self).__init__()
        self.emb = AtomEncoder(embedding_dimension)
        self.conv1 = GCNConv(embedding_dimension, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, representation_size)
        self.lin = Linear(representation_size, output_size)

        self.hyper_params = hyper_params

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
